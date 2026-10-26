#!/usr/bin/env python3
"""Fused cast + transpose for pair-bias tensors (iter-013, generalized iter-014).

Two production strided-bias patterns feed _fuse_sbms_kernel with a
non-contiguous (B, H, N, N2) fp32 view whose innermost logical dim has a
large element stride, forcing a one-sector-per-element gather (~3x penalty,
profiling/bench_sbms.py, bench_transpose.py):

  1. attentionv2.proj_z = Sequential(Linear, Rearrange("b ... h -> b h ..."))
     -> logical (B,H,N,N2) view over a physical (B,N,N2,H) buffer
     (strides (N*N2*H, 1, N2*H, H)); fp32 in the trunk/affinity pairformer.
  2. transformersv2.DiffusionTransformer precomputes an all-layers bias
     (B,N,M,L*H) and slices per layer -> (B,H,N,M) view with
     strides (N*M*L*H, 1, M*L*H, L*H) (stride-384 gather for L=24, H=16).

This kernel reads any such view coalesced along the physically-innermost
H axis (stride(1)==1 in both patterns) and writes a contiguous fp32
(B,H,N,N2) tensor, absorbing the `.float()` pass. A channel-block grid
axis keeps the tile small when H is large (the 384-channel hoist).

Numerics: bf16/fp16 -> fp32 is a widening cast (exact), fp32 -> fp32 is
the identity. Output values are bitwise-identical to `bias.float()`;
only the memory layout changes. Zero numerics risk.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _cast_transpose_kernel(
        src_ptr, o_ptr,
        N, N2, H,
        s_b, s_n, s_n2,
        BLOCK_N2: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        # src: logical (B, H, N, N2) view with unit stride on the H axis
        #      (enforced by the wrapper guard); s_b/s_n/s_n2 are the element
        #      strides of the B/N/N2 axes in the physical buffer.
        # o:   contiguous fp32 (B, H, N, N2).
        bn = tl.program_id(0)
        b = bn // N
        n = bn % N
        pn2 = tl.program_id(1)
        ph = tl.program_id(2)
        n2 = pn2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
        h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
        lmask = (n2 < N2)[:, None] & (h < H)[None, :]
        src_off = b * s_b + n * s_n + n2[:, None] * s_n2 + h[None, :]
        x = tl.load(src_ptr + src_off, mask=lmask, other=0.0)
        x = tl.trans(x.to(tl.float32))  # (BLOCK_H, BLOCK_N2)
        smask = (h < H)[:, None] & (n2 < N2)[None, :]
        dst_off = b * (H * N * N2) + h[:, None] * (N * N2) + n * N2 + n2[None, :]
        tl.store(o_ptr + dst_off, x, mask=smask)


def cast_transpose_bias(
    bias: torch.Tensor, block_n2: int = 64, block_h: int | None = None
) -> torch.Tensor:
    """Contiguous fp32 (B, H, N, N2) copy of a strided pair-bias view.

    Bitwise-identical values to `bias.float()`; returns `bias.float()`
    unchanged when the guard does not match: non-CUDA, wrong dtype/dim,
    H < 8 (atom-window shapes gain nothing), H axis not unit-stride, or
    the tensor is already contiguous (then `.float()` is already optimal).
    """
    if (
        not _HAS_TRITON
        or bias.dim() != 4
        or not bias.is_cuda
        or bias.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return bias.float()
    B, H, N, N2 = bias.shape
    # The strided gather only hurts when the gather footprint (the backing
    # storage the view walks) exceeds L2 (6 MB on A10G): (1,16,219,219) fp32
    # measures strided==contiguous, (1,16,608,608) fp32 measures 3.0x, and
    # the stride-384 diffusion slice of the (B,N,M,384) buffer measures
    # 5.6x (profiling/bench_transpose2.py).
    if (
        H < 8
        or bias.stride(1) != 1
        or bias.is_contiguous()
        or bias.untyped_storage().nbytes() <= (6 << 20)
    ):
        return bias.float()
    out = torch.empty((B, H, N, N2), dtype=torch.float32, device=bias.device)
    if out.numel() == 0:
        return out
    if block_h is None:
        block_h = min(1 << (H - 1).bit_length(), 64)  # min(next_pow2(H), 64)
    grid = (B * N, (N2 + block_n2 - 1) // block_n2, (H + block_h - 1) // block_h)
    _cast_transpose_kernel[grid](
        bias, out, N, N2, H,
        bias.stride(0), bias.stride(2), bias.stride(3),
        BLOCK_N2=block_n2, BLOCK_H=block_h,
        num_warps=4,
    )
    return out
