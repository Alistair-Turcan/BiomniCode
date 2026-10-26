#!/usr/bin/env python3
"""Fused LayerNorm(fp32) -> bf16 cast kernel (iter-012).

Targets the pairformer z-stream sites attention.py:145-153 and
transition.py:64-70, where an fp32 LayerNorm is immediately followed by an
explicit fp32->bf16 cast (the cast-once-and-share pattern from iter-006).
Fusing eliminates one full fp32 write + read of the (B,N,N,128) activation:
eager traffic is LN(read 4B + write 4B) + cast(read 4B + write 2B) = 14B/elem;
fused is read 4B + write 2B = 6B/elem.

Numerics: fp32 accumulation, plain two-pass mean/var. Reduction order differs
from aten's vectorized_layer_norm at ~1e-7 relative, which can flip a bf16
rounding on boundary values (<=1 bf16 ulp). Same numerics class as the
iter-006/iter-010 fused elementwise kernels; admitted under the relaxed
policy and gated by the 3-case correctness suite.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

import os

_NO_FAST_LN = bool(os.environ.get("BOLTZ_NO_FAST_LN"))


if _HAS_TRITON:

    @triton.jit
    def _ln_cast_bf16_kernel(
        x_ptr, w_ptr, b_ptr, o_ptr,
        s0, s1, s2,
        I12, I2, total_rows,
        eps,
        D: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        cmask = cols < D
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
        b = tl.load(b_ptr + cols, mask=cmask, other=0.0)
        for r in tl.static_range(ROWS):
            row = pid * ROWS + r
            rmask = row < total_rows
            i0 = row // I12
            rem = row % I12
            i1 = rem // I2
            i2 = rem % I2
            off = i0 * s0 + i1 * s1 + i2 * s2 + cols  # requires stride(-1) == 1
            m = cmask & rmask
            x = tl.load(x_ptr + off, mask=m, other=0.0)
            mean = tl.sum(x, axis=0) / D
            d = tl.where(m, x - mean, 0.0)
            var = tl.sum(d * d, axis=0) / D
            rstd = 1.0 / tl.sqrt(var + eps)
            y = d * rstd * w + b
            tl.store(o_ptr + row * D + cols, y.to(tl.bfloat16), mask=m)


def layer_norm_cast_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fused LayerNorm + bf16 cast; None when the layout/dtype is unsupported.

    Requires a 4D fp32 CUDA tensor with unit stride in the normalized dim
    (covers both the contiguous and the transposed(-2,-3) z-stream layouts).
    The output is contiguous bf16 with the same logical shape as x.
    """
    if (
        not _HAS_TRITON
        or x.dtype != torch.float32
        or not x.is_cuda
        or x.dim() != 4
        or x.stride(-1) != 1
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.numel() != x.shape[-1]
        or bias.numel() != x.shape[-1]
        or x.shape[-1] > 1024
    ):
        return None
    d = x.shape[-1]
    i0, i1, i2 = x.shape[0], x.shape[1], x.shape[2]
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    total_rows = i0 * i1 * i2
    if total_rows == 0:
        return out
    block = max(16, 1 << (d - 1).bit_length())
    rows = 8
    grid = ((total_rows + rows - 1) // rows,)
    _ln_cast_bf16_kernel[grid](
        x, weight, bias, out,
        x.stride(0), x.stride(1), x.stride(2),
        i1 * i2, i2, total_rows,
        eps,
        D=d, BLOCK=block, ROWS=rows,
        num_warps=max(1, min(8, block // 32)),
    )
    return out


if _HAS_TRITON:

    @triton.jit
    def _ln_fp32_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr, n_rows, eps,
        D: tl.constexpr, ROWS: tl.constexpr,
    ):
        # iter-029: fp32-in fp32-out LayerNorm, one program per ROWS rows.
        # Two-pass mean/var in fp32; differs from aten's Welford at fp32-ulp
        # scale (measured max 2.9e-06 on (1,608,608,128)) — the admitted
        # reassociation class, downstream of TF32 GEMMs whose 10-bit input
        # mantissa absorbs it. aten reaches only ~217 GB/s on this
        # wide-batch/short-row shape; this kernel runs near HBM bandwidth.
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, D)
        offs = rows[:, None] * D + cols[None, :]
        m = rows[:, None] < n_rows
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        mean = tl.sum(x, axis=1) / D
        xc = tl.where(m, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols)
        b = tl.load(b_ptr + cols)
        y = xc * rstd[:, None] * w[None, :] + b[None, :]
        tl.store(y_ptr + offs, y, mask=m)


def layer_norm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """iter-029: fast fp32 LayerNorm (fp32 in/out); None on guard fail.

    fp32-ulp reassociation vs aten (two-pass vs Welford), admitted under the
    relaxed policy. Requires a contiguous fp32 CUDA tensor whose normalized
    dim is a power of two in [16, 1024].
    """
    if (
        _NO_FAST_LN
        or not _HAS_TRITON
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.numel() != x.shape[-1]
        or bias.numel() != x.shape[-1]
    ):
        return None
    d = x.shape[-1]
    if d < 16 or d > 1024 or (d & (d - 1)) != 0:
        return None
    x2 = x.reshape(-1, d)
    n_rows = x2.shape[0]
    if n_rows == 0:
        return torch.empty_like(x)
    out = torch.empty_like(x2)
    rows = 8
    _ln_fp32_kernel[((n_rows + rows - 1) // rows,)](
        x2, weight, bias, out, n_rows, eps,
        D=d, ROWS=rows, num_warps=1,
    )
    return out.reshape(x.shape)


if _HAS_TRITON:

    @triton.jit
    def _add_ln_cast_kernel(
        z_ptr, u_ptr, out_ptr, b_ptr, ln_ptr, w_ptr, bb_ptr, n_rows, eps,
        D: tl.constexpr, ROWS: tl.constexpr,
    ):
        # iter-030: one pass producing (z+upd fp32, bf16 cast, fp32 LN of the
        # sum). Bitwise-identical to add_cast_bf16 followed by
        # layer_norm_fp32 on its fp32 output (the in-register fp32 add is the
        # same value the LN kernel would read back; LN math identical).
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, D)
        offs = rows[:, None] * D + cols[None, :]
        m = rows[:, None] < n_rows
        z = tl.load(z_ptr + offs, mask=m, other=0.0)
        u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
        o = z + u
        tl.store(out_ptr + offs, o, mask=m)
        tl.store(b_ptr + offs, o.to(tl.bfloat16), mask=m)
        mean = tl.sum(o, axis=1) / D
        xc = tl.where(m, o - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols)
        bb = tl.load(bb_ptr + cols)
        y = xc * rstd[:, None] * w[None, :] + bb[None, :]
        tl.store(ln_ptr + offs, y, mask=m)


def add_ln_cast_bf16(
    z: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
):
    """iter-030: fused (z+u, bf16(z+u), LayerNorm(z+u)); None on guard fail.

    Bitwise-identical to add_cast_bf16(z, u) plus layer_norm_fp32 on the
    sum. Eliminates the LN's full re-read of the fp32 sum (189 MB per call
    at (1,608,608,128)).
    """
    if (
        _NO_FAST_LN
        or not _HAS_TRITON
        or z.dtype != torch.float32
        or u.dtype != torch.bfloat16
        or not (z.is_cuda and u.is_cuda)
        or z.shape != u.shape
        or not z.is_contiguous()
        or not u.is_contiguous()
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.numel() != z.shape[-1]
        or bias.numel() != z.shape[-1]
    ):
        return None
    d = z.shape[-1]
    if d < 16 or d > 1024 or (d & (d - 1)) != 0:
        return None
    n = z.numel()
    if n == 0:
        o = z + u
        return o, o.to(torch.bfloat16), torch.nn.functional.layer_norm(
            o, (d,), weight, bias, eps
        )
    n_rows = n // d
    out = torch.empty_like(z)
    b = torch.empty(z.shape, dtype=torch.bfloat16, device=z.device)
    ln = torch.empty_like(z)
    rows = 8
    _add_ln_cast_kernel[((n_rows + rows - 1) // rows,)](
        z, u, out, b, ln, weight, bias, n_rows, eps,
        D=d, ROWS=rows, num_warps=1,
    )
    return out, b, ln


def fast_layer_norm(module: torch.nn.LayerNorm, x: torch.Tensor):
    """iter-030: drop-in fast fp32 LN for an nn.LayerNorm module.

    Returns the fused-kernel output, or None on guard failure (caller falls
    back to `module(x)`). fp32-ulp reassociation class, admitted policy.
    """
    return layer_norm_fp32(x, module.weight, module.bias, module.eps)
