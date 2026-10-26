#!/usr/bin/env python3
"""Fused flash-style pair-bias attention for the fp32 token path (iter-021).

The diffusion token transformer runs 24 layers x 200 diffusion steps of
fp32 self-attention at N=M=608 (S) with a (B,H,N,M) fp32 pair bias. The
iter-011..014 stack still materializes the (B,H,N,M) score tensor in HBM
twice per layer-step: the QK^T einsum writes it, the fused
scale+bias+mask+softmax kernel reads and rewrites it, and the AV einsum
reads the probabilities - ~87 MB of HBM traffic and four kernel launches
per layer-step at S (~9600 layer-steps per predict).

This kernel fuses the whole interior - QK^T, x 1/sqrt(d), + pair bias,
+ key-mask term, online softmax, AV - into one pass over K/V blocks per
query block, so scores/probabilities live in registers and never touch
HBM. K/V per head are 116 KB (fp32, N=608, D=48), L2-resident across the
query-block grid, so the classic flash-attention traffic analysis applies.

Numerics: FP32 dot precision follows torch's matmul policy (IEEE for
"highest", TF32 for explicitly reduced precision); the online
softmax is a reassociation of the iter-011 sbms softmax. Same numerics
class as the shipped kernels, admitted under the run-3 relaxed policy.
The elementwise prologue replicates the sbms kernel exactly:
fma(score, 1/scale, 0) -> + bias -> + (1-mask)*(-inf) with the module's
finite inf=1e6, so fully-masked rows behave identically (softmax over the
raw scores, including the fp32 quantization of x - 1e6).

Guards (wrapper returns None -> caller falls back to einsum + sbms):
non-CUDA, non-fp32 q/k/v, rank/shape mismatch, head_dim outside [16, 128],
bias not a logical (B|1, H, N, M) fp32 tensor, mask not (B|1, M). The
trunk bf16 path and the atom-level gathered-k path self-exclude at the
call site (q.dtype / k_in is s checks in AttentionPairBias.forward).
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
    def _flash_bias_attn_kernel(
        q_ptr, k_ptr, v_ptr, bias_ptr, mask_ptr, o_ptr,
        s_qb, s_qn, s_qh, s_qd,
        s_kb, s_km, s_kh, s_kd,
        s_vb, s_vm, s_vh, s_vd,
        s_bb, s_bh, s_bn, s_bm,
        s_mb, s_mm,
        s_ob, s_on, s_oh, s_od,
        H, N, M, D,
        inv_scale, mask_neg,
        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, D_PAD: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D_PAD)
        nmask = offs_n < N
        dmask = offs_d < D
        qmask = nmask[:, None] & dmask[None, :]

        q = tl.load(
            q_ptr + b * s_qb + h * s_qh + offs_n[:, None] * s_qn + offs_d[None, :] * s_qd,
            mask=qmask, other=0.0,
        )

        m_i = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_N,), tl.float32)
        acc = tl.zeros((BLOCK_N, D_PAD), tl.float32)

        for start_m in range(0, M, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mmask = offs_m < M
            kv_mask = mmask[:, None] & dmask[None, :]
            k = tl.load(
                k_ptr + b * s_kb + h * s_kh + offs_m[:, None] * s_km + offs_d[None, :] * s_kd,
                mask=kv_mask, other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
            # match the iter-011 sbms prologue exactly:
            # fma(score, 1/scale, 0) -> + bias -> + (1-mask)*(-inf)
            scores = tl.math.fma(scores, inv_scale, 0.0)
            bias = tl.load(
                bias_ptr + b * s_bb + h * s_bh + offs_n[:, None] * s_bn + offs_m[None, :] * s_bm,
                mask=nmask[:, None] & mmask[None, :], other=0.0,
            )
            scores = scores + bias
            mv = tl.load(mask_ptr + b * s_mb + offs_m * s_mm, mask=mmask, other=1.0)
            scores = scores + (1.0 - mv.to(tl.float32))[None, :] * mask_neg
            # exclude key-padding lanes from the softmax reduction
            scores = tl.where(mmask[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(
                v_ptr + b * s_vb + h * s_vh + offs_m[:, None] * s_vm + offs_d[None, :] * s_vd,
                mask=kv_mask, other=0.0,
            )
            acc = tl.dot(p, v, acc=acc, input_precision=INPUT_PRECISION)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(
            o_ptr + b * s_ob + h * s_oh + offs_n[:, None] * s_on + offs_d[None, :] * s_od,
            acc, mask=qmask,
        )


def flash_bias_attn(q, k, v, bias, mask, sqrt_d, inf):
    """Fused flash-style attention with pair bias; None if unsupported.

    q: (B, N, H, D) fp32 CUDA (unit-stride head dim; strided views OK).
    k, v: (B, M, H, D) fp32 CUDA, same layout contract as q.
    bias: logical (B|1, H, N, M) fp32 CUDA, arbitrary strides.
    mask: (B|1, M) CUDA, floating or bool; 1 = keep, 0 = mask.
    sqrt_d: attention scale divisor (head_dim ** 0.5).
    inf: finite mask penalty (module's self.inf, 1e6).

    Returns (B, N, H, D) fp32 contiguous, or None on guard failure.
    """
    if (
        not _HAS_TRITON
        or q.dtype != torch.float32
        or k.dtype != torch.float32
        or v.dtype != torch.float32
        or not q.is_cuda
        or q.dim() != 4
        or k.dim() != 4
        or v.dim() != 4
        or bias is None
        or mask is None
        or bias.dim() != 4
        or mask.dim() != 2
        or bias.dtype != torch.float32
        or not bias.is_cuda
        or not mask.is_cuda
    ):
        return None
    B, N, H, D = q.shape
    Bk, M, Hk, Dk = k.shape
    if (Bk, Hk, Dk) != (B, H, D) or v.shape != k.shape:
        return None
    if bias.shape[1:] != (H, N, M) or bias.shape[0] not in (1, B):
        return None
    if mask.shape[0] not in (1, B) or mask.shape[1] != M:
        return None
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return None
    if D < 16 or D > 128:
        return None
    if mask.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.bool):
        return None
    D_PAD = max(16, 1 << (D - 1).bit_length())
    BLOCK_N, BLOCK_M = 32, 64
    o = torch.empty((B, N, H, D), dtype=torch.float32, device=q.device)
    if o.numel() == 0:
        return o
    grid = ((N + BLOCK_N - 1) // BLOCK_N, B * H)
    _flash_bias_attn_kernel[grid](
        q, k, v, bias, mask, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0) if bias.shape[0] == B else 0,
        bias.stride(1), bias.stride(2), bias.stride(3),
        mask.stride(0) if mask.shape[0] == B else 0,
        mask.stride(1),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N, M, D,
        1.0 / sqrt_d, -float(inf),
        BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M, D_PAD=D_PAD,
        INPUT_PRECISION=("ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"),
        num_warps=4,
    )
    return o
