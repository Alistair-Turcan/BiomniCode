"""Fused elementwise kernels (run 3, iter-005).

Each kernel replaces a two-kernel eager chain (silu/sigmoid then mul, or the
chunked SwiGLU pattern) with a single pass, eliminating one intermediate
tensor round-trip and one kernel launch per call.

Numerics: compute is fp32 internally with the same per-op rounding order as
the eager chain (the activation is rounded to the output dtype before the
multiply, matching the eager intermediate tensor). tl.sigmoid can differ from
torch's eager sigmoid by ~1 ulp (measured <=2e-5 relative), which is admitted
under the run-3 relaxed numerical policy. All functions return None when the
inputs are unsupported, so callers keep the exact eager fallback.
"""
from __future__ import annotations

import os

import torch

_NO_ADD_CAST = bool(os.environ.get("BOLTZ_NO_ADD_CAST"))

try:  # pragma: no cover - triton import guarded
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _silu_mul_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(a_ptr + offs, mask=m).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=m).to(tl.float32)
        s = a * tl.sigmoid(a)
        # round like the eager intermediate silu output, then multiply
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + offs, (s * b).to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _sig_mul_kernel(a_ptr, b_ptr, o_ptr, n, D, sa_row, BLOCK: tl.constexpr):
        # iter-018: a may be a last-dim chunk view of a wider fused
        # projection (stride(-1)==1, uniform row stride sa_row); the
        # contiguous case is the same formula with sa_row = D.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        row = offs // D
        col = offs % D
        a = tl.load(a_ptr + row * sa_row + col, mask=m).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=m).to(tl.float32)
        s = tl.sigmoid(a)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + offs, (s * b).to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _sig_mul_add_kernel(g_ptr, t_ptr, r_ptr, o_ptr, n, D, sg_row, BLOCK: tl.constexpr):
        # out = r + sigmoid(g) * t  (eager: sigmoid -> mul -> add chain)
        # iter-018: g may be a last-dim chunk view (uniform row stride
        # sg_row); contiguous g is the same formula with sg_row = D.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        row = offs // D
        col = offs % D
        g = tl.load(g_ptr + row * sg_row + col, mask=m).to(tl.float32)
        t = tl.load(t_ptr + offs, mask=m).to(tl.float32)
        r = tl.load(r_ptr + offs, mask=m).to(tl.float32)
        s = tl.sigmoid(g)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        p = (s * t).to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + offs, (r + p).to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _adaln_sig_mul_add_kernel(g_ptr, a_ptr, b_ptr, o_ptr, n, D, sg_row, sb_row, BLOCK: tl.constexpr):
        # out = sigmoid(g) * a + b  (AdaLN: sigmoid then fused mul+add)
        # iter-018: g and b may be last-dim chunk views of one fused GEMM
        # (uniform row strides sg_row/sb_row); contiguous is sg_row = D.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        row = offs // D
        col = offs % D
        g = tl.load(g_ptr + row * sg_row + col, mask=m).to(tl.float32)
        a = tl.load(a_ptr + offs, mask=m).to(tl.float32)
        b = tl.load(b_ptr + row * sb_row + col, mask=m).to(tl.float32)
        s = tl.sigmoid(g)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        p = (s * a).to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + offs, (p + b).to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _gate_flatten_kernel(
        o_ptr, g_ptr, out_ptr,
        s_lead, sQ, sH, sC,
        sg_lead, sg_q,
        QHC, HC, C, total,
        BLOCK: tl.constexpr,
    ):
        # primitives.py Attention._wrap_up fusion:
        #   g = sigmoid(linear_g(q_x)).view(..., Q, H, C); o = o * g; flatten(o)
        # o is the transposed attention output read through its own strides;
        # out is contiguous (..., Q, H*C). g (the raw linear_g output) may be
        # contiguous or a last-dim chunk view of a wider contiguous tensor
        # (iter-016 qkvg fusion): it then has stride(-1)==1 with row stride
        # sg_q and leading stride sg_lead. Contiguous g is the same formula
        # with sg_q = H*C, sg_lead = Q*H*C.
        #   out[..., q, h*C+c] = o[..., q, h, c] * sigmoid(g[..., q, h*C+c])
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        lead = offs // QHC
        rem = offs % QHC
        q = rem // HC
        rem2 = rem % HC
        h = rem2 // C
        c = rem2 % C
        o_off = lead * s_lead + q * sQ + h * sH + c * sC
        g_off = lead * sg_lead + q * sg_q + rem2
        ov = tl.load(o_ptr + o_off, mask=m, other=0.0).to(tl.float32)
        gv = tl.load(g_ptr + g_off, mask=m, other=0.0).to(tl.float32)
        s = tl.sigmoid(gv)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        p = (ov * s).to(o_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, p, mask=m)

    @triton.jit
    def _swiglu_mul_kernel(x_ptr, t_ptr, o_ptr, n_out, half, sx_row, st_row, BLOCK: tl.constexpr):
        # out = SwiGLU(x) * t = (silu(x[..., half:]) * x[..., :half]) * t
        # iter-018: x and t may be last-dim chunk views of one fused
        # projection (uniform row strides sx_row/st_row); contiguous is
        # sx_row = 2*half, st_row = half.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_out
        row = offs // half
        col = offs % half
        src = row * sx_row + col
        xv = tl.load(x_ptr + src, mask=m).to(tl.float32)
        gates = tl.load(x_ptr + src + half, mask=m).to(tl.float32)
        t = tl.load(t_ptr + row * st_row + col, mask=m).to(tl.float32)
        s = gates * tl.sigmoid(gates)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        p = (s * xv).to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + offs, (p * t).to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_kernel(x_ptr, o_ptr, n_out, half, BLOCK: tl.constexpr):
        # SwiGLU on a contiguous (..., 2*half) tensor without materializing
        # the two strided chunk views. Source convention (utils.py SwiGLU):
        #   x, gates = x.chunk(2, -1); return silu(gates) * x
        # i.e. the SECOND half is the gate, the FIRST half is the content:
        #   out = silu(x[..., half:]) * x[..., :half]
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_out
        row = offs // half
        col = offs % half
        src = row * (2 * half) + col
        xv = tl.load(x_ptr + src, mask=m).to(tl.float32)  # first half: content
        gates = tl.load(x_ptr + src + half, mask=m).to(tl.float32)  # second: gate
        s = gates * tl.sigmoid(gates)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + offs, (s * xv).to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _add_cast_bf16_kernel(z_ptr, u_ptr, out_ptr, b_ptr, n, BLOCK: tl.constexpr):
        # iter-029: z_new = z + upd (fp32 add of exactly-widened bf16 upd,
        # identical to eager type promotion) plus the RNE fp32->bf16 cast of
        # the result in the same pass, so the consumer (tri_mul kernel entry)
        # skips its own x.to(bfloat16). Bitwise vs eager add + .to(bf16)
        # (prototype: torch.equal on both outputs).
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        z = tl.load(z_ptr + offs, mask=m)
        u = tl.load(u_ptr + offs, mask=m).to(tl.float32)
        o = z + u
        tl.store(out_ptr + offs, o, mask=m)
        tl.store(b_ptr + offs, o.to(tl.bfloat16), mask=m)

except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _pair_ok(a, b):
    return (
        _HAS_TRITON
        and a.is_cuda
        and b.is_cuda
        and a.dtype == b.dtype
        and a.dtype in (torch.float32, torch.bfloat16)
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape == b.shape
    )


def _chunk_row_stride(t, last_dim):
    """Uniform row stride for a (..., last_dim) tensor that is contiguous or
    a last-dim chunk view of a contiguous wider row (stride(-1)==1 and the
    leading dims collapse); None otherwise. iter-018."""
    if not _HAS_TRITON or t.dim() < 1 or t.stride(-1) != 1:
        return None
    s = t.stride()
    sh = t.shape
    for i in range(len(s) - 2):
        if s[i] != s[i + 1] * sh[i + 1]:
            return None
    return s[-2] if t.dim() >= 2 else last_dim


def silu_mul(a, b):
    """silu(a) * b in one kernel; None if unsupported."""
    if not _pair_ok(a, b):
        return None
    out = torch.empty_like(a)
    n = a.numel()
    _silu_mul_kernel[((n + 1023) // 1024,)](a, b, out, n, BLOCK=1024, num_warps=4)
    return out


def sig_mul(a, b):
    """sigmoid(a) * b in one kernel; None if unsupported.

    iter-018: a may be a last-dim chunk view of a wider fused projection
    (uniform row stride); b stays strictly contiguous.
    """
    if not (
        _HAS_TRITON
        and a.is_cuda
        and b.is_cuda
        and a.dtype == b.dtype
        and a.dtype in (torch.float32, torch.bfloat16)
        and b.is_contiguous()
        and a.shape == b.shape
    ):
        return None
    sa = _chunk_row_stride(a, a.shape[-1])
    if sa is None:
        return None
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    n = a.numel()
    _sig_mul_kernel[((n + 1023) // 1024,)](
        a, b, out, n, a.shape[-1], sa, BLOCK=1024, num_warps=4
    )
    return out


def _triple_ok(a, b, c):
    return _pair_ok(a, b) and (
        c.is_cuda and c.dtype == a.dtype and c.is_contiguous() and c.shape == a.shape
    )


def sig_mul_add(g, t, r):
    """r + sigmoid(g) * t in one kernel; None if unsupported.

    iter-018: g may be a last-dim chunk view (uniform row stride); t and r
    stay strictly contiguous.
    """
    if not (
        _HAS_TRITON
        and g.is_cuda
        and t.is_cuda
        and r.is_cuda
        and g.dtype == t.dtype
        and g.dtype == r.dtype
        and g.dtype in (torch.float32, torch.bfloat16)
        and t.is_contiguous()
        and r.is_contiguous()
        and g.shape == t.shape
        and g.shape == r.shape
    ):
        return None
    sg = _chunk_row_stride(g, g.shape[-1])
    if sg is None:
        return None
    out = torch.empty(g.shape, dtype=g.dtype, device=g.device)
    n = g.numel()
    _sig_mul_add_kernel[((n + 1023) // 1024,)](
        g, t, r, out, n, g.shape[-1], sg, BLOCK=1024, num_warps=4
    )
    return out


def adaln_sig_mul_add(g, a, b):
    """sigmoid(g) * a + b in one kernel; None if unsupported.

    iter-018: g and b may be last-dim chunk views of one fused GEMM
    (uniform row strides); a stays strictly contiguous.
    """
    if not (
        _HAS_TRITON
        and g.is_cuda
        and a.is_cuda
        and b.is_cuda
        and g.dtype == a.dtype
        and g.dtype == b.dtype
        and g.dtype in (torch.float32, torch.bfloat16)
        and a.is_contiguous()
        and g.shape == a.shape
        and g.shape == b.shape
    ):
        return None
    D = a.shape[-1]
    sg = _chunk_row_stride(g, D)
    sb = _chunk_row_stride(b, D)
    if sg is None or sb is None:
        return None
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    n = a.numel()
    _adaln_sig_mul_add_kernel[((n + 1023) // 1024,)](
        g, a, b, out, n, D, sg, sb, BLOCK=1024, num_warps=4
    )
    return out


def gate_flatten(o, g_lin):
    """flatten(o * sigmoid(g_lin)) for Attention._wrap_up; None if unsupported.

    o: (..., Q, H, C) tensor with arbitrary strides (the transposed attention
    output); g_lin: (..., Q, H*C) raw linear_g output - contiguous, or
    (iter-016) a last-dim chunk view of a wider contiguous fused projection
    (stride(-1)==1, arbitrary row stride). Returns a contiguous (..., Q, H*C)
    tensor, fusing the sigmoid, the gating multiply, and the
    flatten_final_dims copy into one pass.
    """
    if not (
        _HAS_TRITON
        and o.is_cuda
        and g_lin.is_cuda
        and g_lin.stride(-1) == 1
        and o.dtype == g_lin.dtype
        and o.dtype in (torch.float32, torch.bfloat16)
        and o.dim() >= 3
    ):
        return None
    lead, Q, H, C = o.shape[:-3], o.shape[-3], o.shape[-2], o.shape[-1]
    if g_lin.shape != tuple(lead) + (Q, H * C):
        return None
    # leading dims must collapse to a single stride (true for the transposed
    # view: leading strides form a contiguous chain)
    s = o.stride()
    for i in range(len(lead) - 1):
        if s[i] != s[i + 1] * o.shape[i + 1]:
            return None
    s_lead = s[len(lead) - 1] if lead else 0
    # same collapse requirement on g's leading dims; its row stride may
    # exceed H*C when it is a chunk view of the fused qkvg projection
    sg = g_lin.stride()
    for i in range(len(lead) - 1):
        if sg[i] != sg[i + 1] * g_lin.shape[i + 1]:
            return None
    sg_lead = sg[len(lead) - 1] if lead else 0
    sg_q = sg[-2]
    out = torch.empty(tuple(lead) + (Q, H * C), dtype=o.dtype, device=o.device)
    total = out.numel()
    _gate_flatten_kernel[((total + 1023) // 1024,)](
        o, g_lin, out,
        s_lead, o.stride(-3), o.stride(-2), o.stride(-1),
        sg_lead, sg_q,
        Q * H * C, H * C, C, total,
        BLOCK=1024, num_warps=4,
    )
    return out


def swiglu_mul(x, t):
    """SwiGLU(x) * t in one kernel; None if unsupported.

    iter-018: x and t may be last-dim chunk views of one fused projection
    (uniform row strides).
    """
    half = x.shape[-1] // 2 if (x.dim() > 0 and x.shape[-1] % 2 == 0) else 0
    if not (
        _HAS_TRITON
        and x.is_cuda
        and x.dtype in (torch.float32, torch.bfloat16)
        and half > 0
        and t.is_cuda
        and t.dtype == x.dtype
        and t.shape == x.shape[:-1] + (half,)
    ):
        return None
    sx = _chunk_row_stride(x, 2 * half)
    st = _chunk_row_stride(t, half)
    if sx is None or st is None:
        return None
    out = torch.empty(x.shape[:-1] + (half,), dtype=x.dtype, device=x.device)
    n_out = out.numel()
    _swiglu_mul_kernel[((n_out + 1023) // 1024,)](
        x, t, out, n_out, half, sx, st, BLOCK=1024, num_warps=4
    )
    return out


def swiglu(x):
    """SwiGLU(x) = silu(x[..., d//2:]) * x[..., :d//2] in one kernel
    (source convention: x, gates = x.chunk(2, -1); silu(gates) * x)."""
    if not (
        _HAS_TRITON
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float32, torch.bfloat16)
        and x.shape[-1] % 2 == 0
    ):
        return None
    half = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (half,), dtype=x.dtype, device=x.device)
    n_out = out.numel()
    _swiglu_kernel[((n_out + 1023) // 1024,)](x, out, n_out, half, BLOCK=1024, num_warps=4)
    return out
if _HAS_TRITON:

    @triton.jit
    def _adaln_ln_sig_mul_add_kernel(
        g_ptr, a_ptr, b_ptr, o_ptr, D, eps, sg_row, sb_row,
        D_PAD: tl.constexpr,
    ):
        # iter-022: out = sigmoid(g) * LayerNorm(a) + b in one row-wise
        # pass, fusing the AdaLN a_norm. Biased variance, eps-matched to
        # nn.LayerNorm; fp32 stats (tree-reduction reassociation vs
        # torch's block reduction is ulp-level, admitted under the relaxed
        # policy); then the iter-010 per-op rounding chain. g/b may be
        # last-dim chunk views (row strides sg_row/sb_row). One program
        # per row.
        row = tl.program_id(0)
        cols = tl.arange(0, D_PAD)
        m = cols < D
        a = tl.load(a_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
        mean = tl.sum(a, 0) / D
        d = tl.where(m, a - mean, 0.0)
        var = tl.sum(d * d, 0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        a_ln = d * rstd
        g = tl.load(g_ptr + row * sg_row + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * sb_row + cols, mask=m, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        s = s.to(o_ptr.dtype.element_ty).to(tl.float32)
        p = (s * a_ln).to(o_ptr.dtype.element_ty).to(tl.float32)
        tl.store(o_ptr + row * D + cols, (p + b).to(o_ptr.dtype.element_ty), mask=m)


def adaln_ln_sig_mul_add(g, a, b, eps):
    """sigmoid(g) * LayerNorm(a, affine=False, bias=False) + b fused.

    a: (..., D) fp32 CUDA contiguous. g, b: (..., D) fp32 CUDA, unit-stride
    last dim (last-dim chunk views OK). Returns None on guard failure.
    """
    if (
        not _HAS_TRITON
        or a.dtype != torch.float32
        or g.dtype != torch.float32
        or b.dtype != torch.float32
        or not (a.is_cuda and g.is_cuda and b.is_cuda)
        or not a.is_contiguous()
        or a.dim() < 2
        or a.shape != g.shape
        or a.shape != b.shape
        or g.stride(-1) != 1
        or b.stride(-1) != 1
    ):
        return None
    D = a.shape[-1]
    if D < 16 or D > 4096:
        return None
    D_PAD = max(16, 1 << (D - 1).bit_length())
    n_rows = a.numel() // D
    out = torch.empty_like(a)
    _adaln_ln_sig_mul_add_kernel[(n_rows,)](
        g, a, b, out, D, eps, g.stride(-2), b.stride(-2),
        D_PAD=D_PAD, num_warps=4,
    )
    return out


def add_cast_bf16(z: torch.Tensor, u: torch.Tensor):
    """iter-029: fused (z + u, bf16(z + u)) in one pass; None on guard fail.

    Bitwise-identical to `(z + u, (z + u).to(torch.bfloat16))`: the add is
    fp32 with the bf16 update exactly widened (eager type promotion), and the
    cast is round-to-nearest-even like aten. Fusing eliminates the separate
    cast kernel's full re-read of the fp32 sum (189 MB per call at
    (1,608,608,128)).
    """
    if (
        _NO_ADD_CAST
        or not _HAS_TRITON
        or z.dtype != torch.float32
        or u.dtype != torch.bfloat16
        or not (z.is_cuda and u.is_cuda)
        or z.shape != u.shape
        or not z.is_contiguous()
        or not u.is_contiguous()
    ):
        return None
    n = z.numel()
    if n == 0:
        return z + u, (z + u).to(torch.bfloat16)
    out = torch.empty_like(z)
    b = torch.empty(z.shape, dtype=torch.bfloat16, device=z.device)
    BLOCK = 4096
    _add_cast_bf16_kernel[((n + BLOCK - 1) // BLOCK,)](
        z, u, out, b, n, BLOCK=BLOCK, num_warps=8
    )
    return out, b
