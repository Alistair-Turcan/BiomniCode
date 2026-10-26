from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

import boltz.model.layers.initialize as init
from boltz.model.layers.fused_elementwise import gate_flatten, sig_mul
from boltz.model.layers.fused_transpose import cast_transpose_bias
from boltz.model.layers.fused_attention import flash_bias_attn
from boltz.model.layers.fused_norm import layer_norm_fp32



# iter-004: bitwise-exact fused pre-softmax chain ((a*rs)+b)+m.
# torch's tensor/scalar division lowers to multiplication by the reciprocal
# (verified bitwise on real shapes); fma(a, rs, 0.0) provides the
# single-rounded multiply so LLVM cannot contract mul+add into an FMA.
# iter-010: generalized to nonsquare attention (N2 != N, BLOCK sized to
# next_pow2(N2)) and batch-broadcast bias/mask (stride-0 on dim 0 when the
# bias/mask batch is 1, deferring repeat_interleave to the eager fallback).
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _fuse_sbm_kernel(
        a_ptr, b_ptr, m_ptr, o_ptr, rs,
        sb0, sb1, sb2, sb3, sm0,
        HN, N, N2, total_rows,
        ROWS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        cmask = cols < N2
        for r in tl.static_range(ROWS):
            row = pid * ROWS + r
            rmask = cmask & (row < total_rows)
            bidx = row // HN
            rem = row % HN
            hidx = rem // N
            ridx = rem % N
            a_off = row * N2 + cols
            b_off = bidx * sb0 + hidx * sb1 + ridx * sb2 + cols * sb3
            av = tl.load(a_ptr + a_off, mask=rmask)
            bv = tl.load(b_ptr + b_off, mask=rmask)
            mv = tl.load(m_ptr + bidx * sm0 + cols, mask=rmask)
            t = tl.math.fma(av, rs, 0.0)
            t = t + bv
            t = t + mv
            tl.store(o_ptr + a_off, t, mask=rmask)

    @triton.jit
    def _fuse_sbms_kernel(
        a_ptr, b_ptr, m_ptr, o_ptr, rs,
        sb0, sb1, sb2, sb3, sm0,
        HN, N, N2, total_rows,
        ROWS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # iter-011: scale+bias+mask+softmax in one pass (row fits in registers,
        # BLOCK >= N2). Saves a full read+write of the (B,H,N,N2) fp32 matrix
        # vs the separate aten softmax. Reduction order differs from torch's
        # softmax (ulp-level), admitted under the relaxed policy.
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        cmask = cols < N2
        for r in tl.static_range(ROWS):
            row = pid * ROWS + r
            rmask = cmask & (row < total_rows)
            bidx = row // HN
            rem = row % HN
            hidx = rem // N
            ridx = rem % N
            a_off = row * N2 + cols
            b_off = bidx * sb0 + hidx * sb1 + ridx * sb2 + cols * sb3
            av = tl.load(a_ptr + a_off, mask=rmask, other=0.0)
            bv = tl.load(b_ptr + b_off, mask=rmask, other=0.0)
            mv = tl.load(m_ptr + bidx * sm0 + cols, mask=rmask, other=0.0)
            t = tl.math.fma(av, rs, 0.0)
            t = t + bv
            t = t + mv
            # exclude padding lanes from the softmax reduction
            t = tl.where(rmask, t, float("-inf"))
            rmax = tl.max(t, axis=0)
            e = tl.exp(t - rmax)
            e = tl.where(rmask, e, 0.0)
            s = tl.sum(e, axis=0)
            tl.store(o_ptr + a_off, e / s, mask=rmask)

except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _fused_scale_bias_mask(attn, bias, mask, scale, inf):
    """((attn/scale) + bias) + (1-mask)*-inf, bitwise-equal to eager; None if unsupported.

    Supports square and nonsquare (N2 != N) attention, and batch-broadcast
    bias/mask (dim-0 size 1 read with stride 0).
    """
    if (
        not _HAS_TRITON
        or attn.dtype != torch.float32
        or not attn.is_cuda
        or not attn.is_contiguous()
        or bias.dtype != torch.float32
        or mask is None
        or attn.dim() != 4
        or mask.dim() != 2
        or bias.dim() != 4
    ):
        return None
    B, H, N, N2 = attn.shape
    if N2 > 1024 or bias.shape[1:] != (H, N, N2):
        return None
    if bias.shape[0] not in (1, B) or mask.shape[0] not in (1, B):
        return None
    if mask.shape[1] != N2:
        return None
    mt = ((1 - mask[:, None, None].float()) * -inf)[:, 0, 0, :].contiguous()
    out = torch.empty_like(attn)
    total_rows = B * H * N
    ROWS = 8
    BLOCK = max(16, 1 << (N2 - 1).bit_length())
    num_warps = max(1, min(8, BLOCK // 128))
    grid = ((total_rows + ROWS - 1) // ROWS,)
    _fuse_sbm_kernel[grid](
        attn, bias, mt, out, 1.0 / scale,
        bias.stride(0) if bias.shape[0] == B else 0,
        bias.stride(1), bias.stride(2), bias.stride(3),
        mt.stride(0) if mt.shape[0] == B else 0,
        H * N, N, N2, total_rows,
        ROWS=ROWS, BLOCK=BLOCK, num_warps=num_warps,
    )
    return out

def _fused_scale_bias_mask_softmax(attn, bias, mask, scale, inf):
    """softmax((attn/scale) + bias + (1-mask)*-inf) in one kernel; None if unsupported."""
    if (
        not _HAS_TRITON
        or attn.dtype != torch.float32
        or not attn.is_cuda
        or not attn.is_contiguous()
        or bias.dtype != torch.float32
        or mask is None
        or attn.dim() != 4
        or mask.dim() != 2
        or bias.dim() != 4
    ):
        return None
    B, H, N, N2 = attn.shape
    if N2 > 1024 or bias.shape[1:] != (H, N, N2):
        return None
    if bias.shape[0] not in (1, B) or mask.shape[0] not in (1, B):
        return None
    if mask.shape[1] != N2:
        return None
    mt = ((1 - mask[:, None, None].float()) * -inf)[:, 0, 0, :].contiguous()
    out = torch.empty_like(attn)
    total_rows = B * H * N
    ROWS = 8
    BLOCK = max(16, 1 << (N2 - 1).bit_length())
    num_warps = max(1, min(8, BLOCK // 128))
    grid = ((total_rows + ROWS - 1) // ROWS,)
    _fuse_sbms_kernel[grid](
        attn, bias, mt, out, 1.0 / scale,
        bias.stride(0) if bias.shape[0] == B else 0,
        bias.stride(1), bias.stride(2), bias.stride(3),
        mt.stride(0) if mt.shape[0] == B else 0,
        H * N, N, N2, total_rows,
        ROWS=ROWS, BLOCK=BLOCK, num_warps=num_warps,
    )
    return out


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.compute_pair_bias = compute_pair_bias
        # iter-014: set by DiffusionTransformer when it hoists the
        # all-layers bias transpose; z then arrives as (B,H,N,M).
        self.z_pretransposed = False
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def _proj_z_fast(self, z: Tensor, z_ln: Optional[Tensor] = None) -> Tensor:
        """iter-029/030: fast fp32 LayerNorm inside proj_z.

        The proj_z sequential is (LayerNorm, Linear, Rearrange); the aten LN
        reaches only ~217 GB/s on the wide-batch/short-row z shape
        (1,608,608,128), so a bandwidth-rate triton LN saves ~0.2 ms/call
        (x264 layer-passes). fp32-ulp reassociation vs aten (two-pass vs
        Welford), admitted under the relaxed policy; the downstream Linear
        and its autocast behavior are untouched. iter-030: the caller may
        hand in a precomputed z_ln (from the fused add+LN+cast at the
        transition add; bitwise the same values as layer_norm_fp32(z)).
        Exact eager fallback on any guard failure.
        """
        seq = self.proj_z
        if (
            isinstance(seq, nn.Sequential)
            and len(seq) == 3
            and isinstance(seq[0], nn.LayerNorm)
            and isinstance(seq[1], nn.Linear)
            and z.dtype == torch.float32
        ):
            if (
                z_ln is not None
                and z_ln.dtype == torch.float32
                and z_ln.shape == z.shape
                and z_ln.is_contiguous()
            ):
                return seq[2](seq[1](z_ln))
            ln = layer_norm_fp32(z, seq[0].weight, seq[0].bias, seq[0].eps)
            if ln is not None:
                return seq[2](seq[1](ln))
        return seq(z)

    def _fused_qkvg_weight(self):
        """iter-018: cached [4*cs, cs] cat of proj_q/k/v/g weights with the
        proj_q bias padded by exact zeros (value-identical). Keyed by
        parameter data_ptr+version (inference-fixed)."""
        key = (
            self.proj_q.weight.data_ptr(),
            self.proj_q.weight._version,
            self.proj_k.weight.data_ptr(),
            self.proj_k.weight._version,
            self.proj_v.weight.data_ptr(),
            self.proj_v.weight._version,
            self.proj_g.weight.data_ptr(),
            self.proj_g.weight._version,
            self.proj_q.bias.data_ptr(),
            self.proj_q.bias._version,
        )
        cache = getattr(self, "_qkvg_cache", None)
        if cache is None or cache[0] != key:
            w = torch.cat(
                [
                    self.proj_q.weight,
                    self.proj_k.weight,
                    self.proj_v.weight,
                    self.proj_g.weight,
                ],
                dim=0,
            )
            z = torch.zeros(
                3 * self.c_s,
                dtype=self.proj_q.bias.dtype,
                device=self.proj_q.bias.device,
            )
            b = torch.cat([self.proj_q.bias, z])
            cache = (key, w, b)
            self._qkvg_cache = cache
        return cache[1], cache[2]

    def _fused_qg_kv_weight(self):
        """iter-018: cached [2*cs, cs] (q+g) and [2*cs, cs] (k+v) cats for
        the atom-level path where k_in is a gathered tensor distinct from s."""
        key = (
            self.proj_q.weight.data_ptr(),
            self.proj_q.weight._version,
            self.proj_k.weight.data_ptr(),
            self.proj_k.weight._version,
            self.proj_v.weight.data_ptr(),
            self.proj_v.weight._version,
            self.proj_g.weight.data_ptr(),
            self.proj_g.weight._version,
            self.proj_q.bias.data_ptr(),
            self.proj_q.bias._version,
        )
        cache = getattr(self, "_qgkv_cache", None)
        if cache is None or cache[0] != key:
            w_qg = torch.cat([self.proj_q.weight, self.proj_g.weight], dim=0)
            z = torch.zeros(
                self.c_s,
                dtype=self.proj_q.bias.dtype,
                device=self.proj_q.bias.device,
            )
            b_qg = torch.cat([self.proj_q.bias, z])
            w_kv = torch.cat([self.proj_k.weight, self.proj_v.weight], dim=0)
            cache = (key, w_qg, b_qg, w_kv)
            self._qgkv_cache = cache
        return cache[1], cache[2], cache[3]

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
        z_ln: Optional[Tensor] = None,
    ) -> Tensor:
        B = s.shape[0]
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        bias = z if getattr(self, "z_pretransposed", False) else self.proj_z(z)
        # Replication changes layout, not values. Preserve fixed-bias reuse.
        if multiplicity != 1:
            if not self.training and getattr(self, "z_pretransposed", False):
                version = None if torch.is_inference(bias) else bias._version
                key = (bias.data_ptr(), version, bias.device, bias.dtype,
                       tuple(bias.shape), tuple(bias.stride()), multiplicity)
                cache = getattr(self, "_consistent_bias_cache", None)
                if cache is not None and cache[0] == key:
                    bias = cache[2]
                else:
                    repeated = bias.repeat_interleave(multiplicity, 0)
                    self._consistent_bias_cache = (key, bias, repeated)
                    bias = repeated
            else:
                bias = bias.repeat_interleave(multiplicity, 0)
        g = self.proj_g(s).sigmoid()
        with torch.autocast("cuda", enabled=False):
            weights = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            fused = _fused_scale_bias_mask(
                weights, bias.float(), mask, self.head_dim**0.5, self.inf)
            if fused is None:
                weights = weights / (self.head_dim**0.5) + bias.float()
                weights = weights + (1-mask[:, None, None].float()) * -self.inf
            else:
                weights = fused
            weights = weights.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", weights, v.float()).to(v.dtype)
        return self.proj_o(g * o.reshape(B, -1, self.c_s))
