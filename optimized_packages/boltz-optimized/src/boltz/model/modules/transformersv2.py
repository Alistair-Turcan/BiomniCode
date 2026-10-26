# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

import torch
from torch import nn, sigmoid
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
)

from boltz.model.layers.attentionv2 import AttentionPairBias
from boltz.model.layers.fused_transpose import cast_transpose_bias
from boltz.model.layers.fused_elementwise import (
    adaln_ln_sig_mul_add,
    adaln_sig_mul_add,
    sig_mul,
    sig_mul_add,
    swiglu_mul,
)
from boltz.model.modules.utils import LinearNoBias, SwiGLU, default

# iter-027: cap on the persistent all-layers fp32 bias-transpose cache
# (DiffusionTransformer.forward). The cache scales as N^2 (567 MB at S,
# 868 MB at M, 1.85 GB at L=1098 tokens); above this cap the per-layer
# strided-view fallback (pre-iter-014 path) is used instead, trading
# L-size speed for L-size operability on 23 GB cards.
_BIAS_CT_CACHE_CAP_BYTES = 1 << 30  # 1 GiB

try:  # pragma: no cover - triton import guarded
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _adaln_mul_add_kernel(m_ptr, x_ptr, b_ptr, o_ptr, numel, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        mv = tl.load(m_ptr + offs, mask=mask)
        xv = tl.load(x_ptr + offs, mask=mask)
        bv = tl.load(b_ptr + offs, mask=mask)
        # fma(m, x, 0.0) is a single-rounded multiply that the compiler cannot
        # contract with the following add; matches eager mul-then-add bitwise.
        t = tl.math.fma(mv, xv, 0.0)
        t = t + bv
        tl.store(o_ptr + offs, t, mask=mask)

except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _fused_sig_mul_add(m, x, b):
    """Compute m*x + b with exact eager per-op rounding; None if unsupported."""
    if (
        not _HAS_TRITON
        or m.dtype != torch.float32
        or x.dtype != torch.float32
        or b.dtype != torch.float32
        or not (m.is_cuda and x.is_cuda and b.is_cuda)
        or not (m.is_contiguous() and x.is_contiguous() and b.is_contiguous())
        or m.shape != x.shape
        or m.shape != b.shape
    ):
        return None
    out = torch.empty_like(x)
    numel = x.numel()
    BLOCK = 1024
    grid = ((numel + BLOCK - 1) // BLOCK,)
    _adaln_mul_add_kernel[grid](m, x, b, out, numel, BLOCK=BLOCK, num_warps=4)
    return out


def _fused_scond_gb(layer, s, ln_pre=None):
    """iter-023: one shared no-affine LN over s plus one fused [ds, 4*dim]
    GEMM producing both adalns' scale|bias halves (layer adaln first, then
    transition adaln). The two s_norm affine weight vectors differ in the
    trained checkpoint, so they are folded into the fused GEMM weights
    (fp32 precompute, GEMM-weight-refold class, same as iter-018).
    Module-cached on the layer, keyed by data_ptr+version of every weight
    involved (inference-fixed). Returns None on guard failure; the caller
    then falls back to the per-adaln path."""
    ad1 = layer.adaln
    ad2 = layer.transition.adaln
    if s.dtype != torch.float32 or not s.is_cuda:
        return None
    if ad1.s_norm.eps != ad2.s_norm.eps:
        return None
    if ad1.s_norm.weight is None or ad2.s_norm.weight is None:
        return None
    if ad1.s_norm.bias is not None or ad2.s_norm.bias is not None:
        return None
    ws = (
        ad1.s_scale.weight,
        ad1.s_bias.weight,
        ad1.s_norm.weight,
        ad1.s_scale.bias,
        ad2.s_scale.weight,
        ad2.s_bias.weight,
        ad2.s_norm.weight,
        ad2.s_scale.bias,
    )
    if any(w.dtype != torch.float32 for w in ws):
        return None
    key = tuple((w.data_ptr(), w._version) for w in ws)
    cache = getattr(layer, "_scond_cache", None)
    if cache is None or cache[0] != key:
        dim = ad1.s_scale.out_features
        w1 = (
            torch.cat([ad1.s_scale.weight, ad1.s_bias.weight], dim=0)
            * ad1.s_norm.weight
        )
        b1 = torch.cat(
            [
                ad1.s_scale.bias,
                torch.zeros(
                    dim,
                    dtype=ad1.s_scale.bias.dtype,
                    device=ad1.s_scale.bias.device,
                ),
            ]
        )
        w2 = (
            torch.cat([ad2.s_scale.weight, ad2.s_bias.weight], dim=0)
            * ad2.s_norm.weight
        )
        b2 = torch.cat(
            [
                ad2.s_scale.bias,
                torch.zeros(
                    dim,
                    dtype=ad2.s_scale.bias.dtype,
                    device=ad2.s_scale.bias.device,
                ),
            ]
        )
        w_big = torch.cat([w1, w2], dim=0).contiguous()
        b_big = torch.cat([b1, b2], dim=0).contiguous()
        cache = (key, w_big, b_big)
        layer._scond_cache = cache
    # iter-024: ln_pre hoists the shared no-affine LN to one launch per
    # DiffusionTransformer.forward (identical across layers - same s, same
    # eps; bitwise). Fallback: per-layer LN (iter-023 behavior).
    if ln_pre is None:
        ln = nn.functional.layer_norm(s, s.shape[-1:], eps=ad1.s_norm.eps)
    else:
        ln = ln_pre
    return nn.functional.linear(ln, cache[1], cache[2])


class AdaLN(Module):
    """Algorithm 26"""

    def __init__(self, dim, dim_single_cond):
        super().__init__()
        self.a_norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = LayerNorm(dim_single_cond, bias=False)
        self.s_scale = Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def _forward_optimized(self, a, s, gb_pre=None):
        # iter-022: a_norm deferred - fused into the adaln epilogue kernel
        # below (row-wise LN stats + sigmoid-mul-add in one pass) when the
        # fp32 guard passes; the separate LN pass is the fallback.
        a_raw = a
        # iter-018: s_scale and s_bias read the same normed s - one fused
        # [ds, 2*dim] GEMM (s_scale bias + exact zeros for s_bias). The
        # strided g/bias chunks feed the iter-018 strided
        # adaln_sig_mul_add directly. GEMM-shape change only.
        dim = self.s_scale.out_features
        if gb_pre is not None:
            # iter-023: the layer-level fused s-conditioning GEMM supplied
            # this adaln's scale|bias half - skip s_norm and the sb GEMM.
            gb = gb_pre
        else:
            # iter-019: invariant-conditioning cache (atom-level layers only,
            # flag set by AtomTransformer). The fixed per-predict conditioning c
            # reaches these layers on every diffusion step, so the normed-s ->
            # fused scale|bias GEMM is recomputed identically. Keyed by
            # data_ptr+shape+strides+dtype with a strong ref (pins the storage,
            # so the key cannot be recycled while cached); bitwise-identical;
            # the gb consumers are read-only.
            _skey = None
            if getattr(self, "cache_invariant_cond", False):
                _skey = (s.data_ptr(), s.dtype) + tuple(s.shape) + tuple(s.stride())
            _sc = getattr(self, "_s_gb_cache", None)
            _shit = _skey is not None and _sc is not None and _sc[0] == _skey
            if _shit:
                s_raw = None
            else:
                s_raw = s
                s = self.s_norm(s)
            key = (
                self.s_scale.weight.data_ptr(),
                self.s_scale.weight._version,
                self.s_bias.weight.data_ptr(),
                self.s_bias.weight._version,
                self.s_scale.bias.data_ptr(),
                self.s_scale.bias._version,
            )
            cache = getattr(self, "_sb_cache", None)
            if cache is None or cache[0] != key:
                w = torch.cat([self.s_scale.weight, self.s_bias.weight], dim=0)
                b = torch.cat(
                    [
                        self.s_scale.bias,
                        torch.zeros(
                            dim,
                            dtype=self.s_scale.bias.dtype,
                            device=self.s_scale.bias.device,
                        ),
                    ]
                )
                cache = (key, w, b)
                self._sb_cache = cache
            if _shit:
                gb = _sc[2]
            else:
                gb = nn.functional.linear(s, cache[1], cache[2])
                if _skey is not None:
                    self._s_gb_cache = (_skey, s_raw, gb)
        g = gb[..., :dim]
        bias = gb[..., dim:]
        # iter-022: fused LN(a_raw) + sigmoid-mul-add in one row-wise pass
        fused = adaln_ln_sig_mul_add(g, a_raw, bias, self.a_norm.eps)
        if fused is not None:
            return fused
        a = self.a_norm(a_raw)
        # iter-010: sigmoid folded into the fused mul+add (fp32 and bf16)
        fused = adaln_sig_mul_add(g, a, bias)
        if fused is not None:
            return fused
        sig = sigmoid(g)
        fused = _fused_sig_mul_add(sig, a, bias)
        if fused is not None:
            return fused
        return sig * a + bias

    def forward(self, a, s, gb_pre=None):
        # iter-022: a_norm deferred - fused into the adaln epilogue kernel
        # below (row-wise LN stats + sigmoid-mul-add in one pass) when the
        # fp32 guard passes; the separate LN pass is the fallback.
        if not (a.dtype == torch.float32 and not torch.is_autocast_enabled(a.device.type)):
            return self._forward_optimized(a, s, gb_pre=gb_pre)
        a = self.a_norm(a)
        invariant = not self.training and getattr(self, "cache_invariant_cond", False)
        key = _consistent_conditioning_key(self, s) if invariant else None
        cache = getattr(self, "_consistent_sb_cache", None)
        if key is not None and cache is not None and cache[0] == key:
            scale, bias = cache[2], cache[3]
        else:
            normalized = self.s_norm(s)
            scale = sigmoid(self.s_scale(normalized))
            bias = self.s_bias(normalized)
            if key is not None:
                self._consistent_sb_cache = (key, s, scale, bias)
        return scale * a + bias


class ConditionedTransitionBlock(Module):
    """Algorithm 25"""

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        super().__init__()

        self.adaln = AdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        output_projection_linear = Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection_linear.weight)
        nn.init.constant_(output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def _forward_optimized(
        self,
        a,  # Float['... d']
        s,
        residual=None,
        pg=None,
        gb_pre=None,
    ):  # -> Float['... d']:
        a = self.adaln(a, s, gb_pre=gb_pre)
        # iter-018: swish_gate[0] and a_to_b read the same a - one fused
        # [dim, 3*inner] GEMM; the strided xa/ab chunks feed the iter-018
        # strided swiglu_mul. GEMM-shape change only.
        inner = self.a_to_b.out_features
        key = (
            self.swish_gate[0].weight.data_ptr(),
            self.swish_gate[0].weight._version,
            self.a_to_b.weight.data_ptr(),
            self.a_to_b.weight._version,
        )
        cache = getattr(self, "_in_cache", None)
        if cache is None or cache[0] != key:
            w = torch.cat([self.swish_gate[0].weight, self.a_to_b.weight], dim=0)
            cache = (key, w)
            self._in_cache = cache
        xab = nn.functional.linear(a, cache[1])
        xa = xab[..., : 2 * inner]
        ab = xab[..., 2 * inner :]
        # iter-010: SwiGLU interior and the outer multiply in one kernel
        _f = swiglu_mul(xa, ab)
        b = _f if _f is not None else self.swish_gate[1](xa) * ab
        # iter-018: pg may be precomputed by the layer-level fused GEMM (D)
        if pg is None:
            pg = self.output_projection[0](s)
        ba = self.b_to_a(b)
        if residual is not None:
            # iter-010: residual add folded into the gating chain
            _f = sig_mul_add(pg, ba, residual)
            a = _f if _f is not None else residual + self.output_projection[1](pg) * ba
        else:
            _f = sig_mul(pg, ba)
            a = _f if _f is not None else self.output_projection[1](pg) * ba

        return a

    def forward(
        self,
        a,  # Float['... d']
        s,
        residual=None,
        pg=None,
        gb_pre=None,
    ):  # -> Float['... d']:
        if not (a.dtype == torch.float32 and not torch.is_autocast_enabled(a.device.type)):
            return self._forward_optimized(a, s, residual=residual, pg=pg, gb_pre=gb_pre)
        a = self.adaln(a, s)
        x, gates = self.swish_gate[0](a).chunk(2, dim=-1)
        b = (nn.functional.silu(gates) * x) * self.a_to_b(a)
        result = _consistent_output_gate(self, s) * self.b_to_a(b)
        return result if residual is None else residual + result


class DiffusionTransformer(Module):
    """Algorithm 23"""

    def __init__(
        self,
        depth,
        heads,
        dim=384,
        dim_single_cond=None,
        pair_bias_attn=True,
        activation_checkpointing=False,
        post_layer_norm=False,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        dim_single_cond = default(dim_single_cond, dim)
        self.pair_bias_attn = pair_bias_attn

        self.layers = ModuleList()
        for _ in range(depth):
            self.layers.append(
                DiffusionTransformerLayer(
                    heads,
                    dim,
                    dim_single_cond,
                    post_layer_norm,
                )
            )
        if pair_bias_attn:
            # iter-014: forward() hoists the all-layers bias transpose,
            # so layers receive (B,H,N,M) slices directly.
            for layer in self.layers:
                layer.pair_bias_attn.z_pretransposed = True

    def forward(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
    ):
        if self.pair_bias_attn:
            B, N, M, D = bias.shape
            L = len(self.layers)
            # iter-014: transpose the all-layers bias to contiguous
            # (B,L,H,N,M) fp32 once per forward (bitwise-identical
            # values), replacing 24 per-layer stride-D gathers into the
            # fused sbm+softmax kernel with one coalesced pass. On guard
            # fallback the reshape still yields (B,H,N,M)-logical
            # slices and the per-call cast_transpose in attentionv2
            # engages instead.
            # iter-020: the bias is a fixed per-predict conditioning tensor
            # (diffusion_conditioning), but the cast+transpose was still
            # recomputed on every diffusion step (~1.1 GB/step at token
            # level). Cache keyed by the bias storage identity with a
            # strong ref (pins the storage, so the key cannot be recycled
            # while cached); bitwise-identical; per-layer slices are
            # read-only views. GPU-peak-neutral (the copy was already live
            # during every forward); host maxrss unaffected.
            _bkey = (bias.data_ptr(), bias.dtype) + tuple(bias.shape) + tuple(bias.stride())
            _bc = getattr(self, "_bias_ct_cache", None)
            if _bc is not None and _bc[0] == _bkey:
                bias = _bc[2]
            elif B * L * D * N * M * 4 <= _BIAS_CT_CACHE_CAP_BYTES:
                _bt = cast_transpose_bias(bias.permute(0, 3, 1, 2))
                _bt = _bt.reshape(B, L, D // L, N, M)
                self._bias_ct_cache = (_bkey, bias, _bt)
                bias = _bt
            else:
                # iter-027: size-guard the persistent all-layers fp32
                # cache - it scales as N^2 (567 MB at S, 868 MB at M,
                # 1.85 GB at L=1098) and tips the L-size predict into OOM
                # on a 23 GB card. Above the cap, serve per-layer strided
                # (B,H,N,M) views of the source bias (zero persistent
                # memory); attentionv2's cast_transpose_bias makes the
                # transient per-call contiguous fp32 copy (the pre-iter-014
                # path). Values are bitwise-identical to the cached path
                # (same source, same widening cast, same kernel).
                bias = bias.reshape(B, N, M, L, D // L).permute(0, 3, 4, 1, 2)

        # iter-024: hoist the shared no-affine s LN out of the per-layer
        # fused s-conditioning (iter-023) - identical across all layers
        # (same s, same eps), so one LN launch per forward replaces 24.
        # Guarded to token-level fp32 cuda paths with uniform eps across
        # both adalns of every layer; atom transformers
        # (cache_invariant_cond) and any guard failure fall back to the
        # per-layer LN inside _fused_scond_gb. Bitwise.
        ln_pre = None
        if (
            len(self.layers) > 0
            and not getattr(self.layers[0], "cache_invariant_cond", False)
            and s.dtype == torch.float32
            and s.is_cuda
            and torch.is_autocast_enabled(s.device.type)
        ):
            _eps = self.layers[0].adaln.s_norm.eps
            if all(
                _l.adaln.s_norm.eps == _eps
                and _l.transition.adaln.s_norm.eps == _eps
                for _l in self.layers
            ):
                ln_pre = nn.functional.layer_norm(s, s.shape[-1:], eps=_eps)

        for i, layer in enumerate(self.layers):
            if self.pair_bias_attn:
                bias_l = bias[:, i]
            else:
                bias_l = None

            if self.activation_checkpointing and self.training:
                a = torch.utils.checkpoint.checkpoint(
                    layer,
                    a,
                    s,
                    bias_l,
                    mask,
                    to_keys,
                    multiplicity,
                )

            else:
                a = layer(
                    a,  # Float['bm n d'],
                    s,  # Float['bm n ds'],
                    bias_l,  # Float['b n n dp']
                    mask,  # Bool['b n'] | None = None
                    to_keys,
                    multiplicity,
                    ln_pre=ln_pre,
                )
        return a


class DiffusionTransformerLayer(Module):
    """Algorithm 23"""

    def __init__(
        self,
        heads,
        dim=384,
        dim_single_cond=None,
        post_layer_norm=False,
    ):
        super().__init__()

        dim_single_cond = default(dim_single_cond, dim)

        self.adaln = AdaLN(dim, dim_single_cond)
        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, num_heads=heads, compute_pair_bias=False
        )

        self.output_projection_linear = Linear(dim_single_cond, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid()
        )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond
        )

        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim)
        else:
            self.post_lnorm = nn.Identity()

    def _forward_optimized(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
        ln_pre=None,
    ):
        # iter-023: consolidate token-level s-conditioning - one shared
        # no-affine LN over s plus one fused [ds, 4*dim] GEMM with the two
        # adalns' s_norm affine vectors folded into the GEMM weights (the
        # trained s_norm weight vectors DIFFER between the layer adaln and
        # the transition adaln, so the affine cannot be shared; folding is
        # the iter-018 GEMM-weight-refold class). Removes one LN launch and
        # one sb GEMM launch per layer-step. Atom-level layers
        # (cache_invariant_cond) keep the iter-019 cache path untouched.
        gb_attn = gb_trans = None
        if not getattr(self, "cache_invariant_cond", False):
            gb_all = _fused_scond_gb(self, s, ln_pre=ln_pre)
            if gb_all is not None:
                _d2 = 2 * self.adaln.s_scale.out_features
                gb_attn = gb_all[..., :_d2]
                gb_trans = gb_all[..., _d2:]
        b = self.adaln(a, s, gb_pre=gb_attn)

        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)

        if self.pair_bias_attn:
            b = self.pair_bias_attn(
                s=b,
                z=bias,
                mask=mask,
                multiplicity=multiplicity,
                k_in=k_in,
            )
        else:
            b = self.no_pair_bias_attn(s=b, mask=mask, k_in=k_in)

        # iter-018: the attention and transition output_projection[0]
        # Linears both read raw s - one fused [ds, 2*dim] GEMM at layer
        # level; the transition half is passed down via the new pg param.
        # GEMM-shape change only.
        dim = self.output_projection[0].out_features
        key = (
            self.output_projection[0].weight.data_ptr(),
            self.output_projection[0].weight._version,
            self.transition.output_projection[0].weight.data_ptr(),
            self.transition.output_projection[0].weight._version,
            self.output_projection[0].bias.data_ptr(),
            self.output_projection[0].bias._version,
            self.transition.output_projection[0].bias.data_ptr(),
            self.transition.output_projection[0].bias._version,
        )
        cache = getattr(self, "_pg_cache", None)
        if cache is None or cache[0] != key:
            w = torch.cat(
                [
                    self.output_projection[0].weight,
                    self.transition.output_projection[0].weight,
                ],
                dim=0,
            )
            b_ = torch.cat(
                [
                    self.output_projection[0].bias,
                    self.transition.output_projection[0].bias,
                ]
            )
            cache = (key, w, b_)
            self._pg_cache = cache
        # iter-019: same invariant-conditioning cache as AdaLN (atom-level
        # layers only; s is the fixed per-predict c on every diffusion step).
        _skey = None
        if getattr(self, "cache_invariant_cond", False):
            _skey = (s.data_ptr(), s.dtype) + tuple(s.shape) + tuple(s.stride())
        _sc = getattr(self, "_s_pg_cache", None)
        if _skey is not None and _sc is not None and _sc[0] == _skey:
            pg2 = _sc[2]
        else:
            pg2 = nn.functional.linear(s, cache[1], cache[2])
            if _skey is not None:
                self._s_pg_cache = (_skey, s, pg2)
        pg = pg2[..., :dim]
        pg_t = pg2[..., dim:]
        # iter-010: sigmoid + gate + residual add in one kernel
        _f = sig_mul_add(pg, b, a)
        a = _f if _f is not None else a + self.output_projection[1](pg) * b
        a = self.transition(a, s, residual=a, pg=pg_t, gb_pre=gb_trans)

        a = self.post_lnorm(a)
        return a

    def forward(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
        ln_pre=None,
    ):
        # iter-023: consolidate token-level s-conditioning - one shared
        # no-affine LN over s plus one fused [ds, 4*dim] GEMM with the two
        # adalns' s_norm affine vectors folded into the GEMM weights (the
        # trained s_norm weight vectors DIFFER between the layer adaln and
        # the transition adaln, so the affine cannot be shared; folding is
        # the iter-018 GEMM-weight-refold class). Removes one LN launch and
        # one sb GEMM launch per layer-step. Atom-level layers
        # (cache_invariant_cond) keep the iter-019 cache path untouched.
        if not (a.dtype == torch.float32 and not torch.is_autocast_enabled(a.device.type)):
            return self._forward_optimized(a, s, bias, mask, to_keys, multiplicity, ln_pre=ln_pre)
        b = self.adaln(a, s)
        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        if self.pair_bias_attn:
            b = self.pair_bias_attn(s=b, z=bias, mask=mask,
                                  multiplicity=multiplicity, k_in=k_in)
        else:
            b = self.no_pair_bias_attn(s=b, mask=mask, k_in=k_in)
        b = _consistent_output_gate(self, s) * b
        a = a + b
        a = a + self.transition(a, s)
        return self.post_lnorm(a)


class AtomTransformer(Module):
    """Algorithm 7"""

    def __init__(
        self,
        attn_window_queries,
        attn_window_keys,
        **diffusion_transformer_kwargs,
    ):
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(
            **diffusion_transformer_kwargs
        )
        # iter-019: the atom transformers are conditioned on the fixed
        # per-predict c on every diffusion step - enable the AdaLN /
        # output_projection-pair invariant-conditioning caches on all layers
        # (bitwise-identical reuse). Token-level transformers leave the flag
        # False and never allocate cache memory.
        for _layer in self.diffusion_transformer.layers:
            _layer.cache_invariant_cond = True
            _layer.adaln.cache_invariant_cond = True
            _layer.transition.adaln.cache_invariant_cond = True

    def forward(
        self,
        q,  # Float['b m d'],
        c,  # Float['b m ds'],
        bias,  # Float['b m m dp']
        to_keys,
        mask,  # Bool['b m'] | None = None
        multiplicity=1,
    ):
        W = self.attn_window_queries
        H = self.attn_window_keys

        B, N, D = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B * NW, W, -1))
        c = c.view((B * NW, W, -1))
        mask = mask.view(B * NW, W)
        if multiplicity != 1:
            # iter-018: the atom bias is a fixed conditioning tensor,
            # re-expanded every diffusion step; cache keyed by strong-ref
            # source identity + multiplicity (bitwise-identical; consumers
            # only read it). Same pattern as the iter-016 a2t cache.
            cache = getattr(self, "_bias_ri_cache", None)
            if cache is not None and cache[0] is bias and cache[1] == multiplicity:
                bias = cache[2]
            else:
                out = bias.repeat_interleave(multiplicity, 0)
                self._bias_ri_cache = (bias, multiplicity, out)
                bias = out
        bias = bias.view((bias.shape[0] * NW, W, H, -1))

        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)

        # main transformer
        q = self.diffusion_transformer(
            a=q,
            s=c,
            bias=bias,
            mask=mask.float(),
            multiplicity=1, # bias term already expanded with multiplicity
            to_keys=to_keys_new,
        )

        q = q.view((B, NW * W, D))
        return q


def _consistent_tensor_version(tensor):
    # Lightning prediction may construct immutable conditioning in inference mode.
    return None if torch.is_inference(tensor) else tensor._version


def _consistent_conditioning_key(module, s):
    """Identity/version/context key with a strong source reference in the cache."""
    return (
        s.data_ptr(), _consistent_tensor_version(s), s.device, s.dtype, tuple(s.shape), tuple(s.stride()),
        torch.is_autocast_enabled(s.device.type), torch.get_autocast_dtype(s.device.type),
        tuple((p.data_ptr(), _consistent_tensor_version(p), p.dtype) for p in module.parameters()),
    )


def _consistent_output_gate(owner, s):
    invariant = (not owner.training and (
        getattr(owner, "cache_invariant_cond", False)
        or getattr(getattr(owner, "adaln", None), "cache_invariant_cond", False)
    ))
    key = _consistent_conditioning_key(owner.output_projection, s) if invariant else None
    cache = getattr(owner, "_consistent_gate_cache", None)
    if key is not None and cache is not None and cache[0] == key:
        return cache[2]
    gate = owner.output_projection(s)
    if key is not None:
        owner._consistent_gate_cache = (key, s, gate)
    return gate
