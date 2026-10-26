import torch
from contextvars import ContextVar
from functools import wraps
from einops.layers.torch import Rearrange
from torch import Tensor, nn
import boltzgen.model.layers.initialize as init

_design_attention_cache = ContextVar("boltzgen_design_attention_cache", default=None)


def _design_attention_scope(function):
    """Fixed diffusion conditioning is shared only for one design sampler call."""
    @wraps(function)
    def scoped(self, *args, **kwargs):
        enabled = (
            getattr(self, "_design_cuda_graphs", False)
            and not self.training
            and self.device.type == "cuda"
            and not self.score_model.activation_checkpointing
        )
        token = _design_attention_cache.set({} if enabled else None)
        try:
            return function(self, *args, **kwargs)
        finally:
            _design_attention_cache.reset(token)
    return scoped


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: int = None,
        num_heads: int = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
        use_qk_norm: bool = False,
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

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        self.compute_pair_bias = compute_pair_bias
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

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor or bias (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """

        B = s.shape[0]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        """
        TODO
        The k and v part should be done like this instead for efficiency reasons in the next version of boltz
        self.proj_kv = nn.Linear(c_s, 2*c_s, bias=False)
        kv = self.proj_kv(k_in).view(B, -1, self.num_heads, 2*self.head_dim).permute(0, 2, 1, 3)
        k,v = torch.chunk(kv, chunks=2, dim=3) # chunking (B,H,N,2C) into 2x (B,H,N,C)
        """

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cache = _design_attention_cache.get()
        if self.training or self.compute_pair_bias or torch.is_grad_enabled():
            cache = None
        key = None
        if cache is not None:
            key = (
                id(self), tuple(z.shape), tuple(z.stride()), z.dtype, z.device,
                tuple(mask.shape), mask.dtype, mask.device, multiplicity, self.inf,
                torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
            )
        attn_mask = cache.get(key) if cache is not None else None
        if attn_mask is None:
            bias = self.proj_z(z)
            bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s)
        g.sigmoid_()

        if attn_mask is None:
            attn_mask = (1 - mask[:, None, None].float()) * -self.inf
            attn_mask = attn_mask + bias.float()
            if cache is not None:
                cache[key] = attn_mask

        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            o = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attn_mask,
            )

        o = o.permute(0, 2, 1, 3).reshape(B, -1, self.c_s)
        o = o * g
        o = self.proj_o(o)
        return o

