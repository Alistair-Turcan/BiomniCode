import torch
from torch import Tensor, nn

import boltz.model.layers.initialize as init
from boltz.model.layers.fused_elementwise import sig_mul
from boltz.model.layers.consistency_ops import layer_norm_cast_bf16


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        """Initialize the pair weighted averaging layer.

        Parameters
        ----------
        c_m: int
            The dimension of the input sequence.
        c_z: int
            The dimension of the input pairwise tensor.
        c_h: int
            The dimension of the hidden.
        num_heads: int
            The number of heads.
        inf: float
            The value to use for masking, default 1e6.

        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self, m: Tensor, z: Tensor, mask: Tensor, chunk_heads: False = bool
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The input sequence tensor (B, S, N, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor (B, S, N, D)

        """
        # Compute layer norms
        # iter-028: fused LayerNorm(fp32)->bf16 for both norms (same guard
        # and bitwise semantics as the iter-012 sites): the LN outputs feed
        # only Linear/matmul GEMMs that autocast would cast to bf16 anyway,
        # so the GEMM inputs are bitwise-identical while the fp32 LN write
        # + separate cast read passes over z (B,N,N,128) and m (B,S,N,64)
        # are eliminated. Eager fallback preserved for unsupported layouts.
        if (
            z.dtype == torch.float32
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            _fz = layer_norm_cast_bf16(
                z, self.norm_z.weight, self.norm_z.bias, self.norm_z.eps
            )
            z = _fz if _fz is not None else self.norm_z(z).bfloat16()
        else:
            z = self.norm_z(z)
        if (
            m.dtype == torch.float32
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            _fm = layer_norm_cast_bf16(
                m, self.norm_m.weight, self.norm_m.bias, self.norm_m.eps
            )
            m = _fm if _fm is not None else self.norm_m(m).bfloat16()
        else:
            m = self.norm_m(m)

        # iter-015/016: one explicit fp32->bf16 cast of m replaces the
        # duplicate per-callsite autocast casts (proj_m and proj_g each
        # re-cast the same m). The GEMMs receive identical bf16 bits,
        # so outputs are bitwise-identical; one cast pass is saved.
        if (
            m.dtype == torch.float32
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            m = m.to(torch.bfloat16)

        if chunk_heads and not self.training:
            # Compute heads sequentially
            o_chunks = []
            for head_idx in range(self.num_heads):
                sliced_weight_proj_m = self.proj_m.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_g = self.proj_g.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, head_idx * self.c_h : (head_idx + 1) * self.c_h
                ]

                # Project input tensors
                v: Tensor = m @ sliced_weight_proj_m.T
                v = v.reshape(*v.shape[:3], 1, self.c_h)
                v = v.permute(0, 3, 1, 2, 4)

                # Compute weights
                b: Tensor = z @ sliced_weight_proj_z.T
                b = b.permute(0, 3, 1, 2)
                b = b + (1 - mask[:, None]) * -self.inf
                w = torch.softmax(b, dim=-1)

                # Compute gating (sigmoid deferred into the fused multiply)
                g: Tensor = m @ sliced_weight_proj_g.T

                # Compute output
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = o.permute(0, 2, 3, 1, 4)
                o = o.reshape(*o.shape[:3], 1 * self.c_h)
                _f = sig_mul(g, o)
                o_chunks = _f if _f is not None else g.sigmoid() * o
                if head_idx == 0:
                    o_out = o_chunks @ sliced_weight_proj_o.T
                else:
                    o_out += o_chunks @ sliced_weight_proj_o.T
            return o_out
        else:
            # Project input tensors
            v: Tensor = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: Tensor = self.proj_z(z)
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: Tensor = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            o = self.proj_o(g * o)
            return o
