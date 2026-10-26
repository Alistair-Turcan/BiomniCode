from typing import Optional

import torch
from torch import Tensor, nn

from boltz.model.layers.fused_elementwise import silu_mul, swiglu

import boltz.model.layers.initialize as init
from boltz.model.layers.consistency_ops import layer_norm_cast_bf16


class Transition(nn.Module):
    """Perform a two-layer MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: Optional[int] = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        init.bias_init_one_(self.norm.weight)
        init.bias_init_zero_(self.norm.bias)

        init.lecun_normal_init_(self.fc1.weight)
        init.lecun_normal_init_(self.fc2.weight)
        init.final_init_(self.fc3.weight)

    def forward(self, x: Tensor, chunk_size: int = None) -> Tensor:
        if x.dtype == torch.float32 and not torch.is_autocast_enabled(x.device.type):
            return self._forward_reference(x, chunk_size)
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        # iter-012: fused LayerNorm(fp32)->bf16 in one pass (see
        # attention.py). Same guard semantics; eager fallback otherwise.
        if x.dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
            _f = layer_norm_cast_bf16(
                x, self.norm.weight, self.norm.bias, self.norm.eps
            )
            x = _f if _f is not None else self.norm(x).bfloat16()
        else:
            x = self.norm(x)

        if chunk_size is None or self.training:
            # iter-016: fuse fc1+fc2 into one GEMM (input read once, one
            # launch) with the concatenated weight ordered [fc2; fc1], so
            # the iter-005 swiglu kernel - silu(x[..., half:]) * x[..., :half]
            # - computes silu(fc1(x)) * fc2(x) directly on the contiguous
            # fused output (same per-op rounding order as silu_mul). Cached
            # bf16 cat keyed by parameter identity+version. Eager per-branch
            # fallback preserved for non-bf16/non-CUDA/training paths.
            # iter-019: extend the fused fc12 path to fp32 inputs when cuda
            # autocast is disabled (the diffusion-loop transitions run fp32 by
            # design); the cat is cached per input dtype. Autocast-enabled
            # fp32 (trunk) still falls through to the eager bf16-autocast
            # path - unchanged. GEMM-shape reassociation only (TF32).
            if x.is_cuda and not self.training and (
                x.dtype == torch.bfloat16
                or (x.dtype == torch.float32 and not torch.is_autocast_enabled("cuda"))
            ):
                key = (
                    x.dtype,
                    self.fc1.weight.data_ptr(),
                    self.fc1.weight._version,
                    self.fc2.weight.data_ptr(),
                    self.fc2.weight._version,
                )
                cache = getattr(self, "_fc12_cache", None)
                if cache is None or cache[0] != key:
                    with torch.autocast("cuda", enabled=False):
                        w = torch.cat(
                            [
                                self.fc2.weight.to(x.dtype),
                                self.fc1.weight.to(x.dtype),
                            ],
                            dim=0,
                        )
                    cache = (key, w)
                    self._fc12_cache = cache
                with torch.autocast("cuda", enabled=False):
                    x2 = nn.functional.linear(x, cache[1])
                _f = swiglu(x2)
                if _f is not None:
                    return self.fc3(_f)
            _a, _b = self.fc1(x), self.fc2(x)
            _f = silu_mul(_a, _b)
            x = _f if _f is not None else self.silu(_a) * _b
            x = self.fc3(x)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                _a, _b = x @ fc1_slice.T, x @ fc2_slice.T
                _f = silu_mul(_a, _b)
                x_chunk = _f if _f is not None else self.silu(_a) * _b
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out

    def _forward_reference(self, x: Tensor, chunk_size: int = None) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        x = self.norm(x)

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                x_chunk = self.silu((x @ fc1_slice.T)) * (x @ fc2_slice.T)
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out
