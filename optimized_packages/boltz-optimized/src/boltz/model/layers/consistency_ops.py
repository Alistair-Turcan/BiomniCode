"""Measured upstream-consistent normalization without global fusion disabling."""
import torch


def layer_norm_cast_bf16(x, weight, bias, eps):
    if x.dtype != torch.float32 or not x.is_cuda:
        return None
    return torch.nn.functional.layer_norm(
        x, (x.shape[-1],), weight, bias, eps).bfloat16()
