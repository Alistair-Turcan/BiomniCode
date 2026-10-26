"""Mapper-local eager CPU defaults, without importing compiler-aware optimizers.

This is not a general optimizer API. Its elementary operations and lazy state
match PyTorch 2.7.1 single-tensor Adam with the mapper's fixed default options.
"""
import torch


class _EagerCPUAdam:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    if parameter.grad.grad_fn is not None:
                        parameter.grad.detach_()
                    else:
                        parameter.grad.requires_grad_(False)
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError(
                    "Adam does not support sparse gradients, please consider SparseAdam instead")
            if parameter not in self.state:
                scalar_dtype = (torch.float64 if torch.get_default_dtype() == torch.float64
                                else torch.float32)
                self.state[parameter] = {
                    "step": torch.tensor(0.0, dtype=scalar_dtype),
                    "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(parameter, memory_format=torch.preserve_format),
                }
            state = self.state[parameter]
            step = state["step"]
            step += 1
            count = step.item()
            mean, square = state["exp_avg"], state["exp_avg_sq"]
            from ._cpu_update import _moments, _update
            if not _moments(mean, square, gradient):
                mean.lerp_(gradient, 1 - 0.9)
                square.mul_(0.999).addcmul_(gradient, gradient, value=1 - 0.999)
            step_size = self.lr / (1 - 0.9 ** count)
            correction = (1 - 0.999 ** count) ** 0.5
            denominator = square.sqrt()
            if _update(parameter, mean, denominator, correction, -step_size):
                continue
            # sqrt returns an owned temporary; dividing in place preserves the
            # same native operations without allocating a second large tensor.
            denominator.div_(correction).add_(1e-8)
            parameter.addcdiv_(mean, denominator, value=-step_size)
        return None


def _make_adam(parameters, learning_rate):
    supported = (
        torch.__version__.split("+")[0] == "2.7.1"
        and not torch.compiler.is_compiling()
        and isinstance(learning_rate, (int, float)) and 0 <= learning_rate
        and all(p.device.type == "cpu" and p.dtype in (torch.float32, torch.float64)
                for p in parameters))
    if supported:
        return _EagerCPUAdam(parameters, learning_rate)
    return torch.optim.Adam(parameters, lr=learning_rate)
