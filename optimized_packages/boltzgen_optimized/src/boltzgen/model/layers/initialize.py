"""Utility functions for initializing weights and biases."""

import math
import threading
from types import MethodDescriptorType
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
import torch
from scipy.stats import truncnorm
from boltzgen.model.layers.design_rng_advance import _native_design_uniform_advance

_deferred_design_initializers = ContextVar("boltzgen_design_initializers", default=None)
_deferred_design_uniforms = ContextVar("boltzgen_design_uniforms", default=None)
_restoring_design_uniforms = ContextVar("boltzgen_restoring_design_uniforms", default=False)
_design_uniform_advance = ContextVar("boltzgen_design_uniform_advance", default=None)
_original_uniform_method = torch.Tensor.uniform_
_uniform_method_lock = threading.RLock()
_uniform_method_users = 0


def _design_uniform_method(weights, *args, **kwargs):
    registry = _deferred_design_uniforms.get()
    advance = _design_uniform_advance.get()
    if registry is None or advance is None or _restoring_design_uniforms.get():
        return _original_uniform_method(weights, *args, **kwargs)
    a = args[0] if len(args) > 0 else kwargs.get("from", 0.0)
    b = args[1] if len(args) > 1 else kwargs.get("to", 1.0)
    eligible = (
        len(args) <= 2 and set(kwargs) <= {"from", "to", "generator"}
        and type(weights) is torch.nn.Parameter
        and not torch.is_grad_enabled() and not weights.is_inference()
        and weights.device.type == "cpu" and weights.dtype == torch.float32
        and weights.is_contiguous() and kwargs.get("generator") is None
        and isinstance(a, (int, float)) and isinstance(b, (int, float))
        and math.isfinite(a) and math.isfinite(b) and b >= a
        and abs(a) <= torch.finfo(torch.float32).max
        and abs(b) <= torch.finfo(torch.float32).max
        and not (len(args) > 0 and "from" in kwargs)
        and not (len(args) > 1 and "to" in kwargs)
    )
    if not eligible:
        return _original_uniform_method(weights, *args, **kwargs)
    rng = torch.random.get_rng_state()
    version = weights._version + 1
    _forget_deferred(weights)
    advance(weights.numel())
    torch.autograd.graph.increment_version(weights)
    registry[id(weights)] = (weights, a, b, rng, version)
    return weights


@contextmanager
def _design_uniform_method_scope():
    """Reference-counted binding; other threads without context stay eager."""
    global _uniform_method_users
    with _uniform_method_lock:
        eligible = (
            isinstance(_original_uniform_method, MethodDescriptorType)
            and torch.Tensor.uniform_ in (_original_uniform_method, _design_uniform_method)
        )
        if eligible:
            if _uniform_method_users == 0:
                torch.Tensor.uniform_ = _design_uniform_method
            _uniform_method_users += 1
    try:
        yield
    finally:
        if eligible:
            with _uniform_method_lock:
                _uniform_method_users -= 1
                if _uniform_method_users == 0 and torch.Tensor.uniform_ is _design_uniform_method:
                    torch.Tensor.uniform_ = _original_uniform_method


@contextmanager
def _defer_design_checkpoint_initialization(enabled):
    """Private, task-scoped deferral; ordinary constructors stay unchanged."""
    if not enabled:
        yield
        return
    token = _deferred_design_initializers.set({})
    advance = _native_design_uniform_advance()
    uniform_token = _deferred_design_uniforms.set({} if advance is not None else None)
    advance_token = _design_uniform_advance.set(advance)
    try:
        if advance is None:
            yield
        else:
            with _design_uniform_method_scope():
                yield
    finally:
        _design_uniform_advance.reset(advance_token)
        _deferred_design_uniforms.reset(uniform_token)
        _deferred_design_initializers.reset(token)


def _forget_deferred(weights):
    deferred = _deferred_design_initializers.get()
    if deferred is not None:
        deferred.pop(id(weights), None)
    uniforms = _deferred_design_uniforms.get()
    if uniforms is not None:
        uniforms.pop(id(weights), None)


def _restore_unloaded_design_initializers(model, checkpoint_state):
    """Materialize original values if a checkpoint will not replace them."""
    deferred = _deferred_design_initializers.get()
    if deferred is None:
        return
    names = {}
    for name, value in model.state_dict(keep_vars=True).items():
        names.setdefault(id(value), []).append(name)
    current_rng = np.random.get_state()
    try:
        for identity, (weights, shape, std, rng) in deferred.items():
            keys = names.get(identity, ())
            if keys and all(
                key in checkpoint_state and checkpoint_state[key].shape == weights.shape
                for key in keys
            ):
                continue
            np.random.set_state(rng)
            samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=_prod(shape))
            with torch.no_grad():
                weights.copy_(torch.tensor(np.reshape(samples, shape), device=weights.device))
    finally:
        np.random.set_state(current_rng)
    uniforms = _deferred_design_uniforms.get()
    if uniforms is not None:
        token = _restoring_design_uniforms.set(True)
        try:
            with torch.random.fork_rng(devices=[]):
                for identity, (weights, a, b, rng, version) in uniforms.items():
                    if weights._version != version:
                        continue
                    keys = names.get(identity, ())
                    if keys and all(
                        key in checkpoint_state and checkpoint_state[key].shape == weights.shape
                        for key in keys
                    ):
                        continue
                    torch.random.set_rng_state(rng)
                    with torch.no_grad():
                        weights.uniform_(a, b)
        finally:
            _restoring_design_uniforms.reset(token)


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    deferred = _deferred_design_initializers.get()
    if (
        deferred is not None
        and std > 0
        and truncnorm.random_state is np.random.uniform.__self__
    ):
        uniforms = _deferred_design_uniforms.get()
        if uniforms is not None:
            uniforms.pop(id(weights), None)
        # SciPy's default inverse-CDF sampler draws exactly these uniforms.
        # Preserve progression, but evaluate the CDF only for unloaded weights.
        rng = np.random.get_state()
        np.random.uniform(size=size)
        deferred[id(weights)] = (weights, shape, std, rng)
        return
    _forget_deferred(weights)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    _forget_deferred(weights)
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    _forget_deferred(weights)
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    _forget_deferred(weights)
    with torch.no_grad():
        weights.fill_(0.0)


def bias_init_zero_(bias):
    _forget_deferred(bias)
    with torch.no_grad():
        bias.fill_(0.0)


def bias_init_one_(bias):
    _forget_deferred(bias)
    with torch.no_grad():
        bias.fill_(1.0)


def normal_init_(weights):
    _forget_deferred(weights)
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    _forget_deferred(weights)
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)
