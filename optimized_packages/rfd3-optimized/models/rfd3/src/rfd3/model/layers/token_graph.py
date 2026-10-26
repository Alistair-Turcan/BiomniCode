"""Token-stack graphs with exact per-invocation masks outside CUDA capture."""
from contextlib import contextmanager
from collections import OrderedDict
import os
import threading
import time
import traceback
from types import FunctionType, MethodType

import torch
from torch.nn.modules import module as module_hooks
from torch.utils._python_dispatch import _get_current_dispatch_mode

from foundry import DISABLE_CHECKPOINTING


def clear_token_graphs(module):
    module._token_graphs = []
    module._token_weight_key = None
    module._token_graph_invalidated = False
    module.__dict__.pop("_token_config_plan", None)


_WEIGHT_TRAVERSAL_METHODS = tuple(
    (name, getattr(torch.nn.Module, name)) for name in (
        "modules", "named_modules", "parameters", "named_parameters",
        "buffers", "named_buffers", "_named_members",
    )
)
_ORIGINAL_NAMED_MODULES = torch.nn.Module.named_modules


def _ordered_weights(module, children):
    """Use the current module order once, with original separate deduplication."""
    def original():
        return tuple(module.parameters()) + tuple(module.buffers())

    for name, function in _WEIGHT_TRAVERSAL_METHODS:
        method = getattr(module, name)
        if getattr(method, "__func__", method) is not function:
            return original()
    maps = []
    classes = set()
    for child in children:
        cls = type(child)
        if cls not in classes:
            if cls.__getattribute__ is not object.__getattribute__ or any(
                name in parent.__dict__ for parent in cls.__mro__
                for name in ("_modules", "_parameters", "_buffers")
            ):
                return original()
            classes.add(cls)
        method = child.named_modules
        if getattr(method, "__func__", method) is not _ORIGINAL_NAMED_MODULES:
            return original()
        registrations = tuple(child.__dict__.get(name)
                              for name in ("_modules", "_parameters", "_buffers"))
        if any(type(mapping) not in (dict, OrderedDict) for mapping in registrations):
            return original()
        maps.append(registrations)
    parameters, buffers = [], []
    seen = set()
    for modules, values, buffer_values in maps:
        for value in values.values():
            if value is not None and value not in seen:
                seen.add(value)
                parameters.append(value)
    seen = set()
    for modules, values, buffer_values in maps:
        for value in buffer_values.values():
            if value is not None and value not in seen:
                seen.add(value)
                buffers.append(value)
    return tuple(parameters) + tuple(buffers)


def _scalar(value):
    if isinstance(value, (bool, int, float, str, type(None), torch.dtype, torch.device)):
        return True
    return isinstance(value, tuple) and all(_scalar(v) for v in value)


def _tensor_key(tensor):
    return (
        id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
        tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device,
    )


def _dense_layout(tensor):
    if tensor.numel() == 0:
        return False
    expected = 1
    for stride, size in sorted(
        (stride, size) for stride, size in zip(tensor.stride(), tensor.shape) if size > 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _stable_classification(value):
    # Exact immutable builtin scalar/tuple contents cannot change their scalar
    # classification in place. Exotic tuple/subclass iteration is re-evaluated.
    if isinstance(value, tuple):
        return type(value) is tuple and all(
            type(x) in (bool, int, float, str, type(None), torch.dtype, torch.device)
            or (type(x) is tuple and _stable_classification(x)) for x in value
        )
    if _scalar(value):
        return type(value) in (bool, int, float, str, type(None), torch.dtype, torch.device)
    return type(value) in (
        dict, list, set, OrderedDict, torch.Tensor, torch.nn.Parameter,
        FunctionType, MethodType,
    )


def _current_config(module, children, private_prefix="_token_"):
    """Reuse exact tuples only after validating every current field identity."""
    identities = tuple(id(child) for child in children)
    plan_name = private_prefix + "config_plan"
    old_ids, plans = getattr(module, plan_name, (None, None))
    if old_ids != identities:
        plans = [None] * len(children)
    configs = []
    for i, child in enumerate(children):
        data = child.__dict__
        keys = tuple(k for k in data if not k.startswith(private_prefix)) if child is module else tuple(data)
        function = id(getattr(child.forward, "__func__", child.forward))
        plan = plans[i]
        if plan is not None:
            old_keys, fields, old_function, config, stable = plan
        if (
            plan is None or not stable or keys != old_keys or function != old_function
            or any(data[name] is not value for name, value in fields)
        ):
            fields = tuple((name, data[name]) for name in keys)
            values = tuple((name, value) for name, value in fields
                           if not name.startswith(private_prefix) and _scalar(value))
            config = (id(child), function, values)
            stable = all(_stable_classification(value) for name, value in fields
                         if not name.startswith(private_prefix))
            plans[i] = (keys, fields, function, config, stable)
        configs.append(config)
    setattr(module, plan_name, (identities, plans))
    return tuple(configs)


def _run_blocks(module, A, S, Z, indices, full):
    for block in module.blocks:
        A = block(A, S, Z, indices=indices, full=full)
    return A


@contextmanager
def _capture_mask(module, mask):
    # Only LocalAttentionPairBias on the capturing stream consults this private
    # attribute. Eager/other-thread calls still execute original mask creation.
    children = [block.attention_pair_bias for block in module.blocks]
    missing = object()
    previous = [child.__dict__.get("_token_graph_mask", missing) for child in children]
    try:
        for child in children:
            child.__dict__["_token_graph_mask"] = mask
        yield
    finally:
        for child, old in zip(children, previous):
            if old is missing:
                child.__dict__.pop("_token_graph_mask", None)
            else:
                child.__dict__["_token_graph_mask"] = old


def replay_token_blocks(module, A, S, Z, indices, full):
    tensors = (A, S, Z, indices)
    if (
        module.training or torch.is_grad_enabled() or full is not True
        or torch.is_inference_mode_enabled() or torch.compiler.is_compiling()
        or torch.jit.is_tracing() or not len(module.blocks)
        or threading.current_thread() is not threading.main_thread()
        or not all(type(t) is torch.Tensor and t.is_cuda for t in tensors)
        or any(t.device != A.device or not _dense_layout(t) for t in tensors)
        or A.ndim != 3 or S.ndim != 3 or Z.ndim != 4
        or indices.ndim != 3 or indices.dtype != torch.int64
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
        or _get_current_dispatch_mode() is not None
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
        or os.environ.get("RFD3_CUDA_GRAPH_TOKEN_STACK", "1") == "0"
        or os.environ.get("RFD3_LOW_MEMORY_MODE", "0") == "1"
    ):
        module._token_graphs = []
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    children = tuple(module.modules())
    if any(child.training or child._forward_hooks or child._forward_pre_hooks
           or child._backward_hooks or child._backward_pre_hooks for child in children):
        module._token_graphs = []
        return None
    if any((
        module_hooks._global_forward_hooks, module_hooks._global_forward_pre_hooks,
        module_hooks._global_backward_hooks, module_hooks._global_backward_pre_hooks,
    )):
        module._token_graphs = []
        return None
    weights = _ordered_weights(module, children)
    if any(w.requires_grad or w.device != A.device for w in weights):
        module._token_graphs = []
        return None
    try:
        weight_key = tuple(_tensor_key(w) for w in weights)
    except RuntimeError:
        module._token_graphs = []
        return None
    old = getattr(module, "_token_weight_key", None)
    if old is not None and old != weight_key:
        module._token_graph_invalidated = True
        module._token_graphs = []
    module._token_weight_key = weight_key
    if getattr(module, "_token_graph_invalidated", False):
        return None
    for block in module.blocks:
        block.attention_pair_bias.use_checkpointing = not DISABLE_CHECKPOINTING
    from rfd3.model.layers import attention
    heads = tuple(block.attention_pair_bias.n_head for block in module.blocks)
    dense_bytes = 4 * A.shape[0] * max(heads) * A.shape[1] ** 2
    input_bytes = sum(t.numel() * t.element_size() for t in tensors)
    reserve = 2 * input_bytes + 6 * dense_bytes + 64 * A.numel() * 4
    free_bytes, _ = torch.cuda.mem_get_info(A.device)
    if free_bytes <= reserve:
        module._token_graphs = []
        return None
    stream = torch.cuda.current_stream(A.device)
    config = _current_config(module, children)
    context = (
        torch.is_autocast_cache_enabled(), torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32, stream.cuda_stream,
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        os.environ.get("RFD3_DENSE_SDPA_ATTENTION", "1"),
        id(attention.use_dense_sdpa_pairbias),
        id(attention.dense_sdpa_pairbias_attention),
        id(attention.sparse_pairbias_attention),
        id(attention.pairbias_attention_), id(attention.indices_to_mask),
    )
    key = (
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors),
        weight_key, config, context, full,
    )
    # Exactly the original operation on current dynamic neighbors, once per
    # invocation, not inside capture and not memoized across invocations.
    current_mask = attention.indices_to_mask(indices)
    entries = getattr(module, "_token_graphs", [])
    stats = getattr(module, "_token_graph_stats", None)
    if stats is None:
        stats = {"captures": 0, "replays": 0, "capture_sec": 0.0, "failures": []}
        module._token_graph_stats = stats
    for entry_key, graph, buffers, result, references, mask_buffer in entries:
        if entry_key == key:
            for target, source in zip(buffers, tensors):
                target.copy_(source)
            mask_buffer.copy_(current_mask)
            graph.replay()
            output = result.clone()
            for t in (*buffers, mask_buffer, result):
                t.record_stream(stream)
            stats["replays"] += 1
            return output
    start = time.perf_counter()
    buffers = tuple(t.clone(memory_format=torch.preserve_format) for t in tensors)
    if any(target.stride() != source.stride() for target, source in zip(buffers, tensors)):
        return None
    mask_buffer = current_mask.clone()
    capture_stream = torch.cuda.Stream(device=A.device)
    capture_stream.wait_stream(stream)
    try:
        with torch.cuda.stream(capture_stream), torch.autocast(
            "cuda", dtype=torch.bfloat16, cache_enabled=False,
        ):
            for _ in range(2):
                _run_blocks(module, *buffers, full)
        stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with _capture_mask(module, mask_buffer), torch.cuda.graph(
            graph, stream=capture_stream,
        ), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            result = _run_blocks(module, *buffers, full)
        stream.wait_stream(capture_stream)
    except RuntimeError as error:
        stream.wait_stream(capture_stream)
        module._token_graph_invalidated = True
        module._token_graphs = []
        stats["failures"].append(str(error))
        stats["capture_traceback"] = traceback.format_exc()
        stats["capture_sec"] += time.perf_counter() - start
        return None
    graph.replay()
    output = result.clone()
    for t in (*buffers, mask_buffer, result):
        t.record_stream(stream)
    module._token_graphs = (
        entries + [(key, graph, buffers, result, weights, mask_buffer)]
    )[-2:]
    stats["captures"] += 1
    stats["replays"] += 1
    stats["capture_sec"] += time.perf_counter() - start
    return output
