"""Complete uniform-map decoder graphs; original downcast and conditioning math."""
from contextlib import contextmanager
import os
import threading
import time
import traceback

from einops import rearrange
import torch
from torch.nn.modules import module as module_hooks
from torch.utils._python_dispatch import _get_current_dispatch_mode
from rfd3.model.layers.token_graph import (
    _current_config, _dense_layout, _tensor_key, _ordered_weights,
)
from rfd3.model.layers import blocks

_PACKING_FUNCTIONS = (
    blocks.Upcast.forward, blocks.Upcast.forward_,
    blocks.Downcast.forward, blocks.Downcast.forward_,
    blocks.build_valid_mask, blocks.ungroup_atoms, blocks.group_atoms,
)


def clear_decoder_uniform_graphs(module):
    module._decoder_uniform_graphs = []
    module._decoder_uniform_weight_key = None
    module._decoder_uniform_invalidated = False
    module.__dict__.pop("_decoder_uniform_config_plan", None)


def _original_packing(module):
    current = (
        blocks.Upcast.forward, blocks.Upcast.forward_,
        blocks.Downcast.forward, blocks.Downcast.forward_,
        blocks.build_valid_mask, blocks.ungroup_atoms, blocks.group_atoms,
    )
    return (
        type(module) is blocks.CompactStreamingDecoder
        and type(module.downcast) is blocks.Downcast
        and all(type(up) is blocks.Upcast for up in module.upcast)
        and all(a is b for a, b in zip(current, _PACKING_FUNCTIONS))
    )


@contextmanager
def _capture_all_valid(module):
    children = [up.gca for up in module.upcast]
    if module.downcast.method == "cross_attention":
        children.append(module.downcast.gca)
    missing = object()
    previous = [m.__dict__.get("_decoder_uniform_all_queries_valid", missing)
                for m in children]
    try:
        for child in children:
            child.__dict__["_decoder_uniform_all_queries_valid"] = True
        yield
    finally:
        for child, old in zip(children, previous):
            if old is missing:
                child.__dict__.pop("_decoder_uniform_all_queries_valid", None)
            else:
                child.__dict__["_decoder_uniform_all_queries_valid"] = old


def _run_decoder_steps(module, A, Q, C, P, indices, S, masks):
    B, L, c = Q.shape
    I = A.shape[1]
    for up, atom, mask in zip(module.upcast, module.atom_transformer, masks[:-1]):
        Q_IA = Q.reshape(B, I, 14, c)
        tokens = rearrange(A, "b n (s c) -> b n s c", s=up.n_split)
        Q_IA = Q_IA + up.gca(q=Q_IA, kv=tokens, attn_mask=mask)
        Q = Q_IA.reshape(B, L, c)
        Q = atom(Q, C, P, indices=indices)
    # Exactly the original downcast forward_ operations, including process_s.
    final = module.downcast.forward_(
        Q.detach().reshape(B, I, 14, c), A.detach(), S.detach(),
        valid_mask=masks[-1],
    )
    return final, Q


def replay_decoder_uniform(module, A, Q, C, P, tok_idx, indices, chunked, S):
    tensors = (A, Q, C, P, indices, S)
    if (
        chunked is not None or module.training or torch.is_grad_enabled()
        or torch.is_inference_mode_enabled() or torch.compiler.is_compiling()
        or torch.jit.is_tracing() or module.n_blocks != 3
        or threading.current_thread() is not threading.main_thread()
        or not all(type(t) is torch.Tensor and t.is_cuda for t in (*tensors, tok_idx))
        or any(t.device != Q.device or not _dense_layout(t)
               or t.storage_offset() != 0 for t in tensors)
        or A.ndim != 3 or Q.ndim != 3 or C.shape != Q.shape or P.ndim != 3
        or S.ndim != 3 or S.shape[:2] != A.shape[:2]
        or Q.shape[0] != 1 or A.shape[0] != 1
        or Q.shape[1] != 14 * A.shape[1]
        or P.shape[:2] != (Q.shape[1], Q.shape[1])
        or indices.ndim != 3 or indices.shape[:2] != Q.shape[:2]
        or indices.dtype != torch.int64
        or tok_idx.ndim != 1 or len(tok_idx) != Q.shape[1]
        or tok_idx.dtype not in (torch.int32, torch.int64)
        or tok_idx.device != Q.device
        or any(t.dtype not in (torch.float32, torch.bfloat16) for t in (A, Q, C, P, S))
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
        or _get_current_dispatch_mode() is not None
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
        or os.environ.get("RFD3_CUDA_GRAPH_DECODER_UNIFORM", "1") == "0"
        or os.environ.get("RFD3_LOW_MEMORY_MODE", "0") == "1"
        or len(module.upcast) != 3 or len(module.atom_transformer) != 3
        or any(up.method != "cross_attention" or up.n_split <= 0
               or A.shape[-1] % up.n_split for up in module.upcast)
        or not _original_packing(module)
    ):
        module._decoder_uniform_graphs = []
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    expected = torch.arange(A.shape[1], device=tok_idx.device,
                            dtype=tok_idx.dtype).repeat_interleave(14)
    if not torch.equal(tok_idx, expected):
        module._decoder_uniform_graphs = []
        return None
    children = tuple(module.modules())
    if any(
        child.training or child._forward_hooks or child._forward_pre_hooks
        or child._backward_hooks or child._backward_pre_hooks
        or "forward" in child.__dict__ or "forward_" in child.__dict__
        for child in children
    ) or any((
        module_hooks._global_forward_hooks, module_hooks._global_forward_pre_hooks,
        module_hooks._global_backward_hooks, module_hooks._global_backward_pre_hooks,
    )):
        module._decoder_uniform_graphs = []
        return None
    weights = _ordered_weights(module, children)
    if any(w.requires_grad or w.device != Q.device for w in weights):
        module._decoder_uniform_graphs = []
        return None
    try:
        weight_key = tuple((_tensor_key(w), w.is_conj(), w.is_neg()) for w in weights)
    except RuntimeError:
        module._decoder_uniform_graphs = []
        return None
    previous = getattr(module, "_decoder_uniform_weight_key", None)
    if previous is not None and previous != weight_key:
        module._decoder_uniform_invalidated = True
        module._decoder_uniform_graphs = []
    module._decoder_uniform_weight_key = weight_key
    if getattr(module, "_decoder_uniform_invalidated", False):
        return None
    heads = tuple(atom.attention_pair_bias.n_head for atom in module.atom_transformer)
    dense_bytes = 4 * Q.shape[0] * max(heads) * Q.shape[1] ** 2
    reserve = 2 * sum(t.numel() * t.element_size() for t in tensors)
    reserve += 6 * dense_bytes + 64 * Q.numel() * 4
    free, _ = torch.cuda.mem_get_info(Q.device)
    if free <= reserve + dense_bytes / 0.35:
        module._decoder_uniform_graphs = []
        return None
    stream = torch.cuda.current_stream(Q.device)
    from rfd3.model.layers import attention
    config = _current_config(module, children, private_prefix="_decoder_uniform_")
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
        torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(),
        os.environ.get("RFD3_DENSE_SDPA_ATTENTION", "1"),
        id(attention.use_dense_sdpa_pairbias),
        id(attention.dense_sdpa_pairbias_attention),
        id(attention.sparse_pairbias_attention),
    )
    key = (
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors),
        weight_key, config, context,
    )
    entries = getattr(module, "_decoder_uniform_graphs", [])
    stats = getattr(module, "_decoder_uniform_graph_stats", None)
    if stats is None:
        stats = {"captures": 0, "replays": 0, "capture_sec": 0.0, "failures": []}
        module._decoder_uniform_graph_stats = stats
    for old, graph, buffers, result, references, masks in entries:
        if old == key:
            for target, source in zip(buffers, tensors):
                target.copy_(source)
            graph.replay()
            output = tuple(t.clone(memory_format=torch.preserve_format) for t in result)
            for t in (*buffers, *result, *masks):
                t.record_stream(stream)
            stats["replays"] += 1
            return output
    start = time.perf_counter()
    buffers = tuple(t.clone(memory_format=torch.preserve_format) for t in tensors)
    if any(a.stride() != b.stride() for a, b in zip(buffers, tensors)):
        return None
    masks = tuple(
        torch.ones((A.shape[1], 14, up.n_split), dtype=torch.bool, device=Q.device)
        for up in module.upcast
    ) + (torch.ones((A.shape[1], 14), dtype=torch.bool, device=Q.device),)
    capture_stream = torch.cuda.Stream(device=Q.device)
    capture_stream.wait_stream(stream)
    try:
        with _capture_all_valid(module), torch.cuda.stream(capture_stream), torch.autocast(
            "cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            current = buffers[1]
            for up, atom in zip(module.upcast, module.atom_transformer):
                current = up(current, buffers[0], tok_idx=tok_idx)
                current = atom(current, buffers[2], buffers[3], indices=buffers[4])
            module.downcast(current.detach(), buffers[0].detach(), buffers[5].detach(),
                            tok_idx=tok_idx)
        stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with _capture_all_valid(module), torch.cuda.graph(
            graph, stream=capture_stream
        ), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            result = _run_decoder_steps(module, *buffers, masks)
        stream.wait_stream(capture_stream)
    except RuntimeError as error:
        stream.wait_stream(capture_stream)
        module._decoder_uniform_invalidated = True
        module._decoder_uniform_graphs = []
        stats["failures"].append(str(error))
        stats["capture_traceback"] = traceback.format_exc()
        stats["capture_sec"] += time.perf_counter() - start
        return None
    graph.replay()
    output = tuple(t.clone(memory_format=torch.preserve_format) for t in result)
    for t in (*buffers, *result, *masks):
        t.record_stream(stream)
    module._decoder_uniform_graphs = (
        entries + [(key, graph, buffers, result, weights, masks)]
    )[-2:]
    stats["captures"] += 1
    stats["replays"] += 1
    stats["capture_sec"] += time.perf_counter() - start
    return output
