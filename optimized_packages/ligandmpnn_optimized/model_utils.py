from __future__ import print_function

import itertools
import os
import sys

import numpy as np
import torch


# Number of independent decode slots (static buffers + replay stream) that
# consecutive batches round-robin across, so two batches' graph replays
# execute concurrently on the GPU. See _sample_graphed for the RNG argument.
_GRAPH_DECODE_STREAMS = 4

# Shapes (capture key minus slot) whose decode-step kernels already had their
# one-time eager warmup this process. The warmup only initializes cublas
# workspaces/autotune and loads kernel modules — process-wide per shape — so
# later slots and repeat captures skip it. RNG is save/restored around the
# warmup, so skipping never affects sampling.
_graph_warmup_keys = set()


class _GraphDecodeState:
    """Static buffers + captured CUDA graphs for one sample() shape key."""

    pass


def _decode_setup(model, st):
    """Replicates the non-symmetry branch pre-loop of sample() on static
    buffers. Bit-identical ops; run under graph capture or eager warmup."""
    B_decoder = st.B_decoder
    L = st.L
    device = st.device
    E_idx = st.E_idx_in.repeat(B_decoder, 1, 1)
    permutation_matrix_reverse = torch.nn.functional.one_hot(
        st.decoding_order_in, num_classes=L
    ).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = st.mask_in.view([st.B, L, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    # repeat for decoding
    st.S_true = st.S_true_in.repeat(B_decoder, 1)
    st.h_V = st.h_V_in.repeat(B_decoder, 1, 1)
    st.h_E = st.h_E_in.repeat(B_decoder, 1, 1, 1)
    st.chain_mask = st.chain_mask_in.repeat(B_decoder, 1)
    st.mask = st.mask_in.repeat(B_decoder, 1)
    st.bias = st.bias_in.repeat(B_decoder, 1, 1)
    # stacked [B,L,23]: one per-step gather yields chain_mask_t/mask_t/bias_t
    st.cmb_stack = torch.cat(
        (st.chain_mask.unsqueeze(-1), st.mask.unsqueeze(-1), st.bias), dim=-1
    )
    st.E_idx = E_idx

    st.all_probs = torch.zeros(
        (B_decoder, L, 20), device=device, dtype=torch.float32
    )
    st.all_log_probs = torch.zeros(
        (B_decoder, L, 21), device=device, dtype=torch.float32
    )
    st.h_S = torch.zeros_like(st.h_V, device=device)
    st.S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
    st.h_V_stack = [st.h_V] + [
        torch.zeros_like(st.h_V, device=device)
        for _ in range(len(model.decoder_layers))
    ]

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(st.h_S), st.h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(st.h_V, h_EX_encoder, E_idx)
    st.h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    st.mask_bw = mask_bw
    # stacked [B,L,K,513]: one per-step gather yields h_E_t/h_EXV_t/mask_bw_t
    st.heb_stack = torch.cat((st.h_E, st.h_EXV_encoder_fw, mask_bw), dim=-1)


def _decode_step(model, st):
    """One full decode step (reads st.t_buf), including sampling.

    torch.multinomial(probs, 1) is replaced by its exact ATen fast-path
    decomposition (Distributions.cpp multinomial_out: q =
    empty_like(p).exponential_(1); argmax(p / q)) — identical values and
    identical RNG consumption, but graph-capturable (the op's two validation
    D2H syncs are dropped; probs are softmax outputs, finite and non-negative
    in valid operation)."""
    t = st.t_buf  # [B]
    cmb_t = torch.gather(
        st.cmb_stack, 1, t[:, None, None].expand(-1, 1, st.cmb_stack.shape[-1])
    )[:, 0]  # [B,23]
    chain_mask_t = cmb_t[:, 0]  # [B] view
    mask_t = cmb_t[:, 1]  # [B] view
    bias_t = cmb_t[:, 2:]  # [B,21] view

    E_idx_t = torch.gather(
        st.E_idx, 1, t[:, None, None].expand(-1, 1, st.E_idx.shape[-1])
    )
    heb_t = torch.gather(
        st.heb_stack,
        1,
        t[:, None, None, None].expand(
            -1, 1, st.heb_stack.shape[-2], st.heb_stack.shape[-1]
        ),
    )  # [B,1,K,257]
    h_E_t = heb_t[..., : st.h_E.shape[-1]]  # [B,1,K,128] view
    h_ES_t = cat_neighbors_nodes(st.h_S, h_E_t, E_idx_t)
    h_EXV_encoder_t = heb_t[..., st.h_E.shape[-1] : -1]  # [B,1,K,128] view
    mask_bw_t = heb_t[..., -1:]  # [B,1,K,1] view

    h_V_t = torch.gather(
        st.h_V_stack[0],
        1,
        t[:, None, None].expand(-1, 1, st.h_V_stack[0].shape[-1]),
    )
    for l, layer in enumerate(model.decoder_layers):
        h_ESV_decoder_t = cat_neighbors_nodes(st.h_V_stack[l], h_ES_t, E_idx_t)
        h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
        # layer() returns [B,B,C] (mask_V broadcast quirk: [B,1] * [B,1,C]);
        # the scatter reads only its [:, 0:1, :] sub-block, so the next
        # center-node gather would read back exactly those values — reuse the
        # slice instead of re-gathering. Bit-identical.
        h_V_t = layer(h_V_t, h_ESV_t, mask_V=mask_t)[:, 0:1, :]
        st.h_V_stack[l + 1].scatter_(
            1,
            t[:, None, None].expand(-1, 1, st.h_V.shape[-1]),
            h_V_t,
        )

    h_V_t = h_V_t[:, 0]
    logits = model.W_out(h_V_t)  # [B,21]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B,21]

    probs = torch.nn.functional.softmax(
        (logits + bias_t) / st.temperature, dim=-1
    )  # [B,21]
    probs_sample = probs[:, :20] / torch.sum(
        probs[:, :20], dim=-1, keepdim=True
    )  # hard omit X #[B,20]
    # exact decomposition of torch.multinomial(probs_sample, 1)[:, 0]
    q = torch.empty_like(probs_sample).exponential_()
    S_t = torch.argmax(probs_sample / q, dim=-1)  # [B]

    st.all_probs.scatter_(
        1,
        t[:, None, None].expand(-1, 1, 20),
        (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
    )
    st.all_log_probs.scatter_(
        1,
        t[:, None, None].expand(-1, 1, 21),
        (chain_mask_t[:, None, None] * log_probs[:, None, :]).float(),
    )
    S_true_t = torch.gather(st.S_true, 1, t[:, None])[:, 0]
    # chain_mask_t is exactly {0.0, 1.0} (product of binary masks), so a
    # where-select is bit-identical to the float mul/add + .long() roundtrip
    # (ints 0-20 are exact in float32) and costs 2 kernels instead of 5
    S_t = torch.where(chain_mask_t.bool(), S_t, S_true_t)
    st.h_S.scatter_(
        1,
        t[:, None, None].expand(-1, 1, st.h_S.shape[-1]),
        model.W_s(S_t)[:, None, :],
    )
    st.S.scatter_(1, t[:, None], S_t[:, None])


def _graph_decode_capture(model, h_V, h_E, E_idx, mask, chain_mask, bias,
                          S_true, decoding_order, temperature, B_decoder, L,
                          device, warmup=True):
    """Build static buffers, warm up kernels, capture the three graphs."""
    st = _GraphDecodeState()
    st.stream = torch.cuda.Stream(device)  # this slot's replay stream
    st.B_decoder = B_decoder
    st.L = L
    st.B = S_true.shape[0]
    st.temperature = temperature
    st.device = device
    st.h_V_in = h_V.clone()
    st.h_E_in = h_E.clone()
    st.E_idx_in = E_idx.clone()
    st.mask_in = mask.clone()
    st.chain_mask_in = chain_mask.clone()
    st.bias_in = bias.clone()
    st.S_true_in = S_true.clone()
    st.decoding_order_in = decoding_order.clone()
    st.t_buf = torch.zeros(B_decoder, dtype=torch.int64, device=device)
    # remember which objects the static input buffers were captured from
    st.src_h_V = h_V
    st.src_h_E = h_E
    st.src_E_idx = E_idx
    st.src_mask = mask
    st.src_bias = bias
    st.src_S_true = S_true

    entry_state = torch.cuda.get_rng_state(device)
    if warmup:
        # Eager warmup on a side stream: initializes cublas
        # workspaces/autotune and loads the step's kernel modules. A single
        # step exercises every kernel the capture records (the step is
        # structurally t-independent), and only the first capture of a shape
        # needs it (process-wide init).
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            _decode_setup(model, st)
            for t_ in range(min(L, 1)):
                st.t_buf.copy_(st.decoding_order_in[:, t_])
                _decode_step(model, st)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.set_rng_state(entry_state)

    # torch.cuda.graph.__enter__ runs torch.cuda.synchronize() + gc.collect()
    # + torch.cuda.empty_cache() before every capture. All three are pure
    # housekeeping here: the captured region holds refs to every tensor it
    # uses, the caching allocator needs no trimming for correctness, and the
    # device-wide sync only stalls behind the other slot's in-flight decode —
    # capture on a dedicated stream records only this thread's ops (already
    # enqueued work on other streams executes normally and is not captured),
    # and the RNG state capture_begin reads is host-side. Suppress them for
    # these two captures only, then restore. No effect on captured ops,
    # values, or RNG.
    import sys as _sys

    _graphs_mod = _sys.modules["torch.cuda.graphs"]
    _real_gc = _graphs_mod.gc
    _real_empty_cache = torch.cuda.empty_cache
    _real_synchronize = torch.cuda.synchronize

    class _NoGc:
        @staticmethod
        def collect():
            pass

    try:
        _graphs_mod.gc = _NoGc
        torch.cuda.empty_cache = lambda: None
        torch.cuda.synchronize = lambda: None
        pool = torch.cuda.graph_pool_handle()
        # Explicit per-slot capture stream: torch.cuda.graph's class-level
        # default capture stream is shared process-wide, and cublas
        # workspaces are keyed by (handle, stream) — two slots captured on
        # the same stream would bake the same workspace into both graphs and
        # race under concurrent replay. Distinct capture streams give each
        # slot its own workspace; the pool stays per-slot as well.
        capture_stream = torch.cuda.Stream(device)
        st.graph_setup = torch.cuda.CUDAGraph()
        with torch.cuda.graph(st.graph_setup, pool=pool, stream=capture_stream):
            _decode_setup(model, st)
        st.graph_step = torch.cuda.CUDAGraph()
        with torch.cuda.graph(st.graph_step, pool=pool, stream=capture_stream):
            _decode_step(model, st)
    finally:
        _graphs_mod.gc = _real_gc
        torch.cuda.empty_cache = _real_empty_cache
        torch.cuda.synchronize = _real_synchronize
    torch.cuda.set_rng_state(entry_state)
    return st


def _sample_graphed(model, h_V, h_E, E_idx, mask, chain_mask, bias, S_true,
                    decoding_order, temperature, B_decoder, L, device):
    """Graph-replay equivalent of sample()'s non-symmetry branch.

    torch.multinomial stays eager in the original order with identical RNG
    consumption, so sampled sequences match eager execution exactly.

    Consecutive calls round-robin across _GRAPH_DECODE_STREAMS slots, each
    with its own static buffers, memory pool, capture stream (hence its own
    cublas workspace) and replay stream, so two batches decode concurrently
    on the GPU. CUDA graph replays assign philox offsets host-side in replay
    call order (verified empirically on this torch build), and the host call
    order is unchanged — batch c step t draws exactly the offset it would
    draw single-stream — so sampled sequences remain bit-identical."""
    slot = model._graph_decode_slot_counter % _GRAPH_DECODE_STREAMS
    model._graph_decode_slot_counter += 1
    key = (
        B_decoder,
        L,
        float(temperature),
        S_true.shape[0],
        E_idx.shape[-1],
        tuple(h_V.shape),
        tuple(h_E.shape),
        device.index,
        slot,
    )
    st = model._graph_decode_cache.get(key)
    fresh = st is None
    if fresh:
        warm_key = key[:-1]  # drop slot: the warmup is per-shape, not per-slot
        need_warmup = warm_key not in _graph_warmup_keys
        _graph_warmup_keys.add(warm_key)
        st = _graph_decode_capture(
            model, h_V, h_E, E_idx, mask, chain_mask, bias, S_true,
            decoding_order, temperature, B_decoder, L, device,
            warmup=need_warmup,
        )
        model._graph_decode_cache[key] = st
    # Order this slot's stream after the producer of this call's inputs
    # (randn, the mask product, argsort and — on the first batch — encode,
    # all enqueued on the default stream). The default stream carries no
    # other per-batch work (scoring runs on the slot stream too), so this
    # never serializes one slot behind another.
    st.stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(st.stream):
        if not fresh:
            # skip copies for source objects unchanged since capture (identity)
            for name, val in (
                ("h_V", h_V),
                ("h_E", h_E),
                ("E_idx", E_idx),
                ("mask", mask),
                ("bias", bias),
                ("S_true", S_true),
            ):
                if getattr(st, "src_" + name) is not val:
                    getattr(st, name + "_in").copy_(val)
                    setattr(st, "src_" + name, val)
            st.chain_mask_in.copy_(chain_mask)  # fresh mask*chain_mask each call
            st.decoding_order_in.copy_(decoding_order)  # fresh argsort each call

        st.graph_setup.replay()
        for t_ in range(L):
            st.t_buf.copy_(st.decoding_order_in[:, t_])
            st.graph_step.replay()

        out = {
            "S": st.S.clone(),
            "sampling_probs": st.all_probs.clone(),
            "log_probs": st.all_log_probs.clone(),
            "decoding_order": decoding_order,
        }
    out["_stream"] = st.stream
    return out


class ProteinMPNN(torch.nn.Module):
    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
        device=None,
        atom_context_num=0,
        model_type="protein_mpnn",
        ligand_mpnn_use_side_chain_context=False,
    ):
        super(ProteinMPNN, self).__init__()

        self.model_type = model_type
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        if self.model_type == "ligand_mpnn":
            self.features = ProteinFeaturesLigand(
                node_features,
                edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps,
                device=device,
                atom_context_num=atom_context_num,
                use_side_chains=ligand_mpnn_use_side_chain_context,
            )
            self.W_v = torch.nn.Linear(node_features, hidden_dim, bias=True)
            self.W_c = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

            self.W_nodes_y = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.W_edges_y = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

            self.V_C = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_C_norm = torch.nn.LayerNorm(hidden_dim)

            self.context_encoder_layers = torch.nn.ModuleList(
                [
                    DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                    for _ in range(2)
                ]
            )

            self.y_context_encoder_layers = torch.nn.ModuleList(
                [DecLayerJ(hidden_dim, hidden_dim, dropout=dropout) for _ in range(2)]
            )
        elif self.model_type == "protein_mpnn" or self.model_type == "soluble_mpnn":
            self.features = ProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )
        elif (
            self.model_type == "per_residue_label_membrane_mpnn"
            or self.model_type == "global_label_membrane_mpnn"
        ):
            self.W_v = torch.nn.Linear(node_features, hidden_dim, bias=True)
            self.features = ProteinFeaturesMembrane(
                node_features,
                edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps,
                num_classes=3,
            )
        else:
            print("Choose --model_type flag from currently available models")
            sys.exit()

        self.W_e = torch.nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = torch.nn.Embedding(vocab, hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)

        # Encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.W_out = torch.nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def encode(self, feature_dict):
        # xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        # Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        # Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        # Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        # X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
        S_true = feature_dict[
            "S"
        ]  # [B,L] - integer protein sequence encoded using "restype_STRtoINT"
        # R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict[
            "mask"
        ]  # [B,L] - mask for missing regions - should be removed! all ones most of the time
        # chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters

        B, L = S_true.shape
        device = S_true.device

        if self.model_type == "ligand_mpnn":
            V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(feature_dict)
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
            h_E = self.W_e(E)
            h_E_context = self.W_v(V)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

            h_V_C = self.W_c(h_V)
            Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
            Y_nodes = self.W_nodes_y(Y_nodes)
            Y_edges = self.W_edges_y(Y_edges)
            for i in range(len(self.context_encoder_layers)):
                Y_nodes = self.y_context_encoder_layers[i](
                    Y_nodes, Y_edges, Y_m, Y_m_edges
                )
                h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
                h_V_C = self.context_encoder_layers[i](
                    h_V_C, h_E_context_cat, mask, Y_m
                )

            h_V_C = self.V_C(h_V_C)
            h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
        elif self.model_type == "protein_mpnn" or self.model_type == "soluble_mpnn":
            E, E_idx = self.features(feature_dict)
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
            h_E = self.W_e(E)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        elif (
            self.model_type == "per_residue_label_membrane_mpnn"
            or self.model_type == "global_label_membrane_mpnn"
        ):
            V, E, E_idx = self.features(feature_dict)
            h_V = self.W_v(V)
            h_E = self.W_e(E)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V, h_E, E_idx

    def sample(self, feature_dict):
        # xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        # Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        # Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        # Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        # X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict[
            "S"
        ]  # [B,L] - integer proitein sequence encoded using "restype_STRtoINT"
        # R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict[
            "mask"
        ]  # [B,L] - mask for missing regions - should be removed! all ones most of the time
        chain_mask = feature_dict[
            "chain_mask"
        ]  # [B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed
        bias = feature_dict["bias"]  # [B,L,21] - amino acid bias per position
        # chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters
        randn = feature_dict[
            "randn"
        ]  # [B,L] - random numbers for decoding order; only the first entry is used since decoding within a batch needs to match for symmetry
        temperature = feature_dict[
            "temperature"
        ]  # float - sampling temperature; prob = softmax(logits/temperature)
        symmetry_list_of_lists = feature_dict[
            "symmetry_residues"
        ]  # [[0, 1, 14], [10,11,14,15], [20, 21]] #indices to select X over length - L
        symmetry_weights_list_of_lists = feature_dict[
            "symmetry_weights"
        ]  # [[1.0, 1.0, 1.0], [-2.0,1.1,0.2,1.1], [2.3, 1.1]]

        B, L = S_true.shape
        device = S_true.device

        # encode() is deterministic (eval mode, no RNG) and its inputs are the
        # same objects across number_of_batches iterations; only "randn"
        # changes. Cache on object identity, holding strong refs so ids cannot
        # be recycled while cached.
        enc_cache = getattr(self, "_encode_cache", None)
        if (
            enc_cache is not None
            and enc_cache["fd_keys"] == set(feature_dict.keys())
            and all(
                k == "randn" or enc_cache["fd"][k] is feature_dict[k]
                for k in feature_dict
            )
        ):
            h_V, h_E, E_idx = enc_cache["out"]
        else:
            h_V, h_E, E_idx = self.encode(feature_dict)
            self._encode_cache = {
                "fd": dict(feature_dict),
                "fd_keys": set(feature_dict.keys()),
                "out": (h_V, h_E, E_idx),
            }

        chain_mask = mask * chain_mask  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        if len(symmetry_list_of_lists[0]) == 0 and len(symmetry_list_of_lists) == 1:
            if (
                device.type == "cuda"
                and os.environ.get("LIGANDMPNN_DISABLE_GRAPH", "0") != "1"
                and not getattr(self, "_graph_decode_disabled", False)
            ):
                if not hasattr(self, "_graph_decode_cache"):
                    self._graph_decode_cache = {}
                    self._graph_decode_slot_counter = 0
                entry_rng = torch.cuda.get_rng_state(device)
                try:
                    output_dict = _sample_graphed(
                        self,
                        h_V,
                        h_E,
                        E_idx,
                        mask,
                        chain_mask,
                        bias,
                        S_true,
                        decoding_order,
                        temperature,
                        B_decoder,
                        L,
                        device,
                    )
                except Exception:
                    self._graph_decode_disabled = True
                    torch.cuda.set_rng_state(entry_rng, device)
                    torch.cuda.synchronize(device)
                    output_dict = None
                if output_dict is not None:
                    return output_dict
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            # repeat for decoding
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E.repeat(B_decoder, 1, 1, 1)
            chain_mask = chain_mask.repeat(B_decoder, 1)
            mask = mask.repeat(B_decoder, 1)
            bias = bias.repeat(B_decoder, 1, 1)

            all_probs = torch.zeros(
                (B_decoder, L, 20), device=device, dtype=torch.float32
            )
            all_log_probs = torch.zeros(
                (B_decoder, L, 21), device=device, dtype=torch.float32
            )
            h_S = torch.zeros_like(h_V, device=device)
            S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [
                torch.zeros_like(h_V, device=device)
                for _ in range(len(self.decoder_layers))
            ]

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_ in range(L):
                t = decoding_order[:, t_]  # [B]
                chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]  # [B]
                mask_t = torch.gather(mask, 1, t[:, None])[:, 0]  # [B]
                bias_t = torch.gather(bias, 1, t[:, None, None].repeat(1, 1, 21))[
                    :, 0, :
                ]  # [B,21]

                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                )
                h_E_t = torch.gather(
                    h_E,
                    1,
                    t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
                )
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )

                mask_bw_t = torch.gather(
                    mask_bw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                    ),
                )

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                    )
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )

                h_V_t = torch.gather(
                    h_V_stack[-1],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                )[:, 0]
                logits = self.W_out(h_V_t)  # [B,21]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B,21]

                probs = torch.nn.functional.softmax(
                    (logits + bias_t) / temperature, dim=-1
                )  # [B,21]
                probs_sample = probs[:, :20] / torch.sum(
                    probs[:, :20], dim=-1, keepdim=True
                )  # hard omit X #[B,20]
                S_t = torch.multinomial(probs_sample, 1)[:, 0]  # [B]

                all_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 20),
                    (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
                )
                all_log_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 21),
                    (chain_mask_t[:, None, None] * log_probs[:, None, :]).float(),
                )
                S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
                S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
                h_S.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, h_S.shape[-1]),
                    self.W_s(S_t)[:, None, :],
                )
                S.scatter_(1, t[:, None], S_t[:, None])

            output_dict = {
                "S": S,
                "sampling_probs": all_probs,
                "log_probs": all_log_probs,
                "decoding_order": decoding_order,
            }
        else:
            # weights for symmetric design
            symmetry_weights = torch.ones([L], device=device, dtype=torch.float32)
            for i1, item_list in enumerate(symmetry_list_of_lists):
                for i2, item in enumerate(item_list):
                    symmetry_weights[item] = symmetry_weights_list_of_lists[i1][i2]

            new_decoding_order = []
            for t_dec in list(decoding_order[0,].cpu().data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [item for item in symmetry_list_of_lists if t_dec in item]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = torch.tensor(
                list(itertools.chain(*new_decoding_order)), device=device
            )[None,].repeat(B, 1)

            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            # repeat for decoding
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E.repeat(B_decoder, 1, 1, 1)
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            mask_fw = mask_fw.repeat(B_decoder, 1, 1, 1)
            mask_bw = mask_bw.repeat(B_decoder, 1, 1, 1)
            chain_mask = chain_mask.repeat(B_decoder, 1)
            mask = mask.repeat(B_decoder, 1)
            bias = bias.repeat(B_decoder, 1, 1)

            all_probs = torch.zeros(
                (B_decoder, L, 20), device=device, dtype=torch.float32
            )
            all_log_probs = torch.zeros(
                (B_decoder, L, 21), device=device, dtype=torch.float32
            )
            h_S = torch.zeros_like(h_V, device=device)
            S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [
                torch.zeros_like(h_V, device=device)
                for _ in range(len(self.decoder_layers))
            ]

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_list in new_decoding_order:
                total_logits = 0.0
                for t in t_list:
                    chain_mask_t = chain_mask[:, t]  # [B]
                    mask_t = mask[:, t]  # [B]
                    bias_t = bias[:, t]  # [B, 21]

                    E_idx_t = E_idx[:, t : t + 1]
                    h_E_t = h_E[:, t : t + 1]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t : t + 1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(
                            h_V_stack[l], h_ES_t, E_idx_t
                        )
                        h_V_t = h_V_stack[l][:, t : t + 1]
                        h_ESV_t = (
                            mask_bw[:, t : t + 1] * h_ESV_decoder_t + h_EXV_encoder_t
                        )
                        h_V_stack[l + 1][:, t : t + 1, :] = layer(
                            h_V_t, h_ESV_t, mask_V=mask_t[:, None]
                        )

                    h_V_t = h_V_stack[-1][:, t]
                    logits = self.W_out(h_V_t)  # [B,21]
                    log_probs = torch.nn.functional.log_softmax(
                        logits, dim=-1
                    )  # [B,21]
                    all_log_probs[:, t] = (
                        chain_mask_t[:, None] * log_probs
                    ).float()  # [B,21]
                    total_logits += symmetry_weights[t] * logits

                probs = torch.nn.functional.softmax(
                    (total_logits + bias_t) / temperature, dim=-1
                )  # [B,21]
                probs_sample = probs[:, :20] / torch.sum(
                    probs[:, :20], dim=-1, keepdim=True
                )  # hard omit X #[B,20]
                S_t = torch.multinomial(probs_sample, 1)[:, 0]  # [B]
                for t in t_list:
                    chain_mask_t = chain_mask[:, t]  # [B]
                    all_probs[:, t] = (
                        chain_mask_t[:, None] * probs_sample
                    ).float()  # [B,20]
                    S_true_t = S_true[:, t]  # [B]
                    S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
                    h_S[:, t] = self.W_s(S_t)
                    S[:, t] = S_t

            output_dict = {
                "S": S,
                "sampling_probs": all_probs,
                "log_probs": all_log_probs,
                "decoding_order": decoding_order.repeat(B_decoder, 1),
            }
        return output_dict

    def single_aa_score(self, feature_dict, use_sequence: bool):
        """
        feature_dict - input features
        use_sequence - False using backbone info only
        """
        B_decoder = feature_dict["batch_size"]
        S_true_enc = feature_dict[
            "S"
        ]
        mask_enc = feature_dict[
            "mask"
        ]
        chain_mask_enc = feature_dict[
            "chain_mask"
        ]
        randn = feature_dict[
            "randn"
        ]
        B, L = S_true_enc.shape
        device = S_true_enc.device

        h_V_enc, h_E_enc, E_idx_enc = self.encode(feature_dict)
        log_probs_out = torch.zeros([B_decoder, L, 21], device=device).float()
        logits_out = torch.zeros([B_decoder, L, 21], device=device).float()
        decoding_order_out = torch.zeros([B_decoder, L, L], device=device).float()

        for idx in range(L):
            h_V = torch.clone(h_V_enc)
            E_idx = torch.clone(E_idx_enc)
            mask = torch.clone(mask_enc)
            S_true = torch.clone(S_true_enc)
            if not use_sequence:
                order_mask = torch.zeros(chain_mask_enc.shape[1], device=device).float()
                order_mask[idx] = 1.
            else:
                order_mask = torch.ones(chain_mask_enc.shape[1], device=device).float()
                order_mask[idx] = 0.
            decoding_order = torch.argsort(
                (order_mask + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E_enc.repeat(B_decoder, 1, 1, 1)
            mask = mask.repeat(B_decoder, 1)

            h_S = self.W_s(S_true)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            log_probs_out[:,idx,:] = log_probs[:,idx,:]
            logits_out[:,idx,:] = logits[:,idx,:]
            decoding_order_out[:,idx,:] = decoding_order

        output_dict = {
            "S": S_true,
            "log_probs": log_probs_out,
            "logits": logits_out,
            "decoding_order": decoding_order_out,
        }
        return output_dict


    def score(self, feature_dict, use_sequence: bool):
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict[
            "S"
        ]
        mask = feature_dict[
            "mask"
        ]
        chain_mask = feature_dict[
            "chain_mask"
        ]
        randn = feature_dict[
            "randn"
        ]
        symmetry_list_of_lists = feature_dict[
            "symmetry_residues"
        ]
        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask * chain_mask  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        if len(symmetry_list_of_lists[0]) == 0 and len(symmetry_list_of_lists) == 1:
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)
        else:
            new_decoding_order = []
            for t_dec in list(decoding_order[0,].cpu().data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [item for item in symmetry_list_of_lists if t_dec in item]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = torch.tensor(
                list(itertools.chain(*new_decoding_order)), device=device
            )[None,].repeat(B, 1)

            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            E_idx = E_idx.repeat(B_decoder, 1, 1)
            mask_fw = mask_fw.repeat(B_decoder, 1, 1, 1)
            mask_bw = mask_bw.repeat(B_decoder, 1, 1, 1)
            decoding_order = decoding_order.repeat(B_decoder, 1)

        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        mask = mask.repeat(B_decoder, 1)

        h_S = self.W_s(S_true)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        if not use_sequence:
            for layer in self.decoder_layers:
                h_V = layer(h_V, h_EXV_encoder_fw, mask)          
        else:
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        output_dict = {
            "S": S_true,
            "log_probs": log_probs,
            "logits": logits,
            "decoding_order": decoding_order,
        }
        return output_dict


class ProteinFeaturesLigand(torch.nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        device=None,
        atom_context_num=16,
        use_side_chains=False,
    ):
        """Extract protein features"""
        super(ProteinFeaturesLigand, self).__init__()

        self.use_side_chains = use_side_chains

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = torch.nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = torch.nn.LayerNorm(edge_features)

        self.node_project_down = torch.nn.Linear(
            5 * num_rbf + 64 + 4, node_features, bias=True
        )
        self.norm_nodes = torch.nn.LayerNorm(node_features)

        self.type_linear = torch.nn.Linear(147, 64)

        self.y_nodes = torch.nn.Linear(147, node_features, bias=False)
        self.y_edges = torch.nn.Linear(num_rbf, node_features, bias=False)

        self.norm_y_edges = torch.nn.LayerNorm(node_features)
        self.norm_y_nodes = torch.nn.LayerNorm(node_features)

        self.atom_context_num = atom_context_num

        # the last 32 atoms in the 37 atom representation
        self.side_chain_atom_types = torch.tensor(
            [
                6,
                6,
                6,
                8,
                8,
                16,
                6,
                6,
                6,
                7,
                7,
                8,
                8,
                16,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                8,
                8,
                6,
                7,
                7,
                8,
                6,
                6,
                6,
                7,
                8,
            ],
            device=device,
        )

        self.periodic_table_features = torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                [
                    0,
                    1,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
            ],
            dtype=torch.long,
            device=device,
        )

    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        e1 = torch.nn.functional.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = torch.nn.functional.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = torch.cat([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1)
        return f

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        Y = input_features["Y"]
        Y_m = input_features["Y_m"]
        Y_t = input_features["Y_t"]
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            Y = Y + self.augment_eps * torch.randn_like(Y)

        B, L, _, _ = X.shape

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca  # shift from CA

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        if self.use_side_chains:
            xyz_37 = input_features["xyz_37"]
            xyz_37_m = input_features["xyz_37_m"]
            E_idx_sub = E_idx[:, :, :16]  # [B, L, 15]
            mask_residues = input_features["chain_mask"]
            xyz_37_m = xyz_37_m * (1 - mask_residues[:, :, None])
            R_m = gather_nodes(xyz_37_m[:, :, 5:], E_idx_sub)

            X_sidechain = xyz_37[:, :, 5:, :].view(B, L, -1)
            R = gather_nodes(X_sidechain, E_idx_sub).view(
                B, L, E_idx_sub.shape[2], -1, 3
            )
            R_t = self.side_chain_atom_types[None, None, None, :].repeat(
                B, L, E_idx_sub.shape[2], 1
            )

            # Side chain atom context
            R = R.view(B, L, -1, 3)  # coordinates
            R_m = R_m.view(B, L, -1)  # mask
            R_t = R_t.view(B, L, -1)  # atom types

            # Ligand atom context
            Y = torch.cat((R, Y), 2)  # [B, L, atoms, 3]
            Y_m = torch.cat((R_m, Y_m), 2)  # [B, L, atoms]
            Y_t = torch.cat((R_t, Y_t), 2)  # [B, L, atoms]

            Cb_Y_distances = torch.sum((Cb[:, :, None, :] - Y) ** 2, -1)
            mask_Y = mask[:, :, None] * Y_m
            Cb_Y_distances_adjusted = Cb_Y_distances * mask_Y + (1.0 - mask_Y) * 10000.0
            _, E_idx_Y = torch.topk(
                Cb_Y_distances_adjusted, self.atom_context_num, dim=-1, largest=False
            )

            Y = torch.gather(Y, 2, E_idx_Y[:, :, :, None].repeat(1, 1, 1, 3))
            Y_t = torch.gather(Y_t, 2, E_idx_Y)
            Y_m = torch.gather(Y_m, 2, E_idx_Y)

        Y_t = Y_t.long()
        Y_t_g = self.periodic_table_features[1][Y_t]  # group; 19 categories including 0
        Y_t_p = self.periodic_table_features[2][Y_t]  # period; 8 categories including 0

        Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19)  # [B, L, M, 19]
        Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8)  # [B, L, M, 8]
        Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120)  # [B, L, M, 120]

        Y_t_1hot_ = torch.cat(
            [Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1
        )  # [B, L, M, 147]
        Y_t_1hot = self.type_linear(Y_t_1hot_.float())

        D_N_Y = self._rbf(
            torch.sqrt(torch.sum((N[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )  # [B, L, M, num_bins]
        D_Ca_Y = self._rbf(
            torch.sqrt(torch.sum((Ca[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )
        D_C_Y = self._rbf(torch.sqrt(torch.sum((C[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_O_Y = self._rbf(torch.sqrt(torch.sum((O[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_Cb_Y = self._rbf(
            torch.sqrt(torch.sum((Cb[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )

        f_angles = self._make_angle_features(N, Ca, C, Y)  # [B, L, M, 4]

        D_all = torch.cat(
            (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), dim=-1
        )  # [B,L,M,5*num_bins+5]
        V = self.node_project_down(D_all)  # [B, L, M, node_features]
        V = self.norm_nodes(V)

        Y_edges = self._rbf(
            torch.sqrt(
                torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
            )
        )  # [B, L, M, M, num_bins]

        Y_edges = self.y_edges(Y_edges)
        Y_nodes = self.y_nodes(Y_t_1hot_.float())

        Y_edges = self.norm_y_edges(Y_edges)
        Y_nodes = self.norm_y_nodes(Y_nodes)

        return V, E, E_idx, Y_nodes, Y_edges, Y_m


class ProteinFeatures(torch.nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = torch.nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = torch.nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class ProteinFeaturesMembrane(torch.nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
        num_classes=3,
    ):
        """Extract protein features"""
        super(ProteinFeaturesMembrane, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.num_classes = num_classes

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = torch.nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = torch.nn.LayerNorm(edge_features)

        self.node_embedding = torch.nn.Linear(
            self.num_classes, node_features, bias=False
        )
        self.norm_nodes = torch.nn.LayerNorm(node_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]
        membrane_per_residue_labels = input_features["membrane_per_residue_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        C_1hot = torch.nn.functional.one_hot(
            membrane_per_residue_labels, self.num_classes
        ).float()
        V = self.node_embedding(C_1hot)
        V = self.norm_nodes(V)

        return V, E, E_idx


class DecLayerJ(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(
            -1, -1, -1, h_E.size(-2), -1
        )  # the only difference
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class DecLayer(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class EncLayer(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)
        self.norm3 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn
