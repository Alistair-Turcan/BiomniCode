import argparse
import contextlib
import copy
import json
import os.path
import random
import sys

import numpy as np

# --- iter021: defer torch subsystems unused at inference --------------------
# torch/__init__.py eagerly imports torch.jit, torch.ao/quantization,
# torch.fx (via torch.func -> _functorch.eager_transforms), torch.optim,
# torch.distributions, torch.export, torch.profiler, torch.testing, torch.hub,
# torch.masked, torch._meta_registrations (-> _decomp/_prims/_refs) and
# torch._lobpcg. LigandMPNN inference uses none of them. Pre-registering
# stand-ins in sys.modules makes those imports bind lazily; real modules load
# transparently on first actual use (e.g. torch.jit._builtins loads for real,
# under the shimmed parent, when torch.ops first resolves an op).
import types as _ttypes
import importlib as _til
import sysconfig as _tsc


class _TLazyFunc:
    """Callable placeholder that imports the real torch module on first call."""

    def __init__(self, modname, attr):
        self._tm_mod = modname
        self._tm_attr = attr

    def __call__(self, *args, **kwargs):
        return getattr(_tload_real(self._tm_mod), self._tm_attr)(*args, **kwargs)


class _TShimModule(_ttypes.ModuleType):
    """Module stand-in: unknown attribute access loads the real module and
    delegates; pre-populated names are lazy callables that load on call."""

    def __getattr__(self, name):
        if name == "__path__":
            raise AttributeError(name)
        full = self.__name__ + "." + name
        mod = sys.modules.get(full)
        if isinstance(mod, _TShimModule):
            return mod
        return getattr(_tload_real(self.__name__), name)


def _tload_real(name):
    # No parent recursion: the import system loads real submodules under a
    # shimmed parent via the shim's __path__; real-loading the parent too
    # would defeat the deferral (e.g. torch.jit._builtins under torch.jit).
    shim = sys.modules.pop(name, None)
    try:
        return _til.import_module(name)
    except Exception:
        if shim is not None:
            sys.modules[name] = shim
        raise


def _jit_is_scripting(*args, **kwargs):
    # torch.jit.is_scripting is called on hot paths (torch.cuda._utils,
    # torch.nn.functional). It is genuinely defined in the lightweight
    # torch._jit_internal and re-exported by torch.jit, so delegate there
    # instead of loading all of torch.jit.
    from torch._jit_internal import is_scripting as _f

    return _f(*args, **kwargs)


def _install_torch_shim():
    if "torch" in sys.modules:
        return
    torch_dir = os.path.join(_tsc.get_paths()["purelib"], "torch")
    if not os.path.isdir(torch_dir):
        return
    leaves = {
        "torch._meta_registrations": [],
        "torch.jit": [],
        "torch.jit._builtins": [],
        "torch.distributions": [],
        "torch.nn.quantizable": [],
        "torch.nn.quantized": [],
        "torch.nn.qat": [],
        "torch.nn.intrinsic": [],
        "torch.quantization": [],
        "torch.optim": [],
        "torch.optim._multi_tensor": [],
        "torch.hub": [],
        "torch.testing": [],
        "torch.profiler": [],
        "torch.export": [],
        "torch.masked": [],
        "torch._higher_order_ops": ["cond"],
        "torch.fx.experimental.sym_node": [],
        "torch._functorch": [],
        "torch._lobpcg": ["lobpcg"],
        "torch.func": ["vmap"],
        # torch.nn only binds DataParallel from it; the real package pulls in
        # torch.distributed.algorithms.join (~14ms) which inference never uses
        "torch.nn.parallel": ["DataParallel", "DistributedDataParallel"],
    }
    inserted = []
    try:
        for name, attrs in leaves.items():
            if name in sys.modules:
                continue
            mod = _TShimModule(name)
            sub = os.path.join(torch_dir, *name.split(".")[1:])
            if os.path.isdir(sub):
                mod.__path__ = [sub]
            for attr in attrs:
                setattr(mod, attr, _TLazyFunc(name, attr))
            if name == "torch.jit":
                mod.is_scripting = _jit_is_scripting
            sys.modules[name] = mod
            inserted.append(name)
    except Exception:
        for name in inserted:
            sys.modules.pop(name, None)
        raise


try:
    _install_torch_shim()
except Exception:
    pass  # never let the shim break startup; worst case is original behavior
# -----------------------------------------------------------------------------

# --- iter025: pre-initialize the CUDA driver context during torch import ---
# cuInit + primary-context retain via ctypes release the GIL, so they run
# concurrently with torch's import (dominated by the torch._C dlopen). The
# primary context is process-wide, so torch's later lazy init reuses it and
# the warmup below costs ~40 ms (torch's own setup) instead of ~200 ms of
# driver init competing with the main thread's imports for the GIL. No RNG is
# consumed. Any failure (no GPU, driver mismatch) is swallowed; the _warm_cuda
# thread below and the normal device logic are unaffected.
import ctypes as _ctypes
import threading


def _pre_init_cuda():
    try:
        _lib = _ctypes.CDLL("libcuda.so.1")
        if _lib.cuInit(0) == 0:
            _ctx = _ctypes.c_void_p()
            _lib.cuDevicePrimaryCtxRetain(_ctypes.byref(_ctx), 0)
    except Exception:
        pass


_pre_init_cuda_thread = threading.Thread(target=_pre_init_cuda, daemon=True)
_pre_init_cuda_thread.start()

import torch

# Warm the CUDA context on a side thread while the remaining pure-CPU imports
# (prody via data_utils, etc.) run. torch.zeros consumes no RNG and the primary
# CUDA context is process-wide, so later main-thread CUDA use is unaffected.
# Any failure (e.g. no GPU) is swallowed; the normal device logic handles it.
import threading


def _warm_cuda():
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
    except Exception:
        pass


_cuda_warm_thread = threading.Thread(target=_warm_cuda)
_cuda_warm_thread.start()

# --- iter020: defer prody's module-level scipy imports ----------------------
# `from data_utils import ...` triggers `import prody`, which eagerly imports
# scipy.optimize/stats/sparse/spatial/cluster/linalg at module level but only
# *uses* them inside functions LigandMPNN never calls (NMA, clustering, LAPACK
# transforms). Pre-registering stand-ins in sys.modules lets those from-imports
# succeed instantly; the real modules load transparently on first actual use.
# Verified: parsePDB output bit-identical, zero real scipy modules loaded.
import types as _types
import importlib as _il
import importlib.util as _ilu


class _LazyFunc:
    """Callable placeholder that imports the real scipy module on first call."""

    def __init__(self, modname, attr):
        self._lm_mod = modname
        self._lm_attr = attr

    def __call__(self, *args, **kwargs):
        return getattr(_load_real(self._lm_mod), self._lm_attr)(*args, **kwargs)


class _ShimModule(_types.ModuleType):
    """Module stand-in: unknown attribute access loads the real module;
    pre-populated names are lazy callables that load on first call."""

    def __getattr__(self, name):
        if name == "__path__":
            # Non-package shim: the real module has no __path__ either. Raising
            # here stops _handle_fromlist from triggering a real load just to
            # answer hasattr(module, '__path__').
            raise AttributeError(name)
        full = self.__name__ + "." + name
        mod = sys.modules.get(full)
        if isinstance(mod, _ShimModule):
            return mod
        return getattr(_load_real(self.__name__), name)


def _load_real(name):
    parent, _, _ = name.rpartition(".")
    if parent and isinstance(sys.modules.get(parent), _ShimModule):
        _load_real(parent)
    shim = sys.modules.pop(name, None)
    try:
        return _il.import_module(name)
    except Exception:
        if shim is not None:
            sys.modules[name] = shim
        raise


def _install_scipy_shim(top_level=True):
    existing = sys.modules.get("scipy")
    if existing is not None and not isinstance(existing, _ShimModule):
        return
    spec = _ilu.find_spec("scipy")  # top-level find_spec executes no module code
    if spec is None or not spec.submodule_search_locations:
        return
    scipy_dir = list(spec.submodule_search_locations)[0]
    leaves = {
        "scipy.optimize": ["linear_sum_assignment"],
        "scipy.stats": ["zscore", "mode"],
        "scipy.sparse": ["coo_matrix"],
        "scipy.spatial": [],
        "scipy.spatial.distance": ["pdist", "squareform"],
        "scipy.cluster": [],
        "scipy.cluster.hierarchy": ["fcluster", "linkage"],
        # prody calls importLA() at module level (transform.py, exanm.py,
        # exgnm.py, clustenm.py, adaptive.py) and binds LA.inv/LA.pinv/LA.norm
        # at module level (exanm.py:17-19, exgnm.py:15-17, clustenm.py:53,
        # adaptive.py:25), so those must be pre-populated lazy callables.
        "scipy.linalg": ["inv", "pinv", "norm"],
    }
    inserted = []
    try:
        for name, attrs in leaves.items():
            if name in sys.modules:
                continue
            mod = _ShimModule(name)
            pkg_dir = os.path.join(scipy_dir, *name.split(".")[1:])
            if os.path.isdir(pkg_dir):
                mod.__path__ = [pkg_dir]
            for attr in attrs:
                setattr(mod, attr, _LazyFunc(name, attr))
            sys.modules[name] = mod
            inserted.append(name)
        if top_level:
            top = _ShimModule("scipy")
            top.__path__ = [scipy_dir]
            top.sparse = sys.modules["scipy.sparse"]
            sys.modules["scipy"] = top
            inserted.append("scipy")
    except Exception:
        for name in inserted:
            sys.modules.pop(name, None)
        raise


try:
    _install_scipy_shim(top_level=True)
except Exception:
    pass  # never let the shim break startup; worst case is original behavior


def _install_prody_dep_shims():
    """Defer heavy transitive deps that prody imports but our paths never use.

    prody.utilities.TreeConstruction does module-level from-imports of
    Bio.Phylo.BaseTree and Bio.Phylo.TreeConstruction._DistanceMatrix (~66ms
    with the Bio.Phylo._io/NexusIO subtree) yet only uses them inside
    tree-building functions. pkg_resources (~92ms) is pulled in during the
    prody.atomic import chain; prody's only use is a function-level
    resource_filename call in misctools.py. Pre-populated lazy attributes
    keep the from-imports binding correctly; any real attribute access
    delegates to the real module, so worst case is original behavior.
    """
    roots = {}
    for top in ("Bio", "pkg_resources", "prody"):
        try:
            spec = _ilu.find_spec(top)  # top-level find_spec executes no code
        except Exception:
            spec = None
        if spec is not None and spec.submodule_search_locations:
            roots[top] = list(spec.submodule_search_locations)[0]
    leaves = {
        "Bio.Phylo": ["BaseTree"],
        "Bio.Phylo.TreeConstruction": ["_DistanceMatrix"],
        "pkg_resources": [],
    }
    inserted = []
    try:
        for name, attrs in leaves.items():
            if name in sys.modules:
                continue
            root = roots.get(name.split(".")[0])
            if root is None:
                continue
            mod = _ShimModule(name)
            pkg_dir = os.path.join(root, *name.split(".")[1:])
            if os.path.isdir(pkg_dir):
                mod.__path__ = [pkg_dir]
            for attr in attrs:
                setattr(mod, attr, _LazyFunc(name, attr))
            sys.modules[name] = mod
            inserted.append(name)
        pr = sys.modules.get("pkg_resources")
        if isinstance(pr, _ShimModule):
            # prody.atomic.atomic calls getDataPath -> pkg_resources.
            # resource_filename("prody.utilities", ...) at module level, which
            # drags in all of pkg_resources.core (~92 ms with vendored
            # packaging/pyparsing). For a directory-backed package the return
            # value is exactly os.path.join(pkg_dir, resource); provide that
            # directly and delegate to the real module for anything exotic.
            def _resource_filename(package, resource):
                mod2 = sys.modules.get(package)
                base = None
                if mod2 is not None:
                    _p = getattr(mod2, "__path__", None)
                    if _p:
                        base = list(_p)[0]
                    elif getattr(mod2, "__file__", None):
                        base = os.path.dirname(mod2.__file__)
                if base is None or not os.path.isdir(base):
                    return getattr(_load_real("pkg_resources"), "resource_filename")(
                        package, resource
                    )
                return os.path.join(base, *resource.split("/"))

            pr.resource_filename = _resource_filename
        # prody.sequence is star-imported by prody/__init__ and from-imported
        # (MSA) by prody.proteins.compare, but none of its names are used on
        # the PDB-parsing/inference path. Pre-populating its public names as
        # lazy placeholders lets both import styles bind without loading the
        # real module (which pulls in Bio.AlignIO and the Bio.Nexus/
        # CodonTable chain, ~40 ms). Any real call delegates.
        seq_names = [
            "MSA", "refineMSA", "mergeMSA", "specMergeMSA", "MSAFile",
            "splitSeqLabel", "parseMSA", "writeMSA", "calcShannonEntropy",
            "buildMutinfoMatrix", "calcMSAOccupancy", "applyMutinfoCorr",
            "applyMutinfoNorm", "calcRankorder", "filterRankedPairs",
            "buildSeqidMatrix", "uniqueSequences", "buildOMESMatrix",
            "buildSCAMatrix", "buildDirectInfoMatrix", "calcMeff",
            "buildPCMatrix", "buildMSA", "showAlignment",
            "alignTwoSequencesWithBiopython", "alignSequenceToMSA",
            "calcPercentIdentities", "alignSequencesByChain",
            "trimAtomsUsingMSA", "showMSAOccupancy", "showShannonEntropy",
            "showMutinfoMatrix", "showDirectInfoMatrix", "showSCAMatrix",
            "Sequence",
        ]
        if "prody.sequence" not in sys.modules and "prody" in roots:
            mod = _ShimModule("prody.sequence")
            pkg_dir = os.path.join(roots["prody"], "sequence")
            if os.path.isdir(pkg_dir):
                mod.__path__ = [pkg_dir]
            mod.__all__ = list(seq_names)
            for attr in seq_names:
                setattr(mod, attr, _LazyFunc("prody.sequence", attr))
            sys.modules["prody.sequence"] = mod
            inserted.append("prody.sequence")
    except Exception:
        for name in inserted:
            sys.modules.pop(name, None)
        raise


try:
    _install_prody_dep_shims()
except Exception:
    pass  # never let the shim break startup; worst case is original behavior
# -----------------------------------------------------------------------------

from data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN
from prody import writePDB
from fast_pdb import FastPDBWriter


def main(args) -> None:
    """
    Inference function
    """
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    folder_for_outputs = args.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if not os.path.exists(base_folder + "seqs"):
        os.makedirs(base_folder + "seqs", exist_ok=True)
    if not os.path.exists(base_folder + "backbones"):
        os.makedirs(base_folder + "backbones", exist_ok=True)
    if not os.path.exists(base_folder + "packed"):
        os.makedirs(base_folder + "packed", exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + "stats"):
            os.makedirs(base_folder + "stats", exist_ok=True)
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    # load to CPU (plain memcpy into fresh params) then one .to(device) move;
    # avoids per-tensor CUDA restore (validate + sync H2D) + D2H load_state_dict
    # + H2D .to() triple transfer. Bit-identical weights either way.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    # Construct on the meta device: the default parameter init (incl. the
    # xavier pass) is discarded by load_state_dict anyway, and no torch CPU
    # RNG is consumed anywhere after this point (sampling uses the device
    # RNG), so skipping init leaves outputs bit-identical. assign=True adopts
    # the checkpoint tensors directly instead of copying into fresh ones.
    with torch.device("meta"):
        model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=device,
            atom_context_num=atom_context_num,
            model_type=args.model_type,
            ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
        )

    model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    del checkpoint  # free the CPU copy of the weights before the run

    # --- iter025: hide remaining CPU setup behind the CUDA context init ----
    # The warmup thread is still initializing the CUDA context (~200 ms) at
    # this point, so run the arg-derived path setup and the PDB parsing here
    # instead of after the join. parse_PDB's device kwarg only affects its
    # final output tensors; creating them safely serializes with the warmup
    # thread through torch's lazy-init lock and consumes no RNG.
    if args.pdb_path_multi:
        with open(args.pdb_path_multi, "r") as fh:
            pdb_paths = list(json.load(fh))
    else:
        pdb_paths = [args.pdb_path]

    if len(args.parse_these_chains_only) != 0:
        parse_these_chains_only_list = args.parse_these_chains_only.split(",")
    else:
        parse_these_chains_only_list = []

    parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
        args.pack_side_chains and not args.repack_everything
    )
    parsed_pdbs = {}
    for pdb in pdb_paths:
        parsed_pdbs[pdb] = parse_PDB(
            pdb,
            device=device,
            chains=parse_these_chains_only_list,
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )

    _cuda_warm_thread.join()
    model.to(device)
    model.eval()

    if args.pack_side_chains:
        # deferred import: sc_utils pulls in the openfold chain (~55 ms) and
        # is only exercised by side-chain packing mode
        from sc_utils import Packer, pack_side_chains

        model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=device,
            num_mix=3,
        )

        checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device)
        model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        model_sc.to(device)
        model_sc.eval()

    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, "r") as fh:
            fixed_residues_multi = json.load(fh)
            fixed_residues_multi = {key:value.split() for key,value in fixed_residues_multi.items()}
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdb_paths:
            fixed_residues_multi[pdb] = fixed_residues

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, "r") as fh:
            redesigned_residues_multi = json.load(fh)
            redesigned_residues_multi = {key:value.split() for key,value in redesigned_residues_multi.items()}
    else:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_residues_multi = {}
        for pdb in pdb_paths:
            redesigned_residues_multi[pdb] = redesigned_residues

    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_str_to_int[AA]] = a2[i]

    if args.bias_AA_per_residue_multi:
        with open(args.bias_AA_per_residue_multi, "r") as fh:
            bias_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue, "r") as fh:
                bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
            bias_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    if args.omit_AA_per_residue_multi:
        with open(args.omit_AA_per_residue_multi, "r") as fh:
            omit_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
    else:
        if args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue, "r") as fh:
                omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
            omit_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=device,
    )

    # loop over PDB paths
    for pdb in pdb_paths:
        if args.verbose:
            print("Designing protein from this path:", pdb)
        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]
        protein_dict, backbone, other_atoms, icodes, _ = parsed_pdbs[pdb]
        # Prewarm the one-time cublasLt per-shape heuristic lookups on a side
        # thread, overlapped with the CPU-bound residue/bias setup and
        # featurize below. The heuristic cache is process-global (verified:
        # a second thread's same-shape GEMM is warm), and the query path
        # releases the GIL (verified: full overlap with numpy work), so the
        # real encode()/decode-capture on the main thread then runs warm.
        # Only the distinct GEMM shapes are prewarmed (~23 F.linear calls,
        # minimal GIL footprint) -- a full dummy encode here is GIL-bound and
        # nets negative. Shapes derive from L and batch_size via the fixed
        # featurizer structure (k_neighbors=32, atom_context_num=25); a wrong
        # guess only wastes the worker, never changes results. No RNG use.
        _enc_warm_thread = None
        if device.type == "cuda" and args.model_type == "ligand_mpnn":

            def _encode_prewarm():
                try:
                    import torch.nn.functional as _F

                    L = int(protein_dict["X"].shape[-3])
                    B = int(args.batch_size)
                    shapes = [
                        (L * 32, 66, 16),
                        (L * 32, 416, 128),
                        (L * 25, 147, 64),
                        (L * 25, 148, 128),
                        (L * 625, 16, 128),
                        (L * 25, 147, 128),
                        (L * 32, 128, 128),
                        (L * 25, 128, 128),
                        (L * 32, 384, 128),
                        (L, 128, 512),
                        (L, 512, 128),
                        (L, 128, 128),
                        (L * 625, 128, 128),
                        (L * 625, 256, 128),
                        (L * 25, 128, 512),
                        (L * 25, 512, 128),
                        (L * 25, 384, 128),
                        # decode (per-step ops run at M=B; setup at M=B*32)
                        (B * 32, 512, 128),
                        (B * 32, 128, 128),
                        (B, 128, 512),
                        (B, 512, 128),
                        (B, 128, 128),
                        (B, 128, 21),
                    ]
                    for m, k, n in shapes:
                        _F.linear(
                            torch.zeros(m, k, device=device),
                            torch.zeros(n, k, device=device),
                        )
                    torch.cuda.synchronize()
                except Exception:
                    pass

            _enc_warm_thread = threading.Thread(target=_encode_prewarm)
            _enc_warm_thread.start()
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
            bias_dict = bias_AA_per_residue_multi[pdb]
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
            omit_dict = omit_AA_per_residue_multi[pdb]
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                args.global_transmembrane_label + 0 * fixed_positions
            )
        if len(args.chains_to_design) != 0:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]

        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            _chain_mask_list = protein_dict["chain_mask"].cpu().tolist()
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if _chain_mask_list[item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if _chain_mask_list[item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        # specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in args.symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        # specify linking weights
        if args.symmetry_weights:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in args.symmetry_weights.split("|")
            ]
        else:
            symmetry_weights = [[]]

        if args.homo_oligomer:
            if args.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:] for item in encoded_residues if item[:lc] == reference_chain
            ]
            remapped_symmetry_residues = []
            symmetry_weights = []
            for res in residue_indices:
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1 / len(chain_letters_set))
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if args.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif args.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    for i, atom_type in enumerate(atom_types):
                        print(
                            f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                        )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            if _enc_warm_thread is not None:
                _enc_warm_thread.join()
            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            rec_list = []
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            if args.model_type == "ligand_mpnn":
                combined_mask = (
                    feature_dict["mask"]
                    * feature_dict["mask_XY"]
                    * feature_dict["chain_mask"]
                )
            else:
                combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
            # Phase 1: enqueue every batch's GPU work back-to-back so the GPU
            # stays saturated; per-batch draining + writing happens in the
            # output loop below while later batches are still decoding.
            # Graph-decode batches run on per-slot side streams (see
            # model_utils._sample_graphed); the scoring ops stay on the same
            # slot stream so the default stream carries no per-batch work and
            # concurrent slots never serialize behind each other. One event
            # per batch marks its outputs as ready for the drain below.
            batch_events = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                output_dict = model.sample(feature_dict)

                _slot_stream = output_dict.get("_stream", None)
                _score_ctx = (
                    torch.cuda.stream(_slot_stream)
                    if _slot_stream is not None
                    else contextlib.nullcontext()
                )
                with _score_ctx:
                    # compute confidence scores
                    loss, loss_per_residue = get_score(
                        output_dict["S"],
                        output_dict["log_probs"],
                        feature_dict["mask"] * feature_dict["chain_mask"],
                    )
                    loss_XY, _ = get_score(
                        output_dict["S"], output_dict["log_probs"], combined_mask
                    )
                    # sequence recovery vs native; per-row sums are exact (0/1
                    # integer values in fp32), so per-batch rows are bit-identical
                    # to rows of the full-stack computation
                    rec_list.append(
                        get_seq_rec(feature_dict["S"][:1], output_dict["S"], rec_mask)
                    )
                if _slot_stream is not None:
                    _batch_event = torch.cuda.Event()
                    _batch_event.record(_slot_stream)
                    batch_events.append(_batch_event)
                else:
                    batch_events.append(None)
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            # Stacks are only needed for the stats file and the packing path;
            # the output loop below drains per batch.
            if args.save_stats or args.pack_side_chains:
                # stacked/packed paths read every batch's tensors at once;
                # settle the slot streams first (all None on the eager path)
                for _batch_event in batch_events:
                    if _batch_event is not None:
                        _batch_event.synchronize()
                S_stack = torch.cat(S_list, 0)
            if args.save_stats:
                log_probs_stack = torch.cat(log_probs_list, 0)
                sampling_probs_stack = torch.cat(sampling_probs_list, 0)
                decoding_order_stack = torch.cat(decoding_order_list, 0)

            native_seq = "".join(
                [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            mask_c_cpu = [mask.cpu().numpy() for mask in protein_dict["mask_c"]]
            seq_out_str = []
            for mask in mask_c_cpu:
                seq_out_str += list(seq_np[mask])
                seq_out_str += [args.fasta_seq_separation]
            seq_out_str = "".join(seq_out_str)[:-1]

            output_fasta = base_folder + "/seqs/" + name + args.file_ending + ".fa"
            output_backbones = base_folder + "/backbones/"
            output_packed = base_folder + "/packed/"
            output_stats_path = base_folder + "stats/" + name + args.file_ending + ".pt"

            if args.save_stats:
                out_dict = {}
                out_dict["generated_sequences"] = S_stack.cpu()
                out_dict["sampling_probs"] = sampling_probs_stack.cpu()
                out_dict["log_probs"] = log_probs_stack.cpu()
                out_dict["decoding_order"] = decoding_order_stack.cpu()
                out_dict["native_sequence"] = feature_dict["S"][0].cpu()
                out_dict["mask"] = feature_dict["mask"][0].cpu()
                out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
                out_dict["seed"] = seed
                out_dict["temperature"] = args.temperature
                torch.save(out_dict, output_stats_path)

            if args.pack_side_chains:
                if args.verbose:
                    print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=args.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = args.batch_size
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(args.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(args.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            model_sc,
                            args.sc_num_denoising_steps,
                            args.sc_num_samples,
                            args.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)

            fast_pdb_writer = FastPDBWriter(backbone, other_atoms)
            with open(output_fasta, "w") as f:
                f.write(
                    ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, ligand_cutoff_distance={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                        name,
                        args.temperature,
                        seed,
                        torch.sum(rec_mask).cpu().numpy(),
                        torch.sum(combined_mask[:1]).cpu().numpy(),
                        bool(args.ligand_mpnn_use_atom_context),
                        float(args.ligand_mpnn_cutoff_for_score),
                        args.batch_size,
                        args.number_of_batches,
                        checkpoint_path,
                        seq_out_str,
                    )
                )
                # Interleaved drain + write: all batches were enqueued in the
                # loop above, so the GPU keeps decoding later batches while the
                # CPU formats and writes each batch as soon as it is ready.
                # Per-batch rows are identical to slices of the stacked
                # tensors (cat and D2H are pure copies).
                ix = 0
                n_seqs = sum(s.shape[0] for s in S_list)
                for c in range(args.number_of_batches):
                    if batch_events[c] is not None:
                        # wait only for this batch's slot stream; earlier
                        # batches drain while later ones are still decoding
                        batch_events[c].synchronize()
                    S_c = S_list[c].cpu().numpy()
                    loss_c = loss_list[c].cpu().numpy()
                    loss_XY_c = loss_XY_list[c].cpu().numpy()
                    loss_per_residue_c = loss_per_residue_list[c].cpu().numpy()
                    rec_c = rec_list[c].cpu().numpy()
                    for r in range(S_c.shape[0]):
                        ix_suffix = ix
                        if not args.zero_indexed:
                            ix_suffix += 1
                        seq_rec_print = np.format_float_positional(
                            rec_c[r], unique=False, precision=4
                        )
                        loss_np = np.format_float_positional(
                            np.exp(-loss_c[r]), unique=False, precision=4
                        )
                        loss_XY_np = np.format_float_positional(
                            np.exp(-loss_XY_c[r]),
                            unique=False,
                            precision=4,
                        )
                        seq = "".join([restype_int_to_str[AA] for AA in S_c[r]])

                        # write new sequences into PDB with backbone coordinates
                        seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                            None,
                        ].repeat(4, 1)
                        bfactor_prody = loss_per_residue_c[r][None, :].repeat(4, 1)
                        backbone.setResnames(seq_prody)
                        backbone.setBetas(
                            np.exp(-bfactor_prody)
                            * (bfactor_prody > 0.01).astype(np.float32)
                        )
                        fast_pdb_writer.write(
                            output_backbones
                            + name
                            + "_"
                            + str(ix_suffix)
                            + args.file_ending
                            + ".pdb"
                        )

                        # write full PDB files
                        if args.pack_side_chains:
                            for c_pack in range(args.number_of_packs_per_design):
                                X_stack = X_stack_list[c_pack]
                                X_m_stack = X_m_stack_list[c_pack]
                                b_factor_stack = b_factor_stack_list[c_pack]
                                write_full_PDB(
                                    output_packed
                                    + name
                                    + args.packed_suffix
                                    + "_"
                                    + str(ix_suffix)
                                    + "_"
                                    + str(c_pack + 1)
                                    + args.file_ending
                                    + ".pdb",
                                    X_stack[ix].cpu().numpy(),
                                    X_m_stack[ix].cpu().numpy(),
                                    b_factor_stack[ix].cpu().numpy(),
                                    feature_dict["R_idx_original"][0].cpu().numpy(),
                                    protein_dict["chain_letters"],
                                    S_stack[ix].cpu().numpy(),
                                    other_atoms=other_atoms,
                                    icodes=icodes,
                                    force_hetatm=args.force_hetatm,
                                )
                        # -----

                        # write fasta lines
                        seq_np = np.array(list(seq))
                        seq_out_str = []
                        for mask in mask_c_cpu:
                            seq_out_str += list(seq_np[mask])
                            seq_out_str += [args.fasta_seq_separation]
                        seq_out_str = "".join(seq_out_str)[:-1]
                        if ix == n_seqs - 1:
                            # final 2 lines
                            f.write(
                                ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
                                    name,
                                    ix_suffix,
                                    args.temperature,
                                    seed,
                                    loss_np,
                                    loss_XY_np,
                                    seq_rec_print,
                                    seq_out_str,
                                )
                            )
                        else:
                            f.write(
                                ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
                                    name,
                                    ix_suffix,
                                    args.temperature,
                                    seed,
                                    loss_np,
                                    loss_XY_np,
                                    seq_rec_print,
                                    seq_out_str,
                                )
                            )
                        ix += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",
    )
    # protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    # ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    # per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    # global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    # soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    argparser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="./model_params/proteinmpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="./model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="./model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="./model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="./model_params/solublempnn_v_48_020.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--fasta_seq_separation",
        type=str,
        default=":",
        help="Symbol to use between sequences from different chains",
    )
    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument(
        "--pdb_path", type=str, default="", help="Path to the input PDB."
    )
    argparser.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    argparser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Provide fixed residues, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--bias_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'",
    )
    argparser.add_argument(
        "--bias_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}",
    )
    argparser.add_argument(
        "--bias_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}",
    )

    argparser.add_argument(
        "--omit_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'ACG'",
    )
    argparser.add_argument(
        "--omit_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': 'APQ', 'A13': 'QST'}",
    )
    argparser.add_argument(
        "--omit_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': 'QSPC', 'A13': 'AGE'}}",
    )

    argparser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    argparser.add_argument(
        "--symmetry_weights",
        type=str,
        default="",
        help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'",
    )
    argparser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    argparser.add_argument(
        "--out_folder",
        type=str,
        help="Path to a folder to output sequences, e.g. /home/out/",
    )
    argparser.add_argument(
        "--file_ending", type=str, default="", help="adding_string_to_the_end"
    )
    argparser.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="1 - to start output PDB numbering with 0",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for torch, numpy, and python random.",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequence to generate per one pass.",
    )
    argparser.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="Number of times to design sequence using a chosen batch size.",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature to sample sequences.",
    )
    argparser.add_argument(
        "--save_stats", type=int, default=0, help="Save output statistics"
    )

    argparser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use atom context, 0 - do not use atom context.",
    )
    argparser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )
    argparser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",
    )
    argparser.add_argument(
        "--chains_to_design",
        type=str,
        default="",
        help="Specify which chains to redesign, all others will be kept fixed, 'A,B,C,F'",
    )

    argparser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Provide chains letters for parsing backbones, 'A,B,C,F'",
    )

    argparser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    argparser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    argparser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    argparser.add_argument(
        "--pack_side_chains",
        type=int,
        default=0,
        help="1 - to run side chain packer, 0 - do not run it",
    )

    argparser.add_argument(
        "--checkpoint_path_sc",
        type=str,
        default="./model_params/ligandmpnn_sc_v_32_002_16.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--number_of_packs_per_design",
        type=int,
        default=4,
        help="Number of independent side chain packing samples to return per design",
    )

    argparser.add_argument(
        "--sc_num_denoising_steps",
        type=int,
        default=3,
        help="Number of denoising/recycling steps to make for side chain packing",
    )

    argparser.add_argument(
        "--sc_num_samples",
        type=int,
        default=16,
        help="Number of samples to draw from a mixture distribution and then take a sample with the highest likelihood.",
    )

    argparser.add_argument(
        "--repack_everything",
        type=int,
        default=0,
        help="1 - repacks side chains of all residues including the fixed ones; 0 - keeps the side chains fixed for fixed residues",
    )

    argparser.add_argument(
        "--force_hetatm",
        type=int,
        default=0,
        help="To force ligand atoms to be written as HETATM to PDB file after packing.",
    )

    argparser.add_argument(
        "--packed_suffix",
        type=str,
        default="_packed",
        help="Suffix for packed PDB paths",
    )

    argparser.add_argument(
        "--pack_with_ligand_context",
        type=int,
        default=1,
        help="1-pack side chains using ligand context, 0 - do not use it.",
    )

    args = argparser.parse_args()
    main(args)
    # Skip interpreter teardown (torch/CUDA/prody module cleanup, ~0.4 s).
    # All outputs are written and closed via `with` blocks before main()
    # returns, and all GPU work is drained by the final .cpu() transfers, so
    # teardown has no observable effect. Flush std streams and exit directly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
