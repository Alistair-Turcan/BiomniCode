import multiprocessing
import os
import pickle
import platform
import sys
import tarfile
import types
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import click
import torch

# Prevent the torch._dynamo / torch._inductor import chain (~2.4 s of
# interpreter startup) that pytorch_lightning pulls in via
# ``from torch._dynamo import OptimizedModule`` (lightning_fabric/wrappers.py).
# Nothing on the predict path compiles (all compile_* flags default False),
# so a minimal stub is numerically inert: disable / allow_in_graph /
# assume_constant_result become passthroughs (exactly the uncompiled
# behavior), is_compiling / is_exporting correctly return False, and config
# writes are discarded (they only affect compilation).
if "torch._dynamo" not in sys.modules:
    _dynamo_stub = types.ModuleType("torch._dynamo")

    class _OptimizedModuleStub(torch.nn.Module):
        pass

    def _dynamo_passthrough(fn=None, *args, **kwargs):
        if fn is None or not callable(fn):
            return lambda f: f
        return fn

    _dynamo_stub.OptimizedModule = _OptimizedModuleStub
    _dynamo_stub.disable = _dynamo_passthrough
    _dynamo_stub.allow_in_graph = _dynamo_passthrough
    _dynamo_stub.assume_constant_result = _dynamo_passthrough
    _dynamo_stub.is_compiling = lambda: False
    _dynamo_stub.is_exporting = lambda: False
    _dynamo_stub.config = types.SimpleNamespace()
    _dynamo_stub.reset = lambda *a, **k: None
    sys.modules["torch._dynamo"] = _dynamo_stub

# torch._library.triton.find_triton_kernels (used when cuequivariance
# registers its triton ops) does ``from torch._inductor.utils import
# IndentedBuffer``, which would execute the full torch._inductor package
# init (and pull in the real torch._dynamo). IndentedBuffer is just a
# string buffer re-exported from torch.utils._indented_buffer, so provide
# it directly and keep the heavy packages out of the process.
if "torch._inductor" not in sys.modules:
    from torch.utils._indented_buffer import IndentedBuffer as _IndentedBuffer

    _inductor_stub = types.ModuleType("torch._inductor")
    _inductor_utils_stub = types.ModuleType("torch._inductor.utils")
    _inductor_utils_stub.IndentedBuffer = _IndentedBuffer
    sys.modules["torch._inductor"] = _inductor_stub
    sys.modules["torch._inductor.utils"] = _inductor_utils_stub

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.mol import load_canonicals
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz2 import Boltz2

# Deferred imports: these are only executed on first use, which for the parse
# functions is inside process_input - a function warm predict runs against an
# existing processed/ directory never call - so their heavy transitive chains
# (pandas, Bio, sklearn, chembl_structure_pipeline, requests) stay unloaded on
# the measured path:
#   pytorch_lightning.strategies.DDPStrategy       -> multi-device branch
#   boltz.data.module.inference.BoltzInferenceDataModule -> boltz1 branch
#   boltz.data.msa.mmseqs2.run_mmseqs2             -> compute_msa (MSA server)
#   boltz.data.parse.{a3m,csv,fasta,yaml}          -> process_input
#   boltz.model.models.boltz1.Boltz1               -> boltz1 branch

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


@rank_zero_only
def download_boltz1(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ1_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ1_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


@rank_zero_only
def download_boltz2(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not tar_mols.exists():
        click.echo(
            f"Downloading the CCD data to {tar_mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
    if not mols.exists():
        click.echo(
            f"Extracting the CCD data to {mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)  # noqa: S202

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the Boltz-2 weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        click.echo(
            f"Downloading the Boltz-2 affinity weights to {affinity_model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(affinity_model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
    else:
        data = [data]

    return data


def filter_inputs_structure(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Filter the manifest to only include missing predictions.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    # Check if existing predictions are found (only top-level prediction folders)
    pred_dir = outdir / "predictions"
    if pred_dir.exists():
        existing = {d.name for d in pred_dir.iterdir() if d.is_dir()}
    else:
        existing = set()

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        msg = (
            f"Found some existing predictions ({len(existing)}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = f"Found {len(existing)} existing predictions, will override."
        click.echo(msg)

    return manifest


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    click.echo("Checking input data for affinity.")

    # Get all affinity targets
    existing = {
        r.id
        for r in manifest.records
        if r.affinity
        and (outdir / "predictions" / r.id / f"affinity_{r.id}.json").exists()
    }

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        num_skipped = len(existing)
        msg = (
            f"Found some existing affinity predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "affinity predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing affinity predictions, will override."
        click.echo(msg)

    return manifest


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    """
    click.echo(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    click.echo(f"MSA server URL: {msa_server_url}")
    click.echo(f"MSA pairing strategy: {msa_pairing_strategy}")
    
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {
            "Content-Type": "application/json",
            key: value
        }
        click.echo(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        click.echo("Using basic authentication for MSA server")
    else:
        click.echo("No authentication provided for MSA server")
    
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> None:
    product_cache = None
    try:
        from boltz.data.parsed_cache import ParsedProteinCache

        product_cache = ParsedProteinCache(
            path, ccd, mol_dir, boltz2 and not use_msa_server, max_msa_seqs,
            {"msa": processed_msa_dir, "constraints": processed_constraints_dir,
             "templates": processed_templates_dir, "mols": processed_mols_dir,
             "structures": structure_dir, "records": records_dir},
        )
        if product_cache.restore():
            return
    except Exception:
        # Cache failure must never prevent the established parser from running.
        product_cache = None
    try:
        # Parse data
        if path.suffix.lower() in (".fa", ".fas", ".fasta"):
            from boltz.data.parse.fasta import parse_fasta

            target = parse_fasta(path, ccd, mol_dir, boltz2)
        elif path.suffix.lower() in (".yml", ".yaml"):
            from boltz.data.parse.yaml import parse_yaml

            target = parse_yaml(path, ccd, mol_dir, boltz2)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)  # noqa: TRY301
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)  # noqa: TRY301

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)  # noqa: TRY301

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    from boltz.data.parse.a3m import parse_a3m

                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    from boltz.data.parse.csv import parse_csv

                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)  # noqa: TRY301

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)
        if product_cache is not None:
            try:
                product_cache.save()
            except Exception:
                pass

    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"Failed to process {path}. Skipping. Error: {e}.")  # noqa: T201


@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    use_msa_server: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 8192.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    msa_server_username : str, optional
        Username for basic authentication with MSA server, by default None.
    msa_server_password : str, optional
        Password for basic authentication with MSA server, by default None.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_api_key = api_key_value is not None
    
    if has_basic_auth and has_api_key:
        raise ValueError(
            "Cannot use both basic authentication (--msa_server_username/--msa_server_password) "
            "and API key authentication (--api_key_header/--api_key_value). Please use only one authentication method."
        )

    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    if records_dir.exists():
        # Load existing records
        existing = [Record.load(p) for p in records_dir.glob("*.json")]
        processed_ids = {record.id for record in existing}

        # Filter to missing only
        data = [d for d in data if d.stem not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            click.echo(
                f"Found {len(existing)} existing processed inputs, skipping them."
            )
        else:
            click.echo("All inputs are already processed.")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD (skipped when every input is already processed: ccd is only
    # consumed by process_input, which never runs for an empty data list)
    ccd = None
    if data:
        if boltz2:
            ccd = load_canonicals(mol_dir)
        else:
            with ccd_path.open("rb") as file:
                ccd = pickle.load(file)  # noqa: S301

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            list(tqdm(pool.imap(process_input_partial, data), total=len(data)))
    else:
        for path in tqdm(data):
            process_input_partial(path)

    # Load all records and write manifest
    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz."""
    return


class _MmapTorchLoad:
    """Force mmap=True on torch.load for .ckpt files (bitwise-identical tensors).

    Lightning's pl_load calls torch.load without mmap, reading the whole
    archive through intermediate buffers. mmap=True lets the OS page tensor
    storage directly; the resulting tensors are identical.
    """

    def __enter__(self):
        self._orig_load = torch.load

        def _load_mmap(*args, **kwargs):
            f = args[0] if args else kwargs.get("f")
            # Lightning's pl_load opens the checkpoint via fsspec and passes a
            # file OBJECT (lightning_fabric/utilities/cloud_io.py::_load), so a
            # plain str check never matches; resolve the path from .name.
            name = f if isinstance(f, (str, Path)) else getattr(f, "name", None)
            if name is not None and str(name).endswith(".ckpt"):
                kwargs.setdefault("mmap", True)
                if not isinstance(f, (str, Path)):
                    # torch.load(mmap=True) needs a real path: re-open by name.
                    return self._orig_load(str(name), **kwargs)
            return self._orig_load(*args, **kwargs)

        torch.load = _load_mmap
        return self

    def __exit__(self, *exc):
        torch.load = self._orig_load


class _SidecarCkptLoad:
    """Zero-copy sidecar cache for torch.load'ed .ckpt files.

    torch.load on the multi-GB Lightning checkpoints spends ~1.7 s of host
    time unpickling ~5k tensor records out of the zip container. The tensor
    bytes are static, so the first (cold) load additionally writes a sidecar
    next to the checkpoint: ``<name>.ckpt.stcache.bin`` holding the state_dict
    tensors' raw bytes back-to-back (64-byte aligned) and
    ``<name>.ckpt.stcache.meta`` holding a pickle of every other top-level
    entry plus the tensor index. Warm loads rebuild the checkpoint dict with
    mmap-backed copy-on-write tensors via torch.frombuffer - bitwise-identical
    values, and the pages stay file-backed exactly like the mmap'd torch.load
    path, so peak RSS is unchanged. ``optimizer_states`` (training-resume
    state, ~2x the parameters in tensors) is not carried into the sidecar:
    load_from_checkpoint only reads ``state_dict`` and the hyper-parameters
    (pytorch_lightning/core/saving.py), and optimizer state cannot influence
    prediction numerics. Any sidecar staleness (checkpoint size/mtime
    mismatch), corruption, or write failure falls back to the wrapped
    torch.load, so a missing or invalid cache never changes results.
    """

    def __enter__(self):
        self._orig_load = torch.load
        torch.load = self._load
        return self

    def __exit__(self, *exc):
        torch.load = self._orig_load

    def _load(self, *args, **kwargs):
        f = args[0] if args else kwargs.get("f")
        path = None
        # Lightning's pl_load passes an fsspec LocalFileOpener, not a path;
        # resolve the checkpoint path from the file object's .name.
        name = f if isinstance(f, (str, Path)) else getattr(f, "name", None)
        if name is not None and str(name).endswith(".ckpt"):
            path = Path(str(name))
        if path is None:
            return self._orig_load(*args, **kwargs)
        bin_path = path.with_name(path.name + ".stcache.bin")
        meta_path = path.with_name(path.name + ".stcache.meta")
        try:
            st = path.stat()
        except OSError:
            st = None
        if st is not None and bin_path.exists() and meta_path.exists():
            try:
                with meta_path.open("rb") as fh:
                    meta = pickle.load(fh)  # noqa: S301
                if (
                    meta.get("ckpt_size") == st.st_size
                    and meta.get("ckpt_mtime_ns") == st.st_mtime_ns
                ):
                    return self._from_sidecar(meta, bin_path)
            except Exception:  # noqa: BLE001 - corrupt cache: fall back
                pass
        obj = self._orig_load(*args, **kwargs)
        if st is not None:
            try:
                self._write_sidecar(obj, st, bin_path, meta_path)
            except Exception:  # noqa: BLE001 - caching must never break loads
                pass
        return obj

    @staticmethod
    def _strip_tensors(value):
        """Drop tensor leaves outside state_dict (EMA shadow weights).

        The checkpoint's callbacks['EMA']['ema_weights'] aliases the state_dict
        storages in the original archive (pickle memoization keeps the file at
        ~1x the parameters); pickling it standalone would duplicate ~2 GB into
        the meta file. It is training-resume state: boltz2.on_load_checkpoint
        never reads callbacks, and load_from_checkpoint only reads state_dict
        and hyper_parameters, so dropping it cannot change predictions.
        """
        if isinstance(value, dict):
            return {
                k: _SidecarCkptLoad._strip_tensors(v)
                for k, v in value.items()
                if not torch.is_tensor(v)
            }
        if isinstance(value, (list, tuple)):
            return type(value)(_SidecarCkptLoad._strip_tensors(v) for v in value)
        return value

    @staticmethod
    def _write_sidecar(obj, st, bin_path, meta_path):
        if not isinstance(obj, dict):
            return
        state_dict = obj.get("state_dict")
        if not state_dict or not all(
            torch.is_tensor(t) for t in state_dict.values()
        ):
            return
        entries = {
            k: _SidecarCkptLoad._strip_tensors(v)
            for k, v in obj.items()
            if k not in ("state_dict", "optimizer_states")
        }
        index = []
        offset = 0
        tmp_bin = bin_path.with_name(bin_path.name + ".tmp")
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        with tmp_bin.open("wb") as fh:
            for name, tensor in state_dict.items():
                t = tensor.detach().contiguous()
                raw = t.view(torch.uint8).numpy()
                fh.write(raw)
                index.append(
                    (name, str(t.dtype).split(".")[-1], list(t.shape), offset, raw.nbytes)
                )
                pad = (-raw.nbytes) % 64
                if pad:
                    fh.write(b"\0" * pad)
                offset += raw.nbytes + pad
        meta = {
            "version": 1,
            "ckpt_size": st.st_size,
            "ckpt_mtime_ns": st.st_mtime_ns,
            "entries": entries,
            "tensor_index": index,
        }
        with tmp_meta.open("wb") as fh:
            pickle.dump(meta, fh)
        os.replace(tmp_bin, bin_path)
        os.replace(tmp_meta, meta_path)

    def _from_sidecar(self, meta, bin_path):
        import math
        import mmap

        # Per-tensor page-aligned mmap slices instead of one whole-file
        # mapping. The kernel populates an entire VMA on first touch of a
        # page-cache-warm file (measured: a 194 kB copy from a 2 GB mapping
        # makes all 2 GB resident; a 64 MB mapping populates exactly 64 MB),
        # so a single whole-file mapping turns the first HtoD copy into a
        # 2 GB RSS transient. Bounding each mapping to one tensor's page span
        # caps the fault transient at one tensor, and the slice is munmap'd
        # (pages dropped) as soon as the CPU parameter is freed after its
        # HtoD copy — torch.frombuffer keeps the slice alive via the tensor
        # storage. Kill-switch: BOLTZ_NO_SIDECAR_SLICES restores the single
        # whole-file mapping.
        page = 4096
        state = {}
        if os.environ.get("BOLTZ_NO_SIDECAR_SLICES"):
            fh = bin_path.open("rb")
            try:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
            finally:
                fh.close()
            for name, dtype_name, shape, off, nbytes in meta["tensor_index"]:
                dt = getattr(torch, dtype_name)
                count = math.prod(shape) if shape else 1
                t = torch.frombuffer(mm, dtype=dt, count=count, offset=off)
                state[name] = t.view(shape) if shape else t.reshape(())
        else:
            fh = bin_path.open("rb")
            try:
                fileno = fh.fileno()
                fsize = bin_path.stat().st_size
                for name, dtype_name, shape, off, nbytes in meta["tensor_index"]:
                    dt = getattr(torch, dtype_name)
                    count = math.prod(shape) if shape else 1
                    if nbytes == 0:
                        state[name] = torch.empty(shape, dtype=dt)
                        continue
                    lo = off & ~(page - 1)
                    # Clamp to the file size: the page-aligned end of the last
                    # record can exceed it, and Python's mmap rejects a length
                    # larger than the remaining file.
                    hi = min((off + nbytes + page - 1) & ~(page - 1), fsize)
                    mm = mmap.mmap(
                        fileno, hi - lo, access=mmap.ACCESS_COPY, offset=lo
                    )
                    t = torch.frombuffer(mm, dtype=dt, count=count, offset=off - lo)
                    state[name] = t.view(shape) if shape else t.reshape(())
            finally:
                fh.close()
        ckpt = dict(meta["entries"])
        ckpt["state_dict"] = state
        return ckpt


def _trim_host_memory():
    """Return freed heap pages to the OS between pipeline phases.

    gc + malloc_trim(0): peak-RSS hygiene only; no tensor values change.
    Freed construction garbage stays in glibc arenas until malloc_trim
    returns it, so this matters even when the freeing itself is refcounted.
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _materialize_meta_tensors(module):
    """Materialize meta-device tensors left over from a meta construction.

    Strict assign=True loading rebinds every parameter and persistent buffer
    to real checkpoint storage, so only non-persistent state survives a meta
    construction as meta: the two scalar ``zero`` buffers (AtomDiffusion and
    DiffusionModuleV2, both ``torch.tensor(0.0)``) and the torchmetrics
    MeanMetric states (``sum``=0.0, ``count``=0). All are zero-valued, so
    zeros of the same shape/dtype restore them exactly. This runs inside the
    load_state_dict patch so it completes before Lightning's terminal
    ``model.to(device)``, which cannot copy out of a meta tensor. Explicit
    ``device='cpu'`` because the ambient device is still meta here.
    """
    import torchmetrics

    for mod in module.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                mod._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
        if isinstance(mod, torchmetrics.Metric):
            # The metric's _device attribute was set to meta at construction;
            # Metric._apply builds a dummy tensor on self.device, so reset it.
            if getattr(mod, "_device", None) is not None and str(mod._device) == "meta":
                mod._device = torch.device("cpu")
            for key, val in list(mod._defaults.items()):
                if isinstance(val, torch.Tensor) and val.device.type == "meta":
                    mod._defaults[key] = torch.zeros(val.shape, dtype=val.dtype, device="cpu")
                cur = getattr(mod, key, None)
                if isinstance(cur, torch.Tensor) and cur.device.type == "meta":
                    setattr(mod, key, torch.zeros(cur.shape, dtype=cur.dtype, device="cpu"))


class _AssignStateDictLoad:
    """Force load_state_dict(assign=True) within the context.

    assign=True binds the checkpoint tensors directly instead of copy_-ing
    them into the constructed parameters. Values are bitwise-identical when
    checkpoint and constructed dtypes match (verified: the boltz2 checkpoints
    are uniformly float32, matching default construction), but the load skips
    a full extra copy of every parameter, halving the peak-memory transient
    of a concurrent model load.
    """

    def __enter__(self):
        self._orig_lsd = torch.nn.Module.load_state_dict

        def _load_assign(module, state_dict, strict=True, **kwargs):
            kwargs["assign"] = True
            result = self._orig_lsd(module, state_dict, strict=strict, **kwargs)
            _materialize_meta_tensors(module)
            return result

        torch.nn.Module.load_state_dict = _load_assign
        return self

    def __exit__(self, *exc):
        torch.nn.Module.load_state_dict = self._orig_lsd


class _CompilerFingerprintPrefetch:
    """Overlap Triton's original full-content fingerprint with construction.

    The caller must fork any early data workers before entering and join this
    context before prediction can fork another worker. No key is substituted:
    the original functools-cached function hashes exactly the original bytes.
    An optional warmup failure is left to the ordinary first kernel call, which
    retains Triton's normal failure/fallback behavior.
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.thread = None
        self.key = None
        self.error = None

    def _compute(self):
        try:
            from triton.runtime.cache import triton_key

            self.key = triton_key()
        except Exception as exc:  # optional preparation; normal dispatch retries
            self.error = exc

    def __enter__(self):
        if self.enabled:
            import threading

            self.thread = threading.Thread(
                target=self._compute, name="boltz-compiler-fingerprint", daemon=True
            )
            try:
                self.thread.start()
            except (RuntimeError, OSError) as exc:
                self.error = exc
                self.thread = None
        return self

    def __exit__(self, *exc):
        if self.thread is not None:
            self.thread.join()


class _InitRngStateCache:
    """Memoize the CPU RNG-state transition of model construction.

    Strict checkpoint loading overwrites every parameter, so the random init
    fills during construction (nn.Linear kaiming_uniform_, nn.Embedding
    normal_, boltz's glorot/normal_init_) are dead work; only their
    advancement of the default CPU generator must be reproduced so downstream
    draws (trunk randperm, dataloader base seed) stay bitwise identical. On a
    cache hit the fills are skipped and the exact post-construction generator
    state is restored; on a miss construction runs normally and the transition
    is recorded. Keyed by the pre-construction generator state and the
    construction kwargs, so any change to seeding, inputs, or model config
    falls back to a full (correct) construction. Any init draw the no-op set
    misses is still corrected by the explicit state restore, so patch coverage
    only affects speed, never correctness.
    """

    _INIT_FUNCS = (
        "kaiming_uniform_", "kaiming_normal_", "normal_", "uniform_",
        "xavier_uniform_", "xavier_normal_", "trunc_normal_", "orthogonal_",
        "sparse_",
    )

    def __init__(self, cache_path, model_name, construct_kwargs):
        self._cache_path = Path(cache_path)
        self._model_name = model_name
        self._construct_kwargs = construct_kwargs
        self._saved = None
        self._post_state = None
        self._meta_ctx = None

    def peek_post_state(self):
        """Read a known transition without changing RNG, init functions or device."""
        import base64
        import hashlib
        import json
        import numpy as np

        digest = hashlib.sha256()
        digest.update(torch.get_rng_state().numpy().tobytes())
        digest.update(self._model_name.encode())
        digest.update(
            repr(sorted((k, repr(v)) for k, v in self._construct_kwargs.items())).encode()
        )
        try:
            encoded = json.loads(self._cache_path.read_text()).get(digest.hexdigest())
            if encoded is not None:
                return torch.from_numpy(
                    np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).copy())
        except (OSError, ValueError):
            pass
        return None

    def __enter__(self):
        import base64
        import hashlib
        import json

        self._base64 = base64
        self._json = json
        pre_state = torch.get_rng_state()
        digest = hashlib.sha256()
        digest.update(pre_state.numpy().tobytes())
        digest.update(self._model_name.encode())
        digest.update(
            repr(sorted((k, repr(v)) for k, v in self._construct_kwargs.items()))
            .encode()
        )
        self._key = digest.hexdigest()
        try:
            with open(self._cache_path) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
        self._cache = cache
        encoded = cache.get(self._key)
        if encoded is not None:
            import numpy as np

            self._post_state = torch.from_numpy(
                np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).copy()
            )
            # Construct on the meta device: strict assign=True loading rebinds
            # every parameter to the checkpoint storage, so the host storage
            # allocated during construction is dead. Under gVisor the many
            # medium-sized anonymous mmaps backing those parameters are
            # populated eagerly (~85% resident), producing a ~1 GB construction
            # transient that sets the process peak RSS. Meta construction
            # allocates no host storage. RNG-safe: the init fills are no-ops on
            # a hit and the exact post-construction generator state is restored
            # on exit, so construction-time draws (absent on meta) do not
            # affect the stream.
            self._meta_ctx = torch.device("meta")
            self._meta_ctx.__enter__()
            # Skip the dead init fills; the exact post-construction generator
            # state is restored on exit.
            self._saved = {}
            for name in self._INIT_FUNCS:
                fn = getattr(torch.nn.init, name, None)
                if fn is not None:
                    self._saved[name] = fn
                    setattr(torch.nn.init, name, lambda tensor, *a, **k: tensor)
        return self

    def __exit__(self, *exc):
        if self._meta_ctx is not None:
            self._meta_ctx.__exit__(*exc)
            self._meta_ctx = None
        if self._saved is not None:
            for name, fn in self._saved.items():
                setattr(torch.nn.init, name, fn)
            self._saved = None
        if exc[0] is not None:
            return  # propagate; never record or restore on failure
        if self._post_state is not None:
            torch.set_rng_state(self._post_state)
        else:
            post_state = torch.get_rng_state()
            self._cache[self._key] = self._base64.b64encode(
                post_state.numpy().tobytes()
            ).decode()
            try:
                with open(self._cache_path, "w") as f:
                    self._json.dump(self._cache, f)
            except OSError:
                pass


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--msa_server_username",
    type=str,
    help="MSA server username for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_USERNAME environment variable.",
    default=None,
)
@click.option(
    "--msa_server_password",
    type=str,
    help="MSA server password for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_PASSWORD environment variable.",
    default=None,
)
@click.option(
    "--api_key_header",
    type=str,
    help="Custom header key for API key authentication (default: X-API-Key).",
    default=None,
)
@click.option(
    "--api_key_value",
    type=str,
    help="Custom header value for API key authentication.",
    default=None,
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--write_embeddings",
    is_flag=True,
    help=" to dump the s and z embeddings into a npz file. Default is False.",
)
def predict(  # noqa: C901, PLR0915, PLR0912
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 3,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    write_embeddings: bool = False,
) -> None:
    """Run predictions with Boltz."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Supress some lightning warnings
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )

    # Set no grad
    torch.set_grad_enabled(False)

    # Preserve upstream inference precision; speedups must not lower this policy.
    torch.set_float32_matmul_precision("highest")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Get MSA server credentials from environment variables if not provided
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")
        
        click.echo(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            click.echo("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            click.echo("MSA server authentication: using basic auth")
        else:
            click.echo("MSA server authentication: no credentials provided")

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache)
    elif model == "boltz2":
        download_boltz2(cache)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs
    data = check_inputs(data)

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" and platform.system() != "Windows" else "spawn"
        from pytorch_lightning.strategies import DDPStrategy

        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            click.echo(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Set up model parameters
    if model == "boltz2":
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
    else:
        diffusion_params = BoltzDiffusionParams()
        step_scale = 1.638 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgs()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=model == "boltz2",
        write_embeddings=write_embeddings,
    )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )

    # The process exits via os._exit(0) immediately after the predictions below,
    # and every output is written and closed by the writers during the predict
    # loops, so Lightning's per-predict strategy teardown - moving each model
    # back to CPU (~1.5s per model), clearing the CUDA cache, plugin teardown -
    # is dead work. Skip it. (Teardown draws no RNG and touches no output file;
    # Trainer._teardown still runs the loop/logger/signal teardown.)
    trainer.strategy.teardown = lambda: None

    # Holds the affinity-model preload state when the affinity checkpoint load
    # is overlapped with the GPU-bound structure diffusion phase.
    affinity_preload = None

    if filtered_manifest.records:
        msg = f"Running structure prediction for {len(filtered_manifest.records)} input"
        msg += "s." if len(filtered_manifest.records) > 1 else "."
        click.echo(msg)

        # Create data module
        if model == "boltz2":
            data_module = Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        else:
            from boltz.data.module.inference import BoltzInferenceDataModule

            data_module = BoltzInferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
            )

        # Load model
        if checkpoint is None:
            if model == "boltz2":
                checkpoint = cache / "boltz2_conf.ckpt"
            else:
                checkpoint = cache / "boltz1_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials

        if model == "boltz2":
            model_cls = Boltz2
        else:
            from boltz.model.models.boltz1 import Boltz1

            model_cls = Boltz1
        # Strict checkpoint loading below overwrites every parameter, so the
        # scipy-based random weight init during construction is dead work.
        from boltz.model.layers import initialize

        initialize.SKIP_RANDOM_INIT = True
        load_kwargs = {
            "strict": True,
            "predict_args": predict_args,
            "map_location": "cpu",
            "diffusion_process_args": asdict(diffusion_params),
            "ema": False,
            "use_kernels": not no_kernels,
            "pairformer_args": asdict(pairformer_args),
            "msa_args": asdict(msa_args),
            "steering_args": asdict(steering_args),
        }
        # assign=True binds the mmap'd checkpoint tensors directly (values are
        # bitwise-identical: the boltz2 checkpoints are uniformly float32,
        # matching default construction), skipping a full CPU-CPU copy of every
        # parameter. The RNG-state cache skips the dead construction init fills
        # and restores the exact post-construction generator state.
        init_rng_cache = _InitRngStateCache(
            cache / "boltz_init_rng_cache.json", model, load_kwargs
        )
        overlap_enabled = (
            model == "boltz2" and accelerator == "gpu" and devices == 1
            and not os.environ.get("BOLTZ_NO_STARTUP_OVERLAP")
        )
        if overlap_enabled and num_workers > 0:
            future_state = init_rng_cache.peek_post_state()
            if future_state is not None:
                data_module.prime_prediction_loader(future_state)
        # Workers above are already forked. Join hashing on leaving construction,
        # before Trainer prediction (including mismatch-fallback worker forks).
        with _CompilerFingerprintPrefetch(overlap_enabled and not no_kernels), \
                _MmapTorchLoad(), _SidecarCkptLoad(), _AssignStateDictLoad(), init_rng_cache:
            model_module = model_cls.load_from_checkpoint(checkpoint, **load_kwargs)
        model_module.eval()
        if not os.environ.get("BOLTZ_NO_RSS_TRIM"):
            _trim_host_memory()

        # Overlap the affinity checkpoint load with the GPU-bound structure
        # diffusion phase. That window makes no default-CPU-generator draws
        # (diffusion noise, random augmentation, and dropout masks all draw on
        # CUDA; the trunk randperm and dataloader base_seed draws all precede
        # the last structure_module.sample() call), so constructing the
        # affinity model on a daemon thread during that window preserves the
        # exact CPU RNG sequence: the thread's init draws land where the serial
        # affinity construct's draws would, and the thread is joined before the
        # affinity dataloader's base_seed draw.
        if model == "boltz2" and any(r.affinity for r in manifest.records):
            import ctypes
            import threading

            # Pin all threads to the main malloc arena so memory freed after
            # the preload (e.g. when the model moves to the GPU) stays reusable
            # by the main thread instead of being stranded in a per-thread
            # arena, which would force fresh pages at teardown and inflate the
            # peak RSS. Large tensor allocations use mmap directly and are
            # unaffected; only small-object allocation lock contention is added.
            try:
                ctypes.CDLL("libc.so.6").mallopt(-8, 1)  # M_ARENA_MAX = 1
            except OSError:
                pass

            if affinity_checkpoint is None:
                affinity_checkpoint = cache / "boltz2_aff.ckpt"

            preload_affinity_args = {
                "recycling_steps": 5,
                "sampling_steps": sampling_steps_affinity,
                "diffusion_samples": diffusion_samples_affinity,
                "max_parallel_samples": 1,
                "write_confidence_summary": False,
                "write_full_pae": False,
                "write_full_pde": False,
            }
            preload_steering_args = BoltzSteeringParams()
            preload_steering_args.fk_steering = False
            preload_steering_args.physical_guidance_update = False
            preload_steering_args.contact_guidance_update = False

            preload_event = threading.Event()
            preload_result = {}
            # batch_size is 1, so sample() is called once per record; the last
            # call marks the start of the final diffusion phase, by which point
            # every trunk randperm draw has completed.
            preload_state = {"remaining": len(filtered_manifest.records)}
            orig_sample = model_module.structure_module.sample

            def _sample_gate(*args, **kwargs):
                preload_state["remaining"] -= 1
                if preload_state["remaining"] <= 0:
                    preload_event.set()
                return orig_sample(*args, **kwargs)

            model_module.structure_module.sample = _sample_gate

            def _affinity_preload_worker() -> None:
                preload_event.wait()
                try:
                    # assign=True binds the mmap'd checkpoint tensors directly
                    # (no CPU-CPU copy). The RNG-state cache skips the dead
                    # construction init fills on warm runs and restores the
                    # exact post-construction generator state, so the ~2 GB of
                    # soon-discarded construction tensors are never written
                    # (never become resident). Safe here because the diffusion
                    # window makes no default-CPU-generator draws (see the
                    # preload comment above), so the worker's pre-construction
                    # state — and thus the cache key — is deterministic.
                    aff_load_kwargs = {
                        "strict": True,
                        "predict_args": preload_affinity_args,
                        "map_location": "cpu",
                        "diffusion_process_args": asdict(diffusion_params),
                        "ema": False,
                        "pairformer_args": asdict(pairformer_args),
                        "msa_args": asdict(msa_args),
                        "steering_args": asdict(preload_steering_args),
                        "affinity_mw_correction": affinity_mw_correction,
                    }
                    with _MmapTorchLoad(), _SidecarCkptLoad(), _AssignStateDictLoad(), _InitRngStateCache(
                        cache / "boltz_init_rng_cache.json", "boltz2_aff", aff_load_kwargs
                    ):
                        preloaded = Boltz2.load_from_checkpoint(
                            affinity_checkpoint, **aff_load_kwargs
                        )
                    preloaded.eval()
                    if accelerator == "gpu" and torch.cuda.is_available():
                        # iter-011: move the affinity model to the GPU in
                        # this worker so the per-tensor HtoD upload
                        # overlaps the structure phase instead of
                        # serializing inside the affinity trainer.predict
                        # (Lightning's later move becomes a per-parameter
                        # no-op). Numerics-neutral.
                        if not os.environ.get("BOLTZ_NO_RSS_TRIM"):
                            _trim_host_memory()
                        preloaded.cuda()
                        if not os.environ.get("BOLTZ_NO_RSS_TRIM"):
                            _trim_host_memory()
                    preload_result["model"] = preloaded
                except BaseException as exc:  # noqa: BLE001
                    preload_result["error"] = exc

            preload_thread = threading.Thread(
                target=_affinity_preload_worker, daemon=True
            )
            preload_thread.start()
            affinity_preload = {
                "thread": preload_thread,
                "event": preload_event,
                "result": preload_result,
            }

        # Compute structure predictions
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )

        if affinity_preload is not None:
            # Guarantee the worker runs even if the sample() gate miscounted
            # (the load then happens serially, at the same CPU-RNG sequence
            # position as before). Normally the thread finished during the
            # diffusion phase and this join is free.
            affinity_preload["event"].set()
            affinity_preload["thread"].join()

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        # Print header
        click.echo("\nPredicting property: affinity\n")

        # Validate inputs
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        if not manifest_filtered.records:
            click.echo("Found existing affinity predictions for all inputs, skipping.")
            return

        msg = f"Running affinity prediction for {len(manifest_filtered.records)} input"
        msg += "s." if len(manifest_filtered.records) > 1 else "."
        click.echo(msg)

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False
        
        if affinity_preload is not None:
            # The preload thread was already joined right after the structure
            # predictions completed.
            if "error" in affinity_preload["result"]:
                raise affinity_preload["result"]["error"]
            model_module = affinity_preload["result"]["model"]
        else:
            with _MmapTorchLoad():
                model_module = Boltz2.load_from_checkpoint(
                    affinity_checkpoint,
                    strict=True,
                    predict_args=predict_affinity_args,
                    map_location="cpu",
                    diffusion_process_args=asdict(diffusion_params),
                    ema=False,
                    pairformer_args=asdict(pairformer_args),
                    msa_args=asdict(msa_args),
                    steering_args=asdict(steering_args),
                    affinity_mw_correction=affinity_mw_correction,
                )
            model_module.eval()

        trainer.callbacks[0] = pred_writer
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )

    # All prediction outputs are written and closed by the writers' context
    # managers above; skip the remaining interpreter/Lightning teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli()
