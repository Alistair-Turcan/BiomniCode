"""Optional exact initializer accelerator, guarded to the validated Torch ABI."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import torch


def _cache_identity():
    source = Path(__file__).with_suffix(".cpp")
    include = Path(torch.__file__).parent / "include"
    headers = {
        name: hashlib.sha256((include / name).read_bytes()).hexdigest()
        for name in ("ATen/CPUGeneratorImpl.h", "ATen/core/MT19937RNGEngine.h")
    }
    identity = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "python_abi": sys.implementation.cache_tag,
        "platform": platform.system(), "machine": platform.machine(),
        "headers": headers, "flags": ["-O3"], "cache_format": 1,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, key


def _valid_cache(directory, identity):
    library = directory / "boltzgen_design_rng_advance.so"
    receipt = directory / "identity.json"
    if not library.is_file() or not receipt.is_file():
        return False
    try:
        data = json.loads(receipt.read_text())
        return (
            data["identity"] == identity
            and data["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
        )
    except (OSError, ValueError, KeyError):
        return False


@lru_cache(maxsize=1)
def _native_design_uniform_advance():
    if torch.__version__.split("+")[0] != "2.7.1":
        return None
    try:
        identity, key = _cache_identity()
        base = Path(os.environ.get("BOLTZGEN_NATIVE_CACHE",
                                   str(Path.home() / ".cache/boltzgen/design_rng")))
        directory = base / key
        library = directory / "boltzgen_design_rng_advance.so"
        if _valid_cache(directory, identity):
            torch.ops.load_library(str(library))
            return torch.ops.boltzgen_design_rng.advance
        from torch.utils.cpp_extension import load
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="build-", dir=base) as build:
            built = load(
                name="boltzgen_design_rng_advance",
                sources=[str(Path(__file__).with_suffix(".cpp"))],
                extra_cflags=["-O3"],
                build_directory=build,
                is_python_module=False,
                verbose=False,
            )
            directory.mkdir(exist_ok=True)
            staging = Path(build) / "identity.json"
            data = {
                "identity": identity,
                "library_sha256": hashlib.sha256(Path(built).read_bytes()).hexdigest(),
            }
            staging.write_text(json.dumps(data, sort_keys=True) + "\n")
            # Publish complete immutable content before the matching identity.
            binary = Path(build) / "published.so"
            shutil.copyfile(built, binary)
            os.replace(binary, library)
            os.replace(staging, directory / "identity.json")
        return torch.ops.boltzgen_design_rng.advance
    except (ImportError, RuntimeError, OSError):
        # Unsupported builds keep the fully eager public behavior.
        return None
