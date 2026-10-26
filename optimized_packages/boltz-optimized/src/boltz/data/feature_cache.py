"""Exact CPU featurization memoization with explicit global RNG transitions."""
import hashlib
import importlib.metadata
import json
import marshal
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import tempfile

import numpy as np
import torch

from boltz.data import const


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def supported(value):
    if isinstance(value, torch.Tensor):
        return value.device.type == "cpu"
    if isinstance(value, dict):
        return all(isinstance(key, str) and supported(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(supported(item) for item in value)
    return isinstance(value, (np.ndarray, np.generic, str, int, float, bool, type(None)))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CPUFeatureCache:
    def __init__(self, mol_dir, featurizer, input_data, tokenized, molecules, settings):
        self.entry = None
        method = featurizer.process
        if type(featurizer).__module__ != "boltz.data.feature.featurizerv2" or (
            method.__func__.__module__ != "boltz.data.feature.featurizerv2"
        ):
            return
        self.before = rng_state()
        source_root = Path(__file__).resolve().parents[1]
        source = [(str(p.relative_to(source_root)), digest(p))
                  for p in sorted(source_root.rglob("*.py"))]
        versions = [(p, importlib.metadata.version(p)) for p in
                    ("numpy", "scipy", "torch", "rdkit", "biopython", "gemmi")]
        constants = {key: value for key, value in vars(const).items()
                     if not key.startswith("_") and isinstance(
                         value, (dict, list, tuple, set, str, int, float, bool, np.ndarray))}
        # Serialize states as bytes for content identity, never tensor object IDs.
        initial_rng = (self.before[0], self.before[1], self.before[2].numpy().tobytes())
        identity = (source, versions, sys.version, settings, constants,
                    input_data, tokenized, molecules, vars(featurizer),
                    marshal.dumps(method.__func__.__code__), initial_rng,
                    torch.get_num_threads(), torch.get_default_dtype(),
                    torch.are_deterministic_algorithms_enabled())
        key = hashlib.sha256(pickle.dumps(identity, protocol=4)).hexdigest()
        self.entry = Path(mol_dir).parent / "cpu_feature_cache" / key

    def load(self):
        if self.entry is None or not (self.entry / "manifest.json").is_file():
            return None
        try:
            meta = json.loads((self.entry / "manifest.json").read_text())
            path = self.entry / "features.pt"
            if digest(path) != meta["sha256"]:
                return None
            value = torch.load(path, map_location="cpu", weights_only=False)
            if not supported(value["features"]):
                restore_rng(self.before)
                return None
            restore_rng(value["after_rng"])
            return value["features"]
        except Exception:
            restore_rng(self.before)
            return None

    def save(self, features):
        if self.entry is None or self.entry.exists() or not supported(features):
            return
        after = rng_state()
        self.entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=self.entry.parent, prefix=".staging-"))
        try:
            path = temporary / "features.pt"
            torch.save({"features": features, "after_rng": after}, path)
            (temporary / "manifest.json").write_text(json.dumps({"sha256": digest(path)}))
            os.rename(temporary, self.entry)
        except FileExistsError:
            pass
        finally:
            restore_rng(after)
            if temporary.exists():
                shutil.rmtree(temporary)
