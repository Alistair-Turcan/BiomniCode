"""Validated cache of deterministic offline protein input products, never predictions."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import tempfile

import numpy as np
from rdkit import Chem
import torch
import yaml


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _rng():
    return pickle.dumps((random.getstate(), np.random.get_state(),
                         torch.get_rng_state().numpy().tobytes()), protocol=4)


class ParsedProteinCache:
    """Conservative opt-in-by-schema cache; unsupported inputs use normal parsing."""

    def __init__(self, path, ccd, mol_dir, enabled, max_msa_seqs, products):
        self.entry = None
        if not enabled or path.suffix.lower() not in (".yaml", ".yml"):
            return
        raw = path.read_bytes()
        spec = yaml.safe_load(raw)
        if not isinstance(spec, dict) or set(spec) != {"version", "sequences"}:
            return
        dependencies = []
        if not isinstance(spec["sequences"], list) or not spec["sequences"]:
            return
        for item in spec["sequences"]:
            if not isinstance(item, dict) or set(item) != {"protein"}:
                return
            protein = item["protein"]
            if not isinstance(protein, dict) or set(protein) != {"id", "sequence", "msa"}:
                return
            if not isinstance(protein["sequence"], str) or not set(
                protein["sequence"]
            ) <= set("ACDEFGHIKLMNPQRSTVWYX"):
                return
            msa = Path(protein["msa"]).resolve()
            if not msa.is_file():
                return
            dependencies.append((str(msa), _sha(msa.read_bytes())))
        self.before_rng = _rng()
        package_root = Path(__file__).resolve().parents[1]
        source = [(str(p.relative_to(package_root)), _sha(p.read_bytes()))
                  for p in sorted(package_root.rglob("*.py"))]
        versions = [(name, importlib.metadata.version(name)) for name in
                    ("numpy", "pandas", "rdkit", "biopython", "gemmi", "torch", "pyyaml")]
        identity = (raw, str(path.resolve()), dependencies, source, versions,
                    sys.version, int(max_msa_seqs), pickle.dumps(ccd, protocol=4),
                    self.before_rng)
        self.entry = Path(mol_dir).parent / "parsed_protein_cache" / _sha(
            pickle.dumps(identity, protocol=4))
        self.products = products
        self.target_id = path.stem

    def restore(self):
        if self.entry is None or not (self.entry / "manifest.json").is_file():
            return False
        try:
            manifest = json.loads((self.entry / "manifest.json").read_text())
            pending = []
            for group, name, digest in manifest:
                # Cache manifests are local but still cannot escape product roots.
                if group not in self.products or Path(name).name != name:
                    return False
                src = self.entry / group / name
                dst = self.products[group] / name
                if dst.exists() or not src.is_file() or _sha(src.read_bytes()) != digest:
                    return False
                pending.append((src, dst))
            if not pending or not any(group == "records" for group, _, _ in manifest):
                return False
            for src, dst in pending:
                shutil.copyfile(src, dst)
            # Original parser sets this global RDKit serialization option.
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
            return True
        except (OSError, ValueError, TypeError):
            return False

    def save(self):
        # Only store executions proved RNG-neutral; no inferred RNG transition.
        if self.entry is None or self.before_rng != _rng() or self.entry.exists():
            return
        files = []
        for group, folder in self.products.items():
            for file in sorted(folder.glob(self.target_id + "*")):
                matches = (file.stem == self.target_id if group not in ("msa", "templates")
                           else file.name.startswith(self.target_id + "_"))
                if file.is_file() and matches:
                    files.append((group, file.name, file))
        if not any(group == "records" and name == self.target_id + ".json"
                   for group, name, _ in files):
            return
        self.entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=self.entry.parent, prefix=".staging-"))
        try:
            manifest = []
            for group, name, file in files:
                destination = temporary / group / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, destination)
                manifest.append((group, name, _sha(destination.read_bytes())))
            (temporary / "manifest.json").write_text(json.dumps(manifest))
            os.rename(temporary, self.entry)
        except FileExistsError:
            pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
