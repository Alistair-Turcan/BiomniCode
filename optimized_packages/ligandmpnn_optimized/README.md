# LigandMPNN — optimized inference

The current best inference implementation from our LigandMPNN optimization
and evaluation, packaged as a source repository. The inference code is unchanged
from the evaluated version.

Based on [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN),
upstream commit `26ec57ac976ade5379920dbd43c7f97a91cf82de`.
The upstream MIT license and third-party source notices are retained.
Full upstream CLI documentation is in [README_UPSTREAM.md](README_UPSTREAM.md).

## Setup

Validated environment: Linux, Python 3.11, PyTorch 2.2.1 with CUDA 12.1,
NumPy 1.23.5, and ProDy 2.4.1. Performance validation used an NVIDIA A10G.

From this repository's root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Uses the original upstream downloader (requires wget).
bash get_model_params.sh "./model_params"
```

Model weights are intentionally not included in this source archive.
Download them before running inference; the default checkpoint paths expect
the `model_params/` directory in the repository root.

## Run inference

```bash
python run.py \
    --model_type ligand_mpnn \
    --seed 111 \
    --pdb_path "./inputs/1BC8.pdb" \
    --out_folder "./outputs/example" \
    --batch_size 8 \
    --number_of_batches 8
```

This retains the upstream command-line interface. Optimizations include
CUDA-graph decoding, multi-stream replay, encoder caching, deferred imports,
reduced synchronization overhead, and faster PDB writing.

## Validation and limitations

- The delivered code passed the 12-case comparison suite. Subsequent evaluation
  used 1GPK, 1JZ8 chain A, and 2GFB chains A–H, with three timed replicates per
  variant and separate statistics runs. Sampled sequences and decoding orders
  matched exactly, as did the compared log-probability tensors in those runs.
  These are tested-workload results, not a guarantee across all inputs,
  hardware, or library versions.
- Optimization focused on `run.py` sequence design for `ligand_mpnn` and
  `protein_mpnn`. Side-chain packing and `score.py` were not performance targets.
- Four CUDA-graph slots increase GPU memory use. The evaluated 1,730-residue
  workload reserved about 20,244 MiB of GPU memory. Long multi-structure runs
  can also retain graph pools across shapes.
- To disable graph decoding when memory is constrained:

  ```bash
  LIGANDMPNN_DISABLE_GRAPH=1 python run.py \
      --model_type ligand_mpnn --seed 111 \
      --pdb_path "./inputs/1BC8.pdb" --out_folder "./outputs/eager"
  ```

- Graph capture can silently fall back to eager execution. Performance on a
  different GPU or software environment should be measured rather than assumed.
- The CLI uses `os._exit(0)` after flushing outputs to avoid interpreter
  teardown overhead. Treat it as a command-line program; embedding it in a
  long-lived Python process requires care.

## Publish to GitHub

Unzip, enter this directory, and initialize a repository:

```bash
git init
git add .
git commit -m "Add optimized LigandMPNN inference"
```

Then connect your chosen GitHub remote and push. `.gitignore` excludes model
weights, generated outputs, Python environments, caches, and local secrets.
No Git history, model weights, benchmark outputs, or evaluation scripts are
included. Example inputs and upstream helper modules are retained.

`SOURCE_MANIFEST.json` records SHA-256 hashes of the packaged files, excluding
the manifest itself.
