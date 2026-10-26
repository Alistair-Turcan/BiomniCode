# ProteinMPNN — optimized inference

Optimized ProteinMPNN source, packaged from the evaluated **iter-011** release.
The two inference Python files are unchanged from the evaluated version.

Based on [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
upstream commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`.
The original MIT license and attribution are retained in `LICENSE`.
Full upstream documentation is preserved in [README_UPSTREAM.md](README_UPSTREAM.md).

## Install

Tested with Python 3.11.13, PyTorch 2.7.1+cu126 and NumPy 2.1.0 on NVIDIA A10.
The pinned requirements below cover inference; training and notebook extras
are not included in those requirements.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
python -m pip install -r requirements.txt
```

A compatible NVIDIA driver is required for CUDA. For CPU-only use, install
the appropriate PyTorch 2.7.1 build instead; GPU speedups do not apply.

## Quick start

From this repository's root:

```bash
python protein_mpnn_run.py \
  --pdb_path inputs/PDB_monomers/pdbs/5L33.pdb \
  --out_folder results/demo \
  --num_seq_per_target 8 \
  --sampling_temp 0.1 \
  --seed 37 \
  --batch_size 1 \
  --model_name v_48_010 \
  --save_probs 1
```

Sequences are written under `results/demo/seqs/`; probabilities and
log-probabilities under `results/demo/probs/`. Use
`python protein_mpnn_run.py --help` for the full CLI.
All upstream inference weight sets are bundled.

## What changed

- Reduced Python/CPU overhead.
- CUDA graph capture and reuse for autoregressive sampling.
- Encoder caching and invariant/indexing work hoisted out of inner loops.
- CLI-only immediate process exit after output completion and stdio flush.

**Integration caveat:** the standalone CLI uses `os._exit(0)` after `main()`
returns. Python `atexit` handlers and interpreter teardown do not run on that
successful CLI path. Importing the module does not trigger this exit.
Account for this when embedding or instrumenting the CLI.

The tested fixed-seed evaluation matched original sequences, log-probabilities,
and scores exactly. This is evidence for the tested configuration, not a
guarantee across all hardware, PyTorch builds, or untested options.
No approximate precision or reassociation was introduced.

## Repository contents

- `protein_mpnn_run.py`, `protein_mpnn_utils.py`: evaluated inference source.
- `helper_scripts/`, `examples/`, `inputs/`: upstream helpers and example inputs.
- `*_model_weights/`: upstream inference model weights.
- `training/`, `colab_notebooks/`: upstream source/notebooks, retained for context.
- `requirements.txt`: pinned inference dependencies.
- `RELEASE.json`, `SOURCE_SHA256SUMS.txt`: source provenance and integrity hashes.

Generated example outputs, training-run artifacts, benchmark datasets,
evaluation results, virtual environments, and Git history are intentionally
excluded. Some upstream examples reference generated `outputs/` files;
run their prerequisite parsing/design steps to recreate those files.

## Publish to GitHub

Unzip, open the `proteinmpnn_optimized` directory, and initialize a fresh repo:

```bash
git init
git add .
git commit -m "Add optimized ProteinMPNN inference"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Create the empty remote repository first. The archive contains source files
and weights, not a preconfigured `.git` directory or credentials.
