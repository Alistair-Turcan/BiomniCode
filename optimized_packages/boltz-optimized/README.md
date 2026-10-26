# Boltz — optimized source snapshot

This repository contains the **exact optimized scientific source used by
`boltz_optimized_lite.zip`**, with a new landing README and compact provenance.
It is the 170-file source snapshot from that release, **not** the subsequent
round-three/four experimental version. No scientific code was changed while
packaging this repository.

## Install this repository

Use a fresh environment. Python 3.11 is a suitable starting point; the unchanged
package metadata declares Python >=3.10,<3.13.

For a CUDA-enabled machine with a compatible PyTorch/CUDA installation:

```bash
cd boltz-optimized
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[cuda]"
boltz predict --help
```

Install **from this directory**. Installing `boltz` from PyPI or cloning the
upstream repository instead does not install this optimized snapshot.
The CUDA kernels and their binary dependencies require a compatible platform.
The commands above are installation instructions, not a newly tested portable
environment lock.

## Predict

For example, on a configured GPU machine:

```bash
boltz predict examples/prot_no_msa.yaml --out_dir outputs
```

For your own YAML inputs, see [prediction instructions](docs/prediction.md).
Model weights and chemistry assets are not bundled. The existing program uses
its normal asset-download/cache mechanism. No inference is run by this ZIP.

## Contents and provenance

- `src/boltz/`: unchanged optimized implementation.
- `pyproject.toml`: unchanged build metadata, dependency declarations, and CLI.
- `tests/`, `examples/`, `scripts/`, `docs/`: original repository contents.
- `LICENSE`: unchanged MIT license.
- `docs/UPSTREAM_README.md`: byte-identical original repository README, relocated
  here so the landing page installs this snapshot rather than an upstream build.
- `provenance/SOURCE_MANIFEST.json`: original-to-packaged paths, hashes and modes
  for all 170 source files.
- `provenance/source_identity.json`: original release identity. The comparison
  was to upstream Boltz v2.2.1; optimized-source ancestry was not established.
- `provenance/benchmark_environment.txt`: historical measured environment,
  **not** a portable lockfile or a recommended blanket install list.
- `provenance/CHECKSUMS.json`: hashes, sizes and modes of all other ZIP members.

Scientific source, dependency declarations, CLI/API, tests, and file modes are
unchanged. Only the landing README is replaced; the original is preserved.
This package excludes evaluation workloads, predictions, optimization histories,
model weights, virtual environments and caches. Small public examples shipped
with the original repository remain included.

The original lite evaluation is a separate artifact. Packaging integrity and
Python syntax were checked; dependency installation, GPU inference and the
scientific test suite were **not** rerun for this source-only delivery.
Prior evaluation does not establish universal numerical or biological equivalence.
Performance depends on workload, hardware, software versions and cache state.

## Put it on GitHub

Unzip, enter this directory, and initialize your own history:

```bash
git init
git add .
git commit -m "Import optimized Boltz source snapshot"
```

Then add your own GitHub remote and push. No Git history or remote credentials
are included, and no repository was published automatically.

## Attribution

Based on the Boltz project: <https://github.com/jwohlwend/boltz>.
Retain the supplied MIT license and upstream notices when redistributing.
