# RFdiffusion3 — optimized Foundry source

This repository contains the frozen **iter-042** optimized implementation from the completed
RFdiffusion3 optimization and real-data evaluation. The executable Python/model source is
unchanged from the evaluated best version.

- Upstream repository: https://github.com/RosettaCommons/foundry
- Original commit: `4010e3e2e7350edada3e25a45c908c6bf407df4d`
- Optimized source identity: `cf22d613d227118b0970ff22c8518510fa59f18ed87bb7463d113bf7192eb9b8`
- Python 3.12.12; evaluated with Torch 2.8.0+cu128, CUDA 12.8 and AtomWorks 2.1.1.

The complete upstream source, documentation and repository test fixtures are retained.
Evaluation datasets/results, model weights, installed environments, Git history, and local
secrets are not bundled. The upstream license is preserved in `LICENSE.md`.

## Install this optimized checkout

Installing `rc-foundry` from PyPI alone does **not** install this optimized version.
Run these commands from this repository directory on a compatible Linux/CUDA system:

```bash
uv python install 3.12.12
uv venv --python 3.12.12
uv pip install --python .venv/bin/python   --extra-index-url https://download.pytorch.org/whl/cu128   --index-strategy unsafe-best-match   -r requirements-optimized.txt
uv pip install --python .venv/bin/python --no-deps -e '.[rfd3]'
cp .env.example .env
```

`requirements-optimized.txt` is the exact 143-package environment lock used in the evaluation.
Optional mirror/tool paths in `.env` can remain empty unless needed for your task. Do not
commit `.env` or credentials. Existing upstream setup documentation is in `README.upstream.md`.

## Model weights and a design run

Weights must be obtained separately. The evaluated checkpoint is available at:
https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt

Required SHA-256 for that checkpoint:
`9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`.

After downloading it to `checkpoints/rfd3.ckpt`, use a supported input specification:

```bash
.venv/bin/rfd3 design   inputs=/absolute/path/to/design.json   out_dir=outputs   ckpt_path=checkpoints/rfd3.ckpt   diffusion_batch_size=1   n_batches=1
```

See `models/rfd3/README.md` and `models/rfd3/docs/input.md` for task specifications and options.
Native production generation is nondeterministic; controlled audit equality should not be
interpreted as native bitwise reproducibility.

## Publish to your GitHub repository

Unzip the archive using a tool that preserves Unix symlinks (for example, `unzip` on Linux/macOS).
Then, from the extracted `rfd3-optimized` directory:

```bash
git init -b main
git add .
git commit -m "Add optimized RFdiffusion3 iter-042"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Create the empty GitHub repository first. No Git history or remote configuration is embedded
in this archive. Upstream GitHub workflow files are retained; review them before enabling CI
in a new repository.

## Version and packaging records

- `OPTIMIZED_VERSION.json`: frozen version, source/checkpoint identities, and packaging metadata.
- `SOURCE_INVENTORY.json`: exact original best-source inventory and three documentation/template renames.
- `optimization.patch`: cumulative optimized changes against the pinned original commit.
- `MANIFEST.sha256`: checksum of every packaged file except itself; symlink entries hash their
  stored target-string bytes, not target-file contents.

Only packaging documentation, ignore rules and reproducibility metadata were added or moved.
No model implementation was reoptimized, altered, or rerun for this source-only delivery.
