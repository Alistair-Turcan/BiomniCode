# Current best optimized GPN-Star

This is a source-only GitHub handoff of the current best implementation. The
evaluated library source is unchanged; only this handoff documentation and the
local-install section of the README have been added.

## Identity

- Upstream base: GPN 0.9.0, commit
  `007768a31618fa76c2c3db953e69cc2013f5a79c`.
- Delivered best optimization: cumulative patch SHA-256
  `d48eecf93d408a4dad61cc565a848eb8367acb0d82ba660cb6bf3c0b861d95dd`.
- Optimized `src/gpn/star/model.py` SHA-256:
  `7715952adee3f41411393c7b9e921856efeef9d39d37118ce08cd967964dc3cc`.
- Optimized `src/gpn/star/inference.py` SHA-256:
  `f1c5ddcfe5897b828ae88136fb59c66eb4b0afac756a800fe9e8af183eceb259`.

Both optimized files are already applied. No additional patch application,
retraining or model-weight conversion is needed. `OPTIMIZED_SNAPSHOT.json`
contains the machine-readable source identity.

## Install this source

Requires Python 3.13. The repository contains upstream's `pyproject.toml` and
`uv.lock`, source, tests and their small original fixtures, documentation, license
and GitHub workflows. It does not contain the evaluation datasets, prediction
arrays, benchmark logs, local environments or model weights.

```bash
uv venv --python 3.13.11 .venv
uv pip install --python .venv/bin/python -e ".[inference]"
```

For the CUDA/PyTorch build used in the completed evaluation, install torch first:

```bash
uv venv --python 3.13.11 .venv
uv pip install --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org "torch==2.10.0+cu128"
uv pip install --python .venv/bin/python -e ".[inference]"
```

The upstream `uv.lock` selects CPU torch for its default `uv sync` workflow;
the CUDA commands above intentionally use a direct editable installation.
Models and alignments are retrieved separately as described in the model docs.
The evaluated checkpoint was `songlab/gpn-star-hg38-v100-200m`, revision
`0c949f132d35619a3eb188b402848c998a3313ae`.

## Validation scope

The completed evaluation used three datasets, 10,000 distinct real loci/dataset,
three independent fresh-process replicates/version/operation, and VEP, logits
and embeddings: 54 measured processes. FP32, batch size 2, 128-bp windows, all
100 species and one NVIDIA A10 were used.

All 27 paired comparisons passed the frozen numerical tolerances across
92,610,000 values. Mean warmed-pipeline speedups were 2.43–2.50x; peak CUDA
allocation increased approximately 11.2–11.5%. These measurements do not establish
performance on other hardware or workloads, and this is not clinical validation.
Full evidence is in the separately delivered evaluation archives, not this repo.

This validates the API/file-backed inference path, not successful execution of
every upstream CLI path. Existing Trainer/tail/DataFrame CLI limitations were not
fixed as part of this optimization. No new claim of a fully passing CI suite is
made by this packaging step.

## Put this snapshot on GitHub

Extract the ZIP, enter the `gpn_star_optimized` directory, create a new empty GitHub
repository, then initialize and push this directory using your own Git credentials.
The ZIP intentionally has no `.git` directory or credentials.

Upstream author/license attribution and repository URLs are preserved. Replace
links/badges with your fork's URLs as appropriate. Before publishing GitHub
releases, configure or disable the inherited `.github/workflows/release.yml`:
it targets the upstream `gpn` package on PyPI and requires a separately configured
publishing environment. This handoff does not authorize or configure publishing
under the upstream package name.
