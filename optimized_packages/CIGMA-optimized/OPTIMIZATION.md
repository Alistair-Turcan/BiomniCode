# Optimized CIGMA source release

This is the accepted **iter-020** implementation from the CIGMA optimization
run `bc-20261002-155305-eb260d07`.

- Upstream: https://github.com/Minhui-Chen/CIGMA
- Upstream base commit: `5813e4ae84d7b3733dfcd938fe42d12c6b30a8aa`
- Source: the saved `cigma_optimized.zip` from the completed optimization.
- License: the upstream GPL-3.0 license is retained in `LICENSE.txt`.

All retained repository files are byte-for-byte identical to that saved archive.
Only generated coverage data and package build metadata were removed. This note
and `SHA256SUMS` were added for distribution. Original documentation, tests,
example data, analysis notebooks, and workflows are preserved.

## Installation

Use Python 3.11 or newer and follow the upstream `README.rst` for usage.
The project derives its version from Git tags via setuptools-scm. Because this
archive intentionally contains no Git history, supply the same metadata-only
version override used for the benchmark installation:

```bash
python -m venv .venv
source .venv/bin/activate
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 python -m pip install .
```

`0.1.0` here is a packaging override, not a claim of a new upstream release.
Dependencies remain exactly as declared by the saved repository. This archive
does not add a new dependency lockfile or repeat the performance evaluation.

## Push to a new GitHub repository

Unzip, enter the `CIGMA-optimized` directory, and replace the remote placeholder:

```bash
git init -b main
git add .
git commit -m "Add accepted optimized CIGMA implementation"
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

The archive is a clean source snapshot, not a fork with upstream Git history.
Keep upstream authorship and license notices intact.

## Integrity

```bash
sha256sum -c SHA256SUMS
```

The packaging check also verifies that the accepted cumulative optimization patch
can be reverse-applied to this source tree. No scientific implementation changes
were made during packaging.
