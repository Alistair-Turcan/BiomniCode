# SpatialDE — optimized source snapshot

This repository contains the **accepted iter008** optimized Python implementation
of SpatialDE 1.1.3, based on upstream commit
`77f9fa5910838f5ee4308b4334a2e6ce85d1748e`.
The scientific Python implementation is unchanged from the accepted version.

## Install the tested environment

Use **Python 3.11.13** with the pinned dependencies. The implementation retains
legacy SciPy/pandas interfaces, and its specialized L-BFGS-B path uses a private
SciPy ABI. Newer dependencies are **not** certified by this snapshot.

From this repository's root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv --python 3.11.13 .venv
uv pip install --python .venv/bin/python -r requirements-tested.txt
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e ./Python-module
.venv/bin/python -c "import SpatialDE; print(SpatialDE.__file__)"
```

On Windows use `.venv/Scripts/python.exe` in place of `.venv/bin/python`.
The evaluated and locally checked platform was Linux; other platforms need
their own validation.

## Usage and tests

```python
import SpatialDE

# X: locations × spatial dimensions; E: locations × genes, preprocessed as documented.
results = SpatialDE.run(X, E)
```

See the [upstream documentation](README.rst) and the retained `Analysis/` scripts
and notebooks for preprocessing and examples. The upstream `pip install spatialde`
instruction installs the public upstream release, **not** this optimized snapshot;
use the local-source installation above.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
```

`tests/` contains source-install/numerical smoke checks, not the complete previous
real-data evaluation. The retained legacy root `test-FaST-GP.py` imports an absent
historical module and is deliberately outside the configured test directory.
The included GitHub Actions workflow runs the smoke checks in the pinned Python
environment. Its hosted execution is not part of the local packaging check.

## Source-only distribution

- Python, R, Stan, example code, upstream documentation, and license notices are retained.
- No virtual environment, Git history, evaluation outputs, caches, or downloaded
  datasets are included.
- Upstream Git LFS pointer stubs are removed rather than shipped as unusable data
  files. Their original locations and content identities are recorded in
  `docs/omitted-example-assets.json`. The old `git lfs pull` instructions in
  `README.rst` apply to the upstream repository, not this source-only copy.
- Example notebooks retain their original upstream embedded demonstration output;
  those are not the new optimization/evaluation results. Full example execution
  requires separately obtaining the upstream assets.
- No R/Stan implementation or Python public API was changed by this packaging.
- This is a GitHub source repository, not a new PyPI release or an upstream merge.

`SOURCE_PROVENANCE.json` records the accepted source identity and byte-level
implementation checksums. Root `SHA256SUMS.txt` covers the distributed repository
files. Packaging additions are this README, environment pins, smoke tests/CI,
license-copy/manifest declarations, provenance, and source-only asset cleanup;
the accepted Python files are byte-identical.

## Publish to your GitHub repository

Extract the ZIP, enter the `SpatialDE` directory, create an empty GitHub repository,
and run:

```bash
git init
git add .
git commit -m "Add accepted optimized SpatialDE source"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Retain the original MIT attribution in [LICENSE](LICENSE) and the SciPy notice in
`Python-module/SpatialDE/LICENSE.scipy.txt`.
