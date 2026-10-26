# MAGIC optimized

Current-best saved MAGIC implementation, packaged as a clean source repository.
This is the same implementation used in the completed full-gene evaluation,
not the earlier repaired-only release. Both MAGIC and its optimized vendored
graphtools are included. The Python/C implementation, tests, build files, and
dependency pins are unchanged from the verified saved best release.

## Licensing: resolve before public release

The inherited `LICENSE` contains GPL version 3, while **both** `setup.py` files
declare GPL version 2. These files are preserved verbatim, not reconciled or
relicensed by this packaging step. Confirm the applicable upstream and
contributor licensing and reconcile these notices before public publication.
This download is a repository-layout package, not a claim of licensing clearance.

## Install

The validated release instructions target Linux and Python 3.11.13.
From this extracted repository root, with `uv` available:

```bash
uv venv .venv --python 3.11.13
uv pip install --python .venv/bin/python -r requirements_runtime.txt
uv pip install --python .venv/bin/python --no-deps . ./python
.venv/bin/python -c "import magic, graphtools; print(magic.__file__, graphtools.__file__)"
```

Install **both** local packages: `.` installs optimized graphtools and its
optional C extension; `./python` installs MAGIC. Installing stock PyPI
`magic-impute`/`graphtools` does not install this fork. The inherited
`python/README.rst` is kept for package metadata; its upstream installation
instructions do not supersede these local installation commands.

The optional extension uses the CPython buffer API and no NumPy C headers.
The preserved build flags include `-O3 -std=c99 -march=native -fno-fast-math
-ffp-contract=off`. Build on the target CPU; do not treat a locally built native
wheel as universally portable. No compiled binaries are bundled. If compilation
is unavailable or loading the extension fails, eligible operations fall back to
SciPy. The native speed benefit is unavailable in that fallback.

Numba is optional and excluded from the primary benchmark environment. The
separately validated optional version was `numba==0.61.2`; use a separate
environment with the same runtime pins if enabling it.

## Test

From the repository root after installation:

```bash
cd test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python -m pytest -q test.py test_parity_repairs.py
```

`data/test_data.csv` is the small existing software-test fixture required by
these tests, not a full evaluation dataset. The working directory in the
command matters for its relative path.

The saved release validation reported 20 passing installed-package tests,
including the no-compiler fallback. Its broader upstream suite reported
229/231 passing, with the same two failures on the repaired baseline and retained
candidate; it was not an all-upstream-green result. These are **historical saved
results**, not tests rerun during this packaging step.

This source ZIP was checked for source identity, Python syntax, local import
layout, safe archive paths, CRCs, and manifest hashes. Dependencies were not
reinstalled and numerical tests/benchmarks were not rerun for this ZIP.

## Repository and provenance

- `python/magic/`: MAGIC implementation and `python/setup.py`.
- `graphtools/`: optimized graph implementation, installed by root `setup.py`.
- `native/`: optional CSR extension source.
- `test/`, `data/test_data.csv`: existing self-contained test suite and fixture.
- `requirements_runtime.txt`: saved validated numerical/runtime pins.
- `PROVENANCE.json`: source identity and exact inclusion/exclusion inventory.
- `FILE_MANIFEST.json`: hashes and sizes of all other ZIP payload files.

No raw evaluation datasets, computed matrices, benchmark outputs, historical
audit bundles, built binaries, environments, caches, or Git history are included.
The separately delivered evaluation ZIPs retain evaluation evidence.

To create a repository locally after extracting and resolving the licensing
issue, run `git init` in this directory and commit its contents. This ZIP does not
create a GitHub repository, publish to PyPI, or assign a new package version.
