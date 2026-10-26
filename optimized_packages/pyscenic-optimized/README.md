# pySCENIC optimized

The best validated optimized **pySCENIC + ctxcore** implementation, packaged as a
single source repository. Both local packages are required; importing them still
uses `pyscenic` and `ctxcore`. Scientific Python source is unchanged from the
validated best delivery.

## Install

Validated on Linux with Python 3.11. Install [uv](https://docs.astral.sh/uv/)
first, extract this repository, then run from its root:

```bash
bash scripts/install.sh
source .venv/bin/activate
pyscenic --help
```

The helper creates a local Python 3.11 environment, installs the exact dependency
pins in `requirements.lock.txt`, and installs **both local packages** without
letting a dependency resolver substitute the upstream ctxcore release.
During the build it supplies neutral pySCENIC Versioneer metadata (`0+unknown`)
so the monorepo folder name is not mistaken for a version. It restores both
original version modules after installation and verifies the source hashes and
actual import locations.

This root is a two-package monorepo, not a third Python distribution; use the
helper instead of `uv pip install .`. A custom environment can be selected with
`bash scripts/install.sh --env /absolute/path/to/new/env`. For reproducibility,
start with a fresh environment. Other Python versions/platforms are not validated.

Distribution version strings can reflect source-archive build metadata rather
than an optimized release number. Use `provenance/source_identity.json` and the
source hashes to identify this implementation.

## Tests

Existing upstream tests are included unchanged. Scientific test datasets are
not bundled; explicitly fetch the checksum-pinned upstream fixtures first:

```bash
.venv/bin/python scripts/fetch_test_fixtures.py
bash scripts/test.sh -q
```

The downloader reads data from exact upstream Git commits and verifies each
size/SHA256 before use. Data is never executed. Consult the upstream projects
and original datasets for their applicable usage terms; this repository makes
no additional data-licensing claims. Downloaded fixtures are ignored by Git.
If a public URL disappears or its bytes differ, downloading stops rather than
silently using a replacement.

The test helper runs each package in a separate pytest process because both
upstream projects use the top-level name `resources.tests`. It defaults to four
Numba threads and single-threaded BLAS. Tests are ordinary software checks, not
new biological findings. To use a custom installed environment, set
`PYSCENIC_TEST_PYTHON=/absolute/path/to/env/bin/python`.

Verify protected source/assets at any time:

```bash
python3 scripts/verify_source.py
```

## What is included

- `pySCENIC/`, `ctxcore/`: complete implementation, packaging files, existing
  tests and upstream documentation/attribution.
- Small bundled delineation resources required by existing runtime commands.
- Pinned installation/test helpers and source/fixture checksum manifests.
- Compact historical validation evidence and clean-package validation receipt.
- Original package licenses and a root attribution notice.

No virtual environment, Git history, credentials, new CI workflow, evaluation
datasets, large motif databases, score arrays, recovery curves, or bulk benchmark
outputs are included. Real analyses still require their own appropriate
expression data, signatures and reference databases.
Nested upstream CI files are retained as package metadata, not configured as
automation for this two-package repository.

## Validation scope

The optimized source was previously evaluated against untouched upstream on
PBMC, brain and pancreas, three process replicates per version and stage.
All 121,358,937 paired scores and 75,376,372,500 recovery values matched bitwise.
That evidence covers **AUCell and cisTarget enrichment/recovery**, not a new
biological validation or universal/end-to-end GRN performance guarantee.
See `provenance/historical_validation.json` for the recorded scope and
`provenance/package_validation.json` for this source package's installation/tests.
The full numerical evaluation was delivered separately.

## Put it on GitHub

This ZIP is ready to become a new repository; no GitHub repository has been
created and no remote or credentials are configured.

```bash
git init
git add .
git commit -m "Add validated optimized pySCENIC and ctxcore"
```

Create your GitHub repository, add its remote and push using your own account.
Do not commit `.venv`, caches, or downloaded fixtures.

## Attribution and license

Based on [aertslab/pySCENIC](https://github.com/aertslab/pySCENIC) and
[aertslab/ctxcore](https://github.com/aertslab/ctxcore). Original authorship and
copyright notices are retained. See `NOTICE`, root `LICENSE`, each package's
license/metadata, and the exact upstream commits in the provenance.
