# Tangram — optimized fork

The current best delivered Tangram implementation, including numerical repairs
and the accepted further optimizations. This is an **unofficial fork**, not a
new release of the upstream PyPI package.

The Python and C++ implementation, original tests and license are unchanged
from the source package used in our completed CPU/GPU evaluations.
Only repository documentation, build configuration and CI have been prepared
for this GitHub handoff. Exact provenance is in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Install this repository

Python **3.11** is the tested configuration. Create and activate an environment:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

Choose one PyTorch build:

```bash
# CPU
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# OR CUDA 12.6, with a compatible NVIDIA driver
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Then install **this checkout**, rather than the unrelated upstream PyPI release:

```bash
uv pip install -e .
python -c "import tangram; print(tangram.__file__)"
```

For a non-editable installation use `uv pip install .`.

The optional CPU native update path is selected only on supported inputs and
hardware; otherwise the code falls back. Its C++ source is included in built
wheels. Do not assume CPU-specific optimizations also accelerate CUDA.

## Use

The public Tangram API is unchanged:

```python
import tangram as tg

tg.pp_adatas(adata_sc, adata_sp, genes=training_genes)
adata_map = tg.map_cells_to_space(
    adata_sc, adata_sp,
    mode="cells",
    device="cuda:0",  # use "cpu" for CPU execution
    num_epochs=500,
)
adata_projected = tg.project_genes(adata_map, adata_sc)
```

Supply your own single-cell and spatial AnnData objects and training-gene list.
`project_genes` remains a CPU NumPy operation even when mapping runs on CUDA.
Tutorial notebooks, upstream documentation and small test fixtures are retained.
The original project README, attribution, citation and usage details are in
[UPSTREAM_README.md](UPSTREAM_README.md).

## What is optimized

- Cached constant normalization/KL terms and deferred GPU history transfers.
- Avoidance of inactive calculations while preserving validated numerical behavior.
- Sparse/vectorized aggregation, gene matching, scoring and annotation utilities.
- CPU-gated projection/density backward work and native parameter-update paths.

Automatic reduced precision, automatic compilation and forced fused Adam are
not reintroduced. See [VALIDATION.md](VALIDATION.md) for results and limitations;
there is no promise of universal bitwise equivalence.

## Tests

```bash
uv pip install pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -m pytest -q tests/test_numerical_regressions.py
```

The inherited original tests contain six uses affected by the removed `np.float`
alias. To run both suites with their original assertions, apply this compatibility
alias **only inside the test process**, never in production code:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -c 'import numpy as np, pytest; np.float = float; raise SystemExit(pytest.main(["-q", "tests/"]))'
```

The included CPU CI runs that explicitly documented compatibility command on
Python 3.11. It does not silently modify the library or certify GPU performance.

## GitHub handoff

Extract this ZIP and use the contents of `tangram_optimized/` as your repository
root. Dotfiles and `.github/` are included. No `.git` history, credentials,
evaluation datasets, benchmark outputs, environments or compiled artifacts are
included. Small original repository test fixtures and tutorial outputs are not
evaluation payloads and are preserved.

An old upstream PyPI-publishing workflow has been removed. The distribution
metadata still uses the inherited `tangram-sc` name/version: do **not** publish
this fork under the upstream package identity without arranging ownership and
release metadata separately.

`REPOSITORY_MANIFEST.json` records every other included file's SHA-256 and size.
The external download checksum covers the complete ZIP.

## License and attribution

The original [LICENSE.md](LICENSE.md), author metadata and upstream attribution
are preserved. Upstream baseline commit:
`4c68995a418f41dc8caef567598c4d9b47781a13`.
