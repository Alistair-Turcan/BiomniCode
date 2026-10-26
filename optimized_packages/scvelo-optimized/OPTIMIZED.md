# Optimized scVelo

Complete source repository based on upstream scVelo commit
`f63c0e70596ced2f1bee8cf07e8ab66037cf86b2`, with the evaluated
current-best optimization (iteration 19) already applied. No patch application
is required. The original BSD license, documentation, tests, and upstream test
fixtures are retained.

## Installation

The evaluated environment used CPython 3.11.13 on Linux x86-64.
From this directory, using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-evaluated.txt
uv pip install -e .
python -c "import scvelo; print(scvelo.__version__); print(scvelo.__file__)"
```

`requirements-evaluated.txt` preserves the evaluated environment's dependency
pins, excluding the upstream editable scVelo checkout: this repository is
installed instead. It includes profiling/test dependencies used during
development. These pins are an environment record, not a cross-platform lockfile.
The optimized code requires `greenlet`, now declared in `pyproject.toml`.

## Compiled kernels

Two optional, already-built AOT kernel libraries are included in `scvelo/tools/`:

- `_em_kernels_aot_np.cpython-311-x86_64-linux-gnu.so`
- `_em_kernels_aot_py.cpython-311-x86_64-linux-gnu.so`

These exact binaries were evaluated with NumPy 2.4.6 and Numba 0.68.0.
They are specific to CPython 3.11 / Linux x86-64; compatibility with other
machines and dependency versions is not established. Other Python ABI tags
will not load these files. If either library cannot be imported, the code
falls back to its Numba JIT kernels. That fallback is not a promise of the
same performance as the evaluated AOT configuration.

On another supported build environment, rebuild the shipped fixed-signature
kernel sources locally (a working C compiler and the evaluated Numba version
are recommended):

```bash
python tools/build_optimized_kernels.py
```

This helper uses paths relative to the repository, temporarily hides existing
libraries while compiling, and restores the old libraries on failure. Rebuilt
binaries need fresh validation on the target machine; this archive does not
claim cross-platform validation. Use the source/editable install above;
portable binary wheels have not been prepared or validated.

## Use

The scVelo API is unchanged. For an already preprocessed AnnData object:

```python
import scvelo as scv
scv.tl.recover_dynamics(adata, var_names=genes, n_jobs=4)
```

The final three-dataset evaluation used `n_jobs=4`, with BLAS threads pinned
to one. Earlier baseline runs exhausted 64 GiB machines at higher parallelism;
four workers was the tested final protocol, not a universal memory guarantee.
Dataset downloads, fitted arrays, benchmark logs, and optimization history
are intentionally not included here. They remain in the separate evaluation
package.

## Publish to GitHub

The zip intentionally contains no `.git` database or remote configuration.
After extracting, create an empty GitHub repository and run:

```bash
git init -b main
git add .
git commit -m "Add optimized scVelo source"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

The ignore rules explicitly retain the two evaluated libraries. All original
license and attribution files are preserved. `RELEASE_MANIFEST.json` records
upstream provenance and SHA-256 checksums of every other distributed file.
