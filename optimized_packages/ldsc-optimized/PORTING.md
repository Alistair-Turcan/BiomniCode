# Python 3 port notes (opt_py2 → opt_py3)

This tree is a Python 3 port of the optimized LDSC fork (`ldsc_optimized_tree.zip`,
itself a fork of bulik/ldsc v1.0.1). See README.md for the evaluated numerical agreement: full-precision h2 outputs were close but not bitwise-identical across the evaluated stacks. Historical validation notes below are retained for context and should not be interpreted as a cross-platform bitwise guarantee.

## Target stack (validated)

- Python 3.9
- numpy 1.26.4, pandas 1.5.3, scipy 1.10.1, bitarray 2.6.0, nose 1.3.7 (tests)

**Performance note:** numpy 1.26.4 is required for full `--l2` performance. numpy
1.22.x wheels bundle OpenBLAS 0.3.18 (ILP64), whose multi-threaded dgemm saturates
at ~8 threads for LDSC's block shapes (measured 13.4 ms vs 5.1 ms per block gemm);
numpy 1.26.4 bundles OpenBLAS 0.3.23, which scales to 16 threads (2.5 ms). The port
is bit-identical under both numpy versions — only speed differs.

## Changes applied (mechanical port)

1. `2to3` baseline: `print >>` → `print(..., file=...)`, `xrange` → `range`,
   old-style `raise X, msg` → `raise X(msg)`, lazy `map`/`zip` wrapped in
   `list(...)` where consumed eagerly.
2. pandas API removals: `.ix` → `.loc`/`.iloc` (12 sites); `DataFrame.as_matrix()`
   → `.to_numpy()` (matches CBIIT/ldsc main); `pandas.util.testing` →
   `pandas.testing` in tests; fully-qualified `pd.set_option('display.precision')`
   / `('display.max_colwidth')` (ambiguous prefixes raise in pandas 1.5).
3. Bytes/str boundaries: log file opened in text mode (`'w'`, was `'wb'`);
   `traceback.format_exc()` called without the py2-style exception argument (3
   sites); hashlib inputs UTF-8-encoded via a `_b()` helper in `ldscore/parse.py`
   (content-keyed cache); `read_header` opens gzip/bz2 in text mode (`'rt'`);
   `.M`/`.M_5_50` outputs opened in text mode; `make_annot.py` gzip output in
   `'wt'` mode.
4. Tab/space normalization in three blocks (py3 rejects mixed indentation that
   py2 tolerated): `munge_sumstats.py` `--daner-n` block and duplicate-column
   check, one line in `ldscore/ldscore.py`, one comment in
   `test/test_regressions.py`.
5. Test-suite fixes matching upstream CBIIT/ldsc py3 idioms:
   `test_jackknife.py` uses `np.isclose(var.flat, ...)` (py3 `round()` does not
   coerce ndarrays); `test_sumstats.py::test_rg_M` reads the `.M_5_50` file in
   text mode.

## Not changed

- No algorithmic changes. All hot-loop code (2-bit LUT decode, vectorized
  normalization, ring-buffer sliding window, threaded gzip prefetch/parse,
  content-keyed caches) is byte-for-byte the ported py2 logic.
- `bitarray` usage was already 2.x-compatible in the upload (`frombytes`/`tobytes`).
- `pybedtools` is only needed by `make_annot.py` and is now an optional extra.

## Validation summary

- Unit tests: 167/167 pass under the target stack (py2 baseline: 167/167).
- `--l2` outputs bit-identical to py2 at chr22/chr10/chr1 (max abs diff 0.0).
- `munge_sumstats` 10M-row output bit-identical.
- `--h2`/`--rg` chr22 estimates identical at printed (4-decimal) precision.
- Performance: see `validation_py3_port.md`; the py3 port matches py2
  single-threaded (247.0s vs 247.0s at chr22) and is faster as-shipped
  (25.3s vs 41.4s at chr22) thanks to the newer OpenBLAS.
