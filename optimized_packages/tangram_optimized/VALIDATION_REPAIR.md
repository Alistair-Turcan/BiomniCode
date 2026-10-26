# Numerical-compatibility repair

This tree repairs the supplied optimized Tangram 1.0.4 against upstream commit
`4c68995a418f41dc8caef567598c4d9b47781a13`. Public signatures are unchanged.

Retained: cached constant normalization and KL terms; bulk history transfers;
skipping inactive regularizers on ordinary finite inputs; sparse, vectorized
cluster aggregation; vectorized scoring/counting and fast gene-index lookup;
the memory-efficient Geary numerator with a stable translation.

Removed: automatic CUDA BF16, automatic `torch.compile`, and forced fused Adam.
Isolated tests showed these could change long training trajectories beyond the
frozen agreement policy. Default training follows upstream FP32/Adam semantics.

Repairs also preserve gene order/duplicates, one-hot label/index conventions,
float64 cluster outputs and native reduction arithmetic, sparse value-based
sparsity, input ownership, list-valued metric coordinates, empty assignment keys,
and history schemas.

## Install and test

```bash
uv pip install .
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q tests/test_numerical_regressions.py
```

The new standalone regression tests pass (47 tests). The unmodified original
test file uses the removed `np.float` alias: six tests fail on that alias in
both upstream and this tree under NumPy 2.1.0. To run its original assertions
without editing that file:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -c \
  'import numpy as np, pytest; np.float = float; raise SystemExit(pytest.main(["-q", "tests/tangram_test.py"]))'
```

All 21 original assertions pass with that test-process-only compatibility alias.
No alias is installed in production code.

The separate validation bundle contains the differential harness, pinned
environment information, CPU/CUDA comparisons, benchmarks, and reproduction
instructions. Validation covers the bundled data and declared subsets, not all
datasets or hardware configurations. Precision agreement is tolerance-based,
not a promise of universal bitwise identity.
