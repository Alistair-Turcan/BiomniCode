# Recorded validation of ext45

This package contains the current-best ext45 source used in the completed
real-data evaluation, not a newly optimized or newly benchmarked variant.

- Baseline: untouched BayesSpace 1.21.2, commit
  `ba8b42211ef1581d31a659766559bd6e42be3acc`.
- Three complete Visium tissues: adult mouse brain (2,702 spots), breast cancer
  (3,798 spots), and human lymph node (4,035 spots).
- Three paired seeds per tissue (149, 102, 2026); 18 fresh measured processes.
- Student-t/equal precision clustering, four clusters, seven PCs, 64,000
  iterations, burn-in 1,000, thinning 100; identical frozen preprocessing and
  initialization, saved chains enabled.
- Mean process-wall speedups: 2.682x, 2.749x and 2.772x respectively.
  Mean peak-RSS reductions: 19.90%, 12.50% and 13.01%.
- All 41,537,823 compared stored values matched bitwise, with identical final
  labels and final RNG states in all nine pairs.
- Both sources were rebuilt in the recorded existing environment and passed
  the three full-spot 1,000-iteration paired replay checks.

`performance_summary.csv` contains the recorded timing/RSS summary and
`pair_summary.csv` the paired correctness results. These are historical results,
not tests rerun while making this ZIP.

## Scope and limitations

The evaluation used Linux, R 4.5.3, recorded compiler/BLAS/dependency versions,
single-CPU affinity and controlled thread settings. The repetitions are technical,
not biological. Results do not establish convergence, biological accuracy or
universal performance/bitwise identity on other models, dimensions or platforms.
Fresh dependency installation was not validated. The original package tests are
included, but this packaging step did not rerun the full upstream test suite.
This source-only ZIP cannot reproduce the external evaluation without its
separate data and evaluation harness.

## Packaging checks

Before archiving, the delivered source was checked against the recorded optimized
file hashes. The cumulative patch was applied to a disposable copy of the pinned
original source, and the reconstructed tree was checked against those same hashes.
The ZIP preserves file permissions, including the executable `configure` script.
README and ignore-file changes are packaging-only; R/C++ code, package metadata,
documentation, original examples and tests are byte-identical to evaluated ext45.
