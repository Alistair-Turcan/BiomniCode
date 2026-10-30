# Benchmark calibration and the overfitting guard

A benchmark that cannot distinguish a real speedup from noise turns Phase 5 into
random search. Calibrate before optimizing; guard against benchmark-size
overfitting before accepting.

## Scaling Suite construction

- Choose the input dimension that actually drives cost (rows, reads, cells,
  samples, batch size, problem dimension). Record it in the benchmark card.
- 3–5 deterministic sizes, log-spaced.
- The smallest size is the per-iteration benchmark: < 60 s per run, and computation
  (not startup/import/initialization) must account for >= 50% of total runtime. If
  profiling shows startup dominates, increase the size until the algorithmic core
  is the majority cost. Optimizations that only cut startup time do not scale;
  optimizations that cut computation do.
- The largest size is the closest realistic workload that fits the 15-minute
  Phase 6 cap (combined baseline-vs-best at every size plus the largest-size
  protected-original comparison). Drop the largest size, never the rep count, if
  the cap is exceeded — and document the drop.

## Sigma measurement

Measure the per-iteration baseline with >= 5 reps on the benchmark machine under
the same conditions used in Phase 5. The resulting stddev is the benchmark σ used
by `scripts/acceptance.py`. If measurements are noisy, increase reps, pin threads,
isolate warm-up effects, and keep comparing against the current best accepted
implementation — do not gate progress on an arbitrary noise percentage.

## Cold vs warm start

Measure both when both matter (CLI tools vs library kernels). Record which mode the
acceptance decisions use and keep it constant within a run.

## The overfitting guard (subset -> full data, code-optimization form)

Accept decisions are made at the per-iteration size for speed. A speedup that only
exists at that size is the code-optimization analog of overfitting to a validation
split. Therefore:

1. **Close results confirm upward.** If an acceptance clears the runtime branch by
   less than 2x the required margin (i.e. the measured improvement is within 2x of
   the 2σ + 3% threshold), re-run the candidate at the next-larger scaling size
   before final acceptance. A win that vanishes at the larger size is rejected as
   benchmark-size overfitting.
2. **Phase 6 re-validates everything.** The cumulative best is benchmarked against
   the baseline at EVERY scaling size, and against the protected original snapshot
   at the largest size (>= 5 reps per side), with output digests compared.
3. **Benchmark-only hacks are rejected.** Any change that improves the benchmark
   while plausibly hurting realistic workloads (e.g. special-casing the benchmark
   input shape, caching expected outputs, detecting evaluator inputs) is rejected
   regardless of measured speedup.

## Resistance to gaming

The benchmark must exercise the code's real entry points on realistic data. Never
let the optimized code detect the benchmark. Protected benchmark definitions,
fixtures, and expected outputs are hashed at Phase 0
(`scripts/protected_hashes.py create`) and verified after every iteration and in
Phase 6.
