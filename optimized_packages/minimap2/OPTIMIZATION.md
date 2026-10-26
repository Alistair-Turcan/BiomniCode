# Optimized minimap2 (v2.31 base)

This branch (`optimized-v2.31`) contains a cumulative performance optimization of upstream
minimap2 v2.31 (commit `3c28777e7e2dcc90f825de1b9f17a89cca7d4452`), produced by three rounds
of automated benchmark-driven optimization (Biomni `biomni-code`). The diff vs upstream is
exactly one commit; build with the stock Makefile (`make`).

## Performance

Single thread (`-t 1 -ax map-ont`), GRCh38 no-alt index, 16 vCPU / 32 GB sandbox:

| Evaluation | Original v2.31 | This branch | Speedup | Peak RSS Δ |
|---|---|---|---|---|
| Optimization benchmark (synthetic, 50-iteration protocol) | 19.18 s | 8.80 s | 2.18× | +0.7% |
| Real ONT D1 — GM24385 PAF27096, 4,000 reads (122.5 Mb) | 394.95 ± 6.05 s | 194.71 ± 18.13 s | 2.03× | +0.98% |
| Real ONT D2 — GM24385 PAG07165, 4,000 reads (82.4 Mb) | 252.81 ± 12.71 s | 118.18 ± 7.75 s | 2.14× | +0.85% |
| Real ONT D3 — GM24385 PAG07506, 4,000 reads (124.0 Mb) | 568.24 ± 0.53 s | 186.36 ± 0.65 s | 3.05× | +4.34% |

The speedup scales with mapping-phase intensity: longer, harder-to-map reads benefit most
(the optimized chaining/alignment SIMD paths dominate), while very small runs are bounded by
the (also optimized) index-load phase.

## Correctness

Output is **byte-identical** to upstream v2.31 on every evaluation run: raw SAM sha256 matches
across variants for all 36 real-data runs (3 datasets × 2 scales × 3 replicates × 2 variants),
and per-record comparison of 18,804 paired alignments shows 0 mismatches in position, MAPQ,
alignment score (AS), edit distance (NM), and alignment block length.

## What changed (8 files)

`index.c`, `lchain.c`, `seed.c`, `map.c`, `main.c`, `mmpriv.h`, `ksw2_dispatch.c`,
`ksw2_extd2_sse.c` — index-load parallelization/teardown elision, chaining and seed-processing
improvements, and AVX2/AVX-512 dispatch for the DP alignment kernels. See the commit diff for
full detail.

## Known caveat

With AddressSanitizer + LeakSanitizer, the build fails on **leak-at-exit only**: the accepted
exit-time teardown elision intentionally skips freeing the index at process exit (the OS
reclaims it). With `ASAN_OPTIONS=detect_leaks=0` the ASan build passes all 26 correctness
tests with zero memory errors. There are no leaks during operation; steady-state memory is
flat (see RSS deltas above).

## Provenance

- Round 1+2: Biomni run `bc-20261010-000052-acc5903a` (cumulative patch sha256
  `1c870c5e7d7d12ca583ef83a7fc6956aff001cde13634586430bff37164cbde7`)
- Round 3: Biomni run `bc-20261010-162956-34f673c6`, best iteration `r3i022`
  (8.801289 ± 0.066112 s vs baseline 19.1824248766 ± 1.0777 s on the benchmark)
- Full evaluation bundles (all raw timings, digests, paired-value data, plots, scripts):
  `evaluation_real_datasets.zip`, `evaluation_real_datasets_large.zip`,
  `minimap2_optimized_lite.zip` in the round-3 results folder.
