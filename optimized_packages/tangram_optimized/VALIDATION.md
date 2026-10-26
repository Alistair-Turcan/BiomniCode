# Validation and known limitations

## Exact implementation identity

This repository packages the delivered further-optimized current-best candidate
(`iter030`), not just the earlier numerical-repair package. Its original source
archive has SHA-256:

```text
e80640ae91bcb2508c28d8a51991130ca96c64e021bbd2884161fff412405259
```

The original baseline is upstream Tangram commit
`4c68995a418f41dc8caef567598c4d9b47781a13`.
Every Python/C++ source and test file is verified byte-for-byte against that
delivered source archive during packaging. No new optimization, precision
change or model repair is part of this GitHub handoff.

## Completed full-size GPU evaluation

Both unchanged implementations were measured on three real workloads and
three paired technical seeds (0, 42, 1234): 18 successful canonical main runs.
All canonical measurements use one actual NVIDIA A10 at a 150 W cap.
Earlier A10G runs are supplementary and are not pooled into these results.

| Workload | Reference cells | Spatial spots | Common genes | Mapping speedup | Mapping + CPU projection speedup |
|---|---:|---:|---:|---:|---:|
| Mouse cortex | 26,431 | 9,852 | 17,991 | 1.731× | 1.498× |
| Human lymph node | 73,260 | 4,035 | 10,178 | 1.721× | 1.568× |
| Human breast cancer | 100,064 | 3,798 | 17,914 | 1.723× | 1.486× |

These are mean seed-paired ratios, not universal performance guarantees.
Peak allocated CUDA memory decreased by approximately 11.9–12.2%.
Projection remains the unchanged CPU operation.

For all nine canonical pairs, final mappings and all-common-gene projections
were bitwise identical. Loss tolerance checks, probability row-sum checks and
argmax comparisons passed. Matching inactive-regularizer NaNs remain explicit.
This result is specific to the measured workloads, settings and environment.

Settings: FP32, Torch 2.7.1+cu126, CUDA 12.6, Python 3.11; deterministic algorithms;
TF32/autocast disabled; 500 training genes, 500 epochs, learning rate 0.1, cells
mode, RNA-count density, lambda_d=lambda_g1=1, other regularizers zero;
two numerical/intra-op threads and one inter-op thread. Timing synchronizes
CUDA at public API boundaries and includes output transfer.

Frozen acceptance bounds use abs(new−original) ≤ atol + rtol×abs(original):

| Quantity | atol | rtol |
|---|---:|---:|
| Finite loss histories | 1e-6 | 1e-5 |
| Mapping/projected expression | 1e-4 | 1e-3 |
| Mean gene cosine score | 1e-4 | 0 |
| Probability row sums | 1e-6 | 0 |

Row-sum reductions retain inherited FP32 semantics. No tolerance was widened.
The separately delivered `tangram_optimized_lite.zip` contains the full GPU
report, figures, derived diagnostics, logs and provenance, without raw arrays.
That evaluation ZIP is intentionally not duplicated inside this source repo.

## CPU and inherited limitations

The CPU evaluation must not be described as passing every strict numerical
gate: it retained probability-normalization failures shared by both
implementations, and two first-epoch loss-tolerance exceedances for mouse seed
42. These are disclosed findings, not repaired or hidden during packaging.

At the end of the further-optimization campaign, native repository testing
reported 62 passing tests and six failures from legacy `np.float` use. A
separate compatibility retest of those six passed. A complete combined
compatibility rerun was not recorded at that stage. The GitHub CI supplied
here is configured to run the full suite using a test-process-only alias;
its presence is not a claim that a new full test run occurred during packaging.

The earlier repair-only validation is preserved, as historical documentation,
in [VALIDATION_REPAIR.md](VALIDATION_REPAIR.md). Its small warm-training speedups
and test results must not be relabeled as full-size current-package results.

## Interpretation and packaging checks

- These are implementation-equivalence and technical-seed benchmarks, not
  biological ground-truth validation; atlas/spatial samples are not donor matched.
- Three seeds support descriptive performance summaries, not strong
  significance claims.
- CPU native optimizations are gated and are not expected to provide the
  same additional acceleration on CUDA.
- A10/A10G, small/full-size and warm-training/public-API timings are different
  experiments. No matched causal decomposition of those differences was run.
- Packaging checks cover source identity, Python syntax, wheel construction,
  included C++ source, ZIP CRCs and per-file hashes. They are not a fresh
  numerical benchmark or a guarantee across all dependency versions.
- The original license, package metadata, tests and bundled small test data
  are preserved. Distribution metadata is inherited, not authorization to
  publish a new upstream PyPI release.
