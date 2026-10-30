# Optimization notebook

## Header
- Repository URL:
- Commit hash:
- Run ID:
- Environment (runtime versions, compiler flags, pinned deps):
- Build command:
- Benchmark command:
- Correctness command:
- Required budget: 50 substantive attempts by default; record any explicit user override
- Phase 5 start (persisted): <ISO timestamp from `start-phase5`>
- Numerical request: auto (default) | explicit setting: ___
- Frozen policy evidence/hash, per-output bounds and rationale: ___
- Workload-class diagnosis (Phase 2): cpu-bound | io-bound | syscall-heavy | allocation-heavy
- Machine id / size:
- GPU sandbox (only if CUDA-gated signals found):
- Original snapshot path (read-only):

## Baseline (Phase 1)
- Runtime mean +/- std (reps):
- Peak RSS mean +/- std (reps):
- Output digest (SHA-256):
- Determinism check:
- Benchmark sigma (per-iteration size std):

## Ranked hotspot table (Phase 2)
| Rank | Function/region | % runtime | Alloc share | Lines | Callers |
|---|---|---|---|---|---|

## Scaling Suite (Phase 3)
- Scaling axis (the input dimension that drives cost):
- Sizes (smallest = per-iteration, <60s, computation >= 50% of runtime):
- Per-iteration baseline mean +/- std:
- Size-choice justification:

## Correctness Suite (Phase 4)
- Cases (edge / normal / benchmark-like / degenerate):
- Comparison mode: justified frozen tolerance | exact where required/unsupported
- Original comparison maximum absolute/relative/scaled errors and failure counts:
- Suite wall-time (<60s target):

## Iterations (Phase 5) — one entry per iteration, mirrored in iterations/results.jsonl
| Iter | ID | Family | Intervention | Hotspot | Hypothesis | Correctness | Runtime | Peak RSS | Decision | Branch | Metric analysis |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Current best
- Patch:
- Runtime mean +/- std:
- Peak RSS mean +/- std:
- Cumulative speedup vs baseline:
- Cumulative peak-RSS delta vs baseline:

## Accepted changes

## Rejected hypotheses (with why)

## Ranked backlog

| Priority | ID | Family | Hotspot | Hypothesis | Minimal test | Evidence/source | Status |
|---|---|---|---|---|---|---|---|

## Latest learning and the current single hypothesis

Record the preceding feedback ID/hash, selected backlog ID if any, and why this
idea remains justified against current best and the latest result. Reassess and
rerank ideas after every result; admit and execute only one candidate at a time.

## Check-in reports (25% / 50% / 75% / 90% of budget)
