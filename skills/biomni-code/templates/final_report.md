# Optimization report

## Executive summary
Cumulative best runtime improvement vs baseline (mean +/- std, reps); cumulative
peak-RSS change (mean +/- std, reps); one-paragraph story of what won and why.

## Target and environment
Repository, commit, environment, build/run commands, machine, budget, FP tolerance,
workload-class diagnosis.

Required substantive budget: 50 unless explicitly overridden; completed attempts
and any real unfinished block. Frozen numerical policy identity, rationale,
per-output bounds and evidence; explain why looser bounds would be inappropriate.

## Baseline
Runtime, peak RSS, output digest, determinism result.

## Scaling Suite
Sizes, scaling axis, calibration (sigma, startup share), per-iteration size choice.
Reference the Phase 6 scaling figure (`phase6_panels.png`: runtime / peak RSS /
digest-match fraction vs size) and the per-size summaries (`runtime.csv`,
`memory.csv`, `correctness.csv`).

## Headline largest-size result
Protected-original vs optimized: runtime, peak RSS, correctness verdict (digest
equality or frozen-policy equivalence), exact commands, commit, accepted patch list.
Report numerical error maxima and failure counts at every size separately from
the digest-match diagnostic; do not relabel exact-match plots as tolerance passes.

## Accepted changes
Per change: hotspot, mechanism, individual runtime contribution, individual memory
contribution, MEMORY-DRIVEN tags where applicable.

## Rejected changes and why

## Complete iteration history
(Pre-filled from iterations/results.jsonl by `biomni_code_state.py render-report`.)

## Correctness tests used

## Remaining likely hotspots

## Latest learning and ranked remaining ideas

## Blockers, uncertainty, open questions

## Reproduction commands
Exact commands, environment, seeds, commit.
