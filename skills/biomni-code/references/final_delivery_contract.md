# Final delivery contract (Phase 6)

Phase 6 runs only after the termination gate passes (`stop-check` exit 0, with the
`STOP-GATE:` line pasted into the transcript).
An evidenced block is unfinished work, not a passed completion gate. Stage the
final artifacts under `ROOT/delivery/`, verify them there, then publish the same
files to the user-facing results location. Keep that run-specific local delivery
for the hash-bound completion receipt; do not validate unrelated session outputs.

## Required steps

1. Full Correctness Suite against the cumulative best. Also run the repository's
   own test suite end-to-end if it completes in under 10 minutes; otherwise note it
   as a follow-up.
2. Scaling Suite at every size, baseline vs cumulative best, >= 3 reps per size and
   variant, via `scripts/scaling_suite.py`.
3. Largest-size comparison against the protected original snapshot
   (`original_snapshot/`), >= 5 reps per side, outputs compared by digest (or
   justified frozen tolerance). Preserve protected-original comparison evidence,
   policy identity, per-output maximum errors and failure counts at every size.
4. Keep the combined Phase 6 run within the 15-minute cap: drop the largest size,
   never the rep count, and document any drop.
5. Raw measurements to `bench/phase6_results.csv` with a `protected_original` flag.
6. The Phase 6 scaling figure via `scripts/plot_scaling_panels.py`: one figure,
   three square panels in a row — (a) runtime, (b) peak RSS, (c) per-size
   digest-match fraction — vs input size, original in gray, optimized in red,
   mean +/- stddev error bars, no gridlines, bold a/b/c panel labels, saved as
   `phase6_panels.png` and `.svg` in `/mnt/results/`, plus the per-size summary
   CSVs `runtime.csv`, `memory.csv`, and `correctness.csv`. If zero changes were
   accepted, produce baseline-only panels labeled as such.
7. `optimization_report_<repo>.md` in `/mnt/results/` — start from
   `biomni_code_state.py render-report` so the complete iteration history
   (accepted, rejected, memory-driven, infrastructure) is pre-filled, then write
   the prose sections.
8. The best code (patched working copy) and `current_best.patch` in `/mnt/results`.
9. Final history sync: `biomni_code_state.py checkpoint --root ROOT --store STORE`
   followed by a separate `run_checkpoint.py verify` establishes a recoverable
   generation and mirrors the complete
   iteration history (ledgers, status, `bench/phase6_results.csv`) to
   `/mnt/results/biomnicode/<run_id>/history/`; verify `iterations/results.jsonl`
   is present there.
10. `scripts/verify_delivery.py --delivery /mnt/results` must pass.
    Also run it on the staged run-specific `ROOT/delivery/`, record the actual
    validation results and artifact hashes, and pass `run_control.py yield` with
    `reason=complete` as documented in `interruption_recovery.md`.

## Final report contents

- Cumulative best runtime improvement vs baseline (mean +/- std, reps).
- Cumulative best peak-RSS change vs baseline (mean +/- std, reps).
- Workload-class diagnosis from profiling.
- Scaling Suite sizes and the chosen scaling axis.
- Headline largest-size protected-original vs optimized result: runtime, peak RSS,
  correctness verdict, exact commands, commit, accepted patch list.
- Accepted changes: hotspot, mechanism, individual runtime contribution,
  individual memory contribution, MEMORY-DRIVEN tags.
- Rejected changes and why (instructive failures stay visible).
- Commands to reproduce every benchmark.
- Correctness tests used.
- Frozen numerical policy, rationale and actual comparison results, separate
  from exact digest-match diagnostics. No post-failure tolerance loosening.
- Remaining measured hotspots, latest learning, and ranked evidence-grounded remaining ideas.
- Blockers, uncertainty, open questions.

## Packaging

- `scripts/create_manifest.py` over the delivery tree.
- `scripts/safe_zip.py` for a deterministic archive (identical SHA-256 on rebuild).
- Keep `/mnt/results` under 10,000 files; prefer JSONL ledgers and compressed
  artifacts over many small files.
