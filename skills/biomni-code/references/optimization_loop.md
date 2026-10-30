# Optimization loop

This protocol turns an open-ended "make this repo faster" request into a controlled,
auditable search. It runs inside Phase 5 of the skill until the termination gate
passes.

## Entry checklist

1. Restate the performance goal (runtime primary, peak RSS secondary).
2. Identify the entry points and the realistic workload that exercises the
   expensive paths.
3. Freeze the benchmark: per-iteration size, scaling sizes, benchmark command,
   correctness command and protected-file manifest. Numerical tolerance is
   selected from baseline evidence, not guessed before measurements.
4. Run the baseline (>= 5 reps) and record it with `set-baseline`.
5. Record the ranked hotspot table with `add-hotspots`.
6. Seed the ranked backlog with `add-backlog`: at least 5 evidence-grounded ideas
   across at least 3 families where supported; do not invent ideas to fill a quota.
7. Reason and freeze the numerical policy with `set-tolerance` (see
   `numerical_tolerance.md`), then persist the Phase 5 start with `start-phase5`.

## Adaptive loop

Each iteration has exactly one coherent hypothesis grounded in measured hotspots.
After reassessment, reserve the only active candidate with `begin-iteration`.
Implement its scoped change, run the full Correctness Suite, benchmark at
the per-iteration size (>= 3 reps exploration, >= 5 before accepting a close
result), decide with `scripts/acceptance.py`, and record with `add-iteration`
including mandatory `metric_analysis`, `next_action`, and `--evidence-json` — explain WHY the numbers came
out as they did and distinguish evidence from speculation. Record accept/revert
disposition, save a checkpoint, and verify it in a separate call before iteration
N+1. Parallel builds/tests may serve this candidate; no second candidate may start.

Select candidate N+1 only after interpreting candidate N's actual result. Reassess
the selected backlog idea (or new hypothesis) against current accepted code and
latest feedback; revise or retire stale ideas rather than blindly running the queue. Bind
`--feedback-id`, `--feedback-sha256`, and `--learning` to the latest reviewed
record returned by `suggest-next` (baseline for the first). The accepted parent
and latest feedback may differ after a rejection. Use `--backlog-id` when selecting
an existing proposal; `--learning` explains the reassessment. A substantial
algorithm replacement may span files and include necessary preparatory edits;
evaluate it as one coherent candidate, not as unrelated individually winning edits.
Never run multiple optimization iterations in one command, script, notebook cell,
loop, or background job. Repetitions/testing for one hypothesis remain allowed.

Accept/reject logic is mechanical (see `references/acceptance_and_correctness.md`):

- Accept via the runtime branch or the memory branch (tagged MEMORY-DRIVEN).
- Reject on correctness failure, unexpected output change, slowdown beyond the
  memory-branch allowance, memory regression beyond the runtime-branch guardrail,
  benchmark-only wins, or protected-file modification.
- Mark `infrastructure` when the run failed for environment reasons (build break,
  machine death, tool missing) rather than because of the patch's merit.

## Counting substantive iterations

Always complete 50 substantive attempts by default, not up to 50. Only explicit
user budget overrides change this target; sourced stops/unavoidable blocks leave
unfinished work, not successful completion. Never pad the count.

One substantive iteration advances the counter by exactly +1. Substantive =
`decision ∈ {accept, reject, memory-driven}` and not `--non-substantive`. Rejects
count: an informative failure is a real iteration. Pure infrastructure retries do
not count. The counter is derived from the append-only ledger by
`biomni_code_state.py`, so it is self-healing.

## Hard blockers are not successful completion

1. **infra** — compute exhausted or unavailable and cannot be restored (sandbox
   cannot be provisioned, repeated OOM that right-sizing cannot fix).
2. **harness** — the correctness or scaling harness is broken and unrepairable
   within scope, so no new candidate can be evaluated.
3. **user** — the user explicitly instructed a stop.
4. **repo** — the target repository cannot be cloned, built, or run at all after
   honest repair attempts.

Record with `record-blocker --kind <kind> --detail "<exact error>" --evidence-json FILE`
only after actual authority/host evidence establishes an unavoidable block and
feasible alternatives have been exhausted. Clear with evidenced `clear-blocker`
after reconciliation; a user stop requires explicit sourced resume. "Converged",
"diminishing returns", "plateau", "no immediate idea", "benchmark noisy", "already
faster" are NOT blockers — inspect the latest evidence to justify one next test.

## Machine-checkable termination gate

Before Phase 6 you MUST pass:

```bash
python scripts/biomni_code_state.py stop-check --root <run_root>
```

Exit 0 => ledger budget exhausted and eligible for Phase 6. Exit 3 => continue or
recover. Exit 4 => sourced stop/block, not completion. Exit 2 => invalid/missing
state. Paste the printed `STOP-GATE:` line before Phase 6, and use the separate
`run_control.py yield` gate before voluntarily ending a turn.

## Backlog replenishment

Maintain a ranked backlog with `add-backlog`, using priorities, optimization
families, hotspots, hypotheses, minimal tests and evidence sources. If fewer than
3 open ideas remain, inspect the latest result, re-profile the current best,
inspect adjacent call sites or read the relevant language hints and hotspot
taxonomy. Research and replenishment remain non-substantive. Reassess relevance
after every result; update an existing ID to rerank/revise it or mark it retired.
Ideas may be researched independently without implementing candidate variants.
The queue never authorizes execution: admit only one freshly reassessed candidate
after the preceding review and verified checkpoint.

## Diversity floor

If one optimization family accounts for >= 70% of the last 10 substantive
iterations, the next iteration must come from a different family or be a
profiling/calibration investigation. `suggest-next` computes this
advisory for you. Mutating only the current best family risks local optima.

## Check-in reports

At 25%, 50%, 75%, and 90% of the iteration budget, write
`reports/checkin_<pct>.md`: what changed since the last check-in, which families
have been tried, current best vs baseline, expected trajectory, decisions worth
flagging. These are for the user to skim — never a reason to stop.

## Failure discipline

Reconcile actual jobs and files before retrying an interrupted step. Reuse completed
outputs, await tracked callbacks, and preserve the same candidate and authorization.
Use the five-attempt transport-retry protocol in `interruption_recovery.md`; these
retries do not consume substantive iterations. Exhaustion changes recovery route,
not approval. Never silently drop an in-flight iteration or weaken correctness.
