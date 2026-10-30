# State and feedback memory

Durable memory lets the optimizer avoid repeating failed patches and build on
accepted ones. All state lives under the run root
(`/workspace/biomnicode/<repo>/`) and is managed exclusively through
`scripts/biomni_code_state.py` — never edit ledger files by hand.

## Core files

- `state/state.json`: run metadata, repo URL + commit, budgets, persisted Phase 5
  start timestamp, iteration counter, baseline metrics, current best, benchmark
  sigma, cumulative speedup / peak-RSS delta, optional blocker, original numerical
  request, resolved numerical policy evidence and canonical policy digest,
  plus frozen `source_policy` and `source_policy_sha256` for rebuildable outputs.
- `state/strategy_weights.json`: adaptive priorities over optimization families,
  with exploration floors.
- `state/diversity_portfolio.json`: per-family attempts, accepts, best evidence.
- `state/backlog.jsonl`: ranked candidate ideas and append-only revisions/status.
  These are proposals, not authorized execution plans.
- `iterations/results.jsonl`: one record per iteration (the append-only ledger).
- `iterations/leaderboard.csv`: substantive iterations ranked by runtime.
- `feedback/feedback.jsonl`: lessons, failed patterns, promising mechanisms.
- `profiling/hotspots.jsonl`: ranked hotspot tables from each profiling round.
- `logs/iteration_log.jsonl`: the one-line per-iteration heartbeat.
- `state/control.json`: sourced authority, jobs, recovery barrier and persistent retries.
- `state/verified_checkpoint.json`: separately verified immutable snapshot identity.
- Durable snapshots: `/mnt/shared-workspace/biomnicode/<repo>/checkpoints/`.
  Local writes use a single-coordinator lock and rollback journal. Object-backed
  publication uses closed full files and immutable generations, never POSIX append.
- `$BIOMNICODE_RESULTS_ROOT/<run_id>/history/` (default
  `/mnt/results/biomnicode/<run_id>/history/`): user-facing mirror of the complete
  history. Ordinary mutations copy changed files as complete objects — never
  POSIX append, because `/mnt/results` is S3-backed and appends can fail silently.
  Checkpoint and `mirror-history` commands content-check all history objects.
  Use `mirror-history --full` to force a complete rewrite after a mirror failure;
  it does not replay a committed iteration. Read-only commands do not mirror.
  The mirror is accumulative; nothing is deleted during a run.

The 2.5.4 helpers use a bounded write-set undo journal and preserve the 2.5.3
run-policy and checkpoint schemas. Recovery understands legacy blanket journals.
Do not downgrade while an interrupted schema-2 transaction remains unresolved.
See `helper_latency.md` for the authoritative/derived-output boundary.

## Iteration record fields

`timestamp, iteration, iteration_id, parent_id, family, intervention, hotspot,
hypothesis, edited_files, patch_file, correctness (pass|fail|not-run),
runtime_mean, runtime_std, runtime_reps, mem_mean, mem_std, mem_reps,
output_digest, decision (accept|reject|memory-driven|infrastructure), branch
(runtime|memory|none), non_substantive, summary, metric_analysis, next_action,
review, request_sha256, feedback_id, feedback_sha256, learning, numerical_policy_sha256,
source_policy_sha256` and optional `backlog_id, backlog_sha256`. A `begin-iteration` record reserves the sole active
candidate before edits. Completion must match it; identical replay does not count
twice, and conflicting replay is rejected.
Each admission binds to the immediately preceding reviewed record, including
rejects and infrastructure results, or baseline/source identity for the first.
`parent_id` remains the accepted source parent and does not replace feedback.
`suggest-next` returns the exact feedback identity, ranked families/backlog, latest
hotspot table, portfolio and recent results. Unresolved work prevents admission,
even when ideas are available for inspection. `--learning` must explain why the
selected idea remains justified against current best and latest feedback;
`--backlog-id` links an existing proposal to its one admitted candidate.
Missing metric fields are recorded explicitly as `null`; backlog linkage is omitted
when the candidate was not selected from the queue.

## Ranked backlog

Use `add-backlog --priority N --family FAMILY --intervention TYPE --hotspot TEXT
--hypothesis TEXT --test TEXT --source TEXT`. Optional `--backlog-id ID` revises
an existing idea by appending its complete replacement fields; omit it to create
an ID. `--status open|retired` records current
eligibility. Reviewed substantive attempts close their linked idea; infrastructure
results leave it available for reassessment. Preserve history rather than editing
the ledger. Other ideas may be added or reranked during an active candidate; its
selected idea cannot change until resolved. Ranking is advisory; before admission reassess the selected proposal
against the accepted source and immediately preceding reviewed result.

## Iteration counter and termination

- `state.json.iteration` is the number of **substantive** iterations, derived from
  `iterations/results.jsonl` on every `add-iteration`. A record is substantive iff
  `decision ∈ {accept, reject, memory-driven}` and it is not flagged
  `non_substantive`. Infrastructure retries are kept for the audit trail but do not
  advance the counter. One substantive iteration = exactly +1. The counter is
  self-healing; a manual `--iteration` value is advisory only.
- `state.json.budget_iterations` (required default 50, not “up to 50”; explicit
  user override only) is the only budget — it is purely
  iteration-based, with no wall-clock limit. `phase5_start_ts` is persisted by
  `start-phase5` the moment Phase 5 begins and is never reset; it is an
  informational elapsed-time marker (shown in `status`/`summary`) and never
  gates finalization.
- `state.json.blocker` (optional) records an evidenced unavoidable stop/block, not completion:
  `{kind, detail, recorded_at, iteration_at_block}` with
  `kind ∈ {infra, harness, user, repo}`.

## Termination gate contract

`biomni_code_state.py stop-check --root <run_root>`:

- Exit `0` only if the ledger-derived budget is exhausted, approval remains valid,
  and no active candidate, unresolved recovery/job, or blocker remains.
- Exit `3` means continue or recover; prints `remaining_to_floor`.
- Exit `4` means sourced stop/block; never budget completion.
- Exit `2` means uninitialized/invalid state.
- Reports `counter_matches_ledger`; a cached counter never overrides ledger truth.
- Always prints `self_audit_line`:
  `STOP-GATE: iteration=<N>/<max>; may_finalize=<yes|no>`.
  Paste it verbatim before any Phase 6 step.

Full schemas, checkpoint verification, restore, callback reconciliation and
voluntary-yield rules are in `interruption_recovery.md`. Existing 2.4 runs stay
on their bound skill until an explicitly documented migration preserves their
history and original approval; no automatic reset or rewriting of prior records.

## Strategy weight updates

`init --fp-tolerance auto` creates a pending numerical policy. After original
characterization, `set-tolerance --evidence-json FILE` freezes a reasoned policy;
even explicit exact mode records a rationale. Resolution, evidence and comparison
schemas are in `numerical_tolerance.md`. No policy changes after admission,
no automatic migration, and no numerical drift relative to the protected original.

`accept` ×1.25, `memory-driven` ×1.15, `reject` ×0.85, `infrastructure` unchanged;
renormalized over floors after every iteration. Do not penalize infrastructure
failures as optimization failures. Floors keep plausible under-tested families
alive.

## Feedback item fields

Type (success, failure, risk, profiling insight, benchmark insight, ablation
insight, speed insight), component/hotspot, evidence, recommendation, similar
ideas worth trying, ideas to avoid.
