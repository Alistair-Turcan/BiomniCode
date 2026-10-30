# Sequential experiments and interruption recovery (helpers 2.5.4, policy 2.5.3)

Maintain diverse ranked ideas, but admit and execute one candidate at a time.
Reassess each selection against current best and latest feedback. Finish detailed review and accept/revert,
commit once, checkpoint, verify in a separate call, then admit the next candidate.
The original authorized budget remains in force. Do not ask to continue after
routine failures, resets of a tool connection, callbacks, or progress messages.
Actual user stops, safety requirements, permissions and host limits still bind.

## Storage and coordinator

Use local `ROOT=/workspace/biomnicode/<repo>` for active state, Git, work, environment,
and compilation; `STORE=/mnt/shared-workspace/biomnicode/<repo>/checkpoints` for
immutable snapshots. These are distinct roots. Only one coordinator owns ROOT.
State mutations use a local lock and rollback journal: an interrupted *uncommitted*
mutation is rolled back on the next command; committed iteration IDs are replay-safe.
The journal enrolls only actual mutable files before writing them, with prior
existence, bytes and permissions. Immutable evidence is checked, not copied into
undo state. Read-only commands still acquire the lock and recover interrupted writes.
Full checkpoint checks are not skipped: repeated blob references are deduplicated
within a pass and remote reads use at most eight I/O threads. No cross-command
metadata-only verification cache is used. Read `helper_latency.md` when upgrading
helpers, diagnosing I/O limits or repairing a history mirror.
Remote workers publish unique result files, never write the coordinator ledger.
The lock is not cross-host exclusion. After host loss, reconcile old jobs before
restoring one coordinator; never run two restored copies concurrently.

Pin commands to cwd-based, reproducible source/build recipes. A restored source
tree omits `.git`, environments and Python/test caches; reinitialize Git from the
recorded original source when needed and keep the original commit provenance.
Do not run commands that depend on a different absolute worktree without explicit
path reconciliation. Rebuild environments from their recipe; a source checkpoint
is not an environment snapshot.

## Initial setup and authority

Initialize the local run with the existing state CLI. Capture original run-level
approval, not an agent summary, then:

```
python scripts/run_control.py record --root ROOT --record-json AUTHORITY.json
```

AUTHORITY fields: `action=approve|stop|resume`, `scope=run`,
`source_kind=user_message|host_run_control`, `source_ref`, exact `instruction`,
timezone-aware original `timestamp`, run-relative `evidence`, and its `sha256`.
An explicit stop needs a later explicit resume. Hashes prove linkage/integrity,
not the authenticity or semantic meaning of user/host evidence.

Create `protected_files.json` using run-relative keys (run `protected_hashes.py
create` from ROOT). It must contain the protected harness/evaluator paths. Preserve
the original code in `original_snapshot/`, with executable modes intact.
Create `recovery_inputs.json` with:

- `environment_recipe`, `build_recipe`, `correctness_runner`: paths to nonempty
  run-relative reproducibility files.
- `external_inputs`: an explicit list, possibly empty, of objects with absolute
  durable `recovery_location` and `sha256`. Account for every input needed to
  reconstruct/test the run; the helper cannot discover undeclared dependencies.
- Optional `rebuildable_outputs`: exact work-relative output files, declared before
  `set-baseline`, for example:

```json
"rebuildable_outputs": [
  {"path": "build/solver.o", "rationale": "Disposable compiled object",
   "rebuild_recipe": "recipes/rebuild_outputs.sh"}
]
```

Declarations and recipe hashes freeze at baseline. Each output must be absent from
the clean `original_snapshot/`; it may already exist untracked in `work/`.
No directories, patterns, symlinks, tracked/original source, protected files,
required inputs/recipes or hash-linked evidence may be excluded. Git ignore rules
do not authorize exclusions. Use a nonempty run-relative recipe that rebuilds the
output through the recorded build workflow; a wrapper outside `work/` may invoke
the repository's build files, which remain ordinary candidate source. Verify the
recipe actually rebuilds the product. Source identities, cumulative-patch checks
and checkpoints omit only these declared products, which must be rebuilt on recovery.
For dynamically named or later-introduced products, build under the excluded `env/`
or outside ROOT instead of expanding the frozen exclusions during an experiment.

`reports/status.json` and `reports/status.md` remain in snapshots but are derived
views: regenerating them does not stale a verified revision unless a protected or
required path or an authoritative evidence link makes them part of the record.
Source, ledger and other evidence changes still require a new checkpoint and verification.

Required data/harnesses may instead be stored inside ROOT and checkpointed. Avoid
checkpointing live databases or changing files. The checkpoint includes regular
files and safe in-tree symlinks, preserving executable modes. Unsupported/unsafe
links and live database journals are rejected. Store disposal/rebuildable large
caches outside ROOT, or under its excluded `env/`; never hide required inputs there.
The original source is also archived deterministically in
`recovery/original_source.tar`. Record cumulative **binary-safe** patches relative
to that original, including new files and executable-mode changes.

## One-candidate admission and review

Before initial admission or after a completed candidate:

```
python scripts/biomni_code_state.py checkpoint --root ROOT --store STORE
# In a separate execution call, use the returned immutable snapshot ID:
python scripts/run_checkpoint.py verify --root ROOT --store STORE --snapshot ID
python scripts/biomni_code_state.py mirror-history --root ROOT
python scripts/biomni_code_state.py suggest-next --root ROOT
# Read the latest result; reassess and select ONE idea against current best.
python scripts/biomni_code_state.py begin-iteration --root ROOT \
  --iteration-id ID --hypothesis "one measurable hypothesis" [--parent-id BEST_ID] \
  [--backlog-id IDEA_ID] --feedback-id FEEDBACK_ID --feedback-sha256 FEEDBACK_SHA \
  --learning "why this idea remains justified against current best and the latest result"
```

Omit parent only for the original baseline. Otherwise use the latest accepted
iteration ID. Make no unrelated state/evidence/source changes after verification
and before admission; a stale snapshot is rejected. A second candidate cannot
start while the first has an active reservation or unfinished job.
Use the exact feedback identity returned by `suggest-next`: the latest reviewed
iteration, including a rejection/infrastructure result, or `baseline` initially.
The helper hashes the complete latest record, or the baseline metrics and original
source identity. `learning` explains why that evidence motivates this one test,
including reassessment when the idea was queued earlier. `--backlog-id` binds its
current proposal revision; revise an outdated proposal before checkpointing and
admitting it. Other ideas may be researched or reranked during the active candidate,
but its selected proposal cannot change until review completes.
These fields establish traceable linkage, not proof of internal reasoning.
The ranked backlog is advisory, not a runnable batch. Never run multiple optimization iterations in a
single command, script, notebook cell, loop, or background job. Multiple benchmark
repetitions or correctness cases for this one hypothesis are permitted.

Run the original correctness/benchmark/acceptance workflow. Review every substantive
result, including rejects, before committing:

```
python scripts/biomni_code_state.py add-iteration --root ROOT \
  --iteration-id ID [--parent-id BEST_ID] --family FAMILY --intervention TECHNIQUE \
  --decision accept|reject|memory-driven|infrastructure --correctness pass|fail|not-run \
  --metric-analysis "mechanism, observations, uncertainty" \
  --next-action "specific follow-up" --evidence-json REVIEW.json [existing metric arguments]
```

REVIEW contains `iteration_id`, nonempty `mechanism` and `uncertainty`,
`disposition=accepted|reverted`, `tested_patch` (run-relative),
`tested_patch_sha256`, `disposed_tree` (from `run_checkpoint.source_inventory(Path(ROOT))`),
and nonempty `artifacts`. This source inventory applies the frozen rebuildable-output
policy; do not substitute a raw directory inventory. Each artifact is `{evidence, sha256}`.
Keep the actual logs/raw measurements, not only their summary.

For substantive results, `correctness` is an evidence object pointing to JSON
with `correctness=pass|fail` and `tested_patch_sha256`. A failing correctness
check must be rejected; do not invent runtime measurements for it.
For a bitwise pass, the CLI and correctness evidence also need `output_digest`
matching the recorded original. Frozen tolerance mode instead requires
`tolerance_evidence` bound to the policy, tested cumulative patch and protected
original, with per-output error summaries and actual comparison artifacts.
See `numerical_tolerance.md` for the full schema. Before Phase 5, all new runs
must resolve the pending policy using `set-tolerance`, even for exact mode.
Recovery checks the same policy identity; it cannot authorize new bounds.
For passing correctness, also provide:
- `benchmark`: evidence for JSON with `tested_patch_sha256`, `runtime_mean`,
  `runtime_std`, `runtime_reps`, `mem_mean`, `mem_std`, `mem_reps`, matching CLI
  values. Repetition minima and all existing measurement requirements remain.
- `acceptance`: evidence for the exact unchanged `acceptance.py` JSON verdict.
  The state gate reruns that rule from bound baseline/current-best measurements.
- Close acceptances require at least five reps and `next_size_confirmation`
  evidence containing `confirmed=true` and `tested_patch_sha256`, supported by
  the actual next-size measurements. Do not assert confirmation from the small
  workload alone.

If the primary mechanical verdict accepts but a required larger-size confirmation
rejects, preserve the primary verdict and record a truthful stricter rejection
using hash-linked `failed_scaling_evidence`. Its proof binds the same tested patch,
the failed confirmation, declared larger size, parent/candidate raw benchmarks and
the unchanged-rule scaling verdict. Both raw benchmarks require at least five
successful deterministic repetitions with matching commands, digests and summary
statistics. See `helper_latency.md` for the proof keys; this can only support
rejection, never turn a failed primary result into acceptance.

Use `--patch-file` with the same cumulative patch for accepted changes. Rejected
or infrastructure candidates must be back at the recorded best source tree.
Promotion requires a change in authoritative source/build configuration; changing
only a disposable product cannot count as an accepted optimization.
Infrastructure transport repairs themselves do not require fresh candidate IDs:
record their evidence/retry outcomes without consuming substantive iterations.
If abandoning an executed candidate for infrastructure reasons, close it as
infrastructure with a reverted disposition; retain its history and observations.
Preserve the backlog and diagnose what that outcome taught you before reassessing
and admitting one next test. Research ideas without preparing executable variants.

## Job reconciliation, retries and callbacks

Register actual tracked jobs with `run_control.py job --root ROOT --record-json FILE`.
Fields: `job_id`, current `iteration_id` when a candidate is active,
`status=queued|running|unknown|completed|failed|cancelled`, `callback_registered`
(true only for an actual callback), and `{evidence, sha256}`. Register uncertainty
as unknown, not failed. Preserve the platform job identity after connection loss.

On every resume/callback, read this guide, then:

```
python scripts/run_control.py enter --root ROOT --purpose resume
# or --purpose callback
```

Inspect actual live job state, output receipts and partial files. Reuse completed
outputs, await running jobs, and reconcile unknown submissions before any repeat.
Supply `run_control.py reconcile --record-json FILE` with `{evidence, sha256}`,
`jobs` (a map accounting for every registered job, each with status/evidence),
and for an active candidate: `iteration_id`, actual coordinator `owner_id`
(hostname), and earliest incomplete `stage=edit|correctness|benchmark|review`.
Taking over an active candidate requires all its prior jobs to be terminal.
Reconciliation verifies linkage, not truth of claimed execution. Do not fabricate it.

For transient, safe reads/local writes, use `run_control.py retry`. The record has
`operation_id`, `input_fingerprint`, `operation_kind=read|local_write`,
`status=failed`, `transient=true`, `retry_safe=true`, `retry_forbidden=false`,
`reconciliation`, `{evidence, sha256}`, and optional finite `retry_after_seconds`.
The helper reserves at most five retries after the initial attempt and returns
5/15/30/60/120-second delays (or a longer supported server interval). Wait that
delay, perform the actual retry, then record `retry-result` with `operation_id`,
`reservation`, actual `status=completed|failed|unknown`, and source evidence.
Never reserve another retry until the previous outcome is reconciled as failed.

Do not use transport retries for deterministic correctness failures, uncertain
submissions, non-idempotent writes, or concurrent GPU-limit rejections. Counts
survive resume. After exhaustion change the recovery route, restore a verified
generation, or continue feasible diagnostics/evidence inspection; do not reset counts
or spin indefinitely. No automatic retry is scheduled by this script.

## Restore and resume without reset

```
python scripts/run_checkpoint.py restore --store STORE --snapshot ID --destination EMPTY_ROOT
python scripts/auto_resume.py --root ROOT --store STORE \
  --repo-url ORIGINAL_URL --commit-hash ORIGINAL_COMMIT --reconciliation-json FILE
```

Alternatively, auto-resume accepts `--snapshot ID` when ROOT is genuinely empty.
Every blob and manifest is checked before restoration. Never overwrite an unknown
or dirty destination. A partially restored directory is not verified: preserve it
for inspection and retry restoration into a new empty directory.

Auto-resume checks original-source integrity, protected files, authority and job
observations. It reconstructs the cumulative best in a temporary local directory,
requires matching source identity, reruns build when applicable and always runs
the recorded correctness command. A synchronous subprocess timeout is cleaned up
as a process group and reported, not treated as successful recovery. Keep build
products in the declared rebuildable-output policy or outside the source tree;
rebuild them with the recorded workflow before using restored work.

Exit 0: resume the same candidate's earliest incomplete stage, otherwise begin the
next candidate or finish Phase 6 if the ledger budget is exhausted. A pre-policy
run resumes numerical policy resolution before any candidate.
Exit 2: genuinely fresh, no prior state or checkpoint evidence.
Exit 4: recover or honor an evidenced block; **never silently restart**.
Repository mismatch, missing correctness command, bad patch, damaged snapshot,
timeout or wrong source does not erase history, approval, retries or the budget.
Save and separately verify the reconciled revision before the next admission.

Old 2.4/2.5.0/2.5.1/2.5.2 runs stay bound to their old skill. There is no automatic history migration:
to opt into 2.5.3, preserve the original run, recover original approval, explicitly
reconcile IDs/source/receipts and document the policy boundary in a separate
validated copy. Do not rewrite earlier ledger entries or invent missing evidence.

## End-of-turn and completion gates

`stop-check`: 0 = budget exhausted/Phase 6 eligible; 3 = continue/recover;
4 = sourced stop/unavoidable block; 2 = invalid state. Blockers are not completion.
`record-blocker` requires `--evidence-json`; `clear-blocker` requires actual
reconciliation and explicit resume authority when there was a user stop.

Before voluntarily ending, use `run_control.py yield --root ROOT --record-json FILE`.
All records have `{evidence, sha256}` and one reason:
- `complete`: include `delivery` evidence for a run-bound receipt containing
  `run_id`, `verified=true`, `correctness_pass=true`, `protected_pass=true`,
  `scaling_complete=true`, `delivery_check_pass=true`, nonempty hash-linked
  `artifacts`, and run-relative `delivery_dir`. The helper also runs
  `verify_delivery.py` against that directory. Capture actual final checks,
  checkpoint and verify after creating the receipt, then request complete yield.
  The receipt establishes linkage, not independent scientific certification.
- `user_stop`: requires the currently recorded sourced stop.
- `waiting_job`: actual registered queued/running `job_id` with a callback.
- `platform_unavailable`: actual `source_kind=host_status`, relevant exhausted
  retry `operation_ids` or evidenced `retry_prohibited=true`.
- `authorization_block`: actual `source_kind=user_message|host_authorization`.

Wait/block records additionally require `ready_work=[]`, nonempty `affected_work`,
`next_action`, and `alternatives_exhausted=true`. These assertions must reflect
real work state, not convenience. If ready work exists, do it. If every tool is
unavailable, preserve the truthful next action without claiming the gate ran.
Shared checkpoints are session-scoped; copy final recovery artifacts to results
and separately verify them for longer retention.
