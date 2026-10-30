# biomni-code

Optimize real code repositories for runtime and memory efficiency while preserving
public behavior and scientific meaning under a frozen numerical policy. Version 2.0.0 ports tusoskill's executable context-management
machinery (state ledger, mechanical stop gate, adaptive strategy weights, diversity
portfolio, history mirroring, packaging/verification tooling) and re-specializes it
for code optimization: runtime/peak-RSS metrics, bitwise correctness gating, and a
σ-based acceptance rule. Version 2.1.0 adds Go, Java, Julia, and JavaScript hint
files, extends the Rust hints (build flags, allocators, PGO, async, FFI), moves the
per-language profiler toolbox into `references/`, and trims SKILL.md's always-loaded
context (~14% fewer bytes, ~24% fewer lines) with identical functionality.
Version 2.2.0 makes the Phase 6 scaling figure a single Nature-style 3-panel
plot (runtime / memory / correctness vs input size, original gray, optimized
red), adds per-size summary CSVs (`runtime.csv`, `memory.csv`,
`correctness.csv`), and requires a final history sync to `/mnt/results`.
Version 2.3.0 adds a comprehensive repo-root README (how v2 differs from the
original skill, repository layout, results/history structure) and refreshes
the repository metadata; the optimization loop is unchanged. Version 2.4.0
makes the run budget purely iteration-based: the 24h wall-clock budget and
all wall-clock gating in `stop-check`/`auto_resume.py` are removed (the
Phase 5 timestamp stays as an informational elapsed-time marker).

Version 2.5.0 admits one candidate at a time and adds evidence-bound continuation, idempotent result recording, verified source/state snapshots, and truthful resume/stop gates. Acceptance rules and language hints are unchanged. See `references/interruption_recovery.md` for new commands and migration limits.

Version 2.5.1 requires sequential **hypothesis generation and learning**, not
just serialized execution: conceive one hypothesis, test and review it, then
formulate the next from that result. Advance candidate queues and multi-iteration
command/cell/job loops are prohibited. `begin-iteration` binds the latest feedback
identity and learning; `suggest-next` returns at most one measured focus.

Version 2.5.2 requires 50 substantive attempts by default (not merely a ceiling)
and replaces unconditional bitwise comparison with autonomous selection of the
most permissive defensible numerical tolerance. `set-tolerance` freezes its
evidence and rationale before optimization; stricter contracts still win.
Performance thresholds, sequential learning and recovery safeguards are preserved.
See `references/numerical_tolerance.md`.

Version 2.5.3 restores the ranked, diverse idea backlog and broader hotspot/family
suggestions. Reassess one selected idea against the latest reviewed result and
current best before admission; never execute a candidate queue or batch of
iterations. One coherent candidate may span multiple files. Derived status views
no longer invalidate checkpoints; explicitly declared rebuildable output files
are separated from source identity under a frozen baseline policy. Correctness,
isolated timing, per-candidate review, numerical policy and recovery gates remain.

## What is in the package

- `SKILL.md` — the skill instructions: Phases 0–6, correctness rule, protected
  behavior, acceptance rule, iteration-counting rule, hard blockers + STOP-GATE,
  adaptive direction prioritization, single-candidate execution with the
  conservative-timing carve-out, freeze-components discipline, benchmark-overfitting
  guard, long-run reliability.
- `scripts/` — executable machinery:
  - `biomni_code_state.py` — run state: init, set-baseline, set-tolerance, start-phase5,
    begin-iteration, add-iteration, add-feedback, add-hotspots, suggest-next,
    checkpoint, status, stop-check, record-blocker, clear-blocker, render-report,
    summary. The iteration counter is derived from an append-only ledger
    (`iterations/results.jsonl`), so it is self-healing.
    `add-backlog` maintains ranked ideas; `begin-iteration --backlog-id ID` selects
    one after reassessment. A backlog is never an executable batch.
  - `benchmark_runner.py` — N-rep wall-clock + peak-RSS + %CPU + SHA-256 output
    digest + determinism check.
  - `acceptance.py` — mechanical acceptance rule (runtime branch and
    memory-driven branch), exit 0 = accept, 3 = reject, prints a JSON verdict.
  - `numerical_policy.py` — private policy validation/binding helper; repository
    harnesses perform actual domain-specific output comparisons.
  - `scaling_suite.py` — Phase 6 sized-input comparison with a 15-minute cap.
  - `auto_resume.py` — cold-start resume: re-verify build + correctness, print
    the resume directive with remaining budget.
  - `plot_scaling_panels.py` — Phase 6 scaling figure (one 3-panel figure:
    runtime / peak RSS / digest-match fraction vs input size, original in gray,
    optimized in red) plus the per-size summary CSVs `runtime.csv`,
    `memory.csv`, `correctness.csv`.
  - `protected_hashes.py`, `create_manifest.py`, `safe_zip.py`,
    `verify_delivery.py`, `verify_skill_package.py` — integrity and packaging.
- `references/strategy_library/` — `optimization_families.json` (11 intervention families
  with diagnostic questions), `intervention_weights.json` (prior weights +
  exploration floors), `hotspot_taxonomy.json` (hotspot class → evidence → likely
  families).
- `references/` — deep-dive docs: state_and_memory, optimization_loop,
  parallelism_and_machines, benchmark_calibration, acceptance_and_correctness,
  final_delivery_contract, profiler_toolbox.
- `templates/` — optimization notebook, iteration record, benchmark card,
  candidate patch, feedback item, status, final report, reproduce.sh,
  run_manifest.json.
- `python/ r/ c/ c++/ rust/ go/ java/ julia/ javascript/ assembly/` — per-language
  optimization hints.

## Quick start

```bash
ROOT=/workspace/biomnicode/<repo>
STORE=/mnt/shared-workspace/biomnicode/<repo>/checkpoints
python scripts/biomni_code_state.py init --root $ROOT --repo-url <url> --commit-hash <sha> ...
# Follow references/interruption_recovery.md: record original approval, preserve original_snapshot,
# and declare protected files/recovery inputs (including any rebuildable outputs) before baseline.
python scripts/benchmark_runner.py --command "python -m target.cli --size 100000" \
    --repeat 5 --warmup 1 --out bench/baseline.json
python scripts/biomni_code_state.py set-baseline --root $ROOT --runtime-mean <s> \
    --runtime-std <s> --mem-mean <kb> --mem-std <kb> --output-digest <sha256> --determinism <actual_result>
# Reason from baseline/contracts and prepare policy evidence.
python scripts/biomni_code_state.py set-tolerance --root $ROOT --evidence-json <policy.json>
python scripts/biomni_code_state.py start-phase5 --root $ROOT
# Save + separately verify checkpoint.
# ... add-backlog -> suggest-next -> review latest evidence -> reassess ONE idea
# ... begin-iteration --backlog-id ID --feedback-id ID --feedback-sha256 SHA --learning TEXT
# ... patch -> correctness -> benchmark -> acceptance.py
# ... detailed review -> add-iteration --evidence-json FILE -> checkpoint --store $STORE
# ... run_checkpoint.py verify --root $ROOT --store $STORE --snapshot ID
python scripts/biomni_code_state.py stop-check --root $ROOT   # gate before Phase 6
python scripts/scaling_suite.py ...                           # Phase 6
python scripts/biomni_code_state.py render-report --root $ROOT
```

See `references/optimization_loop.md` for the full loop and
`references/final_delivery_contract.md` for the required Phase 6 artifacts.

## Package validation

From this package directory, run:

```bash
python scripts/verify_skill_package.py --skill-root . --compile
python -m unittest discover -s tests -v
```

The regression fixtures use synthetic authorization and measurements to test
backlog/admission, promotion, source identity and recovery gates. They do not
measure optimization quality on a real repository. Both package manifests list
the same files and exclude each other to avoid circular hashes. To refresh them:

```bash
python scripts/create_manifest.py --root . --out manifest.json \
  --exclude manifest.json --exclude references/manifest.json
cp manifest.json references/manifest.json
```
