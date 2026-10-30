# Acceptance rule and correctness contract

## Correctness contract

Preserve exact public/non-numerical behavior and scientific meaning. For eligible
floating-point outputs, choose the most permissive defensible tolerance from
original-baseline evidence, then freeze it before optimization with `set-tolerance`.
See `numerical_tolerance.md` for selection criteria and the evidence contract.

- Default `auto` means reason and record, not use arbitrary `allclose` defaults.
  Existing stricter tests/contracts win; use exact comparison when no nonzero
  bound is defensible. Never loosen frozen bounds to rescue an optimization.
  Reassociation, FMA and reduction reordering may satisfy the frozen policy;
  reduced precision, unsafe compiler flags and approximation are not authorized
  merely by selecting tolerance.
- If Phase 1 shows the ORIGINAL is already non-deterministic, record the observed
  `max_abs_diff` / `max_rel_diff` and surface it before any tolerance-based
  acceptance.
- Preserve return types, shapes, schemas, ordering, dtypes, classes, file formats,
  and random-seed behavior unless a documented public contract permits variation.

The Correctness Suite (target < 60 s) covers: small edge cases, representative
normal cases, a benchmark-like case (may reuse the per-iteration input), empty or
degenerate inputs, dtype/class/shape/schema preservation, order preservation, file
output comparison, and seeded tests for stochastic code. One genuinely slow case
may be isolated as a deep correctness check that runs after every accepted change
and in Phase 6. Run the full suite after EVERY scoped change, before benchmarking.

## Acceptance rule (implemented by scripts/acceptance.py)

Always-applicable rules: correctness passes; outputs identical (or equivalent only
under the justified frozen tolerance); public APIs, CLI flags, file formats, return types,
and seed behavior unchanged; no protected files modified; the change is
maintainable, in-scope, and plausibly benefits realistic workloads.

**Runtime branch (primary)** — accept when ALL hold:
1. `(best_runtime_mean - new_runtime_mean) > 2 * best_runtime_std`
2. relative runtime improvement > 3%
3. peak-RSS regression <= 10% of best beyond `2 * baseline_memory_std`

**Memory branch (secondary)** — accept as MEMORY-DRIVEN when ALL hold:
1. `(best_mem_mean - new_mem_mean) > 2 * best_mem_std`
2. relative peak-RSS reduction >= 25%
3. runtime regression <= 5% of best beyond `2 * best_runtime_std`

**Reject** when: outputs change unexpectedly; public behavior changes; crash;
invalid output; slowdown beyond the memory-branch allowance; substantial complexity
for a tiny gain; benchmark-only improvement; dependence on protected evaluator
details; memory regression beyond the runtime-branch guardrail; introduced
nondeterminism; weakened scientific meaning.

Ties or near-ties: prefer the simpler, safer, more maintainable implementation.

## Usage

```bash
python scripts/acceptance.py \
  --best-runtime-mean 10.0 --best-runtime-std 0.2 \
  --new-runtime-mean 9.0 --new-runtime-std 0.2 \
  --best-mem-mean 500 --best-mem-std 5 \
  --new-mem-mean 500 --new-mem-std 5 \
  --baseline-mem-std 5
# exit 0 = accept, exit 3 = reject; prints a JSON verdict with branch and reasons
```

Record the verdict verbatim in the iteration record (`--decision`, `--branch`),
then record the iteration with `biomni_code_state.py add-iteration`.
