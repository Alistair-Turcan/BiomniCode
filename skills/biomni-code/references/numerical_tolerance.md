# Reasoned numerical equivalence

## Selection before optimization

Default `init --fp-tolerance auto` delegates numerical judgment to the agent,
not to a hard-coded epsilon or permissive library default. It leaves the policy
pending. Keep 50 substantive optimization attempts as the required default;
original characterization and tolerance calibration do not consume those attempts.

Before editing implementation code:

1. Inspect the original outputs and existing contracts/tests. Identify floating
   outputs eligible for tolerance and outputs that require exact preservation.
2. Measure original repeatability across representative, edge, near-zero, large
   and small magnitude cases, using relevant dtypes and units. Preserve the
   actual evidence and observed nondeterminism; never report a false digest pass.
3. Consider conditioning and downstream sensitivity: thresholds, ranks, sign,
   discrete decisions, physical constraints and scientific interpretation. Do not
   infer allowable scientific error solely from observed numerical noise.
4. Choose the **most permissive defensible** per-output bounds supported by the
   contracts, measured numerical behavior and intended use. Explain why these
   bounds are lenient enough to admit harmless roundoff and why looser bounds
   risk meaningful change. Never choose bounds by inspecting candidate errors.
5. Preserve stricter supplied tests/contracts. If no nonzero bound is defensible,
   use exact mode and document the limitation. A real ambiguity in scientific
   meaning requires clarification, not invented justification.
6. Record and freeze the policy before Phase 5:
   `biomni_code_state.py set-tolerance --root ROOT --evidence-json FILE`.
   The file and its linked evidence must be inside ROOT using safe relative paths.
   Save and separately verify the resulting checkpoint before candidate admission.

Explicit `--fp-tolerance bitwise` is respected. Explicit numerical requests use
`rtol=N,atol=N` (comma, semicolon or whitespace separators; either bound may be
omitted). Supplied bounds must match the policy; omitted bounds require explicit
reasoning in the policy. The old free-text setting remains a request string, but
unparseable numerical requests now fail clearly rather than silently implying
unverified bounds. Auto mode may use a descriptive setting label; structured
per-output bounds, not that label, are authoritative.

## Comparison semantics

For finite numerical values:

`abs(candidate - original) <= atol + rtol * abs(original)`

The original is always the protected original implementation, including for
cumulative patches. Use per-output bounds where scales/units differ. Relative
error alone is insufficient near zero. Preserve exact integer/string/byte/object
values, shapes, dtypes, ordering, schemas and seed behavior. NaN locations and
infinity signs must match; new nonfinite values fail. Preserve signed zero when
it affects behavior. Test downstream discrete decisions explicitly where relevant.
For no finite values, report a zero finite count and nonfinite checks truthfully.

Report maximum absolute error, maximum relative error where defined, maximum
scaled error `abs_error / (atol + rtol * abs(original))`, comparison count and
failure count, not merely averages. Zero denominator with zero error has scaled
error zero; positive error with zero allowance fails. Never clip failures or
replace nonfinite diagnostics with invented finite numbers. If relative error is
undefined because of zero references, use null plus an explanation.

Repository-specific harnesses perform actual comparisons. The helper verifies
schema, evidence linkage and frozen bounds; it does not parse arbitrary scientific
outputs or authenticate what a harness claims. Domain invariants and downstream
decisions remain mandatory even when pointwise tolerance passes.

Reassociation, FMA fusion and reordered reductions may be eligible within the
policy. Tolerance does not authorize reduced precision, approximate algorithms,
relaxed convergence, broad unsafe compiler flags, changed statistical meaning,
or new stochastic semantics. Never alter protected tests or expected values.

## Policy evidence schema

The policy JSON contains:

- `mode`: `bitwise` or `tolerance`; `setting`: `bitwise` for exact mode, or a
  nonempty numerical setting label (never `auto`). For explicit requests the
  setting must match the original request.
- `selection_basis`: `automatic` for auto requests, otherwise `explicit`.
- `rationale`, `semantic_limits`: nonempty explanations of calibration, limits,
  intended use, stricter contracts and why more lenient bounds are inappropriate.
- `baseline_sha256`: `run_control.canonical(state["baseline"])`.
- `original_tree_sha256`: `run_control.canonical(state["original_tree"])`.
- `outputs`: nonempty list of uniquely named `{name, comparison, rationale}`.
  `comparison` is `exact` or `tolerance`; the latter additionally requires finite
  nonnegative numeric `rtol` and `atol`. Exact outputs have no nonzero bounds.
  Tolerance mode includes at least one tolerant output.
- `artifacts`: nonempty list of `{evidence, sha256}` references to actual
  baseline/contract/calibration evidence. One consolidated source document may
  cover these, but it must contain the actual evidence, not invented measurements.

State records `fp_tolerance_request`, the resolved `fp_tolerance`,
`numerical_policy` (source reference), and `numerical_policy_sha256` (canonical
policy identity). Candidate admission and committed records carry that identity.
Identical policy replays are no-ops; conflicting replacement fails. The recorded
baseline cannot be reset after policy freeze. Recovery preserves the same policy.
Do not rewrite history or fabricate missing evidence to migrate an old run.

## Per-candidate tolerance-pass evidence

The review's `tolerance_evidence` references comparison JSON containing:

- `setting`, `numerical_policy_sha256`, `tested_patch_sha256`,
  `original_tree_sha256`, and `pass: true`, matching the policy, actual tested
  cumulative patch and protected original.
- `artifacts`: hash-linked actual original-versus-candidate comparison logs,
  output/input identities and diagnostics. Preserve enough to reproduce checks.
- `outputs`: exactly the declared output scope. Every entry has `name`,
  `structure_exact: true`, `nonfinite_match: true`, integer `compared_count >= 0`,
  and integer `failure_count: 0`.
- Exact entries additionally have `exact_match: true`.
- Tolerant entries carry unchanged `rtol`, `atol`, finite nonnegative
  `max_abs_error` and `max_scaled_error <= 1`, plus `max_rel_error` (finite
  nonnegative or null with nonempty `relative_error_note`).

These fields describe successful comparisons; failed comparisons are recorded as
correctness failure and rejection, with actual diagnostics. No benchmark needs
inventing for a correctness failure. Exact mode retains original-digest gates.

Never widen a tolerance to accept a candidate. A policy change requires explicit
authorization and a new run/revalidation; it cannot overwrite an ongoing run.

## Final validation and reporting

Apply the same policy at every final Scaling Suite size and the largest protected-
original comparison. Include the policy identity, rationale, per-output bounds,
actual maxima and failure counts in the final report. The existing digest-match
panel remains an exact-match diagnostic, not a numerical-equivalence verdict.
Do not change performance acceptance thresholds or relabel that panel.
