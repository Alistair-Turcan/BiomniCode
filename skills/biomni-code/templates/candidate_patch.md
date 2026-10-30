# Candidate patch

- Iteration ID:
- Parent ID (the accepted state this builds on):
- Backlog ID (if selected from the ranked portfolio):
- Current-best/latest-feedback reassessment:
- Family / intervention:
- Hotspot targeted:
- Hypothesis:
- Patch file (unified diff against current best):
- Edited files (SHA-256 before/after):

## Scope check
- [ ] One coherent hypothesis (focused, reversible, evidence-driven; necessary multi-file/preparatory edits evaluated together)
- [ ] Function bodies / internals changed, not interfaces
- [ ] No public API, CLI flag, file-format, return-type, or seed-behavior change
- [ ] No protected files touched (protected_hashes verify passes)
- [ ] Frozen numerical policy identity is unchanged; comparisons use protected original
- [ ] No unapproved approximation, reduced precision, convergence change or unsafe compiler flags
- [ ] Any reassociation/FMA/reduction reordering satisfies the justified frozen bounds

## Expected effect
- Runtime mechanism:
- Memory mechanism:
- Risk to correctness:

## Minimal test
- Correctness cases that must pass:
- Benchmark size and reps:
