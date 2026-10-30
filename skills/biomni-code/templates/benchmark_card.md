# Benchmark card (Scaling Suite)

## Target
- Repository / commit:
- Entry point under optimization:
- Realistic workload description:

## Scaling axis
- Input dimension that drives cost (rows, reads, cells, samples, batch, problem size):
- Why this axis:

## Sizes
| Name | Size value | Purpose | Reps | Expected runtime |
|---|---|---|---|---|
| per-iteration | (smallest) | Phase 5 gate; <60s; computation >= 50% of runtime | >=5 | |
| mid-1 | | log-spaced intermediate | >=3 | |
| mid-2 | | log-spaced intermediate | >=3 | |
| largest | | closest realistic workload fitting the 15-min Phase 6 cap | >=3 (>=5 for protected-original comparison) | |

## Calibration
- Baseline per-iteration mean +/- std (>=5 reps):
- Benchmark sigma (std used by the acceptance rule):
- Startup/import share at per-iteration size (must be < 50%):
- Cold-start vs warm-start behavior:
- Noise controls (thread pinning, warm-up, isolated machine):

## Determinism
- Output digest (SHA-256) of canonicalized output:
- Reruns produce identical output: yes | no (if no, record max_abs_diff/max_rel_diff and surface before proceeding)

## Protected files
- Manifest path (from scripts/protected_hashes.py create):
- Verified after every iteration and in Phase 6: yes

## Versioning
- Suite version / hash:
- Created:
- Last updated:
