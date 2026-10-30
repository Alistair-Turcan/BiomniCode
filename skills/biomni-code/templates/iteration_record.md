# Iteration record

- Timestamp:
- Iteration:
- Iteration ID:
- Parent ID:
- Immediately preceding reviewed feedback ID / SHA-256:
- Backlog ID (if any):
- Learning and reassessment against current best / latest result that justified this hypothesis:
- Family:
- Intervention type:
- Hotspot (function/region, % runtime from profile):
- Hypothesis (expected mechanism + risk):
- Edited files (hashes):
- Patch file:
- Active candidate owner / tracked job IDs:
- Tested cumulative patch SHA-256:
- Frozen numerical policy SHA-256:
- Frozen source policy SHA-256:
- Benchmark command:
- Correctness command:

## Measurements
- Correctness: pass | fail
- Runtime mean +/- std (reps):
- Peak RSS mean +/- std (reps):
- Output digest (must equal baseline digest under bitwise rule):
- Tolerance-mode original comparison evidence, maximum absolute/relative/scaled errors and failures:
- Acceptance rule output (from scripts/acceptance.py):

## Metric analysis (REQUIRED)
Why did the runtime/memory numbers come out this way? Which mechanism plausibly
drove the result? Is it consistent with the hypothesis? Could benchmark noise,
size composition, or a measurement artifact explain it instead?

## Decision
- accept | reject | memory-driven | infrastructure
- Branch: runtime | memory | none
- Reason:
- Uncertainty and alternative explanations:
- Hash-linked correctness / benchmark / acceptance evidence:
- Verified disposition (accepted or reverted) and resulting tree identity:
- Checkpoint snapshot and separate verification receipt:

## Next action
What did this result teach us? Update backlog priorities or retire stale ideas.
After this review and checkpoint, reassess and select one next hypothesis against
current best and the latest feedback; do not execute a batch of queued candidates.
