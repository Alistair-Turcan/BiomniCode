# Parallelism, machines, and frozen components

One optimization candidate is active at a time. Finish correctness, measurement,
detailed review, accept/revert, and verified checkpoint before admitting another.
Parallel work may support that candidate or research ideas without implementing
them; protect benchmark timing.

## The conservative-timing rule

Benchmark timing is the acceptance signal. Contention destroys it. Therefore:

- **Timing-critical benchmark reps run on ONE dedicated machine** (the benchmark
  machine), one rep at a time, with nothing else heavy running on it. The σ measured
  in Phase 3 is only valid under the same conditions used in Phase 5.
- Builds, Correctness Suite shards and non-timing diagnostics may run concurrently
  for the same candidate. Independent read-only research may compare approaches
  and feed the ranked backlog, but must not implement, build or benchmark another
  candidate, prepare speculative executable worktrees, or contend with timing.

## Level 1 — session machines

Use a suitable existing worker; add resources only when measured requirements and
authorization justify them. A free worker may inspect evidence or research backlog
ideas; it must not dispatch another candidate. No multi-iteration command/cell/job loops,
even if sequential. One coordinator writes local state under
`biomni_code_state.py`'s lock; workers publish distinct result files for reconciliation.
The lock is local, not a distributed lease. A takeover must reconcile the old jobs
and owner through `run_control.py`, not pretend a second local copy is independent.

## Level 2 — GPU sandboxes

Only when the repo has CUDA signals (torch.cuda, cupy, JAX-GPU, nvcc, *.cu/*.cuh,
CUDA flags in build files, GPU deps in requirements): provision one A10G GPU
sandbox. Mirror all durable state to the run root (the only path visible to both).
Verify durable checkpoints before releasing machines. Never auto-provision a GPU
for CPU-only repos or auto-retry a concurrent GPU-limit rejection.

## Level 3 — inside each machine

Use within-candidate parallelism where behavior is preserved: process-level
parallelism (multiprocessing, joblib, concurrent.futures), threaded/BLAS libraries,
memory-mapped IO, chunked streaming. Profile the first candidate of each family
with `time`, `psutil`, and (if GPU) `nvidia-smi`; record CPU%, RAM peak, VRAM peak
in the iteration record. Treat consistent under-utilization (< ~70% CPU while a
compute candidate runs) as a bug to fix, not a normal state — except on the
benchmark machine, where the benchmark itself is the only workload.

## Freeze components to accelerate iterations

Iteration speed is a first-class resource. Compute once, cache, and reuse:

- Built artifacts (compiled binaries, wheels, release builds) — rebuild only when
  source under the build graph changes.
- Parsed inputs, generated scaling-suite data, warmed caches.
- Baseline benchmark results and the protected-original snapshot.

Cache under `/workspace/` for hot access and under the run root for cross-machine
reuse, keyed by a content hash of the inputs plus the code that produced them, so
stale caches are detected automatically. Mark frozen components in the optimization
notebook. When a candidate genuinely modifies a frozen stage, fork a new cache key
rather than invalidating the shared one. If an iteration is dominated by re-running
unchanged work, freeze more aggressively.

## Memory maximization discipline

When a candidate OOMs, diagnose streaming/chunking opportunities that preserve
identical behavior, or right-size within authorization. Do not reduce precision,
workload size, or validation strength as an unapproved recovery shortcut.
