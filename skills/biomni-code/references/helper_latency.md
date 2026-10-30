# Helper I/O, bounded transactions and compatible upgrades

The 2.5.4 helpers preserve run policy `2.5.3`, scientific acceptance, public
commands and output schemas. The package revision changes bookkeeping and I/O;
it does not reset a run, reopen numerical bounds or authorize parallel candidates.

## Bounded authoritative transactions

Every command takes the same nonblocking local coordinator lock. Interrupted
undo state is recovered before running even a read-only command. A schema-2
journal stores only the explicit mutable write set: previous existence, bytes
and mode are durably enrolled **before** mutation. Local JSON, JSONL, CSV and
status writes are atomic replacements with file/directory synchronization.
Repeated writes to one path retain the original undo value. Commit removes the
journal only after successful writes; failed undo retains it for another recovery
attempt. Legacy full-directory journals remain recoverable.

`summary`, `suggest-next` and `stop-check` create no new journal. They still apply
their normal authorization, history, source or checkpoint gates as applicable.
Do not substitute a summary for an admission or integrity gate.

Checkpoint blobs/manifests are content-addressed external publications rather
than authoritative mutable ledger files. Partial external copies may remain
after a failed save, but no verified local receipt is issued. Restore and source
reconstruction write into new destinations, not the active coordinator state.
An explicitly external `render-report --out` is a derived output and, as before,
is not rolled back with the ledger. Required evidence must be created and linked
before its checkpoint; never treat an incomplete report as completed validation.

Development of new authoritative writers must use `run_control.write`,
`atomic_bytes` or `remove`, or explicitly call `enroll` before a raw mutation.
Directory creation alone is not a committed record. Never mutate a ledger with
an unenrolled raw file write.

## Complete integrity checks without duplicate reads

Each checkpoint reference still undergoes path, link, hash-identity, size and
mode validation. Shared content is hashed once per unique blob per pass.
Existing blobs are fully read once during save; new copies receive a separate
post-copy readback. Save retains an independent second local inventory to catch
source mutation. Independent verification always performs a fresh full content
pass and checks that the snapshot covers the current authoritative revision.

Hash reuse is confined to one command/pass and guarded by local file identity
and modification/change metadata; nothing is persisted as a trusted digest
cache. Content verification is never replaced by timestamp comparisons across
commands. The original-source archive is reused only after comparing actual
archive members, modes, links and contents against a freshly checked source
inventory.

Object-store files never use metadata-based digest reuse: every verification
streams the actual bytes and compares the complete SHA-256. Upload/cache promotion
can change their metadata during readback without changing content, so local
filesystem metadata-stability assumptions are not applied to object-store blobs.
Size/path/reference checks and full post-copy/independent hash verification remain.

`BIOMNICODE_CHECKPOINT_IO_WORKERS` selects **1–8** threads, default **8**, for
independent checkpoint blob reads/copies. This is I/O concurrency, not parallel
optimization or timing. Set it to `1` for sequential diagnosis or storage that
does not benefit from concurrency. Invalid values fail clearly; no retry loop
or background service is introduced. Full integrity checks still scale with
unique evidence bytes and storage throughput, so no fixed latency is promised.

## History mirror synchronization

Ordinary mutations copy only enrolled history files that changed; unchanged
content is not rewritten. Read-only commands do not touch the mirror. A
checkpoint command checks all history files; after independent verification,
synchronize again to include the new verified receipt:

```bash
python scripts/biomni_code_state.py mirror-history --root ROOT
# Force complete object replacement when needed:
python scripts/biomni_code_state.py mirror-history --root ROOT --full
```

An ordinary mirror outage warns without undoing an authoritative commit.
Inspect the committed ledger before replaying a state command. Explicit
`mirror-history` fails on an outage, so its success is distinguishable from a
warning. Full synchronization content-checks or rewrites every history object;
it is not an admission, a checkpoint verification or an iteration commit.

## Upgrade and rollback

Preserve the installed helper package and independently verified run checkpoint
before replacement, including any active-candidate reservation. Reconcile
authority/jobs and check ledger count, active identity, source, protected files
and policy hashes after installation. Save and separately verify the preserved
run with the new helpers. A 2.5.3 run requires no policy/history migration merely
to use these compatible helpers.

Resolve any interrupted schema-2 journal with the new helpers **before**
downgrading: older code does not understand its undo entries. Never overwrite an
unresolved transaction or restore a pre-admission generation over an active
reservation without explicit reconciliation.

## Preserved stricter failed-scaling rejection

A review may include `failed_scaling_evidence: {evidence, sha256}` when required
larger-size testing rejects an otherwise primary-accepted candidate. The linked
proof has `tested_patch_sha256` and evidence references named `confirmation`,
`size_definition`, `parent_benchmark`, `candidate_benchmark` and `acceptance`.
The confirmation names the declared larger `size`, has `confirmed=false`, binds
the tested patch, links both benchmark records and the scaling acceptance
record, and provides `parent_mean` and `candidate_mean`.

Both benchmark records must contain at least five successful deterministic raw
repetitions, consistent commands including the declared `--size`, digests and
reconstructed runtime/RSS means and sample standard deviations. The helper
reruns the unchanged `acceptance.py` from those measurements and requires the
exact linked rejection. The original primary verdict is preserved. This proof
only permits truthful rejection after failed required scaling; it never relaxes
primary correctness or performance gates.
