#!/usr/bin/env python3
"""Single-coordinator continuation gates, not a scheduler or authority authenticator."""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import shlex
import statistics
import stat
import subprocess
import sys
import tempfile
import uuid

DELAYS = (5, 15, 30, 60, 120)
SCRIPT = Path(__file__).resolve().parent
_ACTIVE_TRANSACTION = ContextVar('biomnicode_transaction', default=None)
_HASHES = ContextVar('biomnicode_hashes', default=None)


def digest(path):
    """Fresh per invocation, with within-invocation reuse guarded by file identity."""
    path = Path(path)
    if path.absolute().is_relative_to('/mnt'):
        # Object-store metadata may change during upload/cache promotion without
        # a content change. Never use it for digest reuse or content validity.
        # Every pass reads all bytes; the caller compares the actual SHA-256.
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    before = path.stat()
    signature = (before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns)
    cache = _HASHES.get()
    key = str(path.resolve())
    if cache is not None and key in cache and cache[key][0] == signature:
        return cache[key][1]
    h = hashlib.sha256()
    with path.open('rb') as f:
        opened = os.fstat(f.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
                opened.st_ctime_ns) != signature:
            raise ValueError('File changed before hashing: '+str(path))
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
        after = os.fstat(f.fileno())
    final = path.stat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns) != signature or (
            final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns,
            final.st_ctime_ns) != signature:
        raise ValueError('File changed while hashing: '+str(path))
    result = h.hexdigest()
    if cache is not None:
        cache[key] = (signature, result)
    return result


@contextmanager
def fresh_hashes():
    """Bound hash reuse to one invocation/pass, never a persisted metadata cache."""
    token = _HASHES.set({})
    try:
        yield
    finally:
        _HASHES.reset(token)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path):
    path = Path(path)
    missing = []
    parent = path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)


def enroll(path):
    """Persist undo BEFORE a mutation. All authoritative writers call this."""
    active = _ACTIVE_TRANSACTION.get()
    if active is not None:
        active.enroll(Path(path))


def _atomic_bytes(path, data, mode=None):
    path = Path(path)
    durable_mkdir(path.parent)
    if mode is None:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fchmod(f.fileno(), mode)
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_bytes(path, data):
    enroll(path)
    _atomic_bytes(path, data)


def remove(path):
    enroll(path)
    path = Path(path)
    path.unlink(missing_ok=True)
    fsync_directory(path.parent)


def local_root(root):
    root = Path(root).resolve()
    if root.is_relative_to('/mnt'):
        raise ValueError('Active state/locks require a local root; use /mnt only for checkpoint stores')
    return root


def resolve(root, relative):
    if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Expected a safe run-relative path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Unsafe evidence path')
    return path


def evidence(root, row):
    p = resolve(root, row.get('evidence', ''))
    if not p.is_file() or digest(p) != row.get('sha256'):
        raise ValueError('Missing or changed source evidence')
    from run_checkpoint import rebuildable_paths
    if str(p.resolve().relative_to(root.resolve())) in rebuildable_paths(read(root/'state/state.json', {})):
        raise ValueError('Declared rebuildable products cannot serve as required evidence')
    return p


def protected(root):
    manifest = read(root/'protected_files.json')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Require nonempty run-relative protected_files.json')
    for rel, sha in manifest.items():
        if digest(resolve(root, rel)) != sha:
            raise ValueError('Protected file changed: '+rel)


def records(root):
    p = root/'iterations/results.jsonl'
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    ids = [r['iteration_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate iteration IDs require explicit historical reconciliation')
    return rows


def count(root):
    return sum(r['decision'] in {'accept', 'reject', 'memory-driven'} and not r.get('non_substantive')
               for r in records(root))


def control(root):
    return read(root/'state/control.json', {'authority': [], 'retries': {}, 'jobs': {}})


def save_control(root, value):
    write(root/'state/control.json', value)


def approved(root):
    rows = control(root)['authority']
    for row in rows:
        evidence(root, row)
    return bool(rows) and rows[-1]['action'] in {'approve', 'resume'}


def unfinished_jobs(root):
    return {k: v for k, v in control(root).get('jobs', {}).items()
            if v['status'] in {'queued', 'running', 'unknown'}}


def state_files(root):
    return {str(p.relative_to(root)): base64.b64encode(p.read_bytes()).decode()
            for folder in ('state', 'iterations', 'feedback', 'profiling', 'logs', 'reports')
            for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink()
            and p.name not in {'.transaction.json', '.coordinator.lock'}
            and p.suffix != '.tmp'}


def rollback(root, previous):
    # Legacy journals are recovered without reading every current evidence file.
    current = {str(p.relative_to(root))
               for folder in ('state', 'iterations', 'feedback', 'profiling', 'logs', 'reports')
               for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink()
               and p.name not in {'.transaction.json', '.coordinator.lock'} and p.suffix != '.tmp'}
    for rel in current-set(previous):
        (root/rel).unlink()
    for rel, data in previous.items():
        p = resolve(root, rel)
        _atomic_bytes(p, base64.b64decode(data, validate=True))


class MutationJournal:
    """Schema 2: explicit write set, durable enrollment and idempotent undo."""
    def __init__(self, root, readonly=False):
        self.root = root
        self.path = root/'state/.transaction.json'
        self.before = {}
        self.readonly = readonly

    def enroll(self, path):
        if self.readonly:
            raise ValueError('Read-only command attempted a mutation')
        absolute = path.absolute()
        if not absolute.is_relative_to(self.root):
            raise ValueError('Authoritative mutation is outside the run root')
        rel = str(absolute.relative_to(self.root))
        path = resolve(self.root, rel)
        rel = str(path.resolve().relative_to(self.root))
        if rel in {'state/.transaction.json', 'state/.coordinator.lock'}:
            raise ValueError('Reserved coordinator file cannot be a mutation target')
        if rel in self.before:
            return
        if path.exists():
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Journal supports only regular mutable files')
            row = {'exists': True, 'mode': stat.S_IMODE(info.st_mode),
                   'data': base64.b64encode(path.read_bytes()).decode()}
        else:
            row = {'exists': False}
        self.before[rel] = row
        # Raw writer avoids enrolling the undo journal itself.
        _atomic_bytes(self.path, (json.dumps({'schema': 2, 'before': self.before},
                                           sort_keys=True, allow_nan=False)+'\n').encode())

    def undo(self):
        for rel, row in reversed(list(self.before.items())):
            p = resolve(self.root, rel)
            if row.get('exists') is True:
                mode = row.get('mode')
                if not isinstance(mode, int) or not 0 <= mode <= 0o7777:
                    raise ValueError('Invalid undo mode')
                _atomic_bytes(p, base64.b64decode(row['data'], validate=True), mode)
            elif row == {'exists': False}:
                p.unlink(missing_ok=True)
                if p.parent.exists():
                    fsync_directory(p.parent)
            else:
                raise ValueError('Invalid undo entry')

    def finish(self):
        if self.path.exists():
            self.path.unlink()
            fsync_directory(self.path.parent)


@contextmanager
def transaction(root, readonly=False):
    """Rollback an interrupted local mutation; external jobs are never repeated here."""
    root = local_root(root)
    durable_mkdir(root/'state')
    with (root/'state/.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _ACTIVE_TRANSACTION.get() is not None:
            raise ValueError('Nested coordinator transactions are unsupported')
        journal = MutationJournal(root, readonly=readonly)
        if journal.path.exists():
            pending = read(journal.path)
            if pending.get('schema') == 2:
                journal.before = pending['before']
                journal.undo()
            elif 'schema' not in pending and isinstance(pending.get('before'), dict):
                rollback(root, pending['before'])
            else:
                raise ValueError('Unsupported interrupted transaction journal')
            journal.finish()
            journal.before = {}
        token = _ACTIVE_TRANSACTION.set(journal)
        with fresh_hashes():
            try:
                yield journal
            except BaseException:
                # If undo fails, retain the journal so recovery can retry.
                journal.undo()
                journal.finish()
                raise
            else:
                journal.finish()
            finally:
                _ACTIVE_TRANSACTION.reset(token)


def checkpoint_current(root, full=True):
    from run_checkpoint import verify
    receipt = read(root/'state/verified_checkpoint.json')
    if not receipt:
        raise ValueError('Save and separately verify a checkpoint first')
    m = verify(Path(receipt['store']), receipt['snapshot'])
    # Binding includes all source/evidence bytes, not only the cached counter.
    from run_checkpoint import inventory, revision_inventory
    current = revision_inventory(root, inventory(root))
    saved = revision_inventory(root, m['files'])
    compared = current.keys() | saved.keys() if full else {
        'state/state.json', 'state/control.json', 'iterations/results.jsonl'}
    if any(current.get(p) != saved.get(p) for p in compared):
        raise ValueError('Checkpoint is stale; save and verify the current run revision')
    return receipt


def audit_history(root):
    """Recheck linked committed evidence without treating hashes as authenticity."""
    source_policy = read(root/'state/state.json', {}).get('source_policy_sha256')
    for row in records(root):
        if row.get('source_policy_sha256') != source_policy:
            raise ValueError('Historical source policy mismatch')
        review_row = row.get('review')
        if not review_row:
            raise ValueError('Legacy/missing review requires explicit historical reconciliation')
        for item in review_row.get('artifacts', []):
            evidence(root, item)
        for key in ('correctness', 'benchmark', 'acceptance', 'next_size_confirmation', 'tolerance_evidence'):
            if key in review_row:
                evidence(root, review_row[key])
        if digest(resolve(root, review_row['tested_patch'])) != review_row['tested_patch_sha256']:
            raise ValueError('Historical candidate patch changed')


def feedback_context(root):
    """Bind the immediately preceding reviewed result, not just the winning parent."""
    history = records(root)
    if history:
        last = history[-1]
        if not last.get('review'):
            raise ValueError('Latest result has not been reviewed')
        return {'feedback_id': last['iteration_id'], 'feedback_sha256': canonical(last)}
    state = read(root/'state/state.json', {})
    baseline = state.get('baseline', {})
    if baseline.get('runtime_mean') is None or not state.get('original_tree'):
        raise ValueError('Record original source and measured baseline before proposing')
    return {'feedback_id': 'baseline', 'feedback_sha256': canonical({
        'baseline': baseline, 'original_tree': state['original_tree']})}


def begin(root, args):
    s = read(root/'state/state.json', {})
    c = control(root)
    if s.get('policy_version') != '2.5.3':
        raise ValueError('Legacy run requires explicit preserved-history migration')
    if not approved(root) or c.get('recovery') or s.get('blocker'):
        raise ValueError('Resolve authority, recovery or blocker before admission')
    if s.get('active_candidate') or unfinished_jobs(root):
        raise ValueError('Only one active candidate; reconcile all earlier work')
    if count(root) >= s['budget_iterations']:
        raise ValueError('Budget exhausted; finish Phase 6')
    from numerical_policy import check
    check(root, s)
    if any(r['iteration_id'] == args.iteration_id for r in records(root)):
        raise ValueError('Iteration identity already used')
    parent = s.get('best_iteration_id')
    if args.parent_id != parent:
        raise ValueError('Stale or missing current-best parent')
    if not args.hypothesis.strip():
        raise ValueError('One concrete hypothesis is required')
    feedback = feedback_context(root)
    if any(getattr(args, key, None) != value for key, value in feedback.items()):
        raise ValueError('Missing or stale feedback: use the immediately preceding reviewed result')
    learning = getattr(args, 'learning', '')
    if not isinstance(learning, str) or not learning.strip():
        raise ValueError('Explain learning from the latest result before forming this hypothesis')
    checkpoint_current(root)
    audit_history(root)
    protected(root)
    from run_checkpoint import source_inventory
    if source_inventory(root) != s.get('best_tree'):
        raise ValueError('Working tree differs from current best before admission')
    s['active_candidate'] = {'id': args.iteration_id, 'parent_id': parent,
                            'numerical_policy_sha256': s['numerical_policy_sha256'],
                            'source_policy_sha256': s['source_policy_sha256'],
                            'base': s['current_best'], 'hypothesis': args.hypothesis,
                            **feedback, 'learning': learning,
                            'owner_id': socket.gethostname(), 'stage': 'edit'}
    write(root/'state/state.json', s)
    return s['active_candidate']


def failed_scaling_rejection(root, row, patch_sha256):
    """Replay a stricter next-size rejection; never alter the primary verdict."""
    proof = read(evidence(root, row['failed_scaling_evidence']))
    confirmation = read(evidence(root, proof['confirmation']))
    if (proof.get('tested_patch_sha256') != patch_sha256
            or confirmation.get('tested_patch_sha256') != patch_sha256
            or confirmation.get('confirmed') is not False):
        raise ValueError('Failed scaling evidence must bind the tested patch and failed confirmation')
    sizes = read(evidence(root, proof['size_definition']))
    size = confirmation.get('size')
    if (type(size) is not int or size <= sizes.get('primary_size', math.inf)
            or size not in sizes.get('sizes', [])):
        raise ValueError('Failed scaling must use a declared larger size')
    linked = confirmation.get('artifacts', [])
    raw = []
    for role in ('parent_benchmark', 'candidate_benchmark'):
        ref = proof[role]
        if ref not in linked:
            raise ValueError('Scaling measurements must be linked by the confirmation')
        measured = read(evidence(root, ref))
        results = measured.get('results', [])
        if (len(results) < 5 or measured.get('repeat') != len(results)
                or measured.get('returncodes') != [0]*len(results)
                or measured.get('deterministic') is not True):
            raise ValueError('Failed scaling requires five successful deterministic repetitions')
        tokens = shlex.split(measured.get('command', ''))
        if '--size' not in tokens or tokens[tokens.index('--size')+1] != str(size):
            raise ValueError('Scaling measurement command does not bind the larger size')
        for index, result in enumerate(results):
            if (result.get('returncode') != 0 or result.get('repeat_index') != index
                    or result.get('command') != measured['command']
                    or result.get('output_digest') != measured.get('output_digest')):
                raise ValueError('Inconsistent scaling repetition')
        for key, sample_key in (('runtime', 'runtime_sec'), ('maxrss_kb', 'maxrss_kb')):
            values = [r.get(sample_key) for r in results]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('Scaling measurements must be finite positive values')
            summary = measured.get(key, {})
            if (summary.get('n') != len(values)
                    or not math.isclose(summary.get('mean', math.nan), statistics.fmean(values),
                                        rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(summary.get('std', math.nan), statistics.stdev(values),
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('Scaling summary differs from raw repetitions')
        raw.append(measured)
    parent, candidate = raw
    if (confirmation.get('parent_mean') != parent['runtime']['mean']
            or confirmation.get('candidate_mean') != candidate['runtime']['mean']):
        raise ValueError('Scaling confirmation means do not match measured evidence')
    values = {
        'best-runtime-mean': parent['runtime']['mean'], 'best-runtime-std': parent['runtime']['std'],
        'new-runtime-mean': candidate['runtime']['mean'], 'new-runtime-std': candidate['runtime']['std'],
        'best-mem-mean': parent['maxrss_kb']['mean'], 'best-mem-std': parent['maxrss_kb']['std'],
        'new-mem-mean': candidate['maxrss_kb']['mean'], 'new-mem-std': candidate['maxrss_kb']['std'],
        'baseline-mem-std': parent['maxrss_kb']['std']}
    cmd = [sys.executable, str(SCRIPT/'acceptance.py')]
    for key, value in values.items():
        cmd.extend(['--'+key, str(value)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode not in (0, 3):
        raise ValueError('Scaling acceptance replay failed')
    verdict = json.loads(proc.stdout)
    if proof['acceptance'] not in linked:
        raise ValueError('Scaling acceptance must be linked by the confirmation')
    if verdict != read(evidence(root, proof['acceptance'])) or verdict['decision'] != 'reject':
        raise ValueError('Scaling evidence does not establish an unchanged-rule rejection')
    return verdict


def review(root, args, state):
    from numerical_policy import check, validate_comparison
    policy = check(root, state)
    active = state.get('active_candidate')
    if not active or active['id'] != args.iteration_id or args.parent_id != active['parent_id']:
        raise ValueError('Completion must match the sole active candidate and parent')
    if active.get('source_policy_sha256') != state.get('source_policy_sha256'):
        raise ValueError('Candidate source policy mismatch')
    if args.hypothesis and args.hypothesis != active['hypothesis']:
        raise ValueError('Completion hypothesis differs from the admitted single optimization')
    if not approved(root) or control(root).get('recovery') or unfinished_jobs(root):
        raise ValueError('Reconcile authority and jobs before completion')
    if active['owner_id'] != socket.gethostname():
        raise ValueError('Owner changed; enter recovery and reconcile takeover')
    row = read(args.evidence_json)
    if not isinstance(row, dict) or row.get('iteration_id') != args.iteration_id:
        raise ValueError('Require matching --evidence-json review')
    for name in ('mechanism', 'uncertainty', 'disposition'):
        if not str(row.get(name, '')).strip():
            raise ValueError('Missing detailed review: '+name)
    expected = 'accepted' if args.decision in {'accept', 'memory-driven'} else 'reverted'
    if row['disposition'] != expected:
        raise ValueError('Review must establish accept/revert disposition')
    for item in row.get('artifacts', []):
        evidence(root, item)
    if not row.get('artifacts'):
        raise ValueError('Review requires actual hash-linked evidence')
    protected(root)
    from run_checkpoint import source_inventory
    actual = source_inventory(root, state=state)
    if actual != row.get('disposed_tree'):
        raise ValueError('Working tree does not match verified disposition')
    patch = resolve(root, row.get('tested_patch', ''))
    if digest(patch) != row.get('tested_patch_sha256'):
        raise ValueError('Tested patch changed')
    if expected == 'reverted':
        if actual != state.get('best_tree'):
            raise ValueError('Rejected/infrastructure candidate not reverted to current best')
    if args.decision != 'infrastructure':
        if not args.metric_analysis.strip() or not args.next_action.strip():
            raise ValueError('Substantive review needs analysis and concrete next action')
        corr = read(evidence(root, row['correctness']))
        if corr.get('correctness') != args.correctness or corr.get('tested_patch_sha256') != digest(patch):
            raise ValueError('Correctness evidence does not bind tested patch/result')
        if args.correctness == 'not-run':
            raise ValueError('Unexecuted correctness is infrastructure, not substantive')
        if args.correctness == 'fail' and args.decision != 'reject':
            raise ValueError('Correctness failure must be rejected')
        if args.correctness == 'pass':
            if state.get('fp_tolerance', 'bitwise') == 'bitwise':
                if (not args.output_digest or args.output_digest != state['baseline'].get('output_digest')
                        or corr.get('output_digest') != args.output_digest):
                    raise ValueError('Bitwise correctness requires the bound original output digest')
            else:
                tolerance = read(evidence(root, row['tolerance_evidence']))
                validate_comparison(root, tolerance, policy, state, digest(patch))
            measured = read(evidence(root, row['benchmark']))
            for key in ('runtime_mean', 'runtime_std', 'runtime_reps', 'mem_mean', 'mem_std', 'mem_reps'):
                value = getattr(args, key)
                if value is None or not math.isfinite(value) or value < 0 or measured.get(key) != value:
                    raise ValueError('Missing, nonfinite or mismatched measurement: '+key)
            if min(args.runtime_reps, args.mem_reps) < 3 or args.runtime_mean <= 0:
                raise ValueError('Require at least three valid repetitions')
            if measured.get('tested_patch_sha256') != digest(patch):
                raise ValueError('Measurements do not bind candidate patch')
            best = state['current_best']
            cmd = [sys.executable, str(SCRIPT/'acceptance.py')]
            values = {'best-runtime-mean': best['runtime_mean'], 'best-runtime-std': best['runtime_std'],
                      'new-runtime-mean': args.runtime_mean, 'new-runtime-std': args.runtime_std,
                      'best-mem-mean': best['mem_mean'], 'best-mem-std': best['mem_std'],
                      'new-mem-mean': args.mem_mean, 'new-mem-std': args.mem_std,
                      'baseline-mem-std': state['baseline']['mem_std']}
            for k, v in values.items():
                if v is None:
                    raise ValueError('Complete baseline/current-best measurements required')
                cmd.extend(['--'+k, str(v)])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode not in (0, 3):
                raise ValueError('Acceptance helper failed: '+proc.stderr)
            verdict = json.loads(proc.stdout)
            supplied = read(evidence(root, row['acceptance']))
            if verdict != supplied:
                raise ValueError('Acceptance evidence differs from unchanged rule')
            expected_decision = ('memory-driven' if verdict['memory_driven'] else verdict['decision'])
            scaling_reject = (args.decision == 'reject' and expected_decision in {'accept', 'memory-driven'}
                              and args.branch == verdict['branch'] and 'failed_scaling_evidence' in row)
            if scaling_reject:
                failed_scaling_rejection(root, row, digest(patch))
            if (args.decision != expected_decision and not scaling_reject) or args.branch != verdict['branch']:
                raise ValueError('Decision does not match mechanical acceptance')
            if expected == 'accepted':
                if args.branch == 'runtime':
                    delta = best['runtime_mean']-args.runtime_mean
                    margin = max(2*best['runtime_std'], .03*best['runtime_mean'])
                else:
                    delta = best['mem_mean']-args.mem_mean
                    margin = max(2*best['mem_std'], .25*best['mem_mean'])
                if delta < 2*margin:
                    confirmation = read(evidence(root, row['next_size_confirmation']))
                    if (min(args.runtime_reps, args.mem_reps) < 5
                            or confirmation.get('confirmed') is not True
                            or confirmation.get('tested_patch_sha256') != digest(patch)):
                        raise ValueError('Close acceptance requires five reps and next-size confirmation')
    if expected == 'accepted':
        if actual == state.get('best_tree'):
            raise ValueError('Accepted optimization must change authoritative source/build configuration, not only rebuildable products')
        if args.patch_file != row['tested_patch']:
            raise ValueError('Accepted cumulative patch must be the tested patch')
        from run_checkpoint import reconstruct_best
        if not patch.stat().st_size:
            raise ValueError('An accepted optimization requires a real cumulative patch')
        with tempfile.TemporaryDirectory(prefix='biomnicode-patch-check-') as tmp:
            restored = Path(tmp)/'best'
            reconstruct_best(root, args.patch_file, restored)
            if source_inventory(root, restored, state=state) != actual:
                raise ValueError('Cumulative patch does not reconstruct accepted working tree')
        state['best_tree'] = actual
        state['best_iteration_id'] = args.iteration_id
    state['active_candidate'] = None
    return row


def stop_verdict(root):
    s = read(root/'state/state.json', {})
    if not s:
        return 2, {'may_finalize': False, 'reason': 'missing_state'}
    n = count(root)
    c = control(root)
    stopped = bool(c['authority']) and c['authority'][-1]['action'] == 'stop'
    blocked = s.get('blocker')
    if blocked:
        evidence(root, blocked)
    ready = (approved(root) and not c.get('recovery') and not s.get('active_candidate')
             and not unfinished_jobs(root) and not blocked)
    exhausted = s['budget_iterations'] > 0 and n >= s['budget_iterations']
    if exhausted and ready:
        try:
            checkpoint_current(root, full=False)
            audit_history(root)
        except ValueError:
            ready = False
    code = 4 if stopped or blocked else (0 if exhausted and ready else 3)
    return code, {'may_finalize': code == 0, 'iteration': n, 'budget_iterations': s['budget_iterations'],
                  'remaining_to_floor': max(0, s['budget_iterations']-n),
                  'counter_matches_ledger': n == s.get('iteration'), 'floor_reached': exhausted,
                  'self_audit_line': f"STOP-GATE: iteration={n}/{s['budget_iterations']}; may_finalize={'yes' if code == 0 else 'no'}"}


def record(root, row):
    evidence(root, row)
    c = control(root)
    if row.get('action') not in {'approve', 'stop', 'resume'} or row.get('scope') != 'run':
        raise ValueError('Require run-level approve/stop/resume')
    if row.get('source_kind') not in {'user_message', 'host_run_control'}:
        raise ValueError('Agent prose/tool errors are not run authority')
    if not row.get('source_ref') or not row.get('instruction'):
        raise ValueError('Missing original authority source')
    t = datetime.fromisoformat(row['timestamp'])
    if t.tzinfo is None:
        raise ValueError('Authority timestamp must include timezone')
    previous = c['authority']
    if previous and row == previous[-1]:
        return {'replayed': True}
    if previous and t <= datetime.fromisoformat(previous[-1]['timestamp']):
        raise ValueError('Authority ordering conflict')
    if previous and previous[-1]['action'] == 'stop' and row['action'] != 'resume':
        raise ValueError('Explicit stop requires later explicit resume')
    c['authority'].append(row)
    c.pop('yield', None)
    save_control(root, c)
    return {'recorded': row['action']}


def reconcile(root, row):
    evidence(root, row)
    c = control(root)
    if not approved(root):
        raise ValueError('Recover original approval or honor explicit stop')
    supplied = row.get('jobs', {})
    if set(supplied) != set(c['jobs']):
        raise ValueError('Account for every registered job')
    for key, value in supplied.items():
        evidence(root, value)
        if value.get('status') not in {'queued', 'running', 'unknown', 'completed', 'failed', 'cancelled'}:
            raise ValueError('Invalid observed job status')
    c['jobs'] = supplied
    c.pop('recovery', None)
    c.pop('yield', None)
    c['reconciliation'] = row
    save_control(root, c)
    s = read(root/'state/state.json', {})
    active = s.get('active_candidate')
    if active:
        if row.get('iteration_id') != active['id'] or row.get('stage') not in {'edit', 'correctness', 'benchmark', 'review'}:
            raise ValueError('Reconcile the actual active candidate/stage')
        if row.get('owner_id') != socket.gethostname():
            raise ValueError('Reconciliation must name current coordinator host')
        if any(v['status'] in {'running', 'queued', 'unknown'} for v in supplied.values()):
            raise ValueError('Wait for actual job outcome before taking over candidate')
        active.update(stage=row['stage'], owner_id=row['owner_id'])
        write(root/'state/state.json', s)
    return {'reconciled': True}


def retry(root, row):
    evidence(root, row)
    if not approved(root):
        raise ValueError('No active run approval')
    if (row.get('operation_kind') not in {'read', 'local_write'} or row.get('status') != 'failed'
            or row.get('transient') is not True or row.get('retry_safe') is not True
            or row.get('retry_forbidden') is not False or not row.get('reconciliation')
            or row.get('error_kind') == 'gpu_concurrency_limit'):
        raise ValueError('Reconcile first; unsafe or prohibited retry')
    op, fingerprint = row.get('operation_id'), row.get('input_fingerprint')
    if not op or not fingerprint:
        raise ValueError('Stable operation and input identity required')
    c = control(root)
    history = c['retries'].setdefault(op, {'input_fingerprint': fingerprint, 'attempts': []})
    if history['input_fingerprint'] != fingerprint:
        raise ValueError('Changed inputs require a distinct operation')
    attempts = history['attempts']
    if len(attempts) >= 5:
        raise ValueError('Retry budget exhausted; change recovery route')
    if attempts and attempts[-1].get('outcome', {}).get('status') != 'failed':
        raise ValueError('Reconcile previous reserved retry as failed before reserving another')
    delay = row.get('retry_after_seconds', 0)
    if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
        raise ValueError('Invalid server retry interval')
    result = {'reservation': len(attempts)+1, 'delay_seconds': max(DELAYS[len(attempts)], delay),
              'observation': row}
    attempts.append(result)
    save_control(root, c)
    return result


def yield_run(root, row):
    evidence(root, row)
    c = control(root)
    reason = row.get('reason')
    if reason == 'complete':
        if stop_verdict(root)[0] != 0:
            raise ValueError('Budget/recovery gate not complete')
        receipt = read(evidence(root, row['delivery']))
        if receipt.get('run_id') != read(root/'state/state.json')['run_id'] or receipt.get('verified') is not True:
            raise ValueError('Require run-bound final validation receipt')
        for item in receipt.get('artifacts', []):
            evidence(root, item)
        if not receipt.get('artifacts') or not all(receipt.get(k) is True for k in
                ('correctness_pass', 'protected_pass', 'scaling_complete', 'delivery_check_pass')):
            raise ValueError('Full final validation remains incomplete')
        delivery = resolve(root, receipt.get('delivery_dir', ''))
        checked = subprocess.run([sys.executable, str(SCRIPT/'verify_delivery.py'),
                                  '--delivery', str(delivery)], capture_output=True, text=True, timeout=60)
        if checked.returncode:
            raise ValueError('Delivery artifacts failed verification: '+checked.stdout)
    elif reason == 'user_stop':
        if not c['authority'] or c['authority'][-1]['action'] != 'stop':
            raise ValueError('No sourced run stop')
        evidence(root, c['authority'][-1])
    elif reason in {'waiting_job', 'platform_unavailable', 'authorization_block'}:
        if row.get('ready_work') != [] or not row.get('affected_work') or not row.get('next_action') or row.get('alternatives_exhausted') is not True:
            raise ValueError('Continue ready work or account for all blocked work')
        if reason == 'waiting_job':
            job = c['jobs'].get(row.get('job_id'), {})
            evidence(root, job)
            if job.get('status') not in {'queued', 'running'} or job.get('callback_registered') is not True:
                raise ValueError('No actual tracked callback to await')
        elif reason == 'platform_unavailable':
            if row.get('source_kind') != 'host_status':
                raise ValueError('Require actual host-status evidence')
            exhausted = row.get('operation_ids') and all(
                len(c['retries'].get(i, {}).get('attempts', [])) == 5
                and c['retries'][i]['attempts'][-1].get('outcome', {}).get('status') == 'failed'
                for i in row['operation_ids'])
            if not exhausted and row.get('retry_prohibited') is not True:
                raise ValueError('Exhaust relevant safe retries or bind host prohibition')
        elif row.get('source_kind') not in {'user_message', 'host_authorization'}:
            raise ValueError('Require original authorization evidence')
    else:
        raise ValueError('Discretionary early yield is forbidden')
    c['yield'] = row
    save_control(root, c)
    return {'may_yield': True, 'reason': reason}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['record', 'enter', 'reconcile', 'retry', 'retry-result', 'job', 'yield'])
    p.add_argument('--root', required=True)
    p.add_argument('--record-json')
    p.add_argument('--purpose', choices=['resume', 'callback'])
    a = p.parse_args()
    root = local_root(a.root)
    try:
        with transaction(root):
            row = read(a.record_json) if a.record_json else None
            if a.command == 'enter':
                if not a.purpose:
                    raise ValueError('Require recovery purpose')
                c = control(root)
                c['recovery'] = {'purpose': a.purpose, 'id': uuid.uuid4().hex}
                c.pop('yield', None)
                save_control(root, c)
                result = c['recovery']
            elif a.command in {'job', 'retry-result'}:
                evidence(root, row)
                c = control(root)
                if a.command == 'job':
                    if not row.get('job_id') or row.get('status') not in {'queued', 'running', 'unknown', 'completed', 'failed', 'cancelled'}:
                        raise ValueError('Require actual job identity and status')
                    active = read(root/'state/state.json').get('active_candidate')
                    if active and row.get('iteration_id') != active['id']:
                        raise ValueError('Job must belong to current candidate')
                    c['jobs'][row['job_id']] = row
                else:
                    r = c['retries'][row['operation_id']]['attempts'][row['reservation']-1]
                    if 'outcome' in r and r['outcome'] != row:
                        raise ValueError('Conflicting retry outcome')
                    if row.get('status') not in {'failed', 'completed', 'unknown'}:
                        raise ValueError('Invalid actual retry outcome')
                    r['outcome'] = row
                save_control(root, c)
                result = {'recorded': True}
            else:
                result = {'record': record, 'reconcile': reconcile, 'retry': retry, 'yield': yield_run}[a.command](root, row)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(json.dumps({'error': str(e), 'may_yield': False}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
