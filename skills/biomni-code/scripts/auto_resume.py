#!/usr/bin/env python3
"""Verified recovery, never silent restart. 0=resume, 2=genuinely fresh, 4=recover/block."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tempfile
import shutil
import os
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_control as control
from run_checkpoint import restore, source_inventory, reconstruct_best


def run_cmd(cmd: str, cwd: str | None, timeout: float) -> dict:
    start = time.time()
    # Synchronous process-group supervision; not detached background execution.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout.seek(0); stderr.seek(0)
            return {"cmd": cmd, "returncode": code, "elapsed_sec": round(time.time()-start, 2),
                    "stdout_tail": stdout.read().decode(errors='replace')[-2000:],
                    "stderr_tail": stderr.read().decode(errors='replace')[-2000:]}
        except OSError as e:
            return {"cmd": cmd, "returncode": 127, "error": str(e)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--repo-url", default="")
    p.add_argument("--commit-hash", default="")
    p.add_argument("--build-cmd", default="", help="Must match recorded command; no silent harness override")
    p.add_argument("--correctness-cmd", default="", help="Must match recorded command")
    p.add_argument("--work-dir", default="", help="Must be <root>/work")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--store", required=True)
    p.add_argument("--snapshot", help="Explicit verified generation for empty-root recovery")
    p.add_argument("--reconciliation-json", help="Actual job/owner observations for run_control reconcile")
    args = p.parse_args()

    root = control.local_root(args.root)
    store = Path(args.store)
    try:
        if not (root/'state/state.json').exists():
            if args.snapshot:
                restore(store, args.snapshot, root)
            elif (root.exists() and any(root.iterdir())) or (store.exists() and any(store.iterdir())):
                raise ValueError('Prior run evidence exists; select and verify its checkpoint, never restart')
            else:
                print(json.dumps({'action': 'fresh', 'reason': 'no prior run or checkpoint evidence'}))
                raise SystemExit(2)
        # Resolve any interrupted local state command before inspecting state.
        with control.transaction(root):
            state = control.read(root/'state/state.json')
            if state.get('policy_version') != '2.5.3':
                raise ValueError('Legacy run needs explicit preserved-history migration; stay on its old skill meanwhile')
            from numerical_policy import check as check_numerical_policy
            if state.get('numerical_policy') or state.get('active_candidate') or control.records(root):
                check_numerical_policy(root, state)
            for arg, key in ((args.repo_url, 'repo_url'), (args.commit_hash, 'commit_hash'),
                             (args.build_cmd, 'build_cmd'), (args.correctness_cmd, 'correctness_cmd')):
                if arg and arg != state.get(key):
                    raise ValueError('Recorded identity/command mismatch: '+key)
            if args.work_dir and Path(args.work_dir).resolve() != root/'work':
                raise ValueError('Different work directory requires explicit path reconciliation')
            if not control.approved(root):
                raise ValueError('Honor sourced stop or recover original run approval')
            receipt = control.read(root/'state/verified_checkpoint.json')
            if not receipt:
                raise ValueError('No verified checkpoint available')
            from run_checkpoint import verify, check_dependencies
            manifest = verify(Path(receipt['store']), receipt['snapshot'])
            # Local work newer than the last checkpoint is retained and must be accounted for.
            required = ('original_snapshot/', 'recovery/original_source.tar')
            for rel, info in manifest['files'].items():
                if rel.startswith(required) and 'sha256' in info:
                    if not (root/rel).is_file() or control.digest(root/rel) != info['sha256']:
                        raise ValueError('Original recovery source differs from verified checkpoint')
            check_dependencies(root)
            c = control.control(root)
            c['recovery'] = {'purpose': 'resume', 'id': str(time.time_ns())}
            control.save_control(root, c)
        # Barrier remains set on a failed check; no transaction rollback of this fact.
        if not args.reconciliation_json:
            raise ValueError('Provide actual job/owner reconciliation before resuming')
        if not state.get('correctness_cmd'):
            raise ValueError('Successful correctness check is mandatory for resume')
        if args.timeout <= 0 or not __import__('math').isfinite(args.timeout):
            raise ValueError('Require finite positive recovery timeout')
        observation = control.read(args.reconciliation_json)
        known = control.control(root)['jobs']
        if set(observation.get('jobs', {})) != set(known):
            raise ValueError('Reconcile every registered job before recovery execution')
        for job in observation['jobs'].values():
            control.evidence(root, job)
            if job.get('status') not in {'completed', 'failed', 'cancelled'}:
                raise ValueError('Await/reconcile outstanding jobs before recovery execution')
        # Check current protected paths without rewriting a dirty/partial candidate.
        if (root/'work').exists():
            control.protected(root)
        control.audit_history(root)
        if not state.get('active_candidate') and (root/'work').exists():
            if source_inventory(root, 'work') != state.get('best_tree'):
                raise ValueError('Unknown dirty work tree; preserve and reconcile rather than overwrite')
        with tempfile.TemporaryDirectory(prefix='biomnicode-restore-') as tmp:
            best_dir = Path(tmp)/'best'
            patch = state['current_best'].get('patch')
            reconstruct_best(root, patch, best_dir)
            if source_inventory(root, best_dir) != state.get('best_tree'):
                raise ValueError('Reconstructed best tree differs; include new files and executable modes in cumulative patch')
            # Use cwd-based commands; absolute paths require explicit migration/revalidation.
            for key in ('build_cmd', 'correctness_cmd'):
                if state.get(key):
                    result = run_cmd(state[key], str(best_dir), args.timeout)
                    if result['returncode'] != 0:
                        raise ValueError(key+' failed: '+json.dumps(result))
            # Never replace an existing work tree. Restored snapshots normally include it.
            if not (root/'work').exists():
                shutil.copytree(best_dir, root/'work', symlinks=True)
        control.protected(root)
        if state.get('numerical_policy'):
            check_numerical_policy(root)
        with control.transaction(root):
            control.reconcile(root, observation)
        n = control.count(root)
        state = control.read(root/'state/state.json')
        active = state.get('active_candidate')
        directive = {'action': 'resume', 'iteration': n,
                     'remaining_to_floor': max(0, state['budget_iterations']-n),
                     'phase5_start_ts': state.get('phase5_start_ts'),
                     'next_action': ('resume_candidate_'+active['stage'] if active else
                                     'phase6' if n >= state['budget_iterations'] else
                                     'set_tolerance' if not state.get('numerical_policy') else 'begin_iteration'),
                     'candidate_id': active['id'] if active else None,
                     'note': 'Save and separately verify the reconciled revision before the next admission.'}
        print(json.dumps(directive, indent=2))
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as e:
        print(json.dumps({'action': 'recover_or_block', 'error': str(e), 'restart_allowed': False}))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
