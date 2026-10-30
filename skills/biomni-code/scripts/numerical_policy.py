#!/usr/bin/env python3
"""Frozen, evidence-linked numerical policy. Repository harnesses do comparisons."""
from pathlib import Path
import math
import re
import run_control as rc


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Require nonempty '+name)


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Require finite nonnegative '+name)


def validate(root, policy, state):
    if not isinstance(policy, dict) or policy.get('mode') not in {'bitwise', 'tolerance'}:
        raise ValueError('Require bitwise or tolerance policy mode')
    for key in ('setting', 'rationale', 'semantic_limits'):
        text(policy.get(key), key)
    request = state.get('fp_tolerance_request', 'auto')
    basis = 'automatic' if request == 'auto' else 'explicit'
    if policy.get('selection_basis') != basis:
        raise ValueError('Selection basis differs from original request')
    if request != 'auto' and policy['setting'] != request:
        raise ValueError('Explicit numerical setting cannot be silently overridden')
    if policy['setting'] == 'auto' or (policy['mode'] == 'bitwise') != (policy['setting'] == 'bitwise'):
        raise ValueError('Unresolved or inconsistent numerical setting')
    outputs = policy.get('outputs')
    if not isinstance(outputs, list) or not outputs:
        raise ValueError('Declare nonempty output scope')
    names = []
    for out in outputs:
        if not isinstance(out, dict):
            raise ValueError('Invalid output specification')
        text(out.get('name'), 'output name')
        text(out.get('rationale'), 'output rationale')
        names.append(out['name'])
        if out.get('comparison') not in {'exact', 'tolerance'}:
            raise ValueError('Output comparison must be exact or tolerance')
        if out['comparison'] == 'tolerance':
            if policy['mode'] != 'tolerance':
                raise ValueError('Exact policy cannot contain tolerant outputs')
            for key in ('rtol', 'atol'):
                nonnegative(out.get(key), key)
        elif any(out.get(key, 0) != 0 for key in ('rtol', 'atol')):
            raise ValueError('Exact outputs cannot have nonzero tolerances')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate output scope')
    if policy['mode'] == 'tolerance' and not any(o['comparison'] == 'tolerance' for o in outputs):
        raise ValueError('Tolerance mode requires a numerical output')
    if request not in {'auto', 'bitwise'}:
        # Preserve explicit rtol/atol requests, not just their descriptive label.
        parts = re.split(r'[\s,;]+', request.strip())
        bounds = {}
        for part in parts:
            key, sep, value = part.partition('=')
            if not sep or key not in {'rtol', 'atol'} or key in bounds:
                raise ValueError('Explicit setting requires rtol=N,atol=N (either may be omitted)')
            bounds[key] = float(value)
            nonnegative(bounds[key], key)
        for out in outputs:
            if out['comparison'] == 'tolerance' and any(out[k] != v for k, v in bounds.items()):
                raise ValueError('Policy bounds differ from explicit numerical request')
    if not isinstance(policy.get('artifacts'), list) or not policy['artifacts']:
        raise ValueError('Require baseline/contract/calibration evidence')
    for artifact in policy['artifacts']:
        rc.evidence(root, artifact)
    if (state.get('baseline', {}).get('runtime_mean') is None or not state.get('original_tree')
            or policy.get('baseline_sha256') != rc.canonical(state['baseline'])
            or policy.get('original_tree_sha256') != rc.canonical(state['original_tree'])):
        raise ValueError('Numerical policy must bind the recorded baseline and protected original')
    return policy


def resolve(root, evidence_path):
    state = rc.read(root/'state/state.json', {})
    if state.get('policy_version') != '2.5.3':
        raise ValueError('Legacy policy requires explicit migration')
    if not rc.approved(root) or rc.control(root).get('recovery') or state.get('blocker'):
        raise ValueError('Resolve authorization/recovery before numerical policy selection')
    path = Path(evidence_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Numerical evidence must be inside the run root')
    link = {'evidence': str(path.relative_to(root.resolve())), 'sha256': rc.digest(path)}
    policy = validate(root, rc.read(rc.evidence(root, link)), state)
    identity = rc.canonical(policy)
    if state.get('numerical_policy'):
        current = check(root, state)
        if current != policy:
            raise ValueError('Numerical policy is frozen; use an explicitly authorized new run')
        return {'replayed': True, 'numerical_policy_sha256': identity}
    if state.get('active_candidate') or rc.records(root) or state.get('phase5_start_ts') or rc.unfinished_jobs(root):
        raise ValueError('Freeze numerical policy before Phase 5 or candidate admission')
    state['numerical_policy'] = link
    state['numerical_policy_sha256'] = identity
    state['fp_tolerance'] = policy['setting']
    rc.write(root/'state/state.json', state)
    return {'resolved': True, 'setting': policy['setting'], 'numerical_policy_sha256': identity}


def check(root, state=None):
    state = state if state is not None else rc.read(root/'state/state.json', {})
    if not state.get('numerical_policy'):
        raise ValueError('Numerical policy is pending; run set-tolerance before optimization')
    policy = validate(root, rc.read(rc.evidence(root, state['numerical_policy'])), state)
    identity = rc.canonical(policy)
    if identity != state.get('numerical_policy_sha256') or state.get('fp_tolerance') != policy['setting']:
        raise ValueError('Frozen numerical policy identity changed')
    from run_checkpoint import source_inventory
    if source_inventory(root, 'original_snapshot', state=state) != state['original_tree']:
        raise ValueError('Numerical comparison original source changed')
    active = state.get('active_candidate')
    if active and active.get('numerical_policy_sha256') != identity:
        raise ValueError('Candidate numerical policy mismatch')
    for record in rc.records(root):
        if record.get('numerical_policy_sha256') != identity:
            raise ValueError('Historical numerical policy mismatch')
    return policy


def validate_comparison(root, comparison, policy, state, patch_sha):
    """Verify evidence schema/binding; cannot authenticate the harness's claims."""
    if not isinstance(comparison, dict):
        raise ValueError('Missing numerical comparison evidence')
    for key, expected in (
        ('setting', policy['setting']), ('pass', True),
        ('numerical_policy_sha256', state['numerical_policy_sha256']),
        ('tested_patch_sha256', patch_sha),
        ('original_tree_sha256', rc.canonical(state['original_tree']))):
        if comparison.get(key) != expected:
            raise ValueError('Numerical comparison mismatch: '+key)
    if comparison.get('pass') is not True:
        raise ValueError('Numerical comparison must pass')
    if not comparison.get('artifacts'):
        raise ValueError('Require actual protected-original comparison artifacts')
    for item in comparison['artifacts']:
        rc.evidence(root, item)
    rows = comparison.get('outputs')
    if not isinstance(rows, list) or len(rows) != len(policy['outputs']):
        raise ValueError('Comparison must cover every declared output')
    observed = {row['name']: row for row in rows}
    if len(observed) != len(rows) or set(observed) != {o['name'] for o in policy['outputs']}:
        raise ValueError('Comparison output scope mismatch')
    for spec in policy['outputs']:
        row = observed[spec['name']]
        if row.get('structure_exact') is not True or row.get('nonfinite_match') is not True:
            raise ValueError('Output structure/nonfinite behavior changed')
        n = row.get('compared_count')
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError('Require nonnegative compared count')
        if type(row.get('failure_count')) is not int or row['failure_count'] != 0:
            raise ValueError('Output comparison has failures')
        if spec['comparison'] == 'exact':
            if row.get('exact_match') is not True:
                raise ValueError('Exact output mismatch')
        else:
            for key in ('rtol', 'atol'):
                if row.get(key) != spec[key]:
                    raise ValueError('Comparison widened or changed '+key)
            for key in ('max_abs_error', 'max_scaled_error'):
                nonnegative(row.get(key), key)
            if row['max_scaled_error'] > 1:
                raise ValueError('Numerical error exceeds frozen tolerance')
            if row.get('max_rel_error') is None:
                text(row.get('relative_error_note'), 'undefined relative error explanation')
            else:
                nonnegative(row['max_rel_error'], 'max_rel_error')
    return comparison


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
