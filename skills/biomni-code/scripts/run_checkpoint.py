#!/usr/bin/env python3
"""Immutable snapshots: local writers, content-only publication, separate readback."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import subprocess
import tarfile
from run_control import (canonical, digest, enroll, fresh_hashes, fsync_directory,
                         local_root, read, resolve, transaction, write)

IGNORE = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
STATE_IGNORE = {'.coordinator.lock', '.transaction.json', 'verified_checkpoint.json', 'pending_checkpoint.json'}
DERIVED_STATUS = {'reports/status.json', 'reports/status.md'}
SOURCE_SUFFIXES = {'.py', '.pyi', '.pyx', '.pxd', '.c', '.h', '.cc', '.cpp', '.cxx',
                   '.hpp', '.hh', '.hxx', '.cu', '.cuh', '.rs', '.r', '.m', '.mm',
                   '.f', '.for', '.f90', '.f95', '.java', '.scala', '.js', '.jsx',
                   '.ts', '.tsx', '.sh', '.bash', '.pl', '.jl', '.go', '.cs',
                   '.swift', '.kt', '.sql', '.cmake'}
SHA = re.compile('[0-9a-f]{64}')


def safe_link(root, rel, target):
    normalized = posixpath.normpath(str(Path(rel).parent/target))
    if Path(target).is_absolute():
        raise ValueError('Absolute checkpoint symlink')
    return resolve(root, normalized)


def tree_inventory(root, excluded_top=(), excluded_files=()):
    root = Path(root).resolve()
    result = {}
    paths = []
    for directory, subdirs, filenames in os.walk(root, followlinks=False):
        here = Path(directory)
        subdirs[:] = [name for name in subdirs if name not in IGNORE
                      and not (here == root and name in excluded_top)]
        paths.extend(here/name for name in subdirs if (here/name).is_symlink())
        paths.extend(here/name for name in filenames)
    for p in sorted(paths):
        rel = p.relative_to(root)
        if str(rel) in excluded_files:
            continue
        if any(x in IGNORE for x in rel.parts) or p.suffix in {'.pyc', '.tmp'}:
            continue
        if p.is_symlink():
            target = os.readlink(p)
            safe_link(root, str(rel), target)
            if not p.resolve().is_relative_to(root) or not p.exists():
                raise ValueError('Unsafe or broken symlink: '+str(rel))
            result[str(rel)] = {'link': target}
        elif p.is_file():
            if p.name.endswith(('-wal', '-journal', '-shm')):
                raise ValueError('Close or back up databases before checkpoint')
            result[str(rel)] = {'sha256': digest(p), 'bytes': p.stat().st_size,
                                'mode': 0o755 if p.stat().st_mode & 0o111 else 0o644}
    return result


def rebuildable_paths(state):
    """Run-relative disposable paths from the frozen policy, never from Git ignore rules."""
    return {folder+'/'+row['path'] for row in (state.get('source_policy') or {}).get('rebuildable_outputs', [])
            for folder in ('work', 'original_snapshot')}


def linked_evidence_paths(root, state=None):
    """Follow authoritative evidence links, including links nested in JSON artifacts."""
    from run_control import evidence, records
    pending = [state if state is not None else read(root/'state/state.json', {}),
               read(root/'state/control.json', {}), *records(root)]
    seen = set()
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if 'evidence' in value and 'sha256' in value:
                p = evidence(root, value)
                rel = str(p.relative_to(root))
                if rel not in seen:
                    seen.add(rel)
                    try:
                        raw = p.read_text()
                    except UnicodeError:
                        raw = ''  # Binary artifacts have no structured evidence links.
                    try:
                        pending.append(json.loads(raw))
                    except ValueError:
                        # JSON/JSONL receipts need not have a particular filename suffix.
                        for line in raw.splitlines():
                            if line.lstrip().startswith(('{', '[')):
                                try:
                                    pending.append(json.loads(line))
                                except ValueError:
                                    pass  # Raw logs may contain incomplete JSON.
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return seen


def _declarations(root):
    rows = read(root/'recovery_inputs.json', {}).get('rebuildable_outputs', [])
    if not isinstance(rows, list):
        raise ValueError('rebuildable_outputs must be a list of exact work-relative files')
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'path', 'rationale', 'rebuild_recipe'}:
            raise ValueError('Each rebuildable output requires path, rationale and rebuild_recipe')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError('Rebuildable output fields must be nonempty strings')
        rel = row['path']
        if str(Path(rel)) != rel or any(c in rel for c in '*?[]\\') or rel in {'.', '..'}:
            raise ValueError('Declare exact normalized work-relative output files, not patterns/directories')
        if Path(rel).suffix.lower() in SOURCE_SUFFIXES or Path(rel).name.lower() in {'makefile', 'dockerfile', 'cmakelists.txt'}:
            raise ValueError('Code/build source cannot be a rebuildable exclusion: '+rel)
        for folder in ('work', 'original_snapshot'):
            base = root/folder
            p = resolve(base, rel)
            if p.is_dir() or any((base/part).is_symlink() for part in (Path(rel), *Path(rel).parents)):
                raise ValueError('Rebuildable output must be an exact file with no symlink components')
        recipe = resolve(root, row['rebuild_recipe'])
        if not recipe.is_file() or not recipe.stat().st_size:
            raise ValueError('Require a nonempty rebuild recipe for '+rel)
        result.append(dict(row, recipe_sha256=digest(recipe)))
    if len({r['path'] for r in result}) != len(result):
        raise ValueError('Duplicate rebuildable output path')
    return sorted(result, key=lambda row: row['path'])


def _tracked_files(folder):
    p = subprocess.run(['git', '-C', str(folder), 'ls-files', '-z'],
                       capture_output=True, text=True, timeout=30, env={**os.environ, 'LC_ALL': 'C'})
    if p.returncode:
        if 'not a git repository' in p.stderr and not (folder/'.git').exists():
            return set()
        raise ValueError('Cannot establish tracked source identity: '+p.stderr)
    return set(p.stdout.split('\0'))-{''}


def _required_paths(root, state=None):
    required = set(read(root/'protected_files.json', {})) | linked_evidence_paths(root, state)
    spec = read(root/'recovery_inputs.json', {})
    required.update(spec.get(k, '') for k in ('environment_recipe', 'build_recipe', 'correctness_runner'))
    for item in spec.get('external_inputs', []):
        p = Path(item.get('recovery_location', '')).resolve()
        if p.is_relative_to(root.resolve()):
            required.add(str(p.relative_to(root.resolve())))
    return {str(resolve(root, rel).resolve().relative_to(root.resolve())) for rel in required if rel}


def _check_source_policy(root, state):
    policy = state.get('source_policy')
    if policy is None:
        if state.get('original_tree'):
            raise ValueError('Missing frozen source policy; explicit preserved-history migration required')
        return []
    if policy.get('schema') != 1 or canonical(policy) != state.get('source_policy_sha256'):
        raise ValueError('Frozen source policy identity changed')
    rows = _declarations(root)
    if rows != policy.get('rebuildable_outputs'):
        raise ValueError('Rebuildable declarations/recipes are frozen at baseline')
    if not rows:
        return rows
    required = _required_paths(root, state)
    required.update(str(resolve(root, row['rebuild_recipe']).resolve().relative_to(root.resolve())) for row in rows)
    excluded = rebuildable_paths(state)
    if any(a == b or a.startswith(b+'/') or b.startswith(a+'/')
           for a in excluded for b in required if b):
        raise ValueError('Rebuildable products cannot exclude protected files, required inputs/recipes or evidence')
    for folder in ('work', 'original_snapshot'):
        if {r['path'] for r in rows} & _tracked_files(root/folder):
            raise ValueError('Tracked source cannot be a rebuildable exclusion')
    if {r['path'] for r in rows} & set(state.get('original_tree', {})):
        raise ValueError('Original source cannot be a rebuildable exclusion')
    return rows


def freeze_source_policy(root, state):
    """Freeze optional exact output exclusions before source/baseline identity is recorded."""
    root = Path(root)
    if state.get('source_policy') is not None:
        _check_source_policy(root, state)
        return
    if state.get('original_tree') or state.get('active_candidate') or state.get('baseline', {}).get('runtime_mean') is not None:
        raise ValueError('Declare rebuildable outputs before recording the baseline')
    rows = _declarations(root)
    if any((root/'original_snapshot'/row['path']).exists() for row in rows):
        raise ValueError('Rebuildable outputs must be absent from the clean original_snapshot')
    state['source_policy'] = {'schema': 1, 'rebuildable_outputs': rows}
    state['source_policy_sha256'] = canonical(state['source_policy'])
    _check_source_policy(root, state)


def source_inventory(root, folder='work', state=None):
    root = Path(root)
    state = state if state is not None else read(root/'state/state.json', {})
    rows = _check_source_policy(root, state)
    folder = Path(folder)
    source = folder if folder.is_absolute() else root/folder
    return tree_inventory(source, excluded_files={row['path'] for row in rows})


def inventory(root):
    # Rebuildable environment and Git metadata are deliberately not durability claims.
    state = read(root/'state/state.json', {})
    _check_source_policy(root, state)
    return {p: info for p, info in tree_inventory(root, excluded_top={'env'},
            excluded_files=rebuildable_paths(state)).items()
            if not (
                Path(p).parent == Path('state') and Path(p).name in STATE_IGNORE)}


def revision_inventory(root, files):
    # Keep derived views in recoverable snapshots; only unlinked views may change freely.
    required = _required_paths(root)
    return {p: info for p, info in files.items() if p not in DERIVED_STATUS or p in required}


def source_archive(root):
    source = root/'original_snapshot'
    files = source_inventory(root, 'original_snapshot')
    if not files:
        raise ValueError('Protected original_snapshot source is required')
    recorded = read(root/'state/state.json', {}).get('original_tree')
    if recorded and files != recorded:
        raise ValueError('Original source changed since baseline')
    destination = root/'recovery/original_source.tar'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        # Reuse only after checking actual archive members/content against fresh source.
        try:
            with tarfile.open(destination, 'r') as tar:
                members = tar.getmembers()
                matching = len(members) == len(files) and {m.name for m in members} == set(files)
                for member in members:
                    row = files.get(member.name)
                    if not matching or row is None:
                        matching = False
                        break
                    if 'link' in row:
                        matching = member.issym() and member.linkname == row['link']
                    else:
                        matching = (member.isfile() and member.size == row['bytes']
                                    and member.mode == row['mode'])
                        if matching:
                            h = hashlib.sha256()
                            with tar.extractfile(member) as f:
                                for block in iter(lambda: f.read(1048576), b''):
                                    h.update(block)
                            matching = h.hexdigest() == row['sha256']
                    if not matching:
                        break
                if matching:
                    return destination
        except (tarfile.TarError, OSError, ValueError):
            pass  # Rebuild deterministically, then check it in the full inventory.
    enroll(destination)
    temporary = destination.with_name(destination.name+'.tmp')
    # Reproducible archive; no live Git metadata or environment mutations on S3.
    with tarfile.open(temporary, 'w') as tar:
        for rel, row in files.items():
            info = tarfile.TarInfo(rel)
            if 'link' in row:
                info.type, info.linkname = tarfile.SYMTYPE, row['link']
                tar.addfile(info)
            else:
                info.size, info.mode = row['bytes'], row['mode']
                with (source/rel).open('rb') as f:
                    tar.addfile(info, f)
    with temporary.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(temporary, destination)
    fsync_directory(destination.parent)
    return destination


def check_dependencies(root):
    spec = read(root/'recovery_inputs.json')
    if not isinstance(spec, dict) or not isinstance(spec.get('external_inputs'), list):
        raise ValueError('Declare recovery_inputs.json, including external_inputs (possibly empty)')
    for key in ('environment_recipe', 'build_recipe', 'correctness_runner'):
        p = resolve(root, spec.get(key, ''))
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError('Missing recovery dependency: '+key)
    for row in spec['external_inputs']:
        p = Path(row.get('recovery_location', ''))
        if not p.is_absolute() or not p.is_file() or digest(p) != row.get('sha256'):
            raise ValueError('External recovery input missing or changed')
    if not (root/'protected_files.json').is_file():
        raise ValueError('Missing protected-file manifest')
    return spec


def io_map(function, items):
    """Bound remote I/O concurrency. No background jobs or digest cache on disk."""
    workers = int(os.environ.get('BIOMNICODE_CHECKPOINT_IO_WORKERS', '8'))
    if not 1 <= workers <= 8:
        raise ValueError('BIOMNICODE_CHECKPOINT_IO_WORKERS must be between 1 and 8')
    if workers == 1:
        return list(map(function, items))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


def verify(store, snapshot):
    store = Path(store).resolve()
    path = resolve(store, snapshot)
    m = read(path)
    if (not isinstance(m, dict) or m.get('schema') != 1
            or not isinstance(m.get('files'), dict) or not m['files']):
        raise ValueError('Invalid snapshot manifest')
    if path.stem != canonical(m):
        raise ValueError('Manifest content hash mismatch')
    unique = {}
    for rel, row in m['files'].items():
        if not isinstance(row, dict):
            raise ValueError('Invalid snapshot file entry')
        resolve(Path('/workspace/checkpoint-path-check'), rel)
        if 'link' in row:
            safe_link(Path('/workspace/checkpoint-path-check'), rel, row['link'])
            continue
        if not SHA.fullmatch(row.get('sha256', '')):
            raise ValueError('Invalid blob identity')
        if row.get('mode') not in {0o644, 0o755} or type(row.get('bytes')) is not int or row['bytes'] < 0:
            raise ValueError('Invalid snapshot size or executable mode')
        sha = row['sha256']
        if sha in unique and unique[sha][1] != row['bytes']:
            raise ValueError('Conflicting sizes for shared blob')
        unique.setdefault(sha, (rel, row['bytes']))
    def verify_blob(item):
        sha, (rel, size) = item
        p = store/'blobs'/sha
        info = p.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size or digest(p) != sha:
            raise ValueError('Missing or corrupt blob: '+rel)
    # Each standalone verification always re-reads the unique content in this pass.
    with fresh_hashes():
        io_map(verify_blob, sorted(unique.items()))
    return m


def save(root, store):
    root, store = local_root(root), Path(store).resolve()
    if root == store or root.is_relative_to(store) or store.is_relative_to(root):
        raise ValueError('Source and checkpoint store must not overlap')
    check_dependencies(root)
    source_archive(root)
    files = inventory(root)
    unique = {}
    for rel, row in files.items():
        if 'link' in row:
            continue
        unique.setdefault(row['sha256'], (rel, row['bytes']))
    (store/'blobs').mkdir(parents=True, exist_ok=True)
    def publish(item):
        sha, (rel, size) = item
        p = store/'blobs'/sha
        if p.exists():
            if p.stat().st_size != size or digest(p) != sha:
                raise ValueError('Corrupt existing checkpoint blob')
        else:
            shutil.copyfile(root/rel, p)
            if p.stat().st_size != size or digest(p) != sha:
                raise ValueError('Checkpoint copy failed verification')
    io_map(publish, sorted(unique.items()))
    # Independent local second pass must not reuse hashes from the first inventory.
    with fresh_hashes():
        if inventory(root) != files:
            raise ValueError('Source changed while checkpointing')
    manifest = {'schema': 1, 'source_root': str(root), 'files': files}
    snapshot = 'snapshots/'+canonical(manifest)+'.json'
    p = store/snapshot
    p.parent.mkdir(parents=True, exist_ok=True)
    # Sequential object write; no remote rename.
    p.write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    receipt = {'store': str(store), 'snapshot': snapshot, 'verified': False}
    write(root/'state/pending_checkpoint.json', receipt)
    return receipt


def verify_for_root(root, store, snapshot):
    root = local_root(root)
    m = verify(store, snapshot)
    if revision_inventory(root, inventory(root)) != revision_inventory(root, m['files']):
        raise ValueError('Snapshot does not cover current revision')
    check_dependencies(root)
    receipt = {'store': str(Path(store).resolve()), 'snapshot': snapshot, 'verified': True}
    write(root/'state/verified_checkpoint.json', receipt)
    # An immutable readback receipt names a generation; no "latest" overwrite race.
    p = Path(store)/'verified'/Path(snapshot).name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def restore(store, snapshot, destination):
    destination = local_root(destination)
    m = verify(store, snapshot)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Restore only into an empty directory')
    for rel, row in m['files'].items():
        p = resolve(destination, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        if 'link' in row:
            p.symlink_to(row['link'])
        else:
            shutil.copyfile(Path(store)/'blobs'/row['sha256'], p)
            p.chmod(row['mode'])
    if inventory(destination) != m['files']:
        raise ValueError('Restored files failed verification')
    write(destination/'state/verified_checkpoint.json',
          {'store': str(Path(store).resolve()), 'snapshot': snapshot, 'verified': True})
    return {'restored': str(destination), 'verified': True,
            'note': 'Revalidate commands and path-bearing input records before resuming'}


def unpack_original(archive, destination):
    destination = local_root(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Original restore destination must be empty')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            p = resolve(destination, member.name)
            if not member.isfile() and not member.issym():
                raise ValueError('Unsupported source archive member')
            if member.issym():
                safe_link(destination, member.name, member.linkname)
        for member in members:
            p = resolve(destination, member.name)
            p.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                p.symlink_to(member.linkname)
            else:
                with tar.extractfile(member) as source, p.open('wb') as target:
                    shutil.copyfileobj(source, target)
                p.chmod(member.mode & 0o777)


def reconstruct_best(root, patch, destination):
    unpack_original(root/'recovery/original_source.tar', destination)
    if patch:
        p = subprocess.run(['git', 'apply', '--binary', str(resolve(root, patch))],
                           cwd=destination, capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise ValueError('Cumulative patch cannot reconstruct source: '+p.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['save', 'verify', 'restore'])
    p.add_argument('--root')
    p.add_argument('--store', required=True)
    p.add_argument('--snapshot')
    p.add_argument('--destination')
    a = p.parse_args()
    try:
        if a.command == 'restore':
            result = restore(a.store, a.snapshot, a.destination)
        elif a.command == 'save':
            with transaction(Path(a.root)):
                result = save(Path(a.root), Path(a.store))
        elif a.root:
            with transaction(Path(a.root)):
                result = verify_for_root(Path(a.root), Path(a.store), a.snapshot)
        else:
            m = verify(Path(a.store), a.snapshot)
            result = {'verified': True, 'files': len(m['files'])}
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, TypeError, KeyError) as e:
        print(json.dumps({'error': str(e), 'verified': False}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
