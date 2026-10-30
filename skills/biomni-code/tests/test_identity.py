#!/usr/bin/env python3
"""Focused source/checkpoint regression fixtures; no real authorization or measurements."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--skill-root', type=Path,
                    default=Path(__file__).resolve().parents[1])
options, remaining = parser.parse_known_args()
sys.path.insert(0, str(options.skill_root/'scripts'))
import biomni_code_state as state_tool
import run_checkpoint as cp
import run_control as rc


def put(root, rel, text):
    path = root/rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def link(root, path):
    return {'evidence': str(path.relative_to(root)), 'sha256': rc.digest(path)}


def cli(root, command, *arguments):
    args = state_tool.build_parser().parse_args([command, '--root', str(root), *map(str, arguments)])
    output = io.StringIO()
    with contextlib.redirect_stdout(output), rc.transaction(root):
        args.func(args)
    return json.loads(output.getvalue())


def declaration(path='build/solver.o'):
    return {'path': path, 'rationale': 'Synthetic disposable object; never source/input/evidence',
            'rebuild_recipe': 'recipes/rebuild_outputs.sh'}


def prepare(root, rows=None):
    cli(root, 'init', '--run-id', 'SYNTHETIC-IDENTITY-TEST', '--budget-iterations', '3')
    for folder in ('work', 'original_snapshot'):
        put(root, folder+'/solver.py', 'def solve():\n    return 42\n')
        put(root, folder+'/contract.txt', 'Synthetic protected exact contract.\n')
    evidence = put(root, 'evidence/test.txt', 'SYNTHETIC LOCAL TEST ONLY; no real approval or benchmark.\n')
    rc.record(root, {**link(root, evidence), 'action': 'approve', 'scope': 'run',
                    'source_kind': 'user_message', 'source_ref': 'test://synthetic',
                    'instruction': 'Synthetic local fixture only.', 'timestamp': '2026-10-09T00:00:00+00:00'})
    recipes = {}
    for name in ('environment_recipe', 'build_recipe', 'correctness_runner'):
        recipes[name] = 'recipes/'+name+'.sh'
        put(root, recipes[name], '#!/bin/sh\nexit 0\n')
    put(root, 'recipes/rebuild_outputs.sh', '#!/bin/sh\n# Synthetic recipe; not executed.\nexit 0\n')
    rc.write(root/'protected_files.json', {
        folder+'/contract.txt': rc.digest(root/folder/'contract.txt')
        for folder in ('work', 'original_snapshot')})
    rc.write(root/'recovery_inputs.json', {**recipes, 'external_inputs': [],
                                          'rebuildable_outputs': [declaration()] if rows is None else rows})
    return root


def finish_baseline(root):
    cli(root, 'set-baseline', '--runtime-mean', '100', '--runtime-std', '1',
        '--mem-mean', '1000', '--mem-std', '10', '--output-digest', 'SYNTHETIC')
    state = rc.read(root/'state/state.json')
    policy = {'mode': 'bitwise', 'setting': 'bitwise', 'rationale': 'Synthetic fixture only',
              'semantic_limits': 'Exact scalar', 'selection_basis': 'automatic',
              'outputs': [{'name': 'scalar', 'comparison': 'exact', 'rationale': 'Fixture'}],
              'artifacts': [link(root, root/'evidence/test.txt')],
              'baseline_sha256': rc.canonical(state['baseline']),
              'original_tree_sha256': rc.canonical(state['original_tree'])}
    rc.write(root/'evidence/policy.json', policy)
    cli(root, 'set-tolerance', '--evidence-json', root/'evidence/policy.json')


class IdentityRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='identity-fixture-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        # Isolate reconstruction from an unrelated enclosing skill Git repository.
        # Explicitly initialized fixture repositories remain discoverable.
        ceilings = os.pathsep.join(filter(None, (os.environ.get('GIT_CEILING_DIRECTORIES'), str(self.base))))
        for scope in (patch.object(tempfile, 'tempdir', str(self.base)),
                      patch.dict(os.environ, {'GIT_CEILING_DIRECTORIES': ceilings})):
            scope.start()
            self.addCleanup(scope.stop)
        self.root = prepare(self.base/'run')
        self.store = self.base/'store'
        # Output already generated during baseline build is acceptable if absent clean original.
        put(self.root, 'work/build/solver.o', 'Synthetic compiled output v1\n')
        finish_baseline(self.root)
        self.checkpoint()

    def checkpoint(self):
        saved = cp.save(self.root, self.store)
        cp.verify_for_root(self.root, self.store, saved['snapshot'])
        return saved

    def begin(self, iteration='test-candidate'):
        return rc.begin(self.root, argparse.Namespace(
            iteration_id=iteration, parent_id=None,
            hypothesis='Synthetic coherent algorithm replacement across modules',
            learning='Reassessed original and latest reviewed evidence.',
            **rc.feedback_context(self.root)))

    def test_status_read_and_rebuild_products_do_not_stale_checkpoint_or_admission(self):
        before = cp.source_inventory(self.root)
        cli(self.root, 'status')
        put(self.root, 'work/build/solver.o', 'Synthetic newly rebuilt object v2\n')
        put(self.root, 'original_snapshot/build/solver.o', 'Synthetic original-side build product\n')
        self.assertEqual(before, cp.source_inventory(self.root))
        self.assertEqual(before, cp.source_inventory(self.root, 'original_snapshot'))
        self.assertTrue(rc.checkpoint_current(self.root)['verified'])
        active = self.begin()
        self.assertEqual(active['source_policy_sha256'], rc.read(self.root/'state/state.json')['source_policy_sha256'])

    def test_separate_verify_allows_only_derived_status_drift(self):
        saved = cp.save(self.root, self.store)
        cli(self.root, 'status')
        self.assertTrue(cp.verify_for_root(self.root, self.store, saved['snapshot'])['verified'])
        put(self.root, 'reports/analysis.md', 'Authoritative new report\n')
        with self.assertRaisesRegex(ValueError, 'Snapshot does not cover'):
            cp.verify_for_root(self.root, self.store, saved['snapshot'])

    def test_authoritative_source_and_unlisted_products_remain_guarded(self):
        put(self.root, 'work/solver.py', 'def solve():\n    return 43\n')
        with self.assertRaisesRegex(ValueError, 'Checkpoint is stale'):
            rc.checkpoint_current(self.root)
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, 'Working tree differs'):
            self.begin()
        put(self.root, 'work/solver.py', 'def solve():\n    return 42\n')
        put(self.root, 'work/build/unlisted.o', 'Not explicitly declared\n')
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, 'Working tree differs'):
            self.begin()

    def test_late_declaration_and_recipe_mutation_are_rejected(self):
        spec = rc.read(self.root/'recovery_inputs.json')
        spec['rebuildable_outputs'].append(declaration('build/new.o'))
        rc.write(self.root/'recovery_inputs.json', spec)
        with self.assertRaisesRegex(ValueError, 'frozen at baseline'):
            cp.source_inventory(self.root)
        with self.assertRaisesRegex(ValueError, 'frozen at baseline'):
            self.checkpoint()
        spec['rebuildable_outputs'].pop()
        rc.write(self.root/'recovery_inputs.json', spec)
        put(self.root, 'recipes/rebuild_outputs.sh', '#!/bin/sh\nexit 1\n')
        with self.assertRaisesRegex(ValueError, 'frozen at baseline'):
            cp.source_inventory(self.root)

    def test_undeclared_original_tracked_code_directory_and_symlink_exclusions_rejected(self):
        cases = ['solver.py', 'build/*.o', 'build', '../evidence/test.txt', 'build/linked.o', 'build/original.o', 'build/tracked.o']
        for index, target in enumerate(cases):
            with self.subTest(target=target):
                root = prepare(self.base/('bad'+str(index)), [declaration(target)])
                (root/'work/build').mkdir(exist_ok=True)
                if target == 'build/linked.o':
                    (root/'work/build/linked.o').symlink_to('../solver.py')
                elif target == 'build/original.o':
                    put(root, 'original_snapshot/'+target, 'Authoritative original binary\n')
                elif target == 'build/tracked.o':
                    put(root, 'work/'+target, 'Tracked file\n')
                    subprocess.run(['git', 'init', '-q', str(root/'work')], check=True)
                    subprocess.run(['git', '-C', str(root/'work'), 'add', target], check=True)
                with self.assertRaises(ValueError):
                    finish_baseline(root)

    def test_protected_required_and_known_evidence_are_not_exclusions(self):
        for kind in ('protected', 'input', 'evidence'):
            with self.subTest(kind=kind):
                root = prepare(self.base/kind)
                product = put(root, 'work/build/solver.o', 'Required authoritative bytes\n')
                if kind == 'protected':
                    manifest = rc.read(root/'protected_files.json')
                    manifest['work/build/solver.o'] = rc.digest(product)
                    rc.write(root/'protected_files.json', manifest)
                elif kind == 'input':
                    spec = rc.read(root/'recovery_inputs.json')
                    spec['external_inputs'] = [{'recovery_location': str(product), 'sha256': rc.digest(product)}]
                    rc.write(root/'recovery_inputs.json', spec)
                else:
                    c = rc.control(root)
                    c['diagnostic'] = link(root, product)
                    rc.save_control(root, c)
                with self.assertRaisesRegex(ValueError, 'protected files, required inputs/recipes or evidence'):
                    finish_baseline(root)

    def test_output_cannot_become_evidence_or_tracked_source_after_freeze(self):
        product = self.root/'work/build/solver.o'
        with self.assertRaisesRegex(ValueError, 'cannot serve as required evidence'):
            rc.evidence(self.root, link(self.root, product))
        subprocess.run(['git', 'init', '-q', str(self.root/'work')], check=True)
        subprocess.run(['git', '-C', str(self.root/'work'), 'add', 'build/solver.o'], check=True)
        with self.assertRaisesRegex(ValueError, 'Tracked source'):
            cp.source_inventory(self.root)

    def test_path_aliases_cannot_hide_protected_files_recipes_or_evidence(self):
        for index, alias in enumerate(('work/./build/solver.o', './work/build/solver.o')):
            with self.subTest(alias=alias):
                root = prepare(self.base/('alias'+str(index)))
                product = put(root, 'work/build/solver.o', 'Required bytes\n')
                manifest = rc.read(root/'protected_files.json')
                manifest[alias] = rc.digest(product)
                rc.write(root/'protected_files.json', manifest)
                with self.assertRaisesRegex(ValueError, 'protected files, required inputs/recipes or evidence'):
                    finish_baseline(root)
        for index, alias in enumerate(('work/./build/solver.o', './work/build/solver.o')):
            with self.subTest(recipe_alias=alias):
                row = declaration()
                row['rebuild_recipe'] = alias
                root = prepare(self.base/('recipe-alias'+str(index)), [row])
                put(root, 'work/build/solver.o', '#!/bin/sh\nexit 0\n')
                with self.assertRaisesRegex(ValueError, 'protected files, required inputs/recipes or evidence'):
                    finish_baseline(root)
        (self.root/'work/generated').symlink_to('build', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'cannot serve as required evidence'):
            rc.evidence(self.root, link(self.root, self.root/'work/generated/solver.o'))

    def test_linked_status_is_still_preserved_and_hash_checked(self):
        status = self.root/'reports/status.md'
        c = rc.control(self.root)
        c['diagnostic'] = link(self.root, status)
        rc.save_control(self.root, c)
        saved = self.checkpoint()
        manifest = cp.verify(self.store, saved['snapshot'])
        self.assertIn('reports/status.md', manifest['files'])
        put(self.root, 'reports/status.md', 'Modified linked evidence\n')
        with self.assertRaisesRegex(ValueError, 'Missing or changed source evidence'):
            rc.checkpoint_current(self.root)

    def test_nested_evidence_is_validated_by_content_regardless_of_extension(self):
        status = self.root/'reports/status.md'
        original_status = status.read_text()
        for name, is_jsonl in (('wrapper.txt', False), ('wrapper.json', False), ('wrapper-jsonl.log', True)):
            with self.subTest(wrapper=name):
                status.write_text(original_status)
                nested = {'artifacts': [link(self.root, status)]}
                content = json.dumps(nested)+'\n'
                if is_jsonl:
                    content = json.dumps({'synthetic': 'first JSONL record'})+'\n'+content
                wrapper = put(self.root, 'evidence/'+name, content)
                c = rc.control(self.root)
                c['diagnostic'] = link(self.root, wrapper)
                rc.save_control(self.root, c)
                self.checkpoint()
                self.assertTrue(rc.checkpoint_current(self.root)['verified'])
                status.write_text('Modified evidence reached through a nested wrapper\n')
                with self.assertRaisesRegex(ValueError, 'Missing or changed source evidence'):
                    rc.checkpoint_current(self.root)

    def test_checkpoint_recovery_omits_products_but_reconstructs_source_and_evidence(self):
        saved = self.checkpoint()
        manifest = cp.verify(self.store, saved['snapshot'])
        self.assertNotIn('work/build/solver.o', manifest['files'])
        destination = self.base/'restored'
        cp.restore(self.store, saved['snapshot'], destination)
        self.assertFalse((destination/'work/build/solver.o').exists())
        self.assertEqual(cp.source_inventory(self.root), cp.source_inventory(destination))
        self.assertEqual((destination/'evidence/test.txt').read_bytes(), (self.root/'evidence/test.txt').read_bytes())
        rebuilt = self.base/'reconstructed'
        cp.reconstruct_best(destination, '', rebuilt)
        self.assertEqual(cp.source_inventory(destination, rebuilt), cp.source_inventory(self.root))
        self.assertTrue(rc.checkpoint_current(destination)['verified'])

    def accepted_review_fixture(self, only_product=False):
        self.begin()
        if not only_product:
            put(self.root, 'work/solver.py', 'def solve():\n    return 6 * 7\n')
        put(self.root, 'work/build/solver.o', 'Different compiled product\n')
        patch_text = ('diff --git a/solver.py b/solver.py\n--- a/solver.py\n+++ b/solver.py\n'
                      '@@ -1,2 +1,2 @@\n def solve():\n-    return 42\n+    return 6 * 7\n')
        if only_product:
            patch_text = ('diff --git a/build/solver.o b/build/solver.o\nnew file mode 100644\n'
                          '--- /dev/null\n+++ b/build/solver.o\n@@ -0,0 +1 @@\n+Different compiled product\n')
        patch = put(self.root, 'patches/test.patch', patch_text)
        metrics = dict(runtime_mean=50.0, runtime_std=1.0, runtime_reps=3,
                       mem_mean=1000.0, mem_std=10.0, mem_reps=3)
        correct = self.root/'evidence/correct.json'
        rc.write(correct, {'correctness': 'pass', 'tested_patch_sha256': rc.digest(patch), 'output_digest': 'SYNTHETIC'})
        measured = self.root/'evidence/benchmark.json'
        rc.write(measured, {**metrics, 'tested_patch_sha256': rc.digest(patch)})
        command = [sys.executable, str(options.skill_root/'scripts/acceptance.py')]
        for key, value in {'best-runtime-mean': 100, 'best-runtime-std': 1, 'new-runtime-mean': 50,
                           'new-runtime-std': 1, 'best-mem-mean': 1000, 'best-mem-std': 10,
                           'new-mem-mean': 1000, 'new-mem-std': 10, 'baseline-mem-std': 10}.items():
            command.extend(['--'+key, str(value)])
        verdict = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        acceptance = self.root/'evidence/acceptance.json'
        rc.write(acceptance, verdict)
        review = self.root/'evidence/review.json'
        rc.write(review, {'iteration_id': 'test-candidate', 'mechanism': 'Synthetic exact replacement',
                          'uncertainty': 'Synthetic measurements only', 'disposition': 'accepted',
                          'tested_patch': 'patches/test.patch', 'tested_patch_sha256': rc.digest(patch),
                          'disposed_tree': cp.source_inventory(self.root),
                          'artifacts': [link(self.root, correct), link(self.root, measured)],
                          'correctness': link(self.root, correct), 'benchmark': link(self.root, measured),
                          'acceptance': link(self.root, acceptance)})
        args = argparse.Namespace(iteration_id='test-candidate', parent_id=None, hypothesis=None,
                                  evidence_json=review, decision='accept', correctness='pass',
                                  metric_analysis='Synthetic improvement', next_action='Review remaining ideas',
                                  output_digest='SYNTHETIC', branch='runtime', patch_file='patches/test.patch', **metrics)
        state = rc.read(self.root/'state/state.json')
        return args, state

    def test_accepted_patch_reconstructs_source_without_disposable_products(self):
        args, state = self.accepted_review_fixture()
        rc.review(self.root, args, state)
        self.assertEqual(state['best_iteration_id'], 'test-candidate')
        self.assertIsNone(state['active_candidate'])
        self.assertNotIn('build/solver.o', state['best_tree'])

    def test_product_only_patch_cannot_be_promoted_as_a_source_optimization(self):
        args, state = self.accepted_review_fixture(only_product=True)
        before = state['best_tree']
        self.assertEqual(cp.source_inventory(self.root), before)
        with self.assertRaisesRegex(ValueError, 'source'):
            rc.review(self.root, args, state)
        self.assertEqual(state['best_tree'], before)
        self.assertIsNone(state['best_iteration_id'])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
