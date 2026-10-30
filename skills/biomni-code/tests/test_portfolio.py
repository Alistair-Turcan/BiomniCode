#!/usr/bin/env python3
"""Synthetic local regressions: diverse ideas, one admitted candidate, fresh evidence.

No repository optimization, external service, benchmark or build is performed.
All source, authority, metrics and correctness records are explicit test fixtures.
The accepted-result regression runs the real acceptance helper and local Git patch
reconstruction; it does not measure a real speedup.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
cli = argparse.ArgumentParser()
cli.add_argument("--skill-root", type=Path,
                 default=Path(__file__).resolve().parents[1],
                 help="Package root containing scripts/; override when testing from scratch")
options, remaining = cli.parse_known_args()
sys.path.insert(0, str(options.skill_root / "scripts"))
import biomni_code_state as state_tool
import numerical_policy as np
import run_checkpoint as cp
import run_control as rc


def put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def link(root, path):
    return {"evidence": str(path.relative_to(root)), "sha256": rc.digest(path)}


class PortfolioRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="portfolio-fixture-",
                                               dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        # Keep reconstruction temporaries inside this fixture, where Git cannot
        # discover an unrelated checkout enclosing the installed test package.
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = self.temp.name
        self.addCleanup(setattr, tempfile, "tempdir", previous_tempdir)
        previous_ceiling = os.environ.get("GIT_CEILING_DIRECTORIES")
        os.environ["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(
            value for value in (previous_ceiling, self.temp.name) if value)
        def restore_ceiling():
            if previous_ceiling is None:
                os.environ.pop("GIT_CEILING_DIRECTORIES", None)
            else:
                os.environ["GIT_CEILING_DIRECTORIES"] = previous_ceiling
        self.addCleanup(restore_ceiling)
        self.root = Path(self.temp.name) / "run"
        self.store = Path(self.temp.name) / "store"
        self.call("init", "--run-id", "SYNTHETIC-LOCAL-TEST", "--budget-iterations", "3")
        for folder in ("work", "original_snapshot"):
            put(self.root, folder + "/solver.py", "def solve():\n    return 42\n")
            put(self.root, folder + "/contract.txt", "Synthetic exact-output contract.\n")
        self.evidence = put(self.root, "evidence/synthetic.txt",
                            "SYNTHETIC TEST ONLY: no real authority or measurement.\n")
        rc.record(self.root, {
            **link(self.root, self.evidence), "action": "approve", "scope": "run",
            "source_kind": "user_message", "source_ref": "test://synthetic",
            "instruction": "Synthetic local fixture only.",
            "timestamp": "2026-10-09T00:00:00+00:00",
        })
        rc.write(self.root / "protected_files.json", {
            "work/contract.txt": rc.digest(self.root / "work/contract.txt"),
            "original_snapshot/contract.txt": rc.digest(self.root / "original_snapshot/contract.txt"),
        })
        recipes = {}
        for name in ("environment_recipe", "build_recipe", "correctness_runner"):
            recipes[name] = "recovery/" + name + ".txt"
            put(self.root, recipes[name], "Synthetic placeholder; never executed.\n")
        rc.write(self.root / "recovery_inputs.json", {**recipes, "external_inputs": [],
                                                      "rebuildable_outputs": []})
        self.call("set-baseline", "--runtime-mean", "100", "--runtime-std", "1",
                  "--mem-mean", "1000", "--mem-std", "10", "--output-digest", "SYNTHETIC")
        state = rc.read(self.root / "state/state.json")
        policy = {
            "mode": "bitwise", "setting": "bitwise", "rationale": "Synthetic test only",
            "semantic_limits": "Exact scalar", "selection_basis": "automatic",
            "outputs": [{"name": "scalar", "comparison": "exact", "rationale": "Fixture"}],
            "artifacts": [link(self.root, self.evidence)],
            "baseline_sha256": rc.canonical(state["baseline"]),
            "original_tree_sha256": rc.canonical(state["original_tree"]),
        }
        rc.write(self.root / "evidence/policy.json", policy)
        self.call("set-tolerance", "--evidence-json", str(self.root / "evidence/policy.json"))
        self.call("add-hotspots", "--profiler", "SYNTHETIC", "--entries",
                  *[json.dumps({"function": name, "pct_runtime": share})
                    for name, share in (("primary", 65), ("secondary", 25), ("tertiary", 8))])
        self.checkpoint()

    def call(self, command, *arguments):
        args = state_tool.build_parser().parse_args(
            [command, "--root", str(self.root), *map(str, arguments)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), rc.transaction(self.root):
            args.func(args)
        return json.loads(output.getvalue())

    def checkpoint(self):
        saved = cp.save(self.root, self.store)
        cp.verify_for_root(self.root, self.store, saved["snapshot"])

    def idea(self, identity, priority=1, family="algorithmic", hypothesis=None, status="open"):
        return self.call("add-backlog", "--backlog-id", identity,
                         "--priority", priority, "--family", family, "--intervention", family,
                         "--hotspot", "primary", "--hypothesis", hypothesis or "Synthetic idea " + identity,
                         "--test", "Use the unchanged correctness suite and timing harness",
                         "--source", "Synthetic profile", "--status", status)["added"]

    def begin(self, identity, idea, context=None, hypothesis=None, parent=None, learning=None):
        context = context or rc.feedback_context(self.root)
        command = ["--iteration-id", identity, "--backlog-id", idea,
                   "--hypothesis", hypothesis or "Synthetic idea " + idea,
                   "--feedback-id", context["feedback_id"],
                   "--feedback-sha256", context["feedback_sha256"],
                   "--learning", learning if learning is not None else
                   "Reassessed the idea against current best and the latest synthetic result."]
        if parent is not None:
            command.extend(["--parent-id", parent])
        return self.call("begin-iteration", *command)

    def review(self, identity, decision="reject", family="algorithmic", intervention="algorithmic"):
        patch = put(self.root, "patches/" + identity + ".patch", "")
        correct = self.root / ("evidence/" + identity + "-correctness.json")
        rc.write(correct, {"correctness": "fail", "tested_patch_sha256": rc.digest(patch)})
        row = {
            "iteration_id": identity, "mechanism": "Synthetic hypothesis fails exact check",
            "uncertainty": "Test fixture only", "disposition": "reverted",
            "artifacts": [link(self.root, correct)], "correctness": link(self.root, correct),
            "disposed_tree": cp.source_inventory(self.root),
            "tested_patch": str(patch.relative_to(self.root)), "tested_patch_sha256": rc.digest(patch),
        }
        review_file = self.root / ("evidence/" + identity + "-review.json")
        rc.write(review_file, row)
        return self.call("add-iteration", "--iteration-id", identity,
                         "--family", family, "--intervention", intervention,
                         "--decision", decision, "--correctness", "fail",
                         "--metric-analysis", "Synthetic correctness failure prevents promotion.",
                         "--next-action", "Reassess remaining diverse ideas using this reviewed result.",
                         "--evidence-json", review_file)

    def accept_fixture(self, identity):
        """One coherent two-file change, including a new helper; synthetic metrics."""
        put(self.root, "work/solver.py", "from kernel import result\n\ndef solve():\n    return result()\n")
        put(self.root, "work/kernel.py", "def result():\n    return 6 * 7\n")
        patch = put(self.root, "patches/" + identity + ".patch",
                    "diff --git a/solver.py b/solver.py\n"
                    "--- a/solver.py\n+++ b/solver.py\n@@ -1,2 +1,4 @@\n"
                    "+from kernel import result\n+\n def solve():\n-    return 42\n+    return result()\n"
                    "diff --git a/kernel.py b/kernel.py\nnew file mode 100644\n"
                    "--- /dev/null\n+++ b/kernel.py\n@@ -0,0 +1,2 @@\n"
                    "+def result():\n+    return 6 * 7\n")
        patch_sha = rc.digest(patch)
        measurements = {"runtime_mean": 50.0, "runtime_std": 1.0, "runtime_reps": 3,
                        "mem_mean": 1000.0, "mem_std": 10.0, "mem_reps": 3}
        artifacts = {}
        for name, value in {
            "correctness": {"correctness": "pass", "output_digest": "SYNTHETIC",
                            "tested_patch_sha256": patch_sha},
            "benchmark": {**measurements, "tested_patch_sha256": patch_sha},
            "acceptance": {"decision": "accept", "branch": "runtime", "memory_driven": False,
                           "runtime_rel_improvement": 0.5,
                           "reasons": ["runtime branch satisfied"], "warnings": []},
        }.items():
            path = self.root / ("evidence/" + identity + "-" + name + ".json")
            rc.write(path, value)
            artifacts[name] = link(self.root, path)
        review = {
            "iteration_id": identity, "mechanism": "Synthetic equivalent scalar expression",
            "uncertainty": "Synthetic metrics test the promotion gate; no measured speedup claim",
            "disposition": "accepted", "artifacts": list(artifacts.values()), **artifacts,
            "disposed_tree": cp.source_inventory(self.root),
            "tested_patch": str(patch.relative_to(self.root)), "tested_patch_sha256": patch_sha,
        }
        review_file = self.root / ("evidence/" + identity + "-review.json")
        rc.write(review_file, review)
        metric_args = [arg for key, value in measurements.items()
                       for arg in ("--" + key.replace("_", "-"), str(value))]
        return self.call("add-iteration", "--iteration-id", identity,
                         "--family", "algorithmic", "--intervention", "equivalent_scalar",
                         "--decision", "accept", "--branch", "runtime", "--correctness", "pass",
                         "--output-digest", "SYNTHETIC", "--patch-file", str(patch.relative_to(self.root)),
                         "--edited-files", "solver.py", "kernel.py", *metric_args,
                         "--metric-analysis", "Synthetic 2x metrics satisfy the unchanged acceptance rule.",
                         "--next-action", "Reassess diverse ideas against the new accepted source and result.",
                         "--evidence-json", review_file)

    def test_ranked_revisions_restore_diverse_context(self):
        families = ["algorithmic", "data_structures", "vectorization", "algorithmic", "io_batching"]
        for index, family in enumerate(families):
            self.idea("idea-" + str(index), index + 1, family)
        self.idea("idea-0", 9, "algorithmic", "Replace a quadratic pass with an exact index")
        self.checkpoint()
        suggestion = self.call("suggest-next")
        self.assertTrue(suggestion["may_begin"])
        self.assertTrue(suggestion["may_inspect_ideas"])
        self.assertEqual(len(suggestion["top_families"]), 5)
        self.assertEqual(len(suggestion["top_backlog"]), 5)
        self.assertEqual(suggestion["top_backlog"][0]["backlog_id"], "idea-0")
        self.assertEqual(len(suggestion["latest_hotspot_table"]["entries"]), 3)
        self.assertGreaterEqual(len({x["family"] for x in suggestion["top_backlog"]}), 3)
        self.assertEqual(len(state_tool.read_jsonl(self.root / "state/backlog.jsonl")), 6)
        self.assertEqual(len(state_tool.backlog_items(self.root)), 5)
        self.assertIn("portfolio_families", suggestion)
        self.assertIn("recent_iterations", suggestion)
        self.assertEqual(suggestion["feedback_id"], "baseline")

    def test_ideas_remain_inspectable_but_only_one_candidate_can_begin(self):
        broad = "Replace the exact core algorithm across three modules, including preparatory indexes"
        self.idea("broad", hypothesis=broad)
        self.idea("alternative", family="vectorization")
        self.checkpoint()
        active = self.begin("iteration-1", "broad", hypothesis=broad)
        self.assertEqual(active["backlog_id"], "broad")
        self.assertEqual(len(active["backlog_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "Only one active candidate"):
            self.begin("iteration-2", "alternative")
        suggestion = self.call("suggest-next")
        self.assertFalse(suggestion["may_begin"])
        self.assertTrue(suggestion["may_inspect_ideas"])
        self.assertEqual(len(suggestion["latest_hotspot_table"]["entries"]), 3)
        self.assertEqual([x["backlog_id"] for x in suggestion["top_backlog"]], ["alternative"])
        self.idea("later-insight", family="caching_memoization")
        with self.assertRaisesRegex(ValueError, "Selected/reviewed idea is fixed"):
            self.idea("broad", hypothesis=broad)
        self.assertEqual(rc.count(self.root), 0)

    def test_substantive_review_closes_idea_and_stale_feedback_cannot_admit_next(self):
        self.idea("first")
        self.idea("next", family="vectorization")
        self.checkpoint()
        old_feedback = rc.feedback_context(self.root)
        admitted = self.begin("iteration-1", "first")
        self.review("iteration-1")
        recorded = rc.records(self.root)[0]
        self.assertEqual(recorded["backlog_id"], "first")
        self.assertEqual(recorded["backlog_sha256"], admitted["backlog_sha256"])
        self.assertEqual(state_tool.backlog_depth_by_family(self.root), {"vectorization": 1})
        with self.assertRaisesRegex(ValueError, "Selected/reviewed idea is fixed"):
            self.idea("first")
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, "Select an open backlog idea"):
            self.begin("iteration-repeat", "first")
        with self.assertRaisesRegex(ValueError, "Missing or stale feedback"):
            self.begin("iteration-2", "next", context=old_feedback)
        suggestion = self.call("suggest-next")
        self.assertTrue(suggestion["may_begin"])
        self.assertEqual(suggestion["recent_iterations"][-1]["iteration_id"], "iteration-1")
        self.assertEqual(suggestion["portfolio_families"][0]["attempts"], 1)
        self.assertEqual(self.begin("iteration-2", "next")["feedback_id"], "iteration-1")
        self.assertEqual(rc.count(self.root), 1)

    def test_infrastructure_does_not_consume_idea_or_iteration_budget(self):
        self.idea("retry")
        self.checkpoint()
        original_feedback = rc.feedback_context(self.root)
        self.begin("infrastructure-attempt", "retry")
        self.review("infrastructure-attempt", "infrastructure")
        self.assertEqual(rc.count(self.root), 0)
        self.assertEqual(state_tool.backlog_items(self.root)[0]["status"], "open")
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, "Missing or stale feedback"):
            self.begin("retry-2", "retry", context=original_feedback)
        self.assertEqual(self.begin("retry-2", "retry")["feedback_id"], "infrastructure-attempt")

    def test_accepted_winner_requires_new_parent_feedback_and_verified_serial_admission(self):
        self.idea("winning-idea")
        self.idea("next-idea", family="vectorization")
        self.idea("third-idea", family="data_structures")
        self.checkpoint()
        baseline_feedback = rc.feedback_context(self.root)
        self.begin("winner", "winning-idea")
        result = self.accept_fixture("winner")
        self.assertTrue(result["substantive"])
        self.assertEqual(rc.count(self.root), 1)
        state = rc.read(self.root / "state/state.json")
        self.assertEqual(state["best_iteration_id"], "winner")
        self.assertEqual(state["current_best"]["runtime_mean"], 50.0)
        self.assertEqual(state["best_tree"], cp.source_inventory(self.root))
        self.assertNotEqual(state["best_tree"], state["original_tree"])
        self.assertIsNone(state["active_candidate"])
        self.assertEqual(state_tool.backlog_items(self.root)[0]["status"], "completed")
        self.assertEqual(rc.records(self.root)[0]["decision"], "accept")
        with self.assertRaisesRegex(ValueError, "Checkpoint is stale"):
            self.begin("unsaved-next", "next-idea", parent="winner")
        self.checkpoint()
        suggestion = self.call("suggest-next")
        self.assertTrue(suggestion["may_begin"])
        self.assertEqual(suggestion["parent_id"], "winner")
        self.assertEqual(suggestion["feedback_id"], "winner")
        with self.assertRaisesRegex(ValueError, "current-best parent"):
            self.begin("missing-parent", "next-idea")
        with self.assertRaisesRegex(ValueError, "Missing or stale feedback"):
            self.begin("stale-feedback", "next-idea", parent="winner", context=baseline_feedback)
        next_active = self.begin("next", "next-idea", parent="winner", context=suggestion)
        self.assertEqual(next_active["parent_id"], "winner")
        self.assertEqual(next_active["feedback_sha256"], rc.canonical(rc.records(self.root)[0]))
        self.assertEqual(next_active["base"]["runtime_mean"], 50.0)
        with self.assertRaisesRegex(ValueError, "Only one active candidate"):
            self.begin("third", "third-idea", parent="winner")
        self.assertFalse(self.call("suggest-next")["may_begin"])
        self.assertEqual(rc.count(self.root), 1)

    def test_retirement_revision_and_current_hypothesis_required(self):
        self.idea("idea")
        self.idea("idea", hypothesis="A revised evidence-grounded hypothesis")
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, "Revise the queued idea"):
            self.begin("stale", "idea")
        self.idea("idea", hypothesis="A revised evidence-grounded hypothesis", status="retired")
        self.assertEqual(state_tool.backlog_depth_by_family(self.root), {})
        with self.assertRaisesRegex(ValueError, "Select an open backlog idea"):
            self.begin("retired", "idea", hypothesis="A revised evidence-grounded hypothesis")
        with self.assertRaisesRegex(ValueError, "finite priority"):
            self.idea("bad", priority="nan")

    def test_admission_preserves_parent_learning_jobs_authority_and_checkpoint_gates(self):
        self.idea("idea")
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, "current-best parent"):
            self.begin("wrong-parent", "idea", parent="another-branch")
        with self.assertRaisesRegex(ValueError, "Explain learning"):
            self.begin("no-learning", "idea", learning="")
        original_control = rc.control(self.root)
        for label, extra in (
            ("job", {"jobs": {"unfinished": {"status": "running"}}}),
            ("recovery", {"recovery": {"reason": "Synthetic unresolved recovery"}}),
            ("authority", {"authority": []}),
        ):
            with self.subTest(gate=label):
                rc.save_control(self.root, {**original_control, **extra})
                with self.assertRaises(ValueError):
                    self.begin(label, "idea")
                suggestion = self.call("suggest-next")
                self.assertFalse(suggestion["may_begin"])
                self.assertEqual(suggestion["may_inspect_ideas"], label == "job")
                self.assertEqual(len(suggestion["top_backlog"]), 1)
        rc.save_control(self.root, original_control)
        self.idea("new-evidence")
        with self.assertRaisesRegex(ValueError, "Checkpoint is stale"):
            self.begin("unsaved-evidence", "idea")

    def test_budget_exhaustion_blocks_admission_but_preserves_idea_inspection(self):
        self.idea("tested")
        self.idea("remaining")
        state = rc.read(self.root / "state/state.json")
        state["budget_iterations"] = 1
        rc.write(self.root / "state/state.json", state)
        self.checkpoint()
        self.begin("last", "tested")
        self.review("last")
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, "Budget exhausted"):
            self.begin("extra", "remaining")
        suggestion = self.call("suggest-next")
        self.assertFalse(suggestion["may_begin"])
        self.assertEqual(suggestion["next_action"], "complete_final_validation")
        self.assertEqual(suggestion["top_backlog"][0]["backlog_id"], "remaining")
        self.assertEqual(rc.count(self.root), 1)

    def test_strategy_learning_uses_family_and_legacy_family_fallback(self):
        self.idea("first")
        self.idea("second")
        self.idea("third")
        initial = rc.read(self.root / "state/strategy_weights.json")["weights"]
        self.checkpoint()
        self.begin("technique-result", "first")
        self.review("technique-result", family="algorithmic", intervention="blocked_scan")
        weights = rc.read(self.root / "state/strategy_weights.json")["weights"]
        self.assertNotIn("blocked_scan", weights)
        self.assertLess(weights["algorithmic"], initial["algorithmic"])
        self.assertEqual(rc.records(self.root)[0]["family"], "algorithmic")
        self.checkpoint()
        self.begin("legacy-result", "second")
        self.review("legacy-result", family="unspecified", intervention="algorithmic")
        self.assertEqual(rc.records(self.root)[1]["family"], "algorithmic")
        self.assertEqual(rc.read(self.root / "state/diversity_portfolio.json")["families"][0]["attempts"], 2)
        self.checkpoint()
        self.begin("unclassified-result", "third")
        with self.assertRaisesRegex(ValueError, "Set --family"):
            self.review("unclassified-result", family="unspecified", intervention="blocked_scan")
        self.assertEqual(rc.count(self.root), 2)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
