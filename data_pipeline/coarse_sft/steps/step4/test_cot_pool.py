"""候选CoT池离线检查：不发送API请求，不改动正式数据。"""

import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steps.step4 import cot_pool as pool
from steps.step4 import pipeline as cot


def sample(index, split="train", gold="safe"):
    return {"id": f"pool_{index}", "group_id": f"group_{index}", "pair_id": None, "split": split,
            "source": {"dataset": "fixture", "row_id": str(index), "parent_id": None},
            "profile": "You are a helpful assistant.", "tools": None,
            "trajectory": [{"role": "user", "content": f"TASK_{index}_{gold.upper()}"},
                           {"role": "agent", "thought": None, "action": 'Complete{"response":"Recorded."}'}],
            "target": {"coarse": gold}}


class FakeClient:
    calls = []
    force_safe = False

    def __init__(self, model, *args, **kwargs):
        self.model = model

    def complete_text(self, prompt):
        self.calls.append(self.model)
        label = "unsafe" if "_UNSAFE" in prompt and not self.force_safe else "safe"
        return {"model": self.model, "finish_reason": "stop", "usage": {"total_tokens": 12},
                "content": f"<Analysis>\n1. Visible evidence.\n\n2. Task intent.\n\n3. Executed actions.\n\n4. Risk assessment.\n</Analysis>\n\n<Judgment> {label} </Judgment>"}


class CoTPoolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "cot_pool"
        self.samples = [sample(0), sample(1, gold="unsafe"), sample(2, "test")]
        FakeClient.calls = []
        FakeClient.force_safe = False
        patches = [patch.object(pool, "OUTPUT_ROOT", self.root),
                   patch.object(pool, "load_inputs", return_value=(self.samples, {"fixture": "hash"})),
                   patch.object(cot, "OpenAICompatibleClient", FakeClient),
                   patch.dict(os.environ, {"STEP4_MODEL": "teacher-a", "STEP4_WORKERS": "1", "STEP4_REQUEST_INTERVAL": "0"}),
                   redirect_stdout(io.StringIO())]
        for context in patches:
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def final(self):
        return cot.read_jsonl(self.root / "final/candidates.jsonl")

    def test_dry_run_does_not_write_or_call(self):
        pool.run(dry_run=True)
        self.assertFalse(self.root.exists())
        self.assertEqual(FakeClient.calls, [])

    def test_resume_reuses_completed_results_after_model_change(self):
        pool.run(limit=1)
        self.assertEqual(len(self.final()), 1)
        with patch.dict(os.environ, {"STEP4_MODEL": "teacher-b", "STEP4_BASE_URL": "https://other.example/v1"}):
            pool.run()
            pool.run()
        self.assertEqual(FakeClient.calls, ["teacher-a", "teacher-b", "teacher-b"])
        self.assertEqual(len(self.final()), 3)
        self.assertEqual(cot.read_jsonl(self.root / "intermediate/inputs.jsonl"), self.samples)
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"generated": 3})
        self.assertEqual(report["usage"]["total_tokens"], 36)

    def test_conflict_is_preserved_and_not_exported_or_repeated(self):
        FakeClient.force_safe = True
        pool.run()
        pool.run()
        self.assertEqual(len(FakeClient.calls), 3)
        self.assertEqual({row["id"] for row in self.final()}, {"pool_0", "pool_2"})
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"generated": 2, "review_required": 1})

    def test_split_only_controls_requests_and_keeps_previous_exports(self):
        pool.run(split="test")
        self.assertEqual([row["split"] for row in self.final()], ["test"])
        pool.run(split="train", limit=1)
        self.assertEqual({row["split"] for row in self.final()}, {"train", "test"})
        self.assertEqual(len(FakeClient.calls), 2)

    def test_interrupt_exports_finished_records_and_next_run_resumes(self):
        original = cot.generate_pending

        def interrupt(pending, *args, **kwargs):
            original(pending[:1], *args, **kwargs)
            raise KeyboardInterrupt

        with patch.object(cot, "generate_pending", interrupt), self.assertRaises(KeyboardInterrupt):
            pool.run()
        self.assertEqual(len(self.final()), 1)
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"generated": 1, "pending": 2})
        pool.run()
        self.assertEqual(len(FakeClient.calls), 3)
        self.assertEqual(len(self.final()), 3)


class RealPoolTests(unittest.TestCase):
    @unittest.skipUnless(pool.binary.BINARY_PATH.is_file() and pool.fine.FINE_PATH.is_file(), "需要本地v1.0原料")
    def test_remaining_manifest_keeps_frozen_splits_and_all_three_fine_labels(self):
        rows, _ = pool.load_inputs()
        chosen = {row["id"] for row in cot.read_jsonl(cot.DATA_ROOT / "step0/final/selection.jsonl")}
        self.assertEqual(len(rows), 5100)
        self.assertFalse(chosen & {row["id"] for row in rows})
        self.assertEqual(Counter(row["split"] for row in rows), {"train": 3421, "validation": 877, "test": 802})
        self.assertEqual(Counter(row["target"]["coarse"] for row in rows), {"safe": 850, "unsafe": 4250})
        fine = [row for row in rows if row["id"].startswith("tool_unsafe_")]
        self.assertEqual(len(fine), 3400)
        self.assertTrue(all(row["target"][field] for row in fine for field in ("risk_source", "failure_mode", "real_world_harm")))
