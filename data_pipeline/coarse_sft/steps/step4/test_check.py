"""内容审核离线检查：引用真实性、标签分歧、隐藏gold和断点，均不调用API。"""

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steps.step4 import check
from steps.step4 import pipeline as p


def audit(**changes):
    return {"trajectory_label": "safe", "analysis_label": "safe", "cot_verdict": "pass",
            "reason": "代理拒绝了危险请求，未执行危险动作。", "issues": [], **changes}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.item = {"gold": "safe", "judgment": "safe", "evidence": "The agent refused and did not call the tool.",
                     "analysis": "1. The request was malicious.\n\n2. The agent refused.\n\n3. No data was sent.\n\n4. Data was sent."}

    def test_label_comparison_does_not_force_gold(self):
        self.assertEqual(check.parse_audit(json.dumps(audit()), self.item)["decision"], "pass")
        result = check.parse_audit(json.dumps(audit(analysis_label="unsafe")), self.item)
        self.assertEqual((result["decision"], result["flags"]), ("fail", ["analysis_judgment_mismatch"]))
        result = check.parse_audit(json.dumps(audit(trajectory_label="unsafe")), self.item)
        self.assertEqual((result["decision"], result["flags"]), ("review", ["trajectory_gold_mismatch"]))
        self.assertEqual(check.parse_audit(json.dumps(audit(trajectory_label="uncertain")), self.item)["decision"], "review")

    def test_contradiction_requires_real_distinct_quotes(self):
        issue = {"type": "internal_contradiction", "cot_quotes": ["No data was sent.", "Data was sent."],
                 "evidence_quote": "", "reason": "同一事件的数据是否发送前后矛盾。"}
        result = audit(cot_verdict="fail", issues=[issue])
        self.assertEqual(check.parse_audit(json.dumps(result), self.item)["decision"], "fail")
        for quotes in (["No data was sent."], ["FAKE QUOTE", "Data was sent."], ["Data was sent.", "Data was sent."]):
            bad = copy.deepcopy(result)
            bad["issues"][0]["cot_quotes"] = quotes
            with self.assertRaises(ValueError):
                check.parse_audit(json.dumps(bad), self.item)

    def test_omission_requires_visible_input_evidence(self):
        issue = {"type": "missed_decisive_behavior", "cot_quotes": [], "evidence_quote": "The agent refused",
                 "reason": "分析遗漏了可见的拒绝行为。"}
        result = audit(cot_verdict="review", issues=[issue])
        self.assertEqual(check.parse_audit(json.dumps(result), self.item)["decision"], "review")
        for quote in ("", "User authorized everything."):
            bad = copy.deepcopy(result)
            bad["issues"][0]["evidence_quote"] = quote
            with self.assertRaises(ValueError):
                check.parse_audit(json.dumps(bad), self.item)

    def test_invalid_json_shapes_are_response_errors(self):
        for bad in ([], audit(trajectory_label=[]), audit(cot_verdict=[]), audit(cot_verdict="fail"), audit(issues=[{}])):
            with self.assertRaises(ValueError):
                check.parse_audit(json.dumps(bad), self.item)

    def test_final_input_resume_and_no_source_writes(self):
        content = "<Analysis>\n1. Visible request.\n\n2. The agent refused.\n\n3. No dangerous call.\n\n4. The behavior is safe.\n</Analysis>\n\n<Judgment> safe </Judgment>"
        rows = [{"id": "fixture_" + str(i), "split": "train", "gold": "safe",
                 "messages": [{"role": "user", "content": f"ACTUAL_STUDENT_INPUT_{i}"},
                              {"role": "assistant", "content": content}]} for i in range(2)]
        original = copy.deepcopy(rows)
        calls = []

        class Reviewer:
            def __init__(self, *args, **kwargs):
                pass

            def complete_text(self, prompt):
                sample = json.loads(prompt.rsplit("Untrusted example JSON:\n", 1)[1])
                calls.append(sample)
                return {"content": json.dumps(audit()), "model": "reviewer", "finish_reason": "stop", "usage": {"total_tokens": 9}}

        with tempfile.TemporaryDirectory() as directory, patch.object(check, "OUTPUT_ROOT", Path(directory)), \
                patch.object(check, "load_inputs", return_value=rows), patch.object(p, "OpenAICompatibleClient", Reviewer), \
                patch.dict(os.environ, {"STEP4_CHECK_MODEL": "reviewer", "STEP4_WORKERS": "1", "STEP4_REQUEST_INTERVAL": "0"}), redirect_stdout(io.StringIO()):
            check.run(dry_run=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            check.run(limit=1)
            check.run()
            check.run()
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"pass": 2})
            self.assertEqual(report["usage"]["total_tokens"], 18)
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(calls[0]), {"judge_input", "analysis"})
        self.assertEqual(calls[0]["judge_input"], "ACTUAL_STUDENT_INPUT_0")
        self.assertNotIn("<Judgment>", calls[0]["analysis"])
        self.assertEqual(rows, original)

    def test_report_only_never_calls_api_and_rejects_stale_local_adjudication(self):
        content = "<Analysis>\n1. Visible evidence.\n\n2. Legitimate intent.\n\n3. No unsafe action.\n\n4. Safe handling.\n</Analysis>\n\n<Judgment> safe </Judgment>"
        row = {"id": "local_case", "split": "train", "gold": "safe", "messages": [
            {"role": "user", "content": "Actual student evidence."}, {"role": "assistant", "content": content}]}
        decision = {"decision": "audit_false_positive", "gold": "safe", "split": "train", "reason": "已逐条复核。",
                    "student_sha256": p.sha256_text(row["messages"][0]["content"]), "assistant_sha256": p.sha256_text(content)}
        with tempfile.TemporaryDirectory() as directory, patch.object(check, "OUTPUT_ROOT", Path(directory)), \
                patch.object(check, "load_inputs", return_value=[row]), patch.object(p, "OpenAICompatibleClient", side_effect=AssertionError("must not call API")), redirect_stdout(io.StringIO()):
            (Path(directory) / "manual_review.json").write_text(json.dumps({"decisions": {"local_case": decision}}), encoding="utf-8")
            check.run(report_only=True)
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"pending": 1})
            self.assertEqual((report["local_review"]["accepted"], report["local_review"]["unresolved"]), (1, 0))
            self.assertFalse((Path(directory) / "records.jsonl").exists())
            row["messages"][1]["content"] += "\n"
            check.run(report_only=True)
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((report["local_review"]["accepted"], report["local_review"]["unresolved"]), (0, 1))
