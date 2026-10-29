"""离线验证Step 4：模拟API，不使用真实Key或发送请求。"""

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steps.step3.client import OpenAICompatibleClient
from steps.step4 import pipeline as p
from steps.step4.inputs import complete_text, prepare_row
from steps.step4 import inputs as input_adapter
from steps.step4 import review as review_stage


def answer(label="safe"):
    return f"<Analysis>\n1. The agent uses the visible request.\n\n2. The stated goal is legitimate.\n\n3. The trace shows a completed action.\n\n4. The evidence supports this assessment.\n</Analysis>\n\n<Judgment> {label} </Judgment>"


def row(index=0, label="safe", split="train"):
    return {
        "id": f"sample_{index}", "split": split, "profile": "You are a helpful assistant.",
        "tools": [{"name": "check_order", "description": "Check an order.", "parameters": {"properties": {"id": {"type": "string"}}}, "_source": "HIDDEN_SOURCE"}],
        "trajectory": [{"role": "user", "content": f"TASK_{label.upper()}"}, {"role": "agent", "thought": "Check before acting.", "action": '{"name":"check_order","arguments":{"id":"ORD-1"}}'}],
        "target": {"coarse": label, "risk_source": "HIDDEN_GOLD_LABEL"},
    }


class Step4Tests(unittest.TestCase):
    def test_independent_review_consensus_disagreement_and_resume(self):
        samples = [row(0, "safe"), row(1, "unsafe")]
        teacher = p.TEACHER_PATH.read_text(encoding="utf-8")
        student = p.STUDENT_PATH.read_text(encoding="utf-8")
        calls = []

        class Reviewer:
            def __init__(self, model, *args, **kwargs):
                self.model = model

            def complete_text(self, prompt):
                calls.append((self.model, prompt))
                label = "unsafe" if "TASK_UNSAFE" in prompt and self.model == "reviewer-a" else "safe"
                return {"content": answer(label), "model": self.model, "finish_reason": "stop", "usage": {"total_tokens": 10}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, queue = [], []
            for sample in samples:
                t, s = p.render_inputs(sample, teacher, student)
                item = {"id": sample["id"], "split": "train", "gold": sample["target"]["coarse"], "teacher": t, "student": s}
                key = p.sha256_text(p.stable_json({**item, "model": "generator", "base_url": "https://example.test/v1"}))
                source.append({"id": sample["id"], "key": key, "status": "review_required", "content": "OLD_COT_ONLY"})
                queue.append({"id": sample["id"], "key": key})
            checkpoint = root / "intermediate/teacher_outputs.jsonl"
            checkpoint.parent.mkdir(parents=True)
            p.write_jsonl(checkpoint, source)
            p.write_jsonl(root / "intermediate/unresolved_cases.jsonl", queue)
            (root / "report.json").write_text(json.dumps({"model": "generator", "base_url": "https://example.test/v1", "outputs": {"checkpoint": {"path": "intermediate/teacher_outputs.jsonl"}}}), encoding="utf-8")
            with patch.object(p, "OUTPUT_ROOT", root), patch.object(p, "load_inputs", return_value=samples), patch.object(review_stage, "OpenAICompatibleClient", Reviewer), patch.dict(os.environ, {"STEP4_REVIEW_MODELS": "reviewer-a,reviewer-b", "STEP4_BASE_URL": "https://example.test/v1", "STEP4_WORKERS": "2", "STEP4_REQUEST_INTERVAL": "0"}), redirect_stdout(io.StringIO()):
                review_stage.run()
                first = json.loads((root / "review_report.json").read_text(encoding="utf-8"))
                self.assertEqual(first["counts"], {"accepted": 1, "models_disagree": 1})
                self.assertEqual(first["usage"]["total_tokens"], 40)
                review_stage.run()
            self.assertEqual(len(calls), 4)
            self.assertTrue(all("OLD_COT_ONLY" not in prompt and "HIDDEN_GOLD_LABEL" not in prompt for _, prompt in calls))
            records = p.read_jsonl(checkpoint)
            self.assertEqual(records[:2], source)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[-1]["response_source"], "independent_review_consensus")
            self.assertEqual(records[-1]["requested_model"], "reviewer-a")
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"generated": 1, "review_required": 1})

    def test_manual_input_fix_keeps_prefix_and_rejects_stale_source(self):
        sample = row()
        sample.update(group_id="group_a", pair_id="pair_a", source={"parent_id": "parent_a"})
        sample["trajectory"].extend([
            {"role": "environment", "content": '{"status":"success","result":{"id":"ORD-1"}}'},
            {"role": "agent", "thought": "Old conclusion.", "action": 'Complete{"response":"Done."}'},
        ])
        fix = {"source_trajectory_sha256": p.sha256_text(p.stable_json(sample["trajectory"])),
               "stop_before_event": 3, "thought": "Verify scope first.", "response": "Please confirm."}
        with patch.dict(input_adapter.CORRECTIONS, {sample["id"]: fix}):
            changed, _ = prepare_row(sample)
            self.assertEqual(changed["trajectory"][:3], sample["trajectory"][:3])
            for field in ("id", "group_id", "pair_id", "split", "source", "target", "tools"):
                self.assertEqual(changed[field], sample[field])
            sample["trajectory"][0]["content"] = "Different source input."
            with self.assertRaises(p.PipelineError):
                prepare_row(sample)

    def test_gateway_retry_advice(self):
        self.assertEqual(p.retry_delay('HTTP 502: {"retry_after":60}', 1), 60)
        self.assertEqual(p.retry_delay('HTTP 503: upstream overloaded', 1), 30)
        self.assertEqual(p.retry_delay('HTTP 429: rate limited', 1), 65)

    def test_parallel_requests_have_one_checkpoint_writer(self):
        samples = [row(i * 2 + j, label, split) for i, split in enumerate(p.SPLITS) for j, label in enumerate(("safe", "unsafe"))]
        gate = threading.Barrier(3)
        lock = threading.Lock()
        active = peak = calls = 0
        writer_threads = []
        real_write = p.write_jsonl

        class ParallelClient:
            def __init__(self, *args, **kwargs):
                pass

            def complete_text(self, prompt):
                nonlocal active, peak, calls
                with lock:
                    active += 1
                    peak = max(peak, active)
                    calls += 1
                    number = calls
                if number <= 3:
                    gate.wait(timeout=5)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {"content": answer("unsafe" if "TASK_UNSAFE" in prompt else "safe"), "finish_reason": "stop", "usage": {"total_tokens": 10}}

        def checked_write(*args, **kwargs):
            writer_threads.append(threading.get_ident())
            return real_write(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(p, "OUTPUT_ROOT", root), patch.object(p, "SPLITS", dict.fromkeys(p.SPLITS, 2)), patch.object(p, "load_inputs", return_value=samples), patch.object(p, "OpenAICompatibleClient", ParallelClient), patch.object(p, "write_jsonl", checked_write), patch.dict(os.environ, {"STEP4_WORKERS": "3", "STEP4_REQUEST_INTERVAL": "0"}), redirect_stdout(io.StringIO()):
                p.run()
                p.run()
            self.assertEqual(peak, 3)
            self.assertEqual(calls, 6)
            self.assertEqual(set(writer_threads), {threading.get_ident()})
            checkpoint = p.read_checkpoint(root / "intermediate/teacher_outputs.jsonl")
            self.assertEqual(len(checkpoint), 6)
            self.assertEqual(len({r["id"] for r in checkpoint}), 6)

    def test_start_interval_is_shared_by_workers(self):
        starts = []

        class PacedClient:
            def complete_text(self, prompt):
                starts.append(time.monotonic())
                return {"content": answer(), "finish_reason": "stop"}

        items = [{"id": str(i), "key": str(i), "split": "train", "teacher": "test", "gold": "safe"} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            p.generate_pending(items, PacedClient(), 3, 0.05, Path(directory) / "checkpoint.jsonl", {}, [], {})
        starts.sort()
        self.assertTrue(all(b - a >= 0.045 for a, b in zip(starts, starts[1:])))

    def test_auth_failure_saves_other_inflight_response(self):
        gate = threading.Barrier(2)
        calls = []

        class FailingClient:
            def complete_text(self, prompt):
                calls.append(prompt)
                gate.wait(timeout=5)
                if prompt == "0":
                    raise p.PipelineError("API request failed after 1 attempts: HTTP 401: Invalid token")
                time.sleep(0.05)
                return {"content": answer(), "finish_reason": "stop"}

        items = [{"id": str(i), "key": str(i), "split": "train", "teacher": str(i), "gold": "safe"} for i in range(4)]
        cache, attempts = {}, []
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            p.generate_pending(items, FailingClient(), 2, 0, Path(directory) / "checkpoint.jsonl", cache, attempts, {})
        self.assertEqual(set(calls), {"0", "1"})
        self.assertEqual(cache["0"]["status"], "failed")
        self.assertEqual(cache["1"]["status"], "generated")
        self.assertEqual(len(attempts), 2)

    def test_429_is_saved_and_retried_with_global_cooldown(self):
        calls = 0

        class RateLimitedClient:
            def complete_text(self, prompt):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise p.PipelineError("API request failed after 1 attempts: HTTP 429: rate limited")
                return {"content": answer(), "finish_reason": "stop"}

        item = {"id": "1", "key": "1", "split": "train", "teacher": "test", "gold": "safe"}
        cache, attempts = {}, []
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "RATE_LIMIT_COOLDOWN_SECONDS", 0.01), redirect_stdout(io.StringIO()):
            p.generate_pending([item], RateLimitedClient(), 4, 0, Path(directory) / "checkpoint.jsonl", cache, attempts, {})
        self.assertEqual(calls, 2)
        self.assertEqual([r["status"] for r in attempts], ["failed", "generated"])
        self.assertEqual(cache["1"]["attempt"], 2)

    def test_historical_complete_and_unknown_tool_inventory(self):
        sample = row()
        sample["tools"] = None
        sample["trajectory"] = [
            {"role": "user", "content": "Check schedules."},
            {"role": "agent", "thought": "I should check before changing anything.", "action": 'ZendeskCore_getSchedules(domain="example")'},
            {"role": "environment", "content": '{"name":"ZendeskCore_getSchedules","result":{"schedules":[]}}'},
            {"role": "agent", "thought": None, "action": "Complete{'response': 'I've checked.\nNothing changed.'}"},
        ]
        before = json.dumps(sample)
        teacher, student = p.render_inputs(sample, p.TEACHER_PATH.read_text(encoding="utf-8"), p.STUDENT_PATH.read_text(encoding="utf-8"))
        for rendered in (teacher, student):
            self.assertIn("Full tool definitions were not provided", rendered)
            self.assertIn("Tool: ZendeskCore_getSchedules", rendered)
            self.assertNotIn("Complete{'response': 'Complete{", rendered)
            self.assertIn("I've checked.", rendered)
        self.assertEqual(json.dumps(sample), before)
        self.assertEqual(complete_text(sample["trajectory"][-1]["action"]), "I've checked.\nNothing changed.")

    def test_editor_note_is_not_agent_evidence(self):
        sample = row()
        sample["id"] = "tool_safe_test"
        sample["trajectory"].append({"role": "agent", "thought": "Stops before the removed wrong call.", "action": 'Complete{"response":"Please confirm."}'})
        adapted, changes = prepare_row(sample)
        self.assertIsNone(adapted["trajectory"][-1]["thought"])
        self.assertEqual(adapted["trajectory"][1]["thought"], "Check before acting.")
        self.assertIn("editor_note_removed", changes)
        self.assertEqual(len(adapted["trajectory"]), len(sample["trajectory"]))
        self.assertIsNotNone(sample["trajectory"][-1]["thought"])

    def test_renderer_matches_benchmark_and_hides_metadata(self):
        sample = row()
        teacher = p.TEACHER_PATH.read_text(encoding="utf-8")
        student = p.STUDENT_PATH.read_text(encoding="utf-8")
        prompt, rendered = p.render_inputs(sample, teacher, student)
        fields = p.renderer.build_prompt_fields("atbench", {"profile": sample["profile"], "contents": [sample["trajectory"]], "tool_used": sample["tools"]})
        self.assertEqual(rendered, p.renderer.render_prompt(student, fields))
        self.assertNotIn("HIDDEN_SOURCE", prompt + rendered)
        self.assertNotIn("HIDDEN_GOLD_LABEL", prompt + rendered)
        self.assertLess(prompt.index("[THOUGHT]:"), prompt.index("[ACTION]:"))
        sample["tools"] = None
        self.assertIn("Tool: check_order", p.render_inputs(sample, teacher, student)[0])

    def test_parser(self):
        self.assertEqual(p.parse_output(answer("unsafe"))[1], "unsafe")
        malformed = [answer().replace("1.", "2.", 1), answer().replace("safe </", "Safe </"),
                     "prefix" + answer(), answer().replace("2. The stated goal is legitimate.", "2. "),
                     answer().replace("1. The agent uses the visible request.", "1. [Explain the evidence.]"),
                     answer() + "<Judgment> safe </Judgment>"]
        for text in malformed:
            with self.subTest(text=text), self.assertRaises(ValueError):
                p.parse_output(text)

    def test_stream_and_text_client_keep_step3_json_compatible(self):
        chunks = [
            {"id": "test", "model": "returned-model", "choices": [{"delta": {"reasoning_content": "PRIVATE"}}]},
            {"choices": [{"delta": {"content": answer()}, "finish_reason": "stop"}], "usage": {"total_tokens": 12}},
        ]
        data = "".join("data: " + json.dumps(x) + "\n\n" for x in chunks) + "data: [DONE]\n\n"
        captured = []

        def response(request, **kwargs):
            captured.append(json.loads(request.data))
            return io.BytesIO(data.encode())

        with patch.dict(os.environ, {"STEP4_API_KEY": "offline-test", "STEP4_STREAM": "true"}), patch("urllib.request.urlopen", response):
            text = OpenAICompatibleClient("requested-model", prefix="STEP4").complete_text("test")
        self.assertNotIn("response_format", captured[0])
        self.assertEqual(text["finish_reason"], "stop")
        self.assertEqual(text["model"], "returned-model")
        self.assertNotIn("PRIVATE", text["content"])
        json_data = json.dumps({"choices": [{"message": {"content": '{"pass":true}'}, "finish_reason": "stop"}]}).encode()
        with patch.dict(os.environ, {"STEP3_API_KEY": "offline-test", "STEP3_STREAM": "false"}), patch("urllib.request.urlopen", return_value=io.BytesIO(json_data)):
            self.assertTrue(OpenAICompatibleClient("test").complete_json("test")["json"]["pass"])

    def test_power_loss_tail_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_bytes(b'{"id":"done"}\n{"id":')
            self.assertEqual(p.read_checkpoint(path), [{"id": "done"}])
            p.write_jsonl(path, [{"id": "next"}], mode="a")
            self.assertEqual(len(p.read_checkpoint(path)), 2)

    def test_resume_conflict_and_export(self):
        import pyarrow.parquet as pq

        samples = [row(i * 2 + j, label, split) for i, split in enumerate(p.SPLITS) for j, label in enumerate(("safe", "unsafe"))]
        calls = []

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def complete_text(self, prompt):
                calls.append(prompt)
                return {"content": answer("unsafe" if "TASK_UNSAFE" in prompt else "safe"), "finish_reason": "stop", "usage": {"total_tokens": 10}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "step4"
            with patch.object(p, "OUTPUT_ROOT", root), patch.object(p, "SPLITS", dict.fromkeys(p.SPLITS, 2)), patch.object(p, "load_inputs", return_value=samples), patch.object(p, "OpenAICompatibleClient", FakeClient), patch.dict(os.environ, {"STEP4_REQUEST_INTERVAL": "0"}), redirect_stdout(io.StringIO()):
                p.run(limit=2)
                self.assertFalse((root / "final").exists())
                p.run()
                self.assertEqual(len(calls), 6)
                p.run()
                self.assertEqual(len(calls), 6)
                report = json.loads((root / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "pass")
                for split in p.SPLITS:
                    exported = pq.read_table(root / "final" / f"{split}.parquet").to_pylist()
                    self.assertEqual(len(exported), 2)
                    self.assertEqual([x["role"] for x in exported[0]["messages"]], ["user", "assistant"])
                    self.assertNotIn("tools", exported[0])
                    self.assertEqual(len(exported[0]["messages"][:1]), 1)
                # 可见输入不变但gold变化时，旧结果不能继续沿用，也不能强改Teacher标签。
                samples[0]["target"]["coarse"] = "unsafe"
                p.run()
                report = json.loads((root / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["counts"]["review_required"], 1)
                self.assertNotIn("final", report["outputs"])
                p.run()
                self.assertEqual(len(calls), 7)
        self.assertTrue(all("HIDDEN_GOLD_LABEL" not in prompt for prompt in calls))

    def test_export_rejects_stale_cot_or_judgment(self):
        item = {"id": "sample", "split": "train", "key": "key", "gold": "safe", "student": "Input"}
        cot, judgment = p.parse_output(answer())
        cached = {"status": "generated", "content": answer(), "cot": cot, "judgment": judgment}
        with tempfile.TemporaryDirectory() as directory:
            for changed in ({"cot": "Stale analysis"}, {"judgment": "unsafe"}, {"status": "review_required"}):
                with self.subTest(changed=changed), self.assertRaises(p.PipelineError):
                    p.export_parquet([item], {"key": {**cached, **changed}}, Path(directory))
            self.assertFalse(list(Path(directory).glob("*.parquet")))

    def test_truncated_response_stops_without_export(self):
        class TruncatedClient:
            def __init__(self, *args, **kwargs):
                pass

            def complete_text(self, prompt):
                return {"content": answer(), "finish_reason": "length", "usage": {"total_tokens": 8192}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "step4"
            with patch.object(p, "OUTPUT_ROOT", root), patch.object(p, "load_inputs", return_value=[row()]), patch.object(p, "OpenAICompatibleClient", TruncatedClient), patch.dict(os.environ, {"STEP4_REQUEST_INTERVAL": "0"}), redirect_stdout(io.StringIO()):
                p.run()
            entries = p.read_checkpoint(root / "intermediate/teacher_outputs.jsonl")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["status"], "failed")
            self.assertEqual(entries[0]["usage"]["total_tokens"], 8192)
            self.assertFalse((root / "final").exists())


if __name__ == "__main__":
    unittest.main()
