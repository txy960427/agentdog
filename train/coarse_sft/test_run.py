"""不加载GPU/模型，测试真实上游转换器、编码器算法及本项目长度/掩码约束。"""

import ast
import importlib.util
import json
import types
import unittest
from unittest.mock import patch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sft_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
LF = run.ROOT / "LLaMA-Factory/src/llamafactory"


def upstream_node(file, class_name, method=None):
    """直接编译固定版本源码中的目标定义，隔离其非测试所需的GPU依赖。"""
    node = next(n for n in ast.parse((LF / file).read_text(encoding="utf-8")).body if getattr(n, "name", None) == class_name) if class_name else None
    if method:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec")


class TrainingTests(unittest.TestCase):
    def test_cutoff_rounds_up_and_includes_boundary(self):
        self.assertEqual(run.choose_cutoff([4096, 100]), (4096, 4096))
        self.assertEqual(run.choose_cutoff([4097]), (4352, 4352))
        self.assertEqual(run.choose_cutoff([4097], 8192), (8192, 4352))
        self.assertEqual(run.choose_cutoff([4097], 4096), (4096, 4352))
        with self.assertRaises(ValueError):
            run.choose_cutoff([100], -1)

    def test_real_sharegpt_converter_preserves_all_1500_messages(self):
        namespace = {"dataclass": dataclass, "Any": Any, "Literal": Literal}
        exec(upstream_node("data/parser.py", "DatasetAttr"), namespace)
        roles = {r.upper(): types.SimpleNamespace(value=r) for r in ("user", "assistant", "observation", "function", "system")}
        namespace.update(Role=types.SimpleNamespace(**roles), logger=types.SimpleNamespace(warning_rank0=lambda *a: None))
        exec(upstream_node("data/converter.py", "SharegptDatasetConverter", "__call__"), namespace)
        data = run.ROOT / "data/1.5/coarse_sft/step4/final"
        registry = json.loads((data / "dataset_info.json").read_text(encoding="utf-8"))
        seen = 0
        for split, count in run.SPLITS.items():
            entry = registry[f"agentdog_{split}"]
            attr = namespace["DatasetAttr"]("file", entry["file_name"])
            attr.join(entry)
            converter = types.SimpleNamespace(dataset_attr=attr)
            rows = pq.read_table(data / entry["file_name"]).to_pylist()
            self.assertEqual(len(rows), count)
            for row in rows:
                before = json.dumps(row, ensure_ascii=False)
                result = namespace["__call__"](converter, row)
                self.assertEqual(result["_prompt"] + result["_response"], row["messages"])
                self.assertEqual(result["_system"], "")
                self.assertEqual(result["_tools"], "")
                self.assertEqual(json.dumps(row, ensure_ascii=False), before)
                seen += 1
        self.assertEqual(seen, 1500)

    def test_actual_sft_encoder_detects_mask_changes_and_truncation(self):
        tree = ast.parse((LF / "data/processor/processor_utils.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "infer_seqlen")
        namespace = {"Optional": Optional, "Any": Any, "IGNORE_INDEX": -100}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "infer_seqlen", "exec"), namespace)
        exec(upstream_node("data/processor/supervised.py", "SupervisedDatasetProcessor", "_encode_data_example"), namespace)
        source, target = [10, 11, 12], [20, 21, 22]
        mm = types.SimpleNamespace(process_messages=lambda *a: a[0], process_token_ids=lambda *a: ([], []))
        template = types.SimpleNamespace(mm_plugin=mm, efficient_eos=False, preserve_thinking=False,
            encode_multiturn=lambda *a: [(source[:], target[:])])
        args = types.SimpleNamespace(mask_history=False, train_on_prompt=False, cutoff_len=6)
        encoder = types.SimpleNamespace(template=template, tokenizer=None, processor=None, data_args=args)
        row = {"id": "example", "gold": "safe", "messages": [{"role": "user", "content": "Request"},
            {"role": "assistant", "content": "<Analysis>\n1. Evidence.\n2. Intent.\n3. Effects.\n4. Safe.\n</Analysis>\n<Judgment>safe</Judgment>"}]}
        encode = lambda: dict(zip(("input_ids", "labels"), namespace["_encode_data_example"](encoder, [], [], "", "", [], [], [])))
        run.validate_encoding(row, encode(), source, target, source)
        args.cutoff_len = 5
        with self.assertRaises(ValueError):
            run.validate_encoding(row, encode(), source, target, source)
        args.cutoff_len, args.train_on_prompt = 6, True
        with self.assertRaises(ValueError):
            run.validate_encoding(row, encode(), source, target, source)
        args.train_on_prompt = False
        with self.assertRaises(ValueError):
            run.validate_encoding(row, encode(), source, target, [99])

    def test_template_adapter_matches_official_strip_without_changing_data(self):
        spec = importlib.util.spec_from_file_location("lf_entry", HERE / "lf_entry.py")
        entry = importlib.util.module_from_spec(spec)

        class Formatter:
            def __init__(self, slots):
                self.slots = slots

            def apply(self, **kwargs):
                return [self.slots[0].replace("{{content}}", kwargs["content"])]

        original = types.SimpleNamespace(format_user=Formatter(["USER:{{content}}:END"]), replace_jinja_template=True)
        templates = {"qwen3_5": original}
        modules = {"llamafactory.data.formatter": types.SimpleNamespace(StringFormatter=Formatter),
                   "llamafactory.data.template": types.SimpleNamespace(TEMPLATES=templates)}
        content = "\nActual prompt.\n"
        with patch.dict("sys.modules", modules):
            spec.loader.exec_module(entry)
            entry.setup_template()
        self.assertEqual(templates["qwen3_5"].format_user.apply(content=content), ["USER:Actual prompt.:END"])
        self.assertEqual(original.format_user.apply(content=content), ["USER:\nActual prompt.\n:END"])
        self.assertFalse(templates["qwen3_5"].replace_jinja_template)

    def test_yaml_keeps_fixed_splits_and_answer_loss(self):
        config = yaml.safe_load((HERE / "train.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["dataset"], "agentdog_train")
        self.assertEqual(config["eval_dataset"], "agentdog_validation")
        self.assertEqual(config["val_size"], 0)
        for key in ("train_on_prompt", "packing", "enable_thinking", "mask_history"):
            self.assertIs(config[key], False)
        self.assertEqual(config["save_strategy"], config["eval_strategy"])
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["metric_for_best_model"], "eval_loss")
        self.assertEqual(config["learning_rate"], 1e-5)
        self.assertEqual(config["finetuning_type"], "full")


if __name__ == "__main__":
    unittest.main()
