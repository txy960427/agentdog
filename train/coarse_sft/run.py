"""单一入口：真实LLaMA-Factory长度/掩码预检，通过后试训或正式SFT。"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLITS = {"train": 1000, "validation": 250, "test": 250}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = sorted(values)
    return {"min": values[0], "p50": values[math.ceil(len(values) * .50) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1], "max": values[-1]}


def choose_cutoff(lengths, requested=0):
    if requested < 0:
        raise ValueError("cutoff_len不能为负数；0表示自动选择")
    recommended = math.ceil(max(lengths) / 256) * 256
    return requested or recommended, recommended


def validate_encoding(row, encoded, source, target, official_prefix):
    """检查实际训练输入和labels；分析正文属于普通assistant输出，必须全部学习。"""
    if source != official_prefix:
        at = next((i for i, (a, b) in enumerate(zip(source, official_prefix)) if a != b), min(len(source), len(official_prefix)))
        raise ValueError(f"{row['id']}: LLaMA-Factory与官方非思考推理前缀不一致，首个差异token={at}")
    if encoded["input_ids"] != source + target or encoded["labels"] != [-100] * len(source) + target:
        raise ValueError(f"{row['id']}: 输入被截断或loss掩码错误")
    answer = row["messages"][1]["content"]
    match = re.fullmatch(r"<Analysis>\s*(.+?)\s*</Analysis>\s*<Judgment>\s*(safe|unsafe)\s*</Judgment>", answer, re.S)
    if not match or match[2] != row["gold"]:
        raise ValueError(f"{row['id']}: Analysis/Judgment格式或标签错误")


def preflight(config, run_dir, requested):
    import pyarrow.parquet as pq
    from llamafactory.data.converter import get_dataset_converter
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from lf_entry import setup_template

    dataset_dir = ROOT / config["dataset_dir"]
    manifest = json.loads((dataset_dir.parent / "report.json").read_text(encoding="utf-8"))
    if manifest["status"] != "pass":
        raise ValueError("Step 4数据尚未验收")
    data_keys = {f.name for f in fields(DataArguments)}
    data_args = DataArguments(**{k: v for k, v in config.items() if k in data_keys})
    data_args.dataset_dir = str(dataset_dir)
    # 先取得未截断的真实训练编码，不能用已经截断的序列反推最大长度。
    data_args.cutoff_len = 2**30
    module = load_tokenizer(ModelArguments(model_name_or_path=config["model_name_or_path"], trust_remote_code=False, use_fast_tokenizer=True))
    tokenizer = module["tokenizer"]
    official_template = tokenizer.chat_template
    if not official_template:
        raise ValueError("底座缺少官方chat template")
    setup_template()
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoder = SupervisedDatasetProcessor(template, tokenizer, module["processor"], data_args)
    registry = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    details, sample = [], None
    for split, expected in SPLITS.items():
        name = f"agentdog_{split}"
        path = dataset_dir / registry[name]["file_name"]
        if sha256(path) != manifest["outputs"]["final"][split]["sha256"]:
            raise ValueError(f"{split}数据与验收hash不符")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != expected:
            raise ValueError(f"{split}条数应为{expected}，实际{len(rows)}")
        attr = get_dataset_list([name], str(dataset_dir))[0]
        converter = get_dataset_converter(attr.formatting, attr, data_args)
        for row in rows:
            converted = converter(row)
            if converted["_prompt"] + converted["_response"] != row["messages"] or converted["_system"] or converted["_tools"]:
                raise ValueError(f"{row['id']}: 数据注册改变了原始messages或增加了system/tools")
            processed = template.mm_plugin.process_messages(row["messages"], [], [], [], module["processor"])
            pairs = template.encode_multiturn(tokenizer, processed, "", "")
            if len(pairs) != 1:
                raise ValueError(f"{row['id']}: 必须保持一条user与一条assistant")
            source, target = pairs[0]
            encoded = encoder._encode_data_example(converted["_prompt"], converted["_response"], "", "", [], [], [])
            # 此私有API返回(input_ids, labels)，与框架训练时的同一路径完全一致。
            encoded = {"input_ids": encoded[0], "labels": encoded[1]}
            official = tokenizer.apply_chat_template(row["messages"][:1], chat_template=official_template,
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            validate_encoding(row, encoded, source, target, official)
            expected_target = tokenizer.encode(row["messages"][1]["content"] + "<|im_end|>\n", add_special_tokens=False)
            if target != expected_target:
                raise ValueError(f"{row['id']}: assistant分析/判断/EOS被模板改写")
            details.append({"id": row["id"], "split": split, "prompt_tokens": len(source),
                            "answer_tokens": len(target), "total_tokens": len(source) + len(target)})
            if sample is None:
                sample = f"id: {row['id']}\n\n=== 完整模型输入 ===\n{tokenizer.decode(source + target, skip_special_tokens=False)}\n\n=== 参与loss的内容 ===\n{tokenizer.decode(target, skip_special_tokens=False)}\n"
        print(f"已核对{split}: {len(rows)}条，映射/模板/完整输出/loss掩码通过。", flush=True)
    lengths = [r["total_tokens"] for r in details]
    selected, recommended = choose_cutoff(lengths, requested)
    report = {"status": "pass" if max(lengths) <= selected else "failed", "records": len(details),
        "model": config["model_name_or_path"], "selected_cutoff_len": selected, "recommended_cutoff_len": recommended,
        "selection_rule": "全部1500条完整训练序列的最大长度向上取整到256；只看长度，不使用test分数调参",
        "splits": {s: {"records": n, "prompt_tokens": summary([r['prompt_tokens'] for r in details if r['split'] == s]),
            "answer_tokens": summary([r['answer_tokens'] for r in details if r['split'] == s]),
            "total_tokens": summary([r['total_tokens'] for r in details if r['split'] == s])} for s, n in SPLITS.items()},
        "over_budget": [r for r in details if r["total_tokens"] > selected],
        "candidate_overflow_counts": {str(n): sum(v > n for v in lengths) for n in sorted({2048, 4096, 8192, 12288, 16384, 32768, recommended, selected})},
        "dataset_sha256": {s: sha256(dataset_dir / registry[f'agentdog_{s}']['file_name']) for s in SPLITS},
        "registry_sha256": sha256(dataset_dir / "dataset_info.json"),
        "official_chat_template_sha256": hashlib.sha256(str(official_template).encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "framework_source_sha256": sha256(ROOT / "LLaMA-Factory/UPSTREAM.md"),
        "template_adapter_sha256": sha256(HERE / "lf_entry.py"),
        "versions": {name: importlib.metadata.version(name) for name in ("llamafactory", "transformers", "torch", "datasets")},
        "checks": "1500条映射无变化、官方推理前缀一致、assistant完整、prompt掩码、数据hash均通过；未加载模型权重，未测试显存"}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (run_dir / "lengths.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details), encoding="utf-8", newline="\n")
    (run_dir / "sample.txt").write_text(sample, encoding="utf-8", newline="\n")
    for split, info in report["splits"].items():
        print(f"{split:10} n={info['records']:4} total_tokens={info['total_tokens']}")
    print(f"建议cutoff_len={recommended}；本次采用{selected}；超过预算{len(report['over_budget'])}条。", flush=True)
    if report["over_budget"]:
        raise ValueError("指定cutoff_len不足，已停止；不会截断或删除样本")
    return report


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "train"), nargs="?", default="check")
    parser.add_argument("--model", help="底座模型ID或本地目录，默认读取train.yaml")
    parser.add_argument("--cutoff-len", type=int, default=0, help="0=自动选择；指定值不足时停止")
    parser.add_argument("--resume-from-checkpoint", help="正式训练时恢复的checkpoint目录")
    args = parser.parse_args()
    if args.cutoff_len < 0 or args.resume_from_checkpoint and args.mode != "train":
        parser.error("cutoff-len必须>=0；resume仅用于train")
    os.chdir(ROOT)
    os.environ["USE_V1"] = "0"
    config = yaml.safe_load((HERE / "train.yaml").read_text(encoding="utf-8"))
    config["model_name_or_path"] = args.model or config["model_name_or_path"]
    if Path(config["model_name_or_path"]).expanduser().exists():
        config["model_name_or_path"] = str(Path(config["model_name_or_path"]).expanduser().resolve())
    run_dir = HERE / "runs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.mode}"
    run_dir.mkdir(parents=True)
    print(f"Run: {run_dir}", flush=True)
    # 预检必须与下面的训练配置一致，不能临时绕过mask或自动切分检查。
    if config["template"] != "qwen3_5" or config["enable_thinking"] is not False or config["packing"] or config["train_on_prompt"] or config["mask_history"]:
        raise ValueError("当前入口固定为qwen3_5非思考模式、单轮assistant loss、不开packing")
    try:
        report = preflight(config, run_dir, args.cutoff_len)
    except Exception as error:
        path = run_dir / "report.json"
        failed = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        failed.update(status="failed", phase="preflight", error=str(error), model=config["model_name_or_path"])
        path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        raise
    config["cutoff_len"] = report["selected_cutoff_len"]
    config["output_dir"] = str(run_dir / "model")
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = str(Path(args.resume_from_checkpoint).expanduser().resolve())
    if args.mode == "smoke":
        config.update(max_steps=5, gradient_accumulation_steps=1, eval_strategy="steps", save_strategy="steps",
                      eval_steps=5, save_steps=5, save_total_limit=1)
    config_path = run_dir / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    if args.mode == "check":
        print("仅预检，未加载模型权重、未启动训练。")
        return
    report["training_status"] = "running"
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    # 使用相同Python调用官方run_exp；子进程也先安装相同的模板适配，checkpoint保留优化器。
    with (run_dir / "train.log").open("w", encoding="utf-8", newline="\n") as log:
        try:
            child = subprocess.Popen([sys.executable, "-u", str(HERE / "lf_entry.py"), str(config_path)], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        except OSError as error:
            report.update(training_status="launch_failed", error=str(error))
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            raise
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            code = 130
        finally:
            report["training_status"] = "completed" if child.returncode == 0 else "failed_or_interrupted"
            report["training_exit_code"] = child.returncode
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("请先在当前Python环境安装：python -m pip install -e ./LLaMA-Factory", file=sys.stderr)
        raise SystemExit(1)
