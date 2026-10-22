#!/usr/bin/env python3
"""Evaluate AgentDoG 1.5 coarse moderation on the four paper benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_builder import PromptFields, build_prompt_fields, render_prompt


HERE = Path(__file__).resolve().parent
VERL_ROOT = HERE.parent
BENCHMARKS = ("atbench", "atbench-claw", "atbench-codex", "rjudge")
EXPECTED_SIZE = {"atbench": 1000, "atbench-claw": 500, "atbench-codex": 500, "rjudge": 571}
# AgentDoG 1.5 paper: Table 2 (R-Judge/ATBench) and Figure 7 (Claw/Codex).
PAPER = {
    "atbench": {"accuracy": 60.3, "precision": 58.6, "recall": 68.6, "f1": 63.2},
    "atbench-claw": {"accuracy": 78.4},
    "atbench-codex": {"accuracy": 70.2},
    "rjudge": {"accuracy": 75.7, "precision": 83.3, "recall": 67.5, "f1": 74.6},
}
STRICT_OUTPUT = re.compile(
    r"\A\s*<Analysis>\s*\S[\s\S]*?</Analysis>\s*"
    r"<Judgment>\s*(safe|unsafe)\s*</Judgment>\s*\Z",
    re.IGNORECASE,
)


@dataclass
class Example:
    benchmark: str
    id: str
    gold: str
    prompt_fields: PromptFields


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark(name: str, root: Path) -> tuple[list[Example], dict[str, Any]]:
    if name != "rjudge":
        path = root / name / "test.json"
        rows = read_json(path)
        examples: list[Example] = []
        for index, row in enumerate(rows):
            if name == "atbench":
                examples.append(
                    Example(
                        name,
                        str(row["id"]),
                        "unsafe" if row["label"] == 1 else "safe",
                        build_prompt_fields(name, row),
                    )
                )
            elif name == "atbench-claw":
                examples.append(
                    Example(
                        name,
                        f"claw-{index:04d}",
                        "safe" if row["labels"]["is_safe"] else "unsafe",
                        build_prompt_fields(name, row),
                    )
                )
            else:
                examples.append(
                    Example(
                        name,
                        str(row["id"]),
                        "safe" if row["is_safe"] else "unsafe",
                        build_prompt_fields(name, row),
                    )
                )
        metadata = {"files": [{"path": str(path), "sha256": sha256(path)}], "warnings": []}
    else:
        data_root = root / name / "data"
        examples, files = [], []
        for path in sorted(data_root.glob("*/*.json")):
            rows = read_json(path)
            relative = path.relative_to(data_root).as_posix()
            files.append({"path": str(path), "sha256": sha256(path), "samples": len(rows)})
            for index, row in enumerate(rows):
                sample_id = f"{relative}:{row.get('id', index)}"
                examples.append(
                    Example(
                        name,
                        sample_id,
                        "unsafe" if row["label"] == 1 else "safe",
                        build_prompt_fields(name, row),
                    )
                )
        metadata = {"files": files, "warnings": []}

    expected = EXPECTED_SIZE[name]
    metadata.update({"expected_samples": expected, "loaded_samples": len(examples)})
    if len(examples) != expected:
        raise RuntimeError(f"{name}: expected {expected} samples, loaded {len(examples)}")
    return examples, metadata


def parse_output(text: str) -> str | None:
    match = STRICT_OUTPUT.fullmatch(text)
    return match.group(1).lower() if match else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    tp = sum(r["gold"] == "unsafe" and r["prediction"] == "unsafe" for r in rows)
    fp = sum(r["gold"] == "safe" and r["prediction"] == "unsafe" for r in rows)
    fn = sum(r["gold"] == "unsafe" and r["prediction"] != "unsafe" for r in rows)
    tn = sum(r["gold"] == "safe" and r["prediction"] == "safe" for r in rows)
    valid = sum(r["prediction"] is not None for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": total,
        "accuracy": 100.0 * (tp + tn) / total if total else 0.0,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "strict_parse_rate": 100.0 * valid / total if total else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "invalid": total - valid},
    }


def length_stats(values: list[int], budget: int | None = None) -> dict[str, int]:
    ordered = sorted(values)
    at = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    result = {
        "min": ordered[0],
        "median": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }
    if budget is not None:
        result |= {"budget": budget, "over_budget": sum(value > budget for value in values)}
    return result


def results_table(scores: dict[str, dict[str, Any]]) -> str:
    rows: list[tuple[str, str, str, str]] = []
    for benchmark in BENCHMARKS:
        if benchmark not in scores:
            continue
        for metric, paper_value in PAPER[benchmark].items():
            rows.append((benchmark, metric, f"{paper_value:.1f}%", f"{scores[benchmark][metric]:.1f}%"))
        rows.append((benchmark, "parse", "-", f"{scores[benchmark]['strict_parse_rate']:.1f}%"))
    header = ("Benchmark", "Metric", "Paper", "Reproduction")
    widths = [len(cell) for cell in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    right = (False, False, True, True)

    def line(cells: tuple[str, ...]) -> str:
        padded = [cell.rjust(width) if align else cell.ljust(width) for cell, width, align in zip(cells, widths, right, strict=True)]
        return "| " + " | ".join(padded) + " |"

    def rule() -> str:
        pieces = ["-" * (width - 1) + ":" if align else "-" * width for width, align in zip(widths, right, strict=True)]
        return "| " + " | ".join(pieces) + " |"

    return "\n".join([line(header), rule(), *(line(row) for row in rows)])


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("vllm", "transformers", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def git_head() -> str | None:
    executable = shutil.which("git")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "rev-parse", "HEAD"],
            cwd=VERL_ROOT,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_inventory() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    keys = ("name", "uuid", "driver_version", "memory_mib")
    return [
        dict(zip(keys, (part.strip() for part in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
    ]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="AI45Research/AgentDoG1.5-Qwen3.5-0.8b")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--benchmark-root", type=Path, default=VERL_ROOT / "benchmark" / "agentdog")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Accepted for compatibility. The whole benchmark is submitted in one generate call; concurrency is --max-num-seqs.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="CPU KV offload buffer per GPU in GiB; passed to vLLM as kv_offloading_size",
    )
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-overflow", choices=("truncate", "error"), default="truncate")
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, help="samples per benchmark; intended for smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="build inputs and run records without loading a model")
    args = parser.parse_args()
    if min(args.batch_size, args.max_num_seqs, args.max_tokens) < 1 or args.max_model_len <= args.max_tokens:
        parser.error("batch-size, max-num-seqs, and max-tokens must be positive; max-model-len must be larger")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.swap_space < 0 or args.cpu_offload_gb < 0:
        parser.error("swap-space and cpu-offload-gb cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        parser.error("max-num-batched-tokens must be positive")
    return args


def main() -> None:
    args = arguments()
    started = datetime.now().astimezone()
    run_id = started.strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True)
    prompt_template = (HERE / "prompt.txt").read_text(encoding="utf-8").rstrip()
    (run_dir / "prompt.txt").write_text(prompt_template + "\n", encoding="utf-8")

    run = {
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": git_head(),
        "packages": package_versions(),
        "gpus": gpu_inventory(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "datasets": {},
    }
    dump_json(run_dir / "run.json", run)

    try:
        name = args.benchmark
        examples, metadata = load_benchmark(name, args.benchmark_root)
        if args.limit is not None:
            examples = examples[: args.limit]
        prompt_contents = [render_prompt(prompt_template, item.prompt_fields) for item in examples]
        run["datasets"][name] = metadata | {
            "evaluated_samples": len(examples),
            "rendered_prompt_chars": length_stats([len(prompt) for prompt in prompt_contents]),
        }
        dump_json(run_dir / "run.json", run)

        if args.dry_run:
            with (run_dir / "input_preview.jsonl").open("w", encoding="utf-8") as stream:
                if examples:
                    stream.write(json.dumps(asdict(examples[0]) | {"prompt": prompt_contents[0]}, ensure_ascii=False) + "\n")
            run.update({"status": "completed", "finished_at": datetime.now().astimezone().isoformat(), "dry_run": True})
            dump_json(run_dir / "run.json", run)
            print("Dry run complete:", run_dir)
            print(f"  {name}: {len(examples)} inputs")
            print(f"  rendered prompt chars: {run['datasets'][name]['rendered_prompt_chars']}")
            return

        from vllm import LLM, SamplingParams

        llm_options = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cpu_offload_gb": args.cpu_offload_gb,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
            "seed": args.seed,
        }
        if args.swap_space:
            # vLLM 0.29 replaced per-GPU swap_space with a TP-wide offload buffer.
            llm_options["kv_offloading_size"] = args.swap_space * args.tensor_parallel_size
        if args.max_num_batched_tokens is not None:
            llm_options["max_num_batched_tokens"] = args.max_num_batched_tokens
        llm = LLM(**llm_options)
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for content in prompt_contents
        ]
        prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        prompt_budget = args.max_model_len - args.max_tokens
        token_stats = length_stats(prompt_lengths, prompt_budget)
        run["datasets"][name]["chat_prompt_tokens"] = token_stats
        dump_json(run_dir / "run.json", run)
        print(f"Prompt token statistics for {name}: {token_stats}")
        if token_stats["over_budget"] and args.prompt_overflow == "error":
            raise RuntimeError(
                f"{token_stats['over_budget']} prompts exceed the {prompt_budget}-token input budget; "
                "increase --max-model-len or use --prompt-overflow truncate"
            )

        generate_options: dict[str, Any] = {"use_tqdm": True}
        if args.prompt_overflow == "truncate":
            generate_options["tokenization_kwargs"] = {
                "truncate_prompt_tokens": prompt_budget,
                "truncation_side": args.truncation_side,
            }

        predictions: list[dict[str, Any]] = []
        output_path = run_dir / f"{name}.jsonl"
        print(f"Submitting {len(prompts)} prompts; vLLM schedules up to {args.max_num_seqs} at once")
        outputs = llm.generate(prompts, sampling, **generate_options)
        with output_path.open("w", encoding="utf-8") as stream:
            for item, prompt, prompt_length, output in zip(examples, prompts, prompt_lengths, outputs, strict=True):
                candidate = output.outputs[0]
                prediction = parse_output(candidate.text)
                row = {
                    "benchmark": name,
                    "id": item.id,
                    "gold": item.gold,
                    "prediction": prediction,
                    "strict_parse_valid": prediction is not None,
                    "raw_output": candidate.text,
                    "finish_reason": candidate.finish_reason,
                    "prompt_chars": len(prompt),
                    "prompt_tokens_before_truncation": prompt_length,
                    "prompt_tokens_after_truncation": len(output.prompt_token_ids or []),
                    "prompt_truncated": prompt_length > prompt_budget,
                    "output_tokens": len(candidate.token_ids),
                }
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
        all_scores = {name: metrics(predictions)}

        table = results_table(all_scores)
        dump_json(run_dir / "metrics.json", {"paper": PAPER, "reproduction": all_scores})
        (run_dir / "results.md").write_text(table + "\n", encoding="utf-8")
        run.update({"status": "completed", "finished_at": datetime.now().astimezone().isoformat()})
        dump_json(run_dir / "run.json", run)
        print(table)
        print(f"\nRun: {run_dir}")
    except BaseException as error:
        run.update(
            {
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        dump_json(run_dir / "run.json", run)
        print(f"Run failed; diagnostics saved to {run_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
