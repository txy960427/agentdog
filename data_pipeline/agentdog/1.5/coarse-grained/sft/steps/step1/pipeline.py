"""Step 1内部流程：读取固定配置，完成构建并写出结果。"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .overlap import assign_groups, benchmark_overlap
from .sampling import leakage_report, sampling_report, stratified_sample, validate_acceptance
from .source import SCRIPT_VERSION, PipelineError, load_and_parse_source, sha256_file, stable_json

PROJECT_ROOT = Path(__file__).resolve().parents[7]
CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_QUOTAS = {
    "train": {"safe": 100, "unsafe": 100},
    "validation": {"safe": 12, "unsafe": 13},
    "test": {"safe": 13, "unsafe": 12},
}

# Step 1内部固定执行：解析去重 -> benchmark排重 -> 分层切分 -> 验收 -> 写出。
def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    content = "".join(stable_json(row) + "\n" for row in materialized)
    atomic_write_text(path, content)
    return len(materialized)


def run() -> None:
    """使用固定配置执行Step 1，不接受命令行参数。"""
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot load {CONFIG_PATH}: {exc}") from exc
    quotas = config["quotas"]
    if quotas != DEFAULT_QUOTAS:
        raise PipelineError(f"Step-1 quotas must remain {DEFAULT_QUOTAS}; received {quotas}")
    args = SimpleNamespace(
        input=(PROJECT_ROOT / config["input"]).resolve(),
        output_root=(PROJECT_ROOT / config["output_root"]).resolve(),
        benchmark_root=(PROJECT_ROOT / config["benchmark_root"]).resolve(),
        seed=int(config["seed"]),
        group_near_threshold=float(config["group_near_threshold"]),
        benchmark_near_threshold=float(config["benchmark_near_threshold"]),
        quotas=quotas,
    )
    parsed, quarantine, dedup = load_and_parse_source(args.input)
    grouping = assign_groups(parsed, args.group_near_threshold)

    excluded, benchmark_report = benchmark_overlap(
        parsed, args.benchmark_root, args.benchmark_near_threshold
    )
    for item in parsed:
        item["benchmark_overlap"] = item["canonical"]["id"] in excluded

    eligible = [item for item in parsed if not item["benchmark_overlap"]]
    for item in parsed:
        item["features"].setdefault("length_quartile", None)
    splits, _selection_reasons = stratified_sample(eligible, args.quotas, args.seed)
    sampling = sampling_report(splits, eligible, args.quotas)
    leakage = leakage_report(parsed, splits, benchmark_report)
    leakage["grouping"] = grouping

    validate_acceptance(
        parsed,
        quarantine,
        dedup,
        sampling,
        leakage,
        strict_counts=True,
    )

    # 每个阶段只维护自己的目录；重跑时先清理该阶段的旧结果。
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_root = args.output_root / "step1"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    intermediate_dir = stage_root / "intermediate"
    final_dir = stage_root / "final"
    intermediate_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    # 清理旧版本管线留在SFT根目录下的分散输出。
    for name in (
        "source",
        "intermediate",
        "canonical",
        "metadata",
        "reports",
        "final",
        "manifest.json",
    ):
        legacy = args.output_root / name
        if legacy.is_dir():
            shutil.rmtree(legacy)
        elif legacy.exists():
            legacy.unlink()

    output_files: list[Path] = []
    output_records: dict[Path, int] = {}
    cleaned_path = intermediate_dir / "cleaned.jsonl"
    output_records[cleaned_path] = write_jsonl(
        cleaned_path, (item["canonical"] for item in parsed)
    )
    output_files.append(cleaned_path)

    for split in ("train", "validation", "test"):
        path = final_dir / f"{split}.jsonl"
        ordered = sorted(splits[split], key=lambda item: item["canonical"]["id"])
        output_records[path] = write_jsonl(
            path, (item["canonical"] for item in ordered)
        )
        output_files.append(path)

    report = {
        "stage": "agentdog-1.5/coarse-grained/sft/step1-binary",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "input": str(args.input),
            "input_sha256": dedup["input_sha256"],
            "benchmark_root": str(args.benchmark_root),
            "seed": args.seed,
            "group_near_threshold": args.group_near_threshold,
            "benchmark_near_threshold": args.benchmark_near_threshold,
            "quotas": args.quotas,
        },
        "dedup": dedup,
        "quarantine": quarantine,
        "grouping": grouping,
        "benchmark_overlap": benchmark_report,
        "sampling": sampling,
        "leakage": leakage,
        "outputs": [
            {
                "path": path.relative_to(stage_root).as_posix(),
                "records": output_records[path],
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
    }
    write_json(stage_root / "report.json", report)

    print(f"Step 1 complete: {stage_root}")
    print(
        f"source={dedup['physical_records']} physical -> {dedup['unique_records']} unique; "
        f"benchmark_excluded={benchmark_report['matched_source_records']}; quarantine={len(quarantine)}"
    )
    for split in ("train", "validation", "test"):
        labels = Counter(
            item["canonical"]["target"]["coarse"] for item in splits[split]
        )
        print(
            f"{split}: total={len(splits[split])} safe={labels['safe']} unsafe={labels['unsafe']}"
        )
