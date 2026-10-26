"""Step 2内部流程：清洗带工具unsafe轨迹，均衡抽样并写出三份切分。"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..step1.overlap import assign_groups, benchmark_overlap
from ..step1.pipeline import write_json, write_jsonl
from ..step1.sampling import leakage_report
from ..step1.source import PipelineError, sha256_file
from .sampling import build_quotas, sampling_report, stratified_sample, validate_acceptance
from .source import SCRIPT_VERSION, load_and_parse_source

PROJECT_ROOT = Path(__file__).resolve().parents[7]
CONFIG_PATH = Path(__file__).with_name("config.json")


def intermediate_row(item: dict[str, Any]) -> dict[str, Any]:
    """保留复盘所需的最小构建信息，不复制原始大字段。"""
    features = item["features"]
    return {
        "record": item["canonical"],
        "build": {
            "raw_hash": item["raw_hash"],
            "trajectory_hash": item["trajectory_hash"],
            "tool_audit": item["tool_audit"],
            "benchmark_overlap": item["benchmark_overlap"],
            "features": {
                key: features[key]
                for key in (
                    "token_count_proxy",
                    "event_count",
                    "agent_action_count",
                    "tool_action_count",
                    "available_tool_count",
                    "length_quartile",
                    "has_environment_error",
                    "has_schema_warning",
                    "has_duplicate_tool_name",
                    "has_data_irregularity",
                    "has_trailing_complete",
                    "first_tool_name",
                )
            },
        },
    }


def run() -> None:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot load {CONFIG_PATH}: {exc}") from exc
    args = SimpleNamespace(
        input=(PROJECT_ROOT / config["input"]).resolve(),
        output_root=(PROJECT_ROOT / config["output_root"]).resolve(),
        benchmark_root=(PROJECT_ROOT / config["benchmark_root"]).resolve(),
        seed=int(config["seed"]),
        group_near_threshold=float(config["group_near_threshold"]),
        benchmark_near_threshold=float(config["benchmark_near_threshold"]),
    )
    quotas = build_quotas(config["base_quota"], config["extra_risk_sources"])

    parsed, quarantine, source_report = load_and_parse_source(args.input)
    grouping = assign_groups(parsed, args.group_near_threshold)
    excluded, benchmark_report = benchmark_overlap(
        parsed, args.benchmark_root, args.benchmark_near_threshold
    )
    for item in parsed:
        item["benchmark_overlap"] = item["canonical"]["id"] in excluded
    eligible = [item for item in parsed if not item["benchmark_overlap"]]

    splits = stratified_sample(eligible, quotas, args.seed)
    sampling = sampling_report(splits, eligible, quotas)
    leakage = leakage_report(parsed, splits, benchmark_report)
    leakage["grouping"] = grouping
    validate_acceptance(
        parsed, quarantine, source_report, splits, quotas, leakage
    )

    stage_root = args.output_root / "step2"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    intermediate_dir = stage_root / "intermediate"
    final_dir = stage_root / "final"
    intermediate_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    outputs: list[tuple[Path, int]] = []
    cleaned_path = intermediate_dir / "cleaned.jsonl"
    outputs.append(
        (cleaned_path, write_jsonl(cleaned_path, (intermediate_row(item) for item in parsed)))
    )
    for split in ("train", "validation", "test"):
        path = final_dir / f"{split}.jsonl"
        rows = sorted(splits[split], key=lambda item: item["canonical"]["id"])
        outputs.append(
            (path, write_jsonl(path, (item["canonical"] for item in rows)))
        )

    report = {
        "stage": "agentdog-1.5/coarse-grained/sft/step2-tool-unsafe",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "input": str(args.input),
            "input_sha256": source_report["input_sha256"],
            "benchmark_root": str(args.benchmark_root),
            "seed": args.seed,
            "group_near_threshold": args.group_near_threshold,
            "benchmark_near_threshold": args.benchmark_near_threshold,
            "quotas": quotas,
        },
        "source": source_report,
        "quarantine": quarantine,
        "grouping": grouping,
        "benchmark_overlap": benchmark_report,
        "sampling": sampling,
        "leakage": leakage,
        "outputs": [
            {
                "path": path.relative_to(stage_root).as_posix(),
                "records": records,
                "sha256": sha256_file(path),
            }
            for path, records in outputs
        ],
    }
    write_json(stage_root / "report.json", report)

    print(f"Step 2 complete: {stage_root}")
    print(
        f"source={source_report['physical_records']} physical -> "
        f"{source_report['unique_records']} unique; "
        f"benchmark_excluded={benchmark_report['matched_source_records']}; "
        f"quarantine={len(quarantine)}"
    )
    for split in ("train", "validation", "test"):
        risks = Counter(item["canonical"]["target"]["risk_source"] for item in splits[split])
        print(f"{split}: total={len(splits[split])} risk_sources={dict(sorted(risks.items()))}")
