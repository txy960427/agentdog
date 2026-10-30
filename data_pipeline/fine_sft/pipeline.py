"""AgentDoG 1.5 细粒度 SFT 数据管线统一入口。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "data_pipeline"))

from coarse_sft.steps.step0.pipeline import (  # noqa: E402
    assign_splits,
    group_candidates,
    write_jsonl,
)
from coarse_sft.steps.step0.source import (  # noqa: E402
    FAILURE_MODES,
    REAL_WORLD_HARMS,
    RISK_SOURCES,
    PipelineError,
    assign_length_quartiles,
    load_fine,
    sha256_file,
)
from step1 import run_step1  # noqa: E402
from step2 import run_step2  # noqa: E402

SOURCE = PROJECT_ROOT / "data/1.0/source/AgentDoG1.0-Training-Data/AgentDoG-FineGrainedTaxonomy/train.json"
OUTPUT_ROOT = PROJECT_ROOT / "data/1.5/fine_sft/step0"
EXPECTED_TOTAL = 4000
SPLITS = ("train", "validation", "test")
LABELS = {
    "risk_source": RISK_SOURCES,
    "failure_mode": FAILURE_MODES,
    "real_world_harm": REAL_WORLD_HARMS,
}
STRUCTURE_FIELDS = (
    "length_quartile",
    "event_bin",
    "available_tool_count_bin",
    "has_environment_error",
    "has_trailing_complete",
    "action_format",
    "has_schema_warning",
    "has_duplicate_tool_name",
    "has_risk_structure",
)


def value(item: dict[str, Any], field: str) -> str:
    raw = item[field] if field in item else item["features"][field]
    return str(raw).lower() if isinstance(raw, bool) else str(raw)


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(value(item, field) for item in rows).items()))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "unique_groups": len({item["group_id"] for item in rows}),
        "unique_triples": len({tuple(value(item, field) for field in LABELS) for item in rows}),
        "risk_source": distribution(rows, "risk_source"),
        "failure_mode": distribution(rows, "failure_mode"),
        "risk_consequence": distribution(rows, "real_world_harm"),
        "structure": {field: distribution(rows, field) for field in STRUCTURE_FIELDS},
    }


def candidate_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "source_row_id": item["source_row_id"],
        "raw_hash": item["raw_hash"],
        "trajectory_hash": item["trajectory_hash"],
        "group_id": item["group_id"],
        "split": item["split"],
        "risk_source": item["risk_source"],
        "failure_mode": item["failure_mode"],
        "risk_consequence": item["real_world_harm"],
        "features": item["features"],
    }


def selection_row(item: dict[str, Any]) -> dict[str, Any]:
    row = candidate_row(item)
    return {key: row[key] for key in (
        "id", "source_row_id", "raw_hash", "trajectory_hash", "group_id", "split",
        "risk_source", "failure_mode", "risk_consequence",
    )}


def validate(selected: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    if len(selected) != EXPECTED_TOTAL:
        errors.append(f"selection count is not {EXPECTED_TOTAL}")
    if len({item["id"] for item in selected}) != len(selected):
        errors.append("duplicate selected id")
    if len({item["source_row_id"] for item in selected}) != len(selected):
        errors.append("duplicate selected source row")
    if len({item["raw_hash"] for item in selected}) != len(selected):
        errors.append("duplicate selected raw record")
    if len({item["group_id"] for item in selected}) != len(selected):
        errors.append("duplicate selected group")
    for split in SPLITS:
        rows = [item for item in selected if item["split"] == split]
        if not rows:
            errors.append(f"{split} is empty")
        for field, categories in LABELS.items():
            if {item[field] for item in rows} != set(categories):
                errors.append(f"{split} does not cover all {field} labels")
    if errors:
        raise PipelineError("; ".join(errors))
    return {"status": "pass", "checks": 9, "errors": []}


def run_step0() -> None:
    candidates, source_report, quarantine = load_fine(SOURCE)
    if quarantine or len(candidates) != 4000:
        raise PipelineError(f"source must contain 4000 valid unique rows; got {len(candidates)}, quarantine={len(quarantine)}")
    for item in candidates:
        item["id"] = f"fine_unsafe_{item['trajectory_hash'][:16]}"
    assign_length_quartiles(candidates)
    grouping = group_candidates(candidates)
    assign_splits(candidates)

    selected = list(candidates)
    selected.sort(key=lambda item: (SPLITS.index(item["split"]), item["id"]))
    audit = validate(selected)
    split_counts = {split: sum(item["split"] == split for item in selected) for split in SPLITS}

    expected = (PROJECT_ROOT / "data/1.5/fine_sft/step0").resolve()
    if OUTPUT_ROOT.resolve() != expected or PROJECT_ROOT.resolve() not in expected.parents:
        raise PipelineError("unsafe output path")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    (OUTPUT_ROOT / "intermediate").mkdir(parents=True)
    (OUTPUT_ROOT / "final").mkdir()

    candidate_path = OUTPUT_ROOT / "intermediate/candidates.jsonl"
    selection_path = OUTPUT_ROOT / "final/selection.jsonl"
    write_jsonl(candidate_path, (candidate_row(item) for item in sorted(candidates, key=lambda row: row["id"])))
    write_jsonl(selection_path, (selection_row(item) for item in selected))

    by_split = {}
    for split in SPLITS:
        rows = [item for item in selected if item["split"] == split]
        by_split[split] = summarize(rows)
    report = {
        "stage": "fine_sft/step0",
        "version": "2.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source": source_report,
        "grouping": grouping,
        "sampling": {
            "seed": 42,
            "policy": "use all 4000 unique records; no label resampling",
            "quotas": split_counts,
            "group_before_split": True,
            "by_split": by_split,
        },
        "audit": audit,
        "outputs": {
            "candidates": {"records": len(candidates), "sha256": sha256_file(candidate_path)},
            "selection": {"records": len(selected), "sha256": sha256_file(selection_path)},
        },
    }
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    counts = "/".join(str(split_counts[split]) for split in SPLITS)
    print(f"Fine-SFT Step 0 complete: all 4000 candidates -> {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentDoG Fine-SFT data pipeline")
    parser.add_argument("stage", nargs="?", default="step0", choices=("step0", "step1", "step2", "all"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    stage = args.stage
    if stage != "step2" and (args.dry_run or args.limit is not None):
        parser.error("--dry-run and --limit are only supported by step2")
    if stage in {"step0", "all"}:
        run_step0()
    if stage in {"step1", "all"}:
        run_step1()
    if stage == "step2":
        run_step2(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
