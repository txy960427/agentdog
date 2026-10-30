"""Step 1：把冻结清单转换为细粒度 SFT canonical JSONL。"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from coarse_sft.steps.step0.pipeline import write_jsonl
from coarse_sft.steps.step0.source import (
    FAILURE_MODES,
    HISTORY,
    PROFILE,
    REAL_WORLD_HARMS,
    RISK_SOURCES,
    TOOLS_PREFIX,
    PipelineError,
    parse_events,
    parse_labels,
    sha256_file,
    sha256_text,
    stable_json,
    trajectory_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "data/1.0/source/AgentDoG1.0-Training-Data/AgentDoG-FineGrainedTaxonomy/train.json"
STEP0 = PROJECT_ROOT / "data/1.5/fine_sft/step0"
OUTPUT_ROOT = PROJECT_ROOT / "data/1.5/fine_sft/step1"
EXPECTED_TOTAL = 4000
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise PipelineError(f"missing input: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_record(raw: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    raw_hash = sha256_text(stable_json(raw))
    if raw_hash != selected["raw_hash"]:
        raise PipelineError(f"raw hash mismatch: {selected['id']}")

    text = trajectory_text(raw["instruction"])
    head, history = text.split(HISTORY, 1)
    profile_tools = head.split(PROFILE, 1)[1].strip()
    profile, tools_text = profile_tools.split(TOOLS_PREFIX, 1)
    profile = profile.strip()
    tools = json.loads(tools_text.strip())
    if not profile or not isinstance(tools, list) or not tools:
        raise PipelineError(f"invalid profile or tools: {selected['id']}")

    # 训练时只暴露模型实际可见的三个工具字段，保留原顺序和注入后的说明。
    clean_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or not all(key in tool for key in ("name", "description", "parameters")):
            raise PipelineError(f"invalid tool: {selected['id']}")
        clean_tools.append({key: tool[key] for key in ("name", "description", "parameters")})

    events = parse_events(history)
    trajectory_hash = sha256_text(stable_json(events))
    if trajectory_hash != selected["trajectory_hash"]:
        raise PipelineError(f"trajectory hash mismatch: {selected['id']}")
    risk, failure, harm = parse_labels(raw["output"])
    if (risk, failure, harm) != (
        selected["risk_source"],
        selected["failure_mode"],
        selected["risk_consequence"],
    ):
        raise PipelineError(f"label mismatch: {selected['id']}")

    return {
        "id": selected["id"],
        "group_id": selected["group_id"],
        "split": selected["split"],
        "source": {
            "dataset": "AgentDoG-FineGrainedTaxonomy",
            "row_id": selected["source_row_id"],
            "raw_hash": raw_hash,
        },
        "profile": profile,
        "tools": clean_tools,
        "trajectory": events,
        "target": {
            "risk_source": risk,
            "failure_mode": failure,
            "risk_consequence": harm,
            "cot": None,
        },
    }


def label_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row["target"][field] for row in rows).items()))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counts = [len(row["tools"]) for row in rows]
    event_counts = [len(row["trajectory"]) for row in rows]
    return {
        "records": len(rows),
        "risk_source": label_counts(rows, "risk_source"),
        "failure_mode": label_counts(rows, "failure_mode"),
        "risk_consequence": label_counts(rows, "risk_consequence"),
        "unique_triples": len({
            (
                row["target"]["risk_source"],
                row["target"]["failure_mode"],
                row["target"]["risk_consequence"],
            )
            for row in rows
        }),
        "tool_count": {"min": min(tool_counts), "median": median(tool_counts), "max": max(tool_counts)},
        "event_count": {"min": min(event_counts), "median": median(event_counts), "max": max(event_counts)},
        "agent_thought_events": sum(
            event["role"] == "agent" and event["thought"] is not None
            for row in rows
            for event in row["trajectory"]
        ),
    }


def validate(
    rows: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    step0_report: dict[str, Any],
    split_counts: dict[str, int],
) -> dict[str, Any]:
    errors = []
    if len(rows) != EXPECTED_TOTAL or len(rows) != len(selection):
        errors.append(f"record count is not {EXPECTED_TOTAL}")
    if [row["id"] for row in rows] != [row["id"] for row in selection]:
        errors.append("record order differs from Step 0 selection")
    if len({row["id"] for row in rows}) != len(rows):
        errors.append("duplicate id")
    if len({row["group_id"] for row in rows}) != len(rows):
        errors.append("duplicate group")

    for row in rows:
        if set(row) != {"id", "group_id", "split", "source", "profile", "tools", "trajectory", "target"}:
            errors.append(f"invalid top-level fields: {row['id']}")
            break
        if set(row["target"]) != {"risk_source", "failure_mode", "risk_consequence", "cot"}:
            errors.append(f"invalid target fields: {row['id']}")
            break
        if row["target"]["cot"] is not None:
            errors.append(f"cot is not null: {row['id']}")
            break
        if any(set(tool) != {"name", "description", "parameters"} for tool in row["tools"]):
            errors.append(f"hidden tool metadata remains: {row['id']}")
            break
        if any(
            set(event) != ({"role", "thought", "action"} if event["role"] == "agent" else {"role", "content"})
            for event in row["trajectory"]
        ):
            errors.append(f"invalid trajectory event fields: {row['id']}")
            break
        source = candidates[row["id"]]
        if len(row["tools"]) != source["features"]["available_tool_count"]:
            errors.append(f"tool count mismatch: {row['id']}")
            break
        if len(row["trajectory"]) != source["features"]["event_count"]:
            errors.append(f"event count mismatch: {row['id']}")
            break

    for split, quota in split_counts.items():
        current = [row for row in rows if row["split"] == split]
        summary = summarize(current)
        expected = step0_report["sampling"]["by_split"][split]
        if len(current) != quota:
            errors.append(f"{split} count is not {quota}")
        if set(summary["risk_source"]) != set(RISK_SOURCES):
            errors.append(f"{split} risk source coverage mismatch")
        if set(summary["failure_mode"]) != set(FAILURE_MODES):
            errors.append(f"{split} failure mode coverage mismatch")
        if set(summary["risk_consequence"]) != set(REAL_WORLD_HARMS):
            errors.append(f"{split} risk consequence coverage mismatch")
        for field in ("risk_source", "failure_mode", "risk_consequence", "unique_triples"):
            if summary[field] != expected[field]:
                errors.append(f"{split} {field} differs from Step 0")

    if errors:
        raise PipelineError("; ".join(errors[:10]))
    return {"status": "pass", "checks": 13, "errors": []}


def run_step1() -> None:
    selection_path = STEP0 / "final/selection.jsonl"
    candidates_path = STEP0 / "intermediate/candidates.jsonl"
    report_path = STEP0 / "report.json"
    selection = read_jsonl(selection_path)
    candidates = {row["id"]: row for row in read_jsonl(candidates_path)}
    if not report_path.exists():
        raise PipelineError("missing Step 0 report")
    step0_report = json.loads(report_path.read_text(encoding="utf-8"))
    if step0_report.get("audit", {}).get("status") != "pass":
        raise PipelineError("Step 0 audit has not passed")

    raw_counts = step0_report.get("sampling", {}).get("quotas", {})
    if set(raw_counts) != set(SPLITS) or sum(raw_counts.values()) != EXPECTED_TOTAL:
        raise PipelineError("invalid Step 0 split counts")
    split_counts = {split: int(raw_counts[split]) for split in SPLITS}

    source_rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for selected in selection:
        index = int(selected["source_row_id"])
        if index < 0 or index >= len(source_rows):
            raise PipelineError(f"invalid source row: {selected['id']}")
        rows.append(parse_record(source_rows[index], selected))
    audit = validate(rows, selection, candidates, step0_report, split_counts)

    expected = (PROJECT_ROOT / "data/1.5/fine_sft/step1").resolve()
    if OUTPUT_ROOT.resolve() != expected or PROJECT_ROOT.resolve() not in expected.parents:
        raise PipelineError("unsafe output path")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    final = OUTPUT_ROOT / "final"
    final.mkdir(parents=True)

    outputs = {}
    by_split = {}
    for split, quota in split_counts.items():
        current = [row for row in rows if row["split"] == split]
        path = final / f"{split}.jsonl"
        write_jsonl(path, current)
        outputs[split] = {"records": quota, "sha256": sha256_file(path)}
        by_split[split] = summarize(current)

    report = {
        "stage": "fine_sft/step1",
        "version": "2.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "source": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
            "selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
        },
        "conversion": {
            "profile": "verbatim trimmed",
            "tools": "name/description/parameters only; source order preserved",
            "trajectory": "event order and original thought/action/content preserved",
            "cot": None,
        },
        "by_split": by_split,
        "audit": audit,
        "outputs": outputs,
    }
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Fine-SFT Step 1 complete: " + ", ".join(f"{split}={count}" for split, count in split_counts.items()))
