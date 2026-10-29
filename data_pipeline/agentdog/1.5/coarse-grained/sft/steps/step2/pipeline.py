"""按Step 0清单将FineGrained数据转换为带工具的unsafe统一格式。"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from steps.step0.source import (
    HISTORY,
    PROFILE,
    TOOLS_PREFIX,
    PipelineError,
    parse_events,
    parse_labels,
    read_rows,
    sha256_file,
    sha256_text,
    stable_json,
    trajectory_text,
    validate_tool_calls,
)

PROJECT_ROOT = Path(__file__).resolve().parents[7]
FINE_PATH = PROJECT_ROOT / "data/agentdog/1.0/source/AgentDoG1.0-Training-Data/AgentDoG-FineGrainedTaxonomy/train.json"
SELECTION_PATH = PROJECT_ROOT / "data/agentdog/1.5/coarse-grained/sft/step0/final/selection.jsonl"
STEP0_REPORT_PATH = PROJECT_ROOT / "data/agentdog/1.5/coarse-grained/sft/step0/report.json"
OUTPUT_ROOT = PROJECT_ROOT / "data/agentdog/1.5/coarse-grained/sft/step2"
EXPECTED_SIZE = {"train": 400, "validation": 100, "test": 100}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineError(f"file not found: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid JSONL: {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise PipelineError(f"{path} must contain JSON objects")
    return rows


def clean_tools(value: str, record_id: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        tools = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid tools JSON: {record_id}: {exc}") from exc
    if not isinstance(tools, list) or not tools:
        raise PipelineError(f"tools must be a non-empty list: {record_id}")

    cleaned = []
    audit = {"tools": len(tools), "source_fields_removed": 0, "original_descriptions_removed": 0}
    for tool in tools:
        if not isinstance(tool, dict):
            raise PipelineError(f"tool must be an object: {record_id}")
        name, description, parameters = tool.get("name"), tool.get("description"), tool.get("parameters")
        if not isinstance(name, str) or not isinstance(description, str) or not isinstance(parameters, dict):
            raise PipelineError(f"invalid tool schema: {record_id}")

        # description是轨迹当时真正可见的文本，工具描述注入样本必须原样保留它。
        cleaned.append({"name": name, "description": description, "parameters": parameters})
        audit["source_fields_removed"] += int("_source" in tool)
        audit["original_descriptions_removed"] += int("_original_description" in tool)
    return cleaned, audit


def convert(selection: dict[str, Any], raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_text(stable_json(raw)) != selection["raw_hash"]:
        raise PipelineError(f"raw hash mismatch: {selection['id']}")
    if set(raw) != {"instruction", "input", "output"} or raw["input"] != "":
        raise PipelineError(f"invalid FineGrained row: {selection['source_row_id']}")

    labels = parse_labels(raw["output"])
    expected_labels = (
        selection["risk_source"],
        selection["failure_mode"],
        selection["real_world_harm"],
    )
    if labels != expected_labels or selection["coarse"] != "unsafe":
        raise PipelineError(f"label mismatch: {selection['id']}")

    trajectory = trajectory_text(raw["instruction"])
    if PROFILE not in trajectory or HISTORY not in trajectory:
        raise PipelineError(f"invalid FineGrained trajectory: {selection['id']}")
    before, history = trajectory.split(HISTORY, 1)
    profile_tools = before.split(PROFILE, 1)[1].strip()
    if profile_tools.count(TOOLS_PREFIX) != 1:
        raise PipelineError(f"missing Available tools: {selection['id']}")
    profile, tools_text = profile_tools.split(TOOLS_PREFIX, 1)
    if not profile.strip():
        raise PipelineError(f"empty profile: {selection['id']}")

    tools, tool_audit = clean_tools(tools_text.strip(), selection["id"])
    events = parse_events(history)
    # 参数schema异常和重名工具是风险信号，只审计，不删除样本。
    schema_warning = validate_tool_calls(events, tools)
    duplicate_tool_name = len({tool["name"] for tool in tools}) != len(tools)

    row = {
        "id": selection["id"],
        "group_id": selection["group_id"],
        "pair_id": selection["pair_id"],
        "split": selection["split"],
        "source": {
            "dataset": "AgentDoG-FineGrainedTaxonomy",
            "row_id": selection["source_row_id"],
            "parent_id": None,
        },
        "profile": profile.strip(),
        "tools": tools,
        "trajectory": events,
        "target": {
            "type": "judge",
            "coarse": "unsafe",
            "risk_source": labels[0],
            "failure_mode": labels[1],
            "real_world_harm": labels[2],
            "cot": None,
        },
    }
    tool_audit.update(
        {
            "schema_warning": schema_warning,
            "duplicate_tool_name": duplicate_tool_name,
        }
    )
    return row, tool_audit


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def run() -> None:
    step0_report = json.loads(STEP0_REPORT_PATH.read_text(encoding="utf-8"))
    fine_hash = sha256_file(FINE_PATH)
    selection_hash = sha256_file(SELECTION_PATH)
    if step0_report["audit"]["status"] != "pass":
        raise PipelineError("Step 0 audit did not pass")
    if fine_hash != step0_report["source"]["fine"]["sha256"]:
        raise PipelineError("FineGrainedTaxonomy changed after Step 0")
    if selection_hash != step0_report["outputs"]["selection"]["sha256"]:
        raise PipelineError("Step 0 selection changed after audit")

    source_rows = read_rows(FINE_PATH)
    manifest = [row for row in read_jsonl(SELECTION_PATH) if row.get("stage") == "step2"]
    if len(manifest) != 600 or len({row.get("id") for row in manifest}) != 600:
        raise PipelineError("Step 0 selection must contain 600 unique Step 2 records")
    if len({row.get("pair_id") for row in manifest}) != 600 or any(not row.get("pair_id") for row in manifest):
        raise PipelineError("Step 2 pair IDs must be non-empty and unique")

    converted, audits = [], []
    for selection in manifest:
        try:
            raw = source_rows[int(selection["source_row_id"])]
            row, audit = convert(selection, raw)
            converted.append(row)
            audits.append(audit)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"cannot materialize {selection.get('id')}: {exc}") from exc

    counts = Counter(row["split"] for row in converted)
    risk_counts = {
        split: dict(sorted(Counter(row["target"]["risk_source"] for row in converted if row["split"] == split).items()))
        for split in EXPECTED_SIZE
    }
    failures = []
    if dict(counts) != EXPECTED_SIZE:
        failures.append(f"split sizes: expected {EXPECTED_SIZE}, got {dict(counts)}")
    for split in EXPECTED_SIZE:
        expected_risks = step0_report["selection"]["by_split"][split]["step2_risk_sources"]
        if risk_counts[split] != expected_risks:
            failures.append(f"{split} Risk Source quota mismatch")
        rows = [row for row in converted if row["split"] == split]
        if len({row["target"]["failure_mode"] for row in rows}) != 14:
            failures.append(f"{split} does not cover 14 Failure Modes")
        if len({row["target"]["real_world_harm"] for row in rows}) != 10:
            failures.append(f"{split} does not cover 10 Real-World Harms")
    if failures:
        raise PipelineError("Step 2 acceptance failed:\n- " + "\n- ".join(failures))

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    final = OUTPUT_ROOT / "final"
    final.mkdir(parents=True)

    order = {"train": 0, "validation": 1, "test": 2}
    converted.sort(key=lambda row: (order[row["split"]], row["id"]))
    # final三个split已经完整覆盖全部记录，不再额外保存一份合并副本。
    outputs: dict[str, Any] = {}
    for split in EXPECTED_SIZE:
        path = final / f"{split}.jsonl"
        count = write_jsonl(path, (row for row in converted if row["split"] == split))
        outputs[split] = {"path": f"final/{split}.jsonl", "records": count, "sha256": sha256_file(path)}

    report = {
        "stage": "agentdog-1.5/coarse-grained/sft/step2",
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            "fine": {"path": str(FINE_PATH), "sha256": fine_hash},
            "selection": {"path": str(SELECTION_PATH), "sha256": selection_hash},
            "step0_report": {"path": str(STEP0_REPORT_PATH), "sha256": sha256_file(STEP0_REPORT_PATH)},
        },
        "selection_records": len(manifest),
        "by_split": {
            split: {
                "records": counts[split],
                "risk_sources": risk_counts[split],
                "failure_modes": len({row["target"]["failure_mode"] for row in converted if row["split"] == split}),
                "real_world_harms": len({row["target"]["real_world_harm"] for row in converted if row["split"] == split}),
            }
            for split in EXPECTED_SIZE
        },
        "tool_audit": {
            "tools": sum(item["tools"] for item in audits),
            "source_fields_removed": sum(item["source_fields_removed"] for item in audits),
            "original_descriptions_removed": sum(item["original_descriptions_removed"] for item in audits),
            "schema_warning_records": sum(item["schema_warning"] for item in audits),
            "duplicate_tool_name_records": sum(item["duplicate_tool_name"] for item in audits),
        },
        "audit": {
            "status": "pass",
            "raw_hash_mismatches": 0,
            "quarantine": 0,
            "unique_ids": len({row["id"] for row in converted}),
            "unique_pair_ids": len({row["pair_id"] for row in converted}),
            "tool_metadata_removed": all(set(tool) == {"name", "description", "parameters"} for row in converted for tool in row["tools"]),
            "cot_is_null": all(row["target"]["cot"] is None for row in converted),
        },
        "outputs": outputs,
    }
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"Step 2 complete: {OUTPUT_ROOT}")
    for split in EXPECTED_SIZE:
        print(f"{split}: {outputs[split]['records']} records, {len(risk_counts[split])} Risk Sources")
