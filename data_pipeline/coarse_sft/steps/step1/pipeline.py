"""按Step 0清单将BinarySafety转换为统一轨迹格式。"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from steps.step0.source import (
    PROJECT_ROOT,
    HISTORY,
    PROFILE,
    TOOLS_PREFIX,
    PipelineError,
    parse_events,
    read_rows,
    sha256_file,
    sha256_text,
    stable_json,
    trajectory_text,
)

BINARY_PATH = PROJECT_ROOT / "data/1.0/source/AgentDoG1.0-Training-Data/AgentDoG-BinarySafety/train.json"
SELECTION_PATH = PROJECT_ROOT / "data/1.5/coarse_sft/step0/final/selection.jsonl"
STEP0_REPORT_PATH = PROJECT_ROOT / "data/1.5/coarse_sft/step0/report.json"
OUTPUT_ROOT = PROJECT_ROOT / "data/1.5/coarse_sft/step1"
EXPECTED = {
    "train": {"safe": 100, "unsafe": 100},
    "validation": {"safe": 25, "unsafe": 25},
    "test": {"safe": 25, "unsafe": 25},
}


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


def convert(selection: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """核验清单并从原始instruction中提取Profile和保序事件。"""
    if sha256_text(stable_json(raw)) != selection["raw_hash"]:
        raise PipelineError(f"raw hash mismatch: {selection['id']}")
    if set(raw) != {"instruction", "input", "output"} or raw["input"] != "":
        raise PipelineError(f"invalid BinarySafety row: {selection['source_row_id']}")
    if raw["output"] != selection["coarse"]:
        raise PipelineError(f"label mismatch: {selection['id']}")

    trajectory = trajectory_text(raw["instruction"])
    if PROFILE not in trajectory or HISTORY not in trajectory or TOOLS_PREFIX in trajectory:
        raise PipelineError(f"invalid BinarySafety trajectory: {selection['id']}")
    before, history = trajectory.split(HISTORY, 1)
    profile = before.split(PROFILE, 1)[1].strip()
    if not profile:
        raise PipelineError(f"empty profile: {selection['id']}")

    return {
        "id": selection["id"],
        "group_id": selection["group_id"],
        "pair_id": None,
        "split": selection["split"],
        "source": {
            "dataset": "AgentDoG-BinarySafety",
            "row_id": selection["source_row_id"],
            "parent_id": None,
        },
        "profile": profile,
        "tools": None,
        "trajectory": parse_events(history),
        "target": {
            "type": "judge",
            "coarse": raw["output"],
            "risk_source": None,
            "failure_mode": None,
            "real_world_harm": None,
            "cot": None,
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def run() -> None:
    step0_report = json.loads(STEP0_REPORT_PATH.read_text(encoding="utf-8"))
    binary_hash = sha256_file(BINARY_PATH)
    selection_hash = sha256_file(SELECTION_PATH)
    if step0_report["audit"]["status"] != "pass":
        raise PipelineError("Step 0 audit did not pass")
    if binary_hash != step0_report["source"]["binary"]["sha256"]:
        raise PipelineError("BinarySafety changed after Step 0")
    if selection_hash != step0_report["outputs"]["selection"]["sha256"]:
        raise PipelineError("Step 0 selection changed after audit")

    source_rows = read_rows(BINARY_PATH)
    manifest = [row for row in read_jsonl(SELECTION_PATH) if row.get("stage") == "step1"]
    if len(manifest) != 300 or len({row.get("id") for row in manifest}) != 300:
        raise PipelineError("Step 0 selection must contain 300 unique Step 1 records")

    converted = []
    for selection in manifest:
        try:
            raw = source_rows[int(selection["source_row_id"])]
            converted.append(convert(selection, raw))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"cannot materialize {selection.get('id')}: {exc}") from exc

    counts = {
        split: dict(Counter(row["target"]["coarse"] for row in converted if row["split"] == split))
        for split in EXPECTED
    }
    failures = [
        f"{split}: expected {quota}, got {counts[split]}"
        for split, quota in EXPECTED.items()
        if counts[split] != quota
    ]
    if failures:
        raise PipelineError("Step 1 acceptance failed:\n- " + "\n- ".join(failures))

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    final = OUTPUT_ROOT / "final"
    final.mkdir(parents=True)

    order = {"train": 0, "validation": 1, "test": 2}
    converted.sort(key=lambda row: (order[row["split"]], row["id"]))
    # final三个split已经完整覆盖全部记录，不再额外保存一份合并副本。
    outputs: dict[str, Any] = {}
    for split in EXPECTED:
        path = final / f"{split}.jsonl"
        count = write_jsonl(path, (row for row in converted if row["split"] == split))
        outputs[split] = {
            "path": f"final/{split}.jsonl",
            "records": count,
            "sha256": sha256_file(path),
        }

    report = {
        "stage": "coarse_sft/step1",
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            "binary": {"path": str(BINARY_PATH), "sha256": binary_hash},
            "selection": {"path": str(SELECTION_PATH), "sha256": selection_hash},
            "step0_report": {"path": str(STEP0_REPORT_PATH), "sha256": sha256_file(STEP0_REPORT_PATH)},
        },
        "selection_records": len(manifest),
        "by_split": counts,
        "audit": {
            "status": "pass",
            "raw_hash_mismatches": 0,
            "quarantine": 0,
            "unique_ids": len({row["id"] for row in converted}),
            "tools_are_null": all(row["tools"] is None for row in converted),
            "cot_is_null": all(row["target"]["cot"] is None for row in converted),
        },
        "outputs": outputs,
    }
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"Step 1 complete: {OUTPUT_ROOT}")
    for split in EXPECTED:
        print(f"{split}: {outputs[split]['records']} {counts[split]}")
