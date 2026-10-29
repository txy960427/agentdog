"""Step 3最终验收：校验600条safe twin的血缘、审核状态和可见证据。"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SFT_ROOT))

from steps.step0.source import PROJECT_ROOT, action_object, sha256_file, stable_json  # noqa: E402
from steps.step3.pipeline import (  # noqa: E402
    event_schema_valid,
    is_subsequence,
    load_parents,
    normalized_event,
    read_jsonl,
    same_event,
)

ROOT = PROJECT_ROOT
STEP3 = ROOT / "data/1.5/coarse_sft/step3"
MID = STEP3 / "intermediate"
EXPECTED = {"train": 400, "validation": 100, "test": 100}


def digest(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def tool_pairs(events: list[dict]) -> set[tuple[str, str]]:
    pairs = set()
    for index, event in enumerate(events[:-1]):
        action = action_object(event.get("action")) if event.get("role") == "agent" else None
        if action and events[index + 1].get("role") == "environment":
            pairs.add((stable_json(action), stable_json(normalized_event(events[index + 1]))))
    return pairs


def run() -> None:
    parents = {row["id"]: row for row in load_parents()}
    # finalize清理生成草稿后，直接用final和紧凑血缘复验，不重新调用API。
    final_mode = not (MID / "candidates.jsonl").exists() and (MID / "lineage.jsonl").is_file()
    records_path = MID / ("lineage.jsonl" if final_mode else "candidates.jsonl")
    records = read_jsonl(records_path)
    reviews = {row["parent_id"]: row for row in read_jsonl(MID / "reviews.jsonl")}
    audits = {row["parent_id"]: row for row in read_jsonl(MID / "model_semantic_audit.jsonl")}
    folder = "final" if final_mode else "draft"
    drafts = {
        row["source"]["parent_id"]: row
        for split in EXPECTED
        for row in read_jsonl(STEP3 / folder / f"{split}.jsonl")
    }
    errors, counts = [], Counter()

    if set(parents) != {row["parent_id"] for row in records} or set(parents) != set(drafts):
        errors.append("600条parent、candidate与draft覆盖不一致")

    for record in records:
        parent_id = record["parent_id"]
        chosen = [{"trajectory": drafts[parent_id]["trajectory"]}] if final_mode else [row for row in record["candidates"] if row.get("selected")]
        if len(chosen) != 1:
            errors.append(f"{parent_id}: selected候选不是一个")
            continue
        chosen, parent, safe = chosen[0], parents[parent_id], drafts[parent_id]
        events, parent_events = chosen["trajectory"], parent["trajectory"]
        current_hash = digest(events)
        counts[safe["split"]] += 1
        if final_mode and current_hash != record["candidate_sha256"]:
            errors.append(f"{parent_id}: final与血缘hash不一致")

        if not event_schema_valid(events) or safe["trajectory"] != events:
            errors.append(f"{parent_id}: 事件结构或draft不一致")
        if any(safe[key] != parent[key] for key in ("group_id", "pair_id", "split", "profile", "tools")):
            errors.append(f"{parent_id}: parent血缘字段变化")
        if safe["source"].get("parent_id") != parent_id or safe["target"].get("coarse") != "safe":
            errors.append(f"{parent_id}: source或标签错误")

        facts = [event for event in events if event["role"] != "agent"]
        parent_facts = [event for event in parent_events if event["role"] != "agent"]
        if not is_subsequence(facts, parent_facts):
            errors.append(f"{parent_id}: 新增了user/environment证据")
        first = next(
            (i for i, event in enumerate(events) if i >= len(parent_events) or not same_event(event, parent_events[i])),
            None,
        )
        if first is None or first >= len(parent_events) or parent_events[first]["role"] != "agent":
            errors.append(f"{parent_id}: 首个改写点不是原Agent决策")

        available = {tool["name"] for tool in parent["tools"]}
        original_pairs = tool_pairs(parent_events)
        for index, event in enumerate(events[:-1]):
            if event["role"] != "agent" or event["action"].lstrip().startswith("Complete"):
                continue
            action = action_object(event["action"])
            pair = (
                stable_json(action),
                stable_json(normalized_event(events[index + 1])),
            ) if action and events[index + 1]["role"] == "environment" else None
            if not action or action.get("name") not in available or pair not in original_pairs:
                errors.append(f"{parent_id}: 新增或失配的工具调用/反馈")
                break

        review, audit = reviews.get(parent_id, {}), audits.get(parent_id, {})
        if review.get("status") != "approved" or review.get("candidate_sha256") != current_hash:
            errors.append(f"{parent_id}: 人工审核未覆盖当前候选")
        if audit.get("status") != "approved" or audit.get("candidate_sha256") != current_hash:
            errors.append(f"{parent_id}: 三模型审核未覆盖当前候选")

    if dict(counts) != EXPECTED:
        errors.append(f"split数量错误：{dict(counts)}")

    report = {
        "stage": "coarse_sft/step3/verify",
        "status": "pass" if not errors else "failed",
        "records": len(records),
        "errors": errors,
        "inputs": {
            "lineage" if final_mode else "candidates": sha256_file(records_path),
            "reviews": sha256_file(MID / "reviews.jsonl"),
            "model_semantic_audit": sha256_file(MID / "model_semantic_audit.jsonl"),
            folder: {split: sha256_file(STEP3 / folder / f"{split}.jsonl") for split in EXPECTED},
        },
    }
    (MID / "final_quality_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if final_mode:
        report_path = STEP3 / "report.json"
        summary = json.loads(report_path.read_text(encoding="utf-8"))
        summary["review_required"] = sum(row.get("status") != "approved" for row in reviews.values())
        summary["outputs"]["quality_audit"]["sha256"] = sha256_file(MID / "final_quality_audit.json")
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Step 3 verify: {report['status']}, records={len(records)}, errors={len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
