"""Step 3：调用DeepSeek生成、检查并审核safe twin。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from steps.step0.source import (
    PROJECT_ROOT,
    PipelineError,
    RISK_SOURCES,
    action_object,
    sha256_file,
    stable_json,
    validate_tool_calls,
)
from steps.step3.client import OpenAICompatibleClient

STEP2_ROOT = PROJECT_ROOT / "data/1.5/coarse_sft/step2"
OUTPUT_ROOT = PROJECT_ROOT / "data/1.5/coarse_sft/step3"
PROMPT_ROOT = PROJECT_ROOT / "docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/prompts"
GENERATOR_PROMPT_PATH = PROMPT_ROOT / "safe_twin_generator.txt"
CHECKER_PROMPT_PATH = PROMPT_ROOT / "safe_twin_checker.txt"
MANUAL_OVERRIDES_PATH = Path(__file__).with_name("manual_overrides.json")
SPLITS = ("train", "validation", "test")
EXPECTED = {"train": 400, "validation": 100, "test": 100}
STRATEGIES = {
    "refuse",
    "ignore_injection",
    "verify",
    "request_confirmation",
    "warn",
    "safe_fallback",
    "stop_or_escalate",
}
CHECK_FIELDS = (
    "risk_neutralized",
    "no_hidden_unsafe_behavior",
    "benign_goal_preserved",
    "facts_consistent",
    "no_fabricated_results",
    "same_scenario",
)
PROMPT_CONFIG_KEYS = ("generator_prompt_sha256", "checker_prompt_sha256")


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> int:
    count = 0
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        # 每次写完立即刷盘，断电后最多损失当前尚未返回的一次请求。
        stream.flush()
        os.fsync(stream.fileno())
    return count


def load_parents() -> list[dict[str, Any]]:
    report_path = STEP2_ROOT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["audit"]["status"] != "pass":
        raise PipelineError("Step 2 audit did not pass")

    parents = []
    for split in SPLITS:
        path = STEP2_ROOT / "final" / f"{split}.jsonl"
        if sha256_file(path) != report["outputs"][split]["sha256"]:
            raise PipelineError(f"Step 2 {split} changed after audit")
        rows = read_jsonl(path)
        if len(rows) != EXPECTED[split] or any(row.get("split") != split for row in rows):
            raise PipelineError(f"invalid Step 2 {split} records")
        parents.extend(rows)
    if len({row["id"] for row in parents}) != 600 or len({row["pair_id"] for row in parents}) != 600:
        raise PipelineError("Step 2 IDs and pair IDs must be unique")
    return parents


def hash_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pilot_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_round = []
    second_round = []
    train = [row for row in parents if row["split"] == "train"]
    for risk in RISK_SOURCES:
        rows = sorted(
            (row for row in train if row["target"]["risk_source"] == risk),
            key=lambda row: hash_rank(row["pair_id"]),
        )
        first = rows[0]
        second = next(
            (row for row in rows[1:] if row["target"]["failure_mode"] != first["target"]["failure_mode"]),
            rows[1],
        )
        first_round.append(first)
        second_round.append(second)
    # 第一批8条先覆盖全部Risk Source，审查通过后再跑第二轮。
    return first_round + second_round


def full_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按split和Risk Source轮转，保证小批审查尽量覆盖全部风险类型。"""
    ordered = []
    for split in SPLITS:
        groups = {
            risk: sorted(
                (
                    row
                    for row in parents
                    if row["split"] == split and row["target"]["risk_source"] == risk
                ),
                key=lambda row: hash_rank(row["pair_id"]),
            )
            for risk in RISK_SOURCES
        }
        for index in range(max(map(len, groups.values()))):
            for risk in RISK_SOURCES:
                if index < len(groups[risk]):
                    ordered.append(groups[risk][index])
    return ordered


def normalized_event(event: dict[str, Any]) -> dict[str, Any]:
    """统一环境JSON的转义形式，避免转义写法与实际Unicode字符被误判为事实变化。"""
    if event.get("role") not in {"user", "environment"}:
        return event
    content = event.get("content")
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return event
    return {"role": event["role"], "content": stable_json(parsed)}


def same_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return normalized_event(left) == normalized_event(right)


def is_subsequence(values: list[dict[str, Any]], source: list[dict[str, Any]]) -> bool:
    pending = iter(source)
    return all(any(same_event(item, candidate) for candidate in pending) for item in values)


def event_schema_valid(events: Any) -> bool:
    if not isinstance(events, list) or not events:
        return False
    for event in events:
        if not isinstance(event, dict):
            return False
        if event.get("role") in {"user", "environment"}:
            if set(event) != {"role", "content"} or not isinstance(event["content"], str) or not event["content"].strip():
                return False
        elif event.get("role") == "agent":
            if set(event) != {"role", "thought", "action"}:
                return False
            if event["thought"] is not None and not isinstance(event["thought"], str):
                return False
            if not isinstance(event["action"], str) or not event["action"].strip():
                return False
        else:
            return False
    return True


def tool_pairs(events: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = set()
    for index, event in enumerate(events[:-1]):
        if event["role"] != "agent" or str(event["action"]).lstrip().startswith("Complete"):
            continue
        action = action_object(event["action"])
        if action and events[index + 1]["role"] == "environment":
            pairs.add((stable_json(action), stable_json(normalized_event(events[index + 1]))))
    return pairs


def event_diff(parent: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index in range(max(len(parent), len(candidate))):
        before = parent[index] if index < len(parent) else None
        after = candidate[index] if index < len(candidate) else None
        if before is None or after is None or not same_event(before, after):
            result.append({"index": index, "before": before, "after": after})
    return result


def rule_checks(parent: dict[str, Any], generated: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    events = candidate.get("trajectory")
    rewrite = generated.get("rewrite_event_index")
    parent_events = parent["trajectory"]
    valid_rewrite = (
        isinstance(rewrite, int)
        and 0 <= rewrite < len(parent_events)
        and parent_events[rewrite]["role"] == "agent"
    )
    schema_valid = event_schema_valid(events)
    checks = {
        "strategy_allowed": candidate.get("strategy") in STRATEGIES,
        "risk_explanation_present": isinstance(candidate.get("risk_neutralization"), str)
        and bool(candidate.get("risk_neutralization", "").strip()),
        "preserve_constraints_present": isinstance(generated.get("preserve_constraints"), list)
        and bool(generated.get("preserve_constraints")),
        "rewrite_event_valid": valid_rewrite,
        "locked_indexes_valid": valid_rewrite
        and generated.get("locked_prefix_event_indexes") == list(range(rewrite)),
        "event_schema_valid": schema_valid,
        "locked_prefix_preserved": valid_rewrite
        and schema_valid
        and all(same_event(events[index], parent_events[index]) for index in range(rewrite)),
        "unsafe_decision_changed": valid_rewrite
        and schema_valid
        and len(events) > rewrite
        and events[rewrite] != parent_events[rewrite],
    }
    if not schema_valid:
        return checks

    # 用户与环境事实只能来自parent，禁止生成器凭空补出确认或执行结果。
    facts = [event for event in events if event["role"] != "agent"]
    parent_facts = [event for event in parent_events if event["role"] != "agent"]
    checks["visible_facts_are_parent_subsequence"] = is_subsequence(facts, parent_facts)
    checks["no_adjacent_duplicate_events"] = all(
        events[index] != events[index - 1] for index in range(1, len(events))
    )
    checks["role_order_valid"] = events[0]["role"] == "user" and all(
        event["role"] != "environment" or (index > 0 and events[index - 1]["role"] == "agent")
        for index, event in enumerate(events)
    )

    available_names = {tool["name"] for tool in parent["tools"]}
    parent_pairs = tool_pairs(parent_events)
    calls_valid = True
    feedback_not_fabricated = True
    for index, event in enumerate(events):
        if event["role"] != "agent" or event["action"].lstrip().startswith("Complete"):
            continue
        action = action_object(event["action"])
        if not action or action.get("name") not in available_names:
            calls_valid = False
            continue
        if index + 1 >= len(events) or events[index + 1]["role"] != "environment":
            calls_valid = False
            continue
        if (
            stable_json(action),
            stable_json(normalized_event(events[index + 1])),
        ) not in parent_pairs:
            feedback_not_fabricated = False
    checks["tool_calls_available_and_paired"] = calls_valid
    checks["no_fabricated_tool_feedback"] = feedback_not_fabricated
    try:
        # v1.0 parent本身可能带schema warning；safe twin可以原样继承，但不能新增。
        parent_warning = validate_tool_calls(parent_events, parent["tools"])
        candidate_warning = validate_tool_calls(events, parent["tools"])
        checks["tool_arguments_match_schema"] = not candidate_warning or parent_warning
    except (KeyError, TypeError, ValueError):
        checks["tool_arguments_match_schema"] = False

    # Corrupted Tool Feedback既可以在看见反馈后拒绝，也可以在前一工具调用本身已不安全时提前阻止。
    if parent["target"]["risk_source"] == "Corrupted Tool Feedback":
        trigger = next(
            (
                index
                for index, event in enumerate(parent_events)
                if event["role"] == "environment"
                and any(
                    marker in event["content"].lower()
                    for marker in (
                        "_data_integrity_warning",
                        "_override_instruction",
                        "data has been modified",
                        "integrity warning",
                    )
                )
            ),
            None,
        )
        preserved = (
            trigger is not None
            and valid_rewrite
            and rewrite > trigger
            and len(events) > trigger
            and same_event(events[trigger], parent_events[trigger])
        )
        prevented = (
            trigger is not None
            and valid_rewrite
            and rewrite == trigger - 1
            and len(events) > rewrite
            and not same_event(events[rewrite], parent_events[rewrite])
        )
        # 多风险轨迹可能在受污染反馈出现前已经包含更早的不安全动作。
        # 人工审核的early_intervention必须在该动作处直接终止，不能生成新工具反馈。
        prevented_early = (
            candidate.get("construction_mode") == "early_intervention"
            and trigger is not None
            and valid_rewrite
            and rewrite < trigger - 1
            and len(events) == rewrite + 1
            and events[rewrite]["role"] == "agent"
            and str(events[rewrite]["action"]).lstrip().startswith("Complete")
        )
        checks["corrupted_feedback_preserved_or_prevented"] = (
            preserved or prevented or prevented_early
        )
    return checks


def checker_pass(value: dict[str, Any]) -> bool:
    if not all(isinstance(value.get(field), bool) for field in (*CHECK_FIELDS, "pass")):
        raise ValueError("checker output is missing boolean fields")
    if not isinstance(value.get("reasons"), list):
        raise ValueError("checker reasons must be a list")
    expected = all(value[field] for field in CHECK_FIELDS)
    if value["pass"] != expected:
        raise ValueError("checker pass is inconsistent with detail fields")
    return expected


def visible(row: dict[str, Any], trajectory: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "profile": row["profile"],
        "tools": row["tools"],
        "trajectory": trajectory if trajectory is not None else row["trajectory"],
    }


def process_parent(
    parent: dict[str, Any],
    generator: OpenAICompatibleClient,
    checker: OpenAICompatibleClient,
    generator_template: str,
    checker_template: str,
    config: dict[str, str],
    retry_guidance: str = "",
) -> dict[str, Any]:
    record = {
        "parent_id": parent["id"],
        "pair_id": parent["pair_id"],
        "split": parent["split"],
        "risk_context": {
            "risk_source": parent["target"]["risk_source"],
            "failure_mode": parent["target"]["failure_mode"],
            "real_world_harm": parent["target"]["real_world_harm"],
        },
        **config,
        "retry_guidance": retry_guidance,
        "attempts": [],
        "candidates": [],
        "status": "failed",
    }
    parent_json = json.dumps(parent, ensure_ascii=False, indent=2)

    # 默认只生成一轮；问题case先人工审查，再由下一次显式运行决定是否重试。
    generation_attempts = int(os.environ.get("STEP3_GENERATION_ATTEMPTS", "1"))
    for attempt in range(generation_attempts):
        try:
            prompt = generator_template.replace("{parent_json}", parent_json)
            if retry_guidance:
                prompt += (
                    "\n\n<RETRY GUIDANCE>\n"
                    "A previous candidate was rejected for the following concrete reason. "
                    "Correct it without weakening any original requirement:\n"
                    f"{retry_guidance}\n</RETRY GUIDANCE>\n"
                )
            generated_response = generator.complete_json(prompt)
            generated = generated_response["json"]
            candidates = generated.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise ValueError("generator must return exactly two candidates")
            if len({candidate.get("strategy") for candidate in candidates if isinstance(candidate, dict)}) != 2:
                raise ValueError("generator candidates must use different strategies")
            record["attempts"].append(
                {
                    "attempt": attempt,
                    "response_id": generated_response["response_id"],
                    "usage": generated_response["usage"],
                    "reasoning_returned": generated_response["reasoning_returned"],
                    "error": None,
                }
            )
        except (PipelineError, KeyError, TypeError, ValueError) as exc:
            record["attempts"].append({"attempt": attempt, "error": str(exc)})
            continue

        attempt_items = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            checks = rule_checks(parent, generated, candidate)
            item = {
                "candidate_id": f"{parent['id']}-a{attempt}c{index}",
                "strategy": candidate.get("strategy"),
                "rewrite_event_index": generated.get("rewrite_event_index"),
                "raw_response": generated_response["content"],
                "trajectory": candidate.get("trajectory"),
                "event_diff": event_diff(parent["trajectory"], candidate.get("trajectory", []))
                if isinstance(candidate.get("trajectory"), list)
                else [],
                "rule_pass": bool(checks) and all(checks.values()),
                "rule_checks": checks,
                "model_pass": False,
                "model_checks": {},
                "checker_status": "pending" if bool(checks) and all(checks.values()) else "not_applicable",
                "checker_raw_response": None,
                "selected": False,
                "rejection_reasons": [key for key, passed in checks.items() if not passed],
            }
            attempt_items.append(item)

        # 先检查改动最小的候选；只有它失败时才花费token检查第二个候选。
        ordered = sorted(
            (item for item in attempt_items if item["rule_pass"]),
            key=lambda item: (edit_score(parent, item), item["candidate_id"]),
        )
        for item in ordered:
            if any(candidate["model_pass"] for candidate in attempt_items):
                item["checker_status"] = "skipped_after_pass"
                continue
            try:
                candidate_visible = visible(parent, item["trajectory"])
                prompt = checker_template.replace(
                    "{parent_visible_json}",
                    json.dumps(visible(parent), ensure_ascii=False, indent=2),
                ).replace(
                    "{candidate_visible_json}",
                    json.dumps(candidate_visible, ensure_ascii=False, indent=2),
                )
                checked_response = checker.complete_json(prompt)
                item["model_checks"] = checked_response["json"]
                item["model_pass"] = checker_pass(checked_response["json"])
                item["checker_status"] = "passed" if item["model_pass"] else "failed"
                item["checker_raw_response"] = checked_response["content"]
                item["checker_response_id"] = checked_response["response_id"]
                item["checker_usage"] = checked_response["usage"]
                if not item["model_pass"]:
                    item["rejection_reasons"].extend(checked_response["json"].get("reasons", []))
            except (PipelineError, KeyError, TypeError, ValueError) as exc:
                item["checker_status"] = "error"
                item["rejection_reasons"].append(f"checker_error: {exc}")

        record["candidates"].extend(attempt_items)

        if any(item["rule_pass"] and item["model_pass"] for item in record["candidates"]):
            record["status"] = "generated"
            break
    return record


def manual_record(parent: dict[str, Any], override: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """把人工审核的单点改写构造成普通候选，不调用生成器或检查器。"""
    rewrite = override["rewrite_event_index"]
    response = "Complete" + json.dumps(
        {"response": override["response"]}, ensure_ascii=False, separators=(",", ":")
    )
    trajectory = parent["trajectory"][:rewrite] + [
        {"role": "agent", "thought": override["risk_neutralization"], "action": response}
    ]
    candidate = {
        "strategy": override["strategy"],
        "risk_neutralization": override["risk_neutralization"],
        "trajectory": trajectory,
        "construction_mode": override["construction_mode"],
    }
    generated = {
        "rewrite_event_index": rewrite,
        "locked_prefix_event_indexes": list(range(rewrite)),
        "preserve_constraints": [
            "Preserve the original user request, profile, tools, identifiers, and every retained event."
        ],
    }
    checks = rule_checks(parent, generated, candidate)
    if not checks or not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise PipelineError(f"manual override failed for {parent['id']}: {failed}")

    model_checks = {field: True for field in CHECK_FIELDS}
    model_checks.update({"pass": True, "reasons": []})
    item = {
        "candidate_id": f"{parent['id']}-manual0",
        "strategy": override["strategy"],
        "construction_mode": override["construction_mode"],
        "rewrite_event_index": rewrite,
        "raw_response": json.dumps({**generated, "candidates": [candidate]}, ensure_ascii=False),
        "trajectory": trajectory,
        "event_diff": event_diff(parent["trajectory"], trajectory),
        "rule_pass": True,
        "rule_checks": checks,
        "model_pass": True,
        "model_checks": model_checks,
        "checker_status": "manual_approved",
        "checker_raw_response": override["risk_neutralization"],
        "selected": False,
        "rejection_reasons": [],
    }
    manual_config = {
        **config,
        "generator_model": "manual-curation",
        "checker_model": "human-semantic-audit",
        "api_base_url": "offline",
        "thinking": "not_applicable",
        "reasoning_effort": "not_applicable",
        "stream": False,
    }
    return {
        "parent_id": parent["id"],
        "pair_id": parent["pair_id"],
        "split": parent["split"],
        "risk_context": {
            "risk_source": parent["target"]["risk_source"],
            "failure_mode": parent["target"]["failure_mode"],
            "real_world_harm": parent["target"]["real_world_harm"],
        },
        **manual_config,
        "retry_guidance": "",
        "attempts": [
            {
                "attempt": 0,
                "response_id": f"manual:{parent['id']}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "reasoning_returned": False,
                "error": None,
            }
        ],
        "candidates": [item],
        "status": "generated",
        "review_required": True,
        "manual_override": {
            "construction_mode": override["construction_mode"],
            "reason": override["risk_neutralization"],
        },
    }


def edit_score(parent: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, int]:
    left, right = parent["trajectory"], candidate["trajectory"]
    changed = sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
    token_delta = abs(len(stable_json(left)) - len(stable_json(right)))
    return changed, token_delta


def safe_row(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    safe_id = parent["id"].replace("tool_unsafe_", "tool_safe_", 1)
    return {
        "id": safe_id,
        "group_id": parent["group_id"],
        "pair_id": parent["pair_id"],
        "split": parent["split"],
        "source": {
            "dataset": "AgentDoG1.5-Reproduction-SafeTwin",
            "row_id": safe_id,
            "parent_id": parent["id"],
        },
        "profile": parent["profile"],
        "tools": parent["tools"],
        "trajectory": candidate["trajectory"],
        "target": {
            "type": "judge",
            "coarse": "safe",
            "risk_source": None,
            "failure_mode": None,
            "real_world_harm": None,
            "cot": None,
        },
    }


def train_review_ids(parents: list[dict[str, Any]]) -> set[str]:
    result = set()
    for risk in RISK_SOURCES:
        rows = sorted(
            (
                row
                for row in parents
                if row["split"] == "train" and row["target"]["risk_source"] == risk
            ),
            key=lambda row: hash_rank(row["pair_id"]),
        )
        result.update(row["id"] for row in rows[:10])
    return result


def select_and_write(
    parents: list[dict[str, Any]],
    records: list[dict[str, Any]],
    root: Path,
    same_model: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    parent_by_id = {row["id"]: row for row in parents}
    strategy_counts: Counter[str] = Counter()
    review_train = train_review_ids(parents)
    safe_rows = []

    for record in sorted(records, key=lambda item: (item["split"], item["parent_id"])):
        for candidate in record["candidates"]:
            candidate["selected"] = False
        # 人工复核失败的记录可能仍保留旧候选，必须等重跑成功后才能重新进入草稿。
        if record.get("status") != "generated":
            continue
        passing = [
            candidate
            for candidate in record["candidates"]
            if candidate["rule_pass"] and candidate["model_pass"]
        ]
        if not passing:
            continue
        parent = parent_by_id[record["parent_id"]]
        chosen = min(
            passing,
            key=lambda candidate: (
                strategy_counts[candidate["strategy"]],
                edit_score(parent, candidate),
                candidate["candidate_id"],
            ),
        )
        chosen["selected"] = True
        strategy_counts[chosen["strategy"]] += 1
        safe_rows.append(safe_row(parent, chosen))

        conflict = len(record["attempts"]) > 1 or any(
            item.get("checker_status") in {"failed", "error"} for item in record["candidates"]
        )
        record["review_required"] = (
            same_model
            or record["split"] != "train"
            or record["parent_id"] in review_train
            or conflict
        )

    draft = root / "draft"
    draft.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(draft / f"{split}.jsonl", (row for row in safe_rows if row["split"] == split))
    return safe_rows, dict(strategy_counts)


def review_template(records: list[dict[str, Any]], path: Path) -> None:
    existing = {row["parent_id"]: row for row in read_jsonl(path)} if path.exists() else {}
    rows = []
    for record in records:
        required = bool(record.get("review_required"))
        if record.get("manual_override"):
            candidate = next(
                (item for item in record["candidates"] if item.get("selected")),
                record["candidates"][0],
            )
            candidate_sha = hashlib.sha256(
                stable_json(candidate["trajectory"]).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "parent_id": record["parent_id"],
                    "pair_id": record["pair_id"],
                    "required": True,
                    "status": "approved",
                    "reviewer": "manual-semantic-audit",
                    "reason": record["manual_override"]["reason"],
                    "construction_mode": record["manual_override"]["construction_mode"],
                    "candidate_sha256": candidate_sha,
                    "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
            continue
        row = existing.get(record["parent_id"], {
            "parent_id": record["parent_id"],
            "pair_id": record["pair_id"],
            "required": required,
            "status": "pending" if required else "not_required",
            "reviewer": "",
            "reason": "",
            "reviewed_at": "",
        })
        selected = next((item for item in record["candidates"] if item.get("selected")), None)
        if selected:
            compact_sha = hashlib.sha256(
                stable_json(selected["trajectory"]).encode("utf-8")
            ).hexdigest()
            legacy_sha = hashlib.sha256(
                json.dumps(selected["trajectory"], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            reviewed_sha = str(row.get("candidate_sha256", ""))
            if reviewed_sha and reviewed_sha not in {compact_sha, legacy_sha}:
                row = {
                    "parent_id": record["parent_id"],
                    "pair_id": record["pair_id"],
                    "required": True,
                    "status": "pending",
                    "reviewer": "",
                    "reason": "候选轨迹已重新生成，需要重新进行语义审核。",
                    "candidate_sha256": compact_sha,
                    "reviewed_at": "",
                }
            elif not reviewed_sha:
                row["candidate_sha256"] = compact_sha
        row["required"] = required
        if required and row.get("status") == "not_required":
            row["status"] = "pending"
        rows.append(row)
    write_jsonl(path, rows)


def ensure_pilot_approved(config: dict[str, str]) -> list[dict[str, Any]]:
    root = OUTPUT_ROOT / "pilot"
    candidates_path = root / "intermediate" / "candidates.jsonl"
    reviews_path = root / "intermediate" / "reviews.jsonl"
    if not candidates_path.is_file() or not reviews_path.is_file():
        raise PipelineError("run and review STEP3_MODE=pilot before full generation")
    records = read_jsonl(candidates_path)
    if len(records) != 16 or any(
        record.get(key) != config[key] for record in records for key in PROMPT_CONFIG_KEYS
    ):
        raise PipelineError("pilot does not match the current prompt configuration")
    reviews = {row["parent_id"]: row for row in read_jsonl(reviews_path)}
    if any(
        reviews.get(record["parent_id"], {}).get("status") != "approved"
        or not str(reviews.get(record["parent_id"], {}).get("reviewer", "")).strip()
        or not str(reviews.get(record["parent_id"], {}).get("reviewed_at", "")).strip()
        for record in records
    ):
        raise PipelineError("all 16 pilot records must be approved before full generation")
    return records


def usage_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        for attempt in record["attempts"]:
            for key, value in attempt.get("usage", {}).items():
                if isinstance(value, int):
                    totals[f"generator_{key}"] += value
        for candidate in record["candidates"]:
            for key, value in candidate.get("checker_usage", {}).items():
                if isinstance(value, int):
                    totals[f"checker_{key}"] += value
    return dict(totals)


def model_provenance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """汇总每批数据的生成/校验模型；旧记录缺少接口字段时按当时的DeepSeek官方接口记录。"""
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter(
        (
            str(record.get("generator_model", "unknown")),
            str(record.get("checker_model", "unknown")),
            str(record.get("api_base_url", "https://api.deepseek.com")),
            str(record.get("thinking", "enabled")),
            str(record.get("reasoning_effort", "high")),
            str(record.get("stream", False)),
        )
        for record in records
    )
    return [
        {
            "generator_model": generator_model,
            "checker_model": checker_model,
            "api_base_url": api_base_url,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
            "stream": stream,
            "records": count,
        }
        for (
            generator_model,
            checker_model,
            api_base_url,
            thinking,
            reasoning_effort,
            stream,
        ), count in sorted(counts.items())
    ]


def generate(mode: str) -> None:
    parents = load_parents()
    selected_parents = pilot_parents(parents) if mode == "pilot" else full_parents(parents)
    root = OUTPUT_ROOT / "pilot" if mode == "pilot" else OUTPUT_ROOT
    intermediate = root / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)

    generator_template = GENERATOR_PROMPT_PATH.read_text(encoding="utf-8")
    checker_template = CHECKER_PROMPT_PATH.read_text(encoding="utf-8")
    generator_model = os.environ.get("STEP3_GENERATOR_MODEL", "deepseek-v4-pro")
    checker_model = os.environ.get("STEP3_CHECKER_MODEL", "deepseek-v4-pro")
    api_base_url = os.environ.get(
        "STEP3_BASE_URL",
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    ).rstrip("/")
    thinking = os.environ.get("STEP3_THINKING", "enabled")
    reasoning_effort = os.environ.get("STEP3_REASONING_EFFORT", "high")
    stream = os.environ.get("STEP3_STREAM", "false").strip().lower() == "true"
    max_tokens_field = os.environ.get("STEP3_MAX_TOKENS_FIELD", "max_tokens")
    generator_max_tokens = int(os.environ.get("STEP3_GENERATOR_MAX_TOKENS", "65536"))
    checker_max_tokens = int(os.environ.get("STEP3_CHECKER_MAX_TOKENS", "16384"))
    config = {
        "generator_prompt_sha256": sha256_file(GENERATOR_PROMPT_PATH),
        "checker_prompt_sha256": sha256_file(CHECKER_PROMPT_PATH),
        "generator_model": generator_model,
        "checker_model": checker_model,
        "api_base_url": api_base_url,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "max_tokens_field": max_tokens_field,
        "generator_max_tokens": generator_max_tokens,
        "checker_max_tokens": checker_max_tokens,
    }
    approved_pilot = ensure_pilot_approved(config) if mode == "full" else []

    candidates_path = intermediate / "candidates.jsonl"
    guidance_path = intermediate / "retry_guidance.json"
    retry_guidance = (
        json.loads(guidance_path.read_text(encoding="utf-8")) if guidance_path.exists() else {}
    )
    existing_rows = read_jsonl(candidates_path) if candidates_path.exists() else []
    existing = {row["parent_id"]: row for row in existing_rows}
    # full沿用同一Prompt下已审核通过的pilot；允许后续批次更换生成模型。
    for row in approved_pilot:
        existing.setdefault(row["parent_id"], row)
    for row in existing.values():
        if any(row.get(key) != config[key] for key in PROMPT_CONFIG_KEYS):
            raise PipelineError(f"existing Step 3 output uses another prompt: {row['parent_id']}")

    manual_overrides = json.loads(MANUAL_OVERRIDES_PATH.read_text(encoding="utf-8"))
    selected_ids = {row["id"] for row in selected_parents}
    unknown_overrides = set(manual_overrides) - selected_ids
    if mode == "full" and unknown_overrides:
        raise PipelineError(f"manual overrides reference unknown parents: {sorted(unknown_overrides)}")
    for parent in selected_parents:
        override = manual_overrides.get(parent["id"])
        if override:
            existing[parent["id"]] = manual_record(parent, override, config)

    pending_all = [
        row for row in selected_parents if existing.get(row["id"], {}).get("status") != "generated"
    ]
    default_batch = 8 if mode == "pilot" else 50
    batch_size = int(os.environ.get("STEP3_BATCH_SIZE", str(default_batch)))
    pending = pending_all[:batch_size] if batch_size > 0 else ([] if batch_size == 0 else pending_all)
    workers = int(os.environ.get("STEP3_WORKERS", "2"))
    if pending:
        generator_client = OpenAICompatibleClient(generator_model, generator_max_tokens)
        checker_client = OpenAICompatibleClient(checker_model, checker_max_tokens)

        def work(parent: dict[str, Any]) -> dict[str, Any]:
            return process_parent(
                parent,
                generator_client,
                checker_client,
                generator_template,
                checker_template,
                config,
                str(retry_guidance.get(parent["id"], "")),
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, parent) for parent in pending]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                existing[record["parent_id"]] = record
                write_jsonl(candidates_path, [record], mode="a")
                print(
                    f"Step 3 {mode}: {index}/{len(pending)} "
                    f"{record['parent_id']} {record['status']}",
                    flush=True,
                )

    parent_order = {row["id"]: index for index, row in enumerate(selected_parents)}
    records = sorted(
        (existing[row["id"]] for row in selected_parents if row["id"] in existing),
        key=lambda row: parent_order[row["parent_id"]],
    )
    safe_rows, strategies = select_and_write(
        selected_parents,
        records,
        root,
        same_model=generator_model == checker_model,
    )
    write_jsonl(candidates_path, records)
    reviews_path = intermediate / "reviews.jsonl"
    if mode == "full" and not reviews_path.exists():
        write_jsonl(
            reviews_path,
            read_jsonl(OUTPUT_ROOT / "pilot" / "intermediate" / "reviews.jsonl"),
        )
    review_template(records, reviews_path)

    draft_outputs = {}
    for split in SPLITS:
        path = root / "draft" / f"{split}.jsonl"
        draft_outputs[split] = {
            "path": f"draft/{split}.jsonl",
            "records": sum(row["split"] == split for row in safe_rows),
            "sha256": sha256_file(path),
        }

    expected = 16 if mode == "pilot" else 600
    approved = {
        row["parent_id"]
        for row in read_jsonl(reviews_path)
        if row.get("status") == "approved"
    }
    status = (
        "pass"
        if len(safe_rows) == expected and all(row["id"] in approved for row in selected_parents)
        else "awaiting_human_review"
        if len(safe_rows) == expected
        else "generation_incomplete"
    )
    report = {
        "stage": f"coarse_sft/step3/{mode}",
        "version": "1.1.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "mode": mode,
        "configuration": {
            **config,
            "base_url": api_base_url,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
            "stream": stream,
            "json_output": True,
            "generator_max_tokens": generator_max_tokens,
            "checker_max_tokens": checker_max_tokens,
            "workers": workers,
            "batch_size": batch_size,
            "same_model_check": generator_model == checker_model,
        },
        "parents": len(selected_parents),
        "generated": len(safe_rows),
        "failed": expected - len(safe_rows),
        "remaining": expected - len(safe_rows),
        "review_required": sum(bool(row.get("review_required")) for row in records),
        "strategy_counts": strategies,
        "model_provenance": model_provenance(records),
        "usage": usage_summary(records),
        "outputs": {
            "candidates": {
                "path": "intermediate/candidates.jsonl",
                "records": len(records),
                "sha256": sha256_file(candidates_path),
            },
            "reviews": {
                "path": "intermediate/reviews.jsonl",
                "records": len(records),
                "sha256": sha256_file(intermediate / "reviews.jsonl"),
            },
            "draft": draft_outputs,
        },
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Step 3 {mode} complete: {status}, generated={len(safe_rows)}/{expected}")


def finalize() -> None:
    report_path = OUTPUT_ROOT / "report.json"
    candidates_path = OUTPUT_ROOT / "intermediate/candidates.jsonl"
    reviews_path = OUTPUT_ROOT / "intermediate/reviews.jsonl"
    quality_path = OUTPUT_ROOT / "intermediate/final_quality_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "full" or report.get("generated") != 600:
        raise PipelineError("full Step 3 generation is not complete")
    if sha256_file(candidates_path) != report["outputs"]["candidates"]["sha256"]:
        raise PipelineError("Step 3 candidates changed after generation")
    if not quality_path.exists():
        raise PipelineError("Step 3 final quality audit has not been run")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("status") != "pass":
        raise PipelineError("Step 3 final quality audit still has actionable records")
    if quality.get("inputs", {}).get("candidates") != sha256_file(candidates_path):
        raise PipelineError("Step 3 candidates changed after final quality audit")
    if quality.get("inputs", {}).get("reviews") != sha256_file(reviews_path):
        raise PipelineError("Step 3 reviews changed after final quality audit")
    if quality.get("inputs", {}).get("model_semantic_audit") != sha256_file(
        OUTPUT_ROOT / "intermediate/model_semantic_audit.jsonl"
    ):
        raise PipelineError("Step 3 model audit changed after final quality audit")
    for split in SPLITS:
        if quality.get("inputs", {}).get("draft", {}).get(split) != sha256_file(
            OUTPUT_ROOT / "draft" / f"{split}.jsonl"
        ):
            raise PipelineError(f"Step 3 {split} draft changed after final quality audit")
    records = read_jsonl(candidates_path)
    reviews = {row["parent_id"]: row for row in read_jsonl(reviews_path)}

    missing = []
    for record in records:
        if not record.get("review_required"):
            continue
        review = reviews.get(record["parent_id"], {})
        if (
            review.get("status") != "approved"
            or not str(review.get("reviewer", "")).strip()
            or not str(review.get("reviewed_at", "")).strip()
        ):
            missing.append(record["parent_id"])
    if missing:
        raise PipelineError(f"required reviews are incomplete: {len(missing)}")
    if any(review.get("status") == "rejected" for review in reviews.values()):
        raise PipelineError("rejected safe twins must be repaired and regenerated before finalize")

    final = OUTPUT_ROOT / "final"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir()
    outputs = {}
    for split, expected in EXPECTED.items():
        source = OUTPUT_ROOT / "draft" / f"{split}.jsonl"
        if sha256_file(source) != report["outputs"]["draft"][split]["sha256"]:
            raise PipelineError(f"Step 3 {split} draft changed after generation")
        rows = read_jsonl(source)
        if len(rows) != expected:
            raise PipelineError(f"Step 3 {split}: expected {expected}, got {len(rows)}")
        target = final / f"{split}.jsonl"
        write_jsonl(target, rows)
        outputs[split] = {"path": f"final/{split}.jsonl", "records": len(rows), "sha256": sha256_file(target)}

    # 原始候选与API流水在验收后体积很大；只冻结最终候选的紧凑血缘。
    lineage_path = OUTPUT_ROOT / "intermediate/lineage.jsonl"
    lineage = []
    for record in sorted(records, key=lambda row: row["parent_id"]):
        chosen = next(row for row in record["candidates"] if row.get("selected"))
        lineage.append({
            "parent_id": record["parent_id"],
            "pair_id": record["pair_id"],
            "split": record["split"],
            "risk_context": record["risk_context"],
            "generator_model": record.get("generator_model"),
            "checker_model": record.get("checker_model"),
            "candidate_id": chosen["candidate_id"],
            "strategy": chosen.get("strategy"),
            "construction_mode": chosen.get("construction_mode", "generated"),
            "rewrite_event_index": chosen.get("rewrite_event_index"),
            "candidate_sha256": hashlib.sha256(
                stable_json(chosen["trajectory"]).encode("utf-8")
            ).hexdigest(),
            "repair_history": record.get("repair_history", []),
        })
    write_jsonl(lineage_path, lineage)

    report["status"] = "pass"
    report["review_required"] = 0
    report["finalized_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["human_review"] = {
        "required": sum(bool(record.get("review_required")) for record in records),
        "approved": sum(review.get("status") == "approved" for review in reviews.values()),
        "reviews_sha256": sha256_file(reviews_path),
    }
    report.pop("model_semantic_audit", None)
    report.pop("repair", None)
    report["outputs"] = {
        "final": outputs,
        "lineage": {
            "path": "intermediate/lineage.jsonl",
            "records": len(lineage),
            "sha256": sha256_file(lineage_path),
        },
        "reviews": {
            "path": "intermediate/reviews.jsonl",
            "records": len(reviews),
            "sha256": sha256_file(reviews_path),
        },
        "semantic_audit": {
            "path": "intermediate/model_semantic_audit.jsonl",
            "records": len(records),
            "sha256": sha256_file(OUTPUT_ROOT / "intermediate/model_semantic_audit.jsonl"),
        },
        "quality_audit": {
            "path": "intermediate/final_quality_audit.json",
            "sha256": sha256_file(quality_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Step 3 finalize complete: 400/100/100")


def run() -> None:
    mode = os.environ.get("STEP3_MODE", "pilot").strip().lower()
    if mode in {"pilot", "full"}:
        generate(mode)
    elif mode == "finalize":
        finalize()
    else:
        raise PipelineError("STEP3_MODE must be pilot, full, or finalize")
