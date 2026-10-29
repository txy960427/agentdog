"""仅修复三模型语义审核打回的Step 3候选。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SFT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SFT_ROOT))

from steps.step0.source import PipelineError, sha256_file, stable_json  # noqa: E402
from steps.step3.client import OpenAICompatibleClient  # noqa: E402
from steps.step3.pipeline import (  # noqa: E402
    OUTPUT_ROOT,
    PROMPT_ROOT,
    STRATEGIES,
    event_diff,
    load_parents,
    read_jsonl,
    safe_row,
    write_jsonl,
)

INTERMEDIATE = OUTPUT_ROOT / "intermediate"
PROMPT_PATH = PROMPT_ROOT / "safe_twin_repair.txt"
CANDIDATES_PATH = INTERMEDIATE / "candidates.jsonl"
FAILED_PATH = INTERMEDIATE / "model_semantic_failed.jsonl"
REVIEWS_PATH = INTERMEDIATE / "reviews.jsonl"
ATTEMPTS_PATH = INTERMEDIATE / "repair_attempts.jsonl"
REPAIRS_PATH = INTERMEDIATE / "repair_candidates.jsonl"
REPAIR_REPORT_PATH = INTERMEDIATE / "repair_report.json"
STEP3_REPORT_PATH = OUTPUT_ROOT / "report.json"
SPLITS = ("train", "validation", "test")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def selected(record: dict[str, Any]) -> dict[str, Any]:
    rows = [item for item in record["candidates"] if item.get("selected")]
    if len(rows) != 1:
        raise PipelineError(f"expected one selected candidate: {record['parent_id']}")
    return rows[0]


def trajectory_hash(trajectory: list[dict[str, Any]]) -> str:
    return hashlib.sha256(stable_json(trajectory).encode("utf-8")).hexdigest()


def append_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        write_jsonl(path, [row], mode="a")


def validate_output(result: dict[str, Any], allowed: list[int]) -> dict[str, Any]:
    rewrite = result.get("rewrite_event_index")
    strategy = result.get("strategy")
    explanation = result.get("risk_neutralization")
    response = result.get("response")
    if rewrite not in allowed:
        raise ValueError(f"rewrite_event_index must be one of {allowed}")
    if strategy not in STRATEGIES:
        raise ValueError("invalid strategy")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("risk_neutralization is required")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response is required")
    if response.lstrip().startswith(("{\"name\"", "{'name'", "Complete")):
        raise ValueError("response must be plain user-facing text")
    return {
        "rewrite_event_index": rewrite,
        "strategy": strategy,
        "risk_neutralization": explanation.strip(),
        "response": response.strip(),
    }


def build_candidate(parent_id: str, current: dict[str, Any], repair: dict[str, Any], model: str) -> dict[str, Any]:
    rewrite = repair["rewrite_event_index"]
    action = "Complete" + json.dumps(
        {"response": repair["response"]}, ensure_ascii=False, separators=(",", ":")
    )
    trajectory = current["trajectory"][:rewrite] + [
        # risk_neutralization是审核/返修说明，仅保留在元数据，不能充当Agent当时的thought。
        {"role": "agent", "thought": None, "action": action}
    ]
    return {
        "candidate_id": f"{parent_id}-repair-{trajectory_hash(trajectory)[:12]}",
        "strategy": repair["strategy"],
        "construction_mode": "semantic_repair",
        "rewrite_event_index": rewrite,
        "raw_response": json.dumps(repair, ensure_ascii=False),
        "trajectory": trajectory,
        "event_diff": event_diff(current["trajectory"], trajectory),
        "rule_pass": True,
        "rule_checks": {
            "prefix_preserved_from_rejected_candidate": trajectory[:rewrite] == current["trajectory"][:rewrite],
            "terminal_complete_only": len(trajectory) == rewrite + 1 and action.startswith("Complete"),
        },
        "model_pass": True,
        "model_checks": {"pending_three_model_semantic_audit": True},
        "checker_status": "repair_pending_semantic_audit",
        "checker_raw_response": repair["risk_neutralization"],
        "selected": True,
        "rejection_reasons": [],
        "repair_model": model,
    }


def run() -> None:
    models = [x.strip() for x in os.environ.get("STEP3_REPAIR_MODELS", "gpt-5.6-sol,gpt-6.1-sol").split(",") if x.strip()]
    workers = int(os.environ.get("STEP3_REPAIR_WORKERS", "4"))
    batch_size = int(os.environ.get("STEP3_REPAIR_BATCH_SIZE", "0"))
    max_tokens = int(os.environ.get("STEP3_REPAIR_MAX_TOKENS", "4096"))
    interval = float(os.environ.get("STEP3_REPAIR_REQUEST_INTERVAL", "7.5"))
    transient_retries = int(os.environ.get("STEP3_REPAIR_TRANSIENT_RETRIES", "3"))
    cooldown = float(os.environ.get("STEP3_REPAIR_TRANSIENT_COOLDOWN", "65"))
    dry_run = os.environ.get("STEP3_REPAIR_DRY_RUN", "false").lower() == "true"
    if not models or workers < 1 or batch_size < 0:
        raise PipelineError("invalid STEP3_REPAIR_* configuration")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    parents = {row["id"]: row for row in load_parents()}
    records = read_jsonl(CANDIDATES_PATH)
    by_id = {row["parent_id"]: row for row in records}
    failures = read_jsonl(FAILED_PATH)
    failure_ids = {row["parent_id"] for row in failures}
    if len(failure_ids) != len(failures):
        raise PipelineError("duplicate parent_id in model_semantic_failed.jsonl")

    applied = {}
    if REPAIRS_PATH.exists():
        applied = {row["parent_id"]: row for row in read_jsonl(REPAIRS_PATH)}
    completed = {}
    if ATTEMPTS_PATH.exists():
        for row in read_jsonl(ATTEMPTS_PATH):
            if row.get("status") == "completed" and row.get("prompt_sha256") == prompt_sha:
                completed[(row["parent_id"], row["source_candidate_sha256"])] = row

    items = []
    for failure in failures:
        record = by_id[failure["parent_id"]]
        current = selected(record)
        current_sha = trajectory_hash(current["trajectory"])
        first = failure["failures"][0]
        earliest = first["earliest_bad_event_index"]
        allowed = [
            index
            for index, event in enumerate(current["trajectory"])
            if event["role"] == "agent" and index <= earliest
        ]
        if not allowed:
            raise PipelineError(f"no repair point: {failure['parent_id']}")
        payload = {
            "parent_id": failure["parent_id"],
            "risk_context": record["risk_context"],
            "profile": parents[failure["parent_id"]]["profile"],
            "available_tools": parents[failure["parent_id"]]["tools"],
            "rejected_candidate_trajectory": [
                {"event_index": i, **event} for i, event in enumerate(current["trajectory"])
            ],
            "allowed_rewrite_event_indexes": allowed,
            "earliest_bad_event_index": earliest,
            "audit_failures": failure["failures"],
        }
        items.append(
            {
                "parent_id": failure["parent_id"],
                "pair_id": failure["pair_id"],
                "source_candidate_sha256": current_sha,
                "current": current,
                "allowed": allowed,
                "prompt": prompt_template.replace(
                    "{{REPAIR_INPUT_JSON}}",
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            }
        )

    pending = [item for item in items if (item["parent_id"], item["source_candidate_sha256"]) not in completed]
    if batch_size:
        pending = pending[:batch_size]
    if dry_run:
        print(
            f"Repair dry-run: failed={len(failures)} "
            f"already_completed={len(items)-len(pending)} pending={len(pending)}"
        )
        return

    clients = {model: OpenAICompatibleClient(model, max_tokens) for model in models} if pending else {}
    file_lock = threading.Lock()
    rate_lock = threading.Lock()
    next_request = [0.0]

    def wait_rate() -> None:
        with rate_lock:
            scheduled = max(time.monotonic(), next_request[0])
            next_request[0] = scheduled + interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def repair_item(item: dict[str, Any]) -> str:
        last_error = "unknown"
        for model in models:
            for attempt in range(transient_retries + 1):
                try:
                    wait_rate()
                    response = clients[model].complete_json(item["prompt"])
                    result = validate_output(response["json"], item["allowed"])
                    candidate = build_candidate(item["parent_id"], item["current"], result, model)
                    row = {
                        "parent_id": item["parent_id"],
                        "pair_id": item["pair_id"],
                        "source_candidate_sha256": item["source_candidate_sha256"],
                        "repaired_candidate_sha256": trajectory_hash(candidate["trajectory"]),
                        "prompt_sha256": prompt_sha,
                        "model": model,
                        "status": "completed",
                        "repair": result,
                        "candidate": candidate,
                        "usage": response["usage"],
                        "response_id": response["response_id"],
                        "created_at": now(),
                    }
                    append_row(ATTEMPTS_PATH, row, file_lock)
                    completed[(item["parent_id"], item["source_candidate_sha256"])] = row
                    return "repaired"
                except (PipelineError, KeyError, TypeError, ValueError) as exc:
                    last_error = str(exc)
                    # 该中转接口偶尔会在上游切换时返回 401 Invalid token；
                    # /models 与随后请求仍可正常访问，因此仅对此特定响应短暂重试一次。
                    invalid_token = "HTTP 401" in last_error and "Invalid token" in last_error
                    transient = invalid_token or any(
                        code in last_error for code in ("HTTP 429", "HTTP 502", "HTTP 503", "HTTP 504")
                    )
                    if transient and attempt < transient_retries:
                        if invalid_token and attempt >= 1:
                            break
                        time.sleep(15 if invalid_token else cooldown)
                        continue
                    break
        append_row(
            ATTEMPTS_PATH,
            {
                "parent_id": item["parent_id"],
                "pair_id": item["pair_id"],
                "source_candidate_sha256": item["source_candidate_sha256"],
                "prompt_sha256": prompt_sha,
                "status": "error",
                "error": last_error,
                "created_at": now(),
            },
            file_lock,
        )
        return f"error: {last_error[:240]}"

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(repair_item, item) for item in pending]
            for index, future in enumerate(as_completed(futures), 1):
                print(f"Repair: {index}/{len(pending)} {future.result()}", flush=True)

    source_by_parent = {item["parent_id"]: item["source_candidate_sha256"] for item in items}
    current_repairs = sorted(
        (
            row
            for row in completed.values()
            if row.get("prompt_sha256") == prompt_sha
            and row["parent_id"] in failure_ids
            and row["source_candidate_sha256"] == source_by_parent[row["parent_id"]]
        ),
        key=lambda row: row["parent_id"],
    )
    latest = {row["parent_id"]: row for row in current_repairs}
    repair_rows = sorted({**applied, **latest}.values(), key=lambda row: row["parent_id"])
    repaired_ids = set(latest)
    for record in records:
        repair = latest.get(record["parent_id"])
        if not repair:
            continue
        current = selected(record)
        current_sha = trajectory_hash(current["trajectory"])
        if current_sha == repair["repaired_candidate_sha256"]:
            continue
        if current_sha != repair["source_candidate_sha256"]:
            raise PipelineError(f"candidate changed outside repair pipeline: {record['parent_id']}")
        for candidate in record["candidates"]:
            candidate["selected"] = False
        record["candidates"].append(repair["candidate"])
        record["status"] = "generated"
        record["review_required"] = True
        record.setdefault("repair_history", []).append(
            {
                "source_candidate_sha256": repair["source_candidate_sha256"],
                "repaired_candidate_sha256": repair["repaired_candidate_sha256"],
                "model": repair["model"],
                "created_at": repair["created_at"],
            }
        )
    write_jsonl(CANDIDATES_PATH, records)
    write_jsonl(REPAIRS_PATH, repair_rows)

    parent_map = parents
    safe_rows = [safe_row(parent_map[record["parent_id"]], selected(record)) for record in records]
    for split in SPLITS:
        write_jsonl(OUTPUT_ROOT / "draft" / f"{split}.jsonl", (row for row in safe_rows if row["split"] == split))

    reviews = read_jsonl(REVIEWS_PATH)
    for review in reviews:
        if review["parent_id"] not in repaired_ids:
            continue
        candidate = selected(by_id[review["parent_id"]])
        review.update(
            {
                "status": "pending",
                "reviewer": "semantic-repair",
                "reason": "候选已按三模型失败证据修复，等待重新语义审核。",
                "candidate_sha256": trajectory_hash(candidate["trajectory"]),
                "reviewed_at": "",
            }
        )
    write_jsonl(REVIEWS_PATH, reviews)

    remaining = len(failures) - len(repaired_ids)
    report = {
        "stage": "coarse_sft/step3/repair",
        "created_at": now(),
        "status": "pass" if remaining == 0 else "incomplete",
        "failed_input": len(failures),
        "repaired": len(repaired_ids),
        "remaining": remaining,
        "models": models,
        "prompt_sha256": prompt_sha,
        "outputs": {
            "attempts": sha256_file(ATTEMPTS_PATH),
            "repairs": sha256_file(REPAIRS_PATH),
            "candidates": sha256_file(CANDIDATES_PATH),
            "reviews": sha256_file(REVIEWS_PATH),
        },
    }
    REPAIR_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    step3 = json.loads(STEP3_REPORT_PATH.read_text(encoding="utf-8"))
    step3["status"] = "awaiting_semantic_audit" if remaining == 0 else "repair_incomplete"
    step3["repair"] = report
    step3["outputs"]["candidates"]["sha256"] = sha256_file(CANDIDATES_PATH)
    step3["outputs"]["reviews"]["sha256"] = sha256_file(REVIEWS_PATH)
    for split in SPLITS:
        step3["outputs"]["draft"][split]["sha256"] = sha256_file(OUTPUT_ROOT / "draft" / f"{split}.jsonl")
    STEP3_REPORT_PATH.write_text(json.dumps(step3, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Repair complete: repaired={len(repaired_ids)} remaining={remaining}")


if __name__ == "__main__":
    try:
        run()
    except (PipelineError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
