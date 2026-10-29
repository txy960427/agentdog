"""Step 3多模型语义复核：全部检查通过才放行，任一失败即打回。"""

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
from steps.step3.pipeline import OUTPUT_ROOT, PROMPT_ROOT, load_parents, read_jsonl, write_jsonl  # noqa: E402

PROMPT_PATH = PROMPT_ROOT / "safe_twin_semantic_auditor.txt"
INTERMEDIATE = OUTPUT_ROOT / "intermediate"
CANDIDATES_PATH = INTERMEDIATE / "candidates.jsonl"
REVIEWS_PATH = INTERMEDIATE / "reviews.jsonl"
ATTEMPTS_PATH = INTERMEDIATE / "semantic_check_attempts.jsonl"
AUDIT_PATH = INTERMEDIATE / "model_semantic_audit.jsonl"
FAILED_PATH = INTERMEDIATE / "model_semantic_failed.jsonl"
REPORT_PATH = INTERMEDIATE / "model_semantic_audit_report.json"
STEP3_REPORT_PATH = OUTPUT_ROOT / "report.json"

CHECKS = (
    "risk_blocked_before_side_effect",
    "retained_prefix_safe",
    "identifiers_and_facts_preserved",
    "no_fabricated_result_or_success",
    "no_unauthorized_action_or_disclosure",
    "prompt_injection_resisted",
    "useful_safe_resolution",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def selected_candidate(record: dict[str, Any]) -> dict[str, Any]:
    selected = [item for item in record["candidates"] if item.get("selected")]
    if len(selected) != 1:
        raise PipelineError(f"expected one selected candidate: {record['parent_id']}")
    return selected[0]


def candidate_hash(candidate: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(candidate["trajectory"]).encode("utf-8")).hexdigest()


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    checks = result.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError("checks字段不完整")
    if any(not isinstance(checks[name], bool) for name in CHECKS):
        raise ValueError("checks必须全部是boolean")
    passed = result.get("pass")
    failures = result.get("failure_types")
    evidence = result.get("evidence")
    if not isinstance(passed, bool) or not isinstance(failures, list) or not isinstance(evidence, list):
        raise ValueError("pass/failure_types/evidence类型错误")
    if any(not isinstance(item, str) or not item.strip() for item in failures):
        raise ValueError("failure_types必须是非空字符串列表")
    normalized_evidence = []
    for item in evidence:
        if isinstance(item, str) and item.strip():
            normalized_evidence.append(item.strip())
        elif isinstance(item, (dict, list)) and item:
            normalized_evidence.append(stable_json(item))
        else:
            raise ValueError("evidence必须是非空字符串或JSON对象")
    if passed != all(checks.values()) or (passed and failures) or (not passed and not failures):
        raise ValueError("pass、checks和failure_types互相矛盾")
    bad_index = result.get("earliest_bad_event_index")
    if bad_index is not None and (not isinstance(bad_index, int) or bad_index < 0):
        raise ValueError("earliest_bad_event_index无效")
    reason = result.get("reason")
    repair = result.get("repair_instruction")
    if not isinstance(reason, str) or not reason.strip() or not isinstance(repair, str):
        raise ValueError("reason或repair_instruction无效")
    return {
        "pass": passed,
        "checks": checks,
        "failure_types": failures,
        "earliest_bad_event_index": bad_index,
        "evidence": normalized_evidence,
        "reason": reason.strip(),
        "repair_instruction": repair.strip(),
    }


def append_attempt(row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        write_jsonl(ATTEMPTS_PATH, [row], mode="a")


def completed_attempts(prompt_sha: str) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    result = {}
    if not ATTEMPTS_PATH.exists():
        return result
    for row in read_jsonl(ATTEMPTS_PATH):
        if row.get("status") != "completed" or row.get("prompt_sha256") != prompt_sha:
            continue
        key = (row["parent_id"], row["candidate_sha256"], row["model"], row["round"])
        result[key] = row
    return result


def audit_input(parent: dict[str, Any], record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_id": parent["id"],
        "risk_context": record["risk_context"],
        "profile": parent["profile"],
        "available_tools": parent["tools"],
        "original_unsafe_trajectory": [dict(event_index=i, **event) for i, event in enumerate(parent["trajectory"])],
        "candidate_safe_trajectory": [dict(event_index=i, **event) for i, event in enumerate(candidate["trajectory"])],
        "rewrite_event_index": candidate["rewrite_event_index"],
        "construction_mode": candidate.get("construction_mode", "unspecified"),
    }


def aggregate(
    items: list[dict[str, Any]],
    attempts: dict[tuple[str, str, str, int], dict[str, Any]],
    models: list[str],
    rounds: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        checks = [
            attempts.get((item["parent_id"], item["candidate_sha256"], model, round_no))
            for model in models
            for round_no in range(1, rounds + 1)
        ]
        completed = [row for row in checks if row]
        failures = [row for row in completed if not row["result"]["pass"]]
        status = "failed" if failures else "approved" if len(completed) == len(checks) else "pending"
        rows.append(
            {
                "parent_id": item["parent_id"],
                "pair_id": item["pair_id"],
                "split": item["split"],
                "candidate_sha256": item["candidate_sha256"],
                "status": status,
                "required_checks": len(checks),
                "completed_checks": len(completed),
                "models": models,
                "rounds": rounds,
                "failures": [
                    {
                        "model": row["model"],
                        "round": row["round"],
                        **row["result"],
                    }
                    for row in failures
                ],
                "updated_at": now(),
            }
        )
    return rows


def update_reviews(audits: list[dict[str, Any]], prompt_sha: str) -> None:
    old = {row["parent_id"]: row for row in read_jsonl(REVIEWS_PATH)}
    rows = []
    for audit in audits:
        row = old[audit["parent_id"]]
        status = audit["status"]
        if status == "approved":
            review_status = "approved"
            reason = f"多模型语义复核全部通过（{audit['completed_checks']}/{audit['required_checks']}）。"
        elif status == "failed":
            review_status = "rejected"
            first = audit["failures"][0]
            reason = f"{first['model']} round {first['round']}: {first['reason']}"
        else:
            review_status = "pending"
            reason = f"多模型语义复核未完成（{audit['completed_checks']}/{audit['required_checks']}）。"
        rows.append(
            {
                "parent_id": audit["parent_id"],
                "pair_id": audit["pair_id"],
                "required": True,
                "status": review_status,
                "reviewer": f"model-semantic-audit:{prompt_sha[:12]}",
                "reason": reason,
                "candidate_sha256": audit["candidate_sha256"],
                "reviewed_at": now() if review_status != "pending" else "",
            }
        )
    write_jsonl(REVIEWS_PATH, rows)


def update_reports(audits: list[dict[str, Any]], models: list[str], rounds: int, prompt_sha: str) -> None:
    counts = Counter(row["status"] for row in audits)
    report = {
        "stage": "agentdog-1.5/coarse-grained/sft/step3/model-semantic-audit",
        "created_at": now(),
        "status": "pass" if counts["approved"] == len(audits) else "failed" if counts["failed"] else "pending",
        "models": models,
        "rounds": rounds,
        "strict_policy": "all configured checks must pass; any semantic failure rejects the candidate",
        "prompt_sha256": prompt_sha,
        "records": len(audits),
        "counts": dict(counts),
        "outputs": {
            "attempts": {"path": "intermediate/semantic_check_attempts.jsonl", "sha256": sha256_file(ATTEMPTS_PATH)},
            "audit": {"path": "intermediate/model_semantic_audit.jsonl", "sha256": sha256_file(AUDIT_PATH)},
            "failed": {"path": "intermediate/model_semantic_failed.jsonl", "sha256": sha256_file(FAILED_PATH)},
            "reviews": {"path": "intermediate/reviews.jsonl", "sha256": sha256_file(REVIEWS_PATH)},
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    step3 = json.loads(STEP3_REPORT_PATH.read_text(encoding="utf-8"))
    step3["status"] = "pass" if report["status"] == "pass" else "semantic_audit_failed" if counts["failed"] else "awaiting_semantic_audit"
    step3["model_semantic_audit"] = report
    step3["outputs"]["reviews"]["sha256"] = sha256_file(REVIEWS_PATH)
    STEP3_REPORT_PATH.write_text(json.dumps(step3, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def run() -> None:
    models = [item.strip() for item in os.environ.get("STEP3_AUDIT_MODELS", "gpt-6.1-sol,gpt-5.6-sol").split(",") if item.strip()]
    rounds = int(os.environ.get("STEP3_AUDIT_ROUNDS", "2"))
    workers = int(os.environ.get("STEP3_AUDIT_WORKERS", "4"))
    batch_size = int(os.environ.get("STEP3_AUDIT_BATCH_SIZE", "0"))
    max_tokens = int(os.environ.get("STEP3_AUDIT_MAX_TOKENS", "4096"))
    request_interval = float(os.environ.get("STEP3_AUDIT_REQUEST_INTERVAL", "0"))
    rate_limit_retries = int(os.environ.get("STEP3_AUDIT_429_RETRIES", "2"))
    rate_limit_cooldown = float(os.environ.get("STEP3_AUDIT_429_COOLDOWN", "65"))
    dry_run = os.environ.get("STEP3_AUDIT_DRY_RUN", "false").strip().lower() == "true"
    if not models or rounds < 1 or workers < 1 or batch_size < 0:
        raise PipelineError("invalid STEP3_AUDIT_* configuration")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    parents = {row["id"]: row for row in load_parents()}
    records = read_jsonl(CANDIDATES_PATH)
    if len(records) != 600:
        raise PipelineError(f"expected 600 candidates, got {len(records)}")

    items = []
    for record in records:
        candidate = selected_candidate(record)
        parent = parents[record["parent_id"]]
        items.append(
            {
                "parent_id": record["parent_id"],
                "pair_id": record["pair_id"],
                "split": record["split"],
                "candidate_sha256": candidate_hash(candidate),
                "prompt": prompt_template.replace(
                    "{{AUDIT_INPUT_JSON}}",
                    json.dumps(audit_input(parent, record, candidate), ensure_ascii=False, separators=(",", ":")),
                ),
            }
        )

    done = completed_attempts(prompt_sha)
    initial = aggregate(items, done, models, rounds)
    pending_ids = {row["parent_id"] for row in initial if row["status"] == "pending"}
    pending = [item for item in items if item["parent_id"] in pending_ids]
    if batch_size:
        pending = pending[:batch_size]
    if dry_run:
        print(
            f"Semantic audit dry-run: records={len(items)} pending={len(pending_ids)} "
            f"selected={len(pending)} checks_per_record={len(models) * rounds}"
        )
        return

    clients = {model: OpenAICompatibleClient(model, max_tokens) for model in models} if pending else {}
    lock = threading.Lock()
    rate_lock = threading.Lock()
    next_request_at = [0.0]

    def wait_for_rate_limit() -> None:
        if request_interval <= 0:
            return
        with rate_lock:
            scheduled = max(time.monotonic(), next_request_at[0])
            next_request_at[0] = scheduled + request_interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def check_item(item: dict[str, Any]) -> str:
        for model in models:
            for round_no in range(1, rounds + 1):
                key = (item["parent_id"], item["candidate_sha256"], model, round_no)
                previous = done.get(key)
                if previous:
                    if not previous["result"]["pass"]:
                        return "failed"
                    continue
                base = {
                    "parent_id": item["parent_id"],
                    "pair_id": item["pair_id"],
                    "candidate_sha256": item["candidate_sha256"],
                    "prompt_sha256": prompt_sha,
                    "model": model,
                    "round": round_no,
                    "created_at": now(),
                }
                for gateway_attempt in range(rate_limit_retries + 1):
                    response = None
                    try:
                        wait_for_rate_limit()
                        prompt = item["prompt"] + f"\n\nAUDIT_RUN: independent pass {round_no} with model {model}."
                        response = clients[model].complete_json(prompt)
                        result = normalize_result(response["json"])
                        row = {
                            **base,
                            "status": "completed",
                            "result": result,
                            "response_id": response["response_id"],
                            "usage": response["usage"],
                        }
                        append_attempt(row, lock)
                        done[key] = row
                        if not result["pass"]:
                            return "failed"
                        break
                    except (PipelineError, KeyError, TypeError, ValueError) as exc:
                        transient = any(code in str(exc) for code in ("HTTP 429", "HTTP 502", "HTTP 503", "HTTP 504"))
                        if transient and gateway_attempt < rate_limit_retries:
                            print(
                                f"gateway cooldown: {item['parent_id']} {model} "
                                f"retry {gateway_attempt + 1}/{rate_limit_retries}",
                                flush=True,
                            )
                            time.sleep(rate_limit_cooldown)
                            continue
                        error = {**base, "status": "error", "error": str(exc)}
                        if response:
                            error["raw_response"] = response.get("content", "")
                        append_attempt(error, lock)
                        return "pending"
        return "approved"

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(check_item, item) for item in pending]
            for index, future in enumerate(as_completed(futures), 1):
                print(f"Semantic audit: {index}/{len(pending)} {future.result()}", flush=True)

    done = completed_attempts(prompt_sha)
    audits = aggregate(items, done, models, rounds)
    write_jsonl(AUDIT_PATH, audits)
    write_jsonl(FAILED_PATH, (row for row in audits if row["status"] == "failed"))
    update_reviews(audits, prompt_sha)
    update_reports(audits, models, rounds, prompt_sha)
    counts = Counter(row["status"] for row in audits)
    print(f"Semantic audit complete: {dict(counts)}")


if __name__ == "__main__":
    try:
        run()
    except (PipelineError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
