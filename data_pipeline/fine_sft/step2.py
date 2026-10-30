"""Step 2：调用Teacher生成细粒度CoT，并导出SFT Parquet。"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data_pipeline/coarse_sft"))

from judge_eval import prompt_builder as renderer  # noqa: E402
from steps.step0.source import PipelineError, sha256_file, sha256_text, stable_json  # noqa: E402
from steps.step3.client import OpenAICompatibleClient  # noqa: E402

STEP1 = PROJECT_ROOT / "data/1.5/fine_sft/step1"
OUTPUT_ROOT = PROJECT_ROOT / "data/1.5/fine_sft/step2"
TEACHER_PATH = PROJECT_ROOT / "docs/agentdog15_reproduction/stages/02_judge_fine_sft_data/prompts/fine_cot_teacher.txt"
STUDENT_PATH = PROJECT_ROOT / "docs/agentdog15_reproduction/stages/02_judge_fine_sft_data/prompts/fine_sft_student.txt"
EXPECTED_TOTAL = 4000
SPLITS = ("train", "validation", "test")
FIELDS = ("risk_source", "failure_mode", "risk_consequence")
CATEGORIES = {
    "risk_source": renderer.RISK_SOURCES,
    "failure_mode": renderer.FAILURE_MODES,
    "risk_consequence": renderer.REAL_WORLD_HARMS,
}
OUTPUT_RE = re.compile(
    r"\A<Analysis>\s*\r?\n"
    r"What happened:[ \t]*(.*?)\r?\n"
    r"Failure Mode:[ \t]*(.*?)\r?\n"
    r"Risk Consequence:[ \t]*(.*?)\r?\n"
    r"Risk Source:[ \t]*(.*?)\r?\n"
    r"</Analysis>\s*\r?\n"
    r"Failure Mode:[ \t]*([^\r\n]+)\s*\r?\n"
    r"Risk Consequence:[ \t]*([^\r\n]+)\s*\r?\n"
    r"Risk Source:[ \t]*([^\r\n]+)\s*\Z",
    re.DOTALL,
)
LEAK_RE = re.compile(
    r"\b(?:oracle|confuser|gold|given labels?|correct labels?|internal guidance|plausible alternative|hint(?:ed)? labels?)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise PipelineError(f"missing input: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    """保留所有完整记录，只移除断电留下的不完整末行。"""
    if not path.exists():
        return []
    rows, offset = [], 0
    with path.open("rb+") as stream:
        lines = stream.readlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(lines) - 1:
                    raise PipelineError("checkpoint中间存在损坏记录")
                stream.truncate(offset)
                break
            offset += len(line)
        if lines and rows and not lines[-1].endswith(b"\n"):
            stream.seek(0, 2)
            stream.write(b"\n")
    return rows


def load_rows() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    report_path = STEP1 / "report.json"
    if not report_path.exists():
        raise PipelineError("Step 1 report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("audit", {}).get("status") != "pass":
        raise PipelineError("Step 1 audit has not passed")
    split_counts = {split: int(report.get("outputs", {}).get(split, {}).get("records", 0)) for split in SPLITS}
    if sum(split_counts.values()) != EXPECTED_TOTAL or any(count < 1 for count in split_counts.values()):
        raise PipelineError("Step 1 must contain all 4000 records")
    rows = []
    for split, expected in split_counts.items():
        path = STEP1 / f"final/{split}.jsonl"
        info = report["outputs"][split]
        if sha256_file(path) != info["sha256"]:
            raise PipelineError(f"Step 1/{split} hash mismatch")
        current = read_jsonl(path)
        if len(current) != expected or any(row["split"] != split for row in current):
            raise PipelineError(f"Step 1/{split} count or split mismatch")
        rows.extend(current)
    return rows, report, split_counts


def choose_confuser(row: dict[str, Any], field: str, train: list[dict[str, Any]]) -> str:
    """只用Train统计：先固定另外两维，再放宽到固定任意一维。"""
    gold = row["target"][field]
    others = [name for name in FIELDS if name != field]
    exact = Counter(
        item["target"][field]
        for item in train
        if item["target"][field] != gold
        and all(item["target"][name] == row["target"][name] for name in others)
    )
    relaxed = Counter(
        item["target"][field]
        for item in train
        if item["target"][field] != gold
        and any(item["target"][name] == row["target"][name] for name in others)
    )
    counts = exact or relaxed
    order = {name: index for index, name in enumerate(CATEGORIES[field])}
    if counts:
        return min(counts, key=lambda name: (-counts[name], order[name]))
    names = CATEGORIES[field]
    return names[(names.index(gold) + 1) % len(names)]


def render_item(
    row: dict[str, Any], train: list[dict[str, Any]], teacher_template: str, student_template: str,
    model: str, base_url: str,
) -> dict[str, Any]:
    prompt_row = {"profile": row["profile"], "tool_used": row["tools"], "contents": [row["trajectory"]]}
    fields = renderer.build_prompt_fields("atbench", prompt_row, task="fine")
    student = renderer.render_prompt(student_template, fields)
    target = row["target"]
    confusers = {field: choose_confuser(row, field, train) for field in FIELDS}
    teacher = teacher_template.format(
        failure_mode_categories=renderer.category_block(renderer.FAILURE_MODES),
        risk_consequence_categories=renderer.category_block(renderer.REAL_WORLD_HARMS),
        risk_source_categories=renderer.risk_source_categories_text(),
        tools_json=fields.tool_list_text,
        trajectory=fields.trajectory,
        risk_source=target["risk_source"],
        failure_mode=target["failure_mode"],
        real_world_harm=target["risk_consequence"],
        rs_confuser=confusers["risk_source"],
        fm_confuser=confusers["failure_mode"],
        rwh_confuser=confusers["risk_consequence"],
    )
    item = {
        "id": row["id"], "split": row["split"], "gold": {field: target[field] for field in FIELDS},
        "teacher": teacher, "student": student, "confusers": confusers,
    }
    item["key"] = sha256_text(stable_json({**item, "model": model, "base_url": base_url}))
    return item


def parse_output(content: str) -> tuple[str, dict[str, str], str]:
    text = content.strip()
    if text.count("<Analysis>") != 1 or text.count("</Analysis>") != 1:
        raise ValueError("Analysis标签缺失或重复")
    match = OUTPUT_RE.fullmatch(text)
    if not match:
        raise ValueError("输出结构错误或标签行之外存在文字")
    answers = [match[index].strip() for index in range(1, 5)]
    if any(len(answer) < 20 or answer.startswith("[") and answer.endswith("]") for answer in answers):
        raise ValueError("Analysis为空、过短或仍是占位文字")
    analysis = (
        f"What happened: {answers[0]}\n"
        f"Failure Mode: {answers[1]}\n"
        f"Risk Consequence: {answers[2]}\n"
        f"Risk Source: {answers[3]}"
    )
    if LEAK_RE.search(analysis):
        raise ValueError("Analysis泄漏Teacher内部提示")
    labels = {
        "failure_mode": match[5].strip(),
        "risk_consequence": match[6].strip(),
        "risk_source": match[7].strip(),
    }
    for field, label in labels.items():
        if label not in CATEGORIES[field]:
            raise ValueError(f"unknown {field}: {label}")
    canonical = (
        f"<Analysis>\n{analysis}\n</Analysis>\n\n"
        f"Failure Mode: {labels['failure_mode']}\n"
        f"Risk Consequence: {labels['risk_consequence']}\n"
        f"Risk Source: {labels['risk_source']}"
    )
    return analysis, labels, canonical


def gold_contradictions(analysis: str, gold: dict[str, str]) -> list[str]:
    """只拦截明确否定gold的措辞；一般语义质量留给独立审核。"""
    warnings = []
    for field, label in gold.items():
        pattern = re.compile(
            re.escape(label) + r".{0,60}\b(?:is|was|are|were)\s+not\s+"
            r"(?:demonstrated|supported|established|evidenced|shown|present)\b",
            re.IGNORECASE | re.DOTALL,
        )
        if pattern.search(analysis):
            warnings.append(field)
    return warnings


def classify_analysis(analysis: str, labels: dict[str, str], gold: dict[str, str]) -> tuple[str, str | None, list[str]]:
    if labels != gold:
        return "review_required", "Teacher labels conflict with gold", []
    warnings = gold_contradictions(analysis, gold)
    if warnings:
        return "review_required", f"Analysis explicitly rejects gold: {','.join(warnings)}", warnings
    return "generated", None, []


def retry_delay(error: str, attempt: int) -> float:
    http = re.search(r"HTTP (\d{3})", error)
    if http and http[1] == "429":
        seconds = 65
    elif http and http[1] in {"502", "503"}:
        seconds = 30
    else:
        seconds = 2 ** attempt
    advised = re.search(r'"retry_after"\s*:\s*(\d+)', error)
    return max(seconds, int(advised[1])) if advised else seconds


def generate(
    pending: list[dict[str, Any]], client: OpenAICompatibleClient, checkpoint: Path,
    cache: dict[str, dict[str, Any]], workers: int, interval: float, metadata: dict[str, Any],
) -> None:
    """有界并发、全局限速；每条最终尝试立即写入断点。"""
    if not pending:
        return
    client.retries = 1
    queue = deque(pending)
    active: dict[Any, dict[str, Any]] = {}
    stop = threading.Event()
    rate_lock = threading.Lock()
    next_request = 0.0
    finished = 0

    def wait_slot() -> bool:
        nonlocal next_request
        while not stop.is_set():
            with rate_lock:
                delay = max(0.0, next_request - time.monotonic())
                if delay == 0:
                    next_request = time.monotonic() + interval
                    return True
            stop.wait(min(delay, 1.0))
        return False

    def worker(item: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        nonlocal next_request
        attempts = []
        for attempt in range(1, 4):
            if not wait_slot():
                return attempts, False
            result = {
                **metadata, "id": item["id"], "split": item["split"], "key": item["key"],
                "attempt": attempt, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            started = time.monotonic()
            try:
                response = client.complete_text(item["teacher"])
                result.update(response)
                if response["finish_reason"] != "stop":
                    raise ValueError(f"generation finish_reason={response['finish_reason']}")
                analysis, labels, canonical = parse_output(response["content"])
                status, error, warnings = classify_analysis(analysis, labels, item["gold"])
                result.update(
                    analysis=analysis, labels=labels, canonical=canonical, labels_source="teacher",
                    semantic_warnings=warnings, status=status,
                )
                if error:
                    result["error"] = error
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
                attempts.append(result)
                return attempts, False
            except ValueError as exc:
                result.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - started, 3))
                attempts.append(result)
                if attempt == 3:
                    return attempts, False
            except PipelineError as exc:
                error = str(exc)
                result.update(status="failed", error=error, elapsed_seconds=round(time.monotonic() - started, 3))
                attempts.append(result)
                http = re.search(r"HTTP (\d{3})", error)
                transient = not http or int(http[1]) in {408, 409, 429, 500, 502, 503, 504}
                if not transient or attempt == 3:
                    return attempts, not transient
                cooldown = retry_delay(error, attempt)
                with rate_lock:
                    next_request = max(next_request, time.monotonic() + cooldown)
        return attempts, False

    def collect(future: Any) -> bool:
        nonlocal finished
        item = active.pop(future)
        attempts, fatal = future.result()
        if attempts:
            append_jsonl(checkpoint, attempts)
            cache[item["key"]] = attempts[-1]
            finished += 1
            print(f"Fine Step 2: {finished}/{len(pending)} {item['id']} {attempts[-1]['status']}", flush=True)
        return fatal

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while (queue and not stop.is_set()) or active:
            while queue and len(active) < workers and not stop.is_set():
                item = queue.popleft()
                active[pool.submit(worker, item)] = item
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            if any(collect(future) for future in done):
                stop.set()
                print("检测到不可重试的API错误，已停止派发新请求。", flush=True)
    except KeyboardInterrupt:
        stop.set()
        print("已停止派发新请求，等待在途结果保存。", flush=True)
    finally:
        stop.set()
        for future in list(active):
            if not future.cancel():
                collect(future)
        pool.shutdown(wait=True, cancel_futures=True)


def export_parquet(
    items: list[dict[str, Any]], cache: dict[str, dict[str, Any]], split_counts: dict[str, int]
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    final = OUTPUT_ROOT / "final"
    final.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, expected in split_counts.items():
        rows = []
        for item in items:
            if item["split"] != split:
                continue
            result = cache[item["key"]]
            if result["status"] != "generated" or result["labels"] != item["gold"]:
                raise PipelineError(f"cannot export unresolved item: {item['id']}")
            rows.append({
                "id": item["id"], "split": split, "gold": item["gold"], "enable_thinking": False,
                "messages": [
                    {"role": "user", "content": item["student"]},
                    {"role": "assistant", "content": result["canonical"]},
                ],
            })
        if len(rows) != expected:
            raise PipelineError(f"{split} export count mismatch")
        path = final / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(path)
        outputs[split] = {"records": len(rows), "sha256": sha256_file(path)}
    return outputs


def run_step2(*, dry_run: bool = False, limit: int | None = None) -> None:
    if limit is not None and limit < 1:
        raise PipelineError("--limit must be greater than zero")
    rows, step1_report, split_counts = load_rows()
    train = [row for row in rows if row["split"] == "train"]
    teacher_template = TEACHER_PATH.read_text(encoding="utf-8")
    student_template = STUDENT_PATH.read_text(encoding="utf-8")
    model = os.environ.get("FINE_STEP2_MODEL", "gpt-6.1-sol")
    base_url = os.environ.get("FINE_STEP2_BASE_URL", "https://new.mrxiwang.eu.cc/v1").rstrip("/")
    workers = int(os.environ.get("FINE_STEP2_WORKERS", "4"))
    interval = float(os.environ.get("FINE_STEP2_REQUEST_INTERVAL", "8"))
    max_tokens = int(os.environ.get("FINE_STEP2_MAX_TOKENS", "8192"))
    if workers < 1 or interval < 0 or max_tokens < 1:
        raise PipelineError("invalid workers, request interval, or max tokens")
    # 小批试跑交错覆盖三个split；不改变任何样本的冻结归属。
    ordered_rows = [
        row
        for group in zip_longest(*([item for item in rows if item["split"] == split] for split in SPLITS))
        for row in group if row is not None
    ]
    items = [render_item(row, train, teacher_template, student_template, model, base_url) for row in ordered_rows]

    if dry_run:
        lengths = sorted(len(item["teacher"]) for item in items)
        print(
            f"Fine Step 2 dry-run: {len(items)} items; "
            f"teacher chars min/median/p95/max={lengths[0]}/{lengths[len(lengths)//2]}/"
            f"{lengths[int(len(lengths)*0.95)]}/{lengths[-1]}"
        )
        print("Prompt rendering and confusers passed; no API call and no output created.")
        return

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise PipelineError("install export dependency first: python -m pip install pyarrow") from exc

    intermediate = OUTPUT_ROOT / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    checkpoint = intermediate / "teacher_outputs.jsonl"
    attempts = read_checkpoint(checkpoint)
    cache = {row["key"]: row for row in attempts}
    pending = [item for item in items if cache.get(item["key"], {}).get("status") not in {"generated", "review_required"}]
    if limit:
        pending = pending[:limit]
    counts = Counter(cache.get(item["key"], {}).get("status", "pending") for item in items)
    print(
        f"Fine Step 2 resume: generated={counts['generated']}, review_required={counts['review_required']}, "
        f"this_run={len(pending)}, workers={workers}, global_interval={interval}s",
        flush=True,
    )
    metadata = {
        "requested_model": model, "base_url": base_url, "max_tokens": max_tokens,
        "teacher_prompt_sha256": sha256_file(TEACHER_PATH),
        "student_prompt_sha256": sha256_file(STUDENT_PATH),
        "renderer_sha256": sha256_file(Path(renderer.__file__)),
    }
    if pending:
        client = OpenAICompatibleClient(model, max_tokens, prefix="FINE_STEP2")
        generate(pending, client, checkpoint, cache, workers, interval, metadata)

    attempts = read_checkpoint(checkpoint)
    counts = Counter(cache.get(item["key"], {}).get("status", "pending") for item in items)
    reviews = [
        {"id": item["id"], "split": item["split"], "key": item["key"], "gold": item["gold"],
         "labels": cache[item["key"]]["labels"], "error": cache[item["key"]].get("error")}
        for item in items if cache.get(item["key"], {}).get("status") == "review_required"
    ]
    review_path = intermediate / "reviews.jsonl"
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reviews),
        encoding="utf-8", newline="\n",
    )
    status = "ready_to_export" if counts["generated"] == len(items) else "review_required" if reviews else "incomplete"
    if status != "ready_to_export" and (OUTPUT_ROOT / "final").exists():
        shutil.rmtree(OUTPUT_ROOT / "final")
    report = {
        "stage": "fine_sft/step2", "version": "2.0.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "status": status,
        "counts": dict(counts), "model": model, "base_url": base_url,
        "workers": workers, "request_interval_seconds": interval, "max_tokens": max_tokens,
        "inputs": {split: step1_report["outputs"][split] for split in SPLITS},
        "teacher_prompt_sha256": sha256_file(TEACHER_PATH),
        "student_prompt_sha256": sha256_file(STUDENT_PATH),
        "renderer_sha256": sha256_file(Path(renderer.__file__)),
        "checks": "format, three-label space, gold consistency, leakage words; semantic audit is separate",
        "usage": {
            name: sum(
                int(row.get("usage", {}).get(name, 0) or 0)
                for row in attempts if row.get("key") in {item["key"] for item in items}
            )
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "outputs": {
            "checkpoint": {"records": len(attempts), "sha256": sha256_file(checkpoint) if checkpoint.exists() else None},
            "reviews": {"records": len(reviews), "sha256": sha256_file(review_path)},
        },
    }
    if status == "ready_to_export":
        report["outputs"]["final"] = export_parquet(items, cache, split_counts)
        report["status"] = "pass"
    report_path = OUTPUT_ROOT / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(report_path)
    print(f"Fine Step 2 complete: {report['status']}, {dict(counts)}")
