"""Step 4：四问CoT生成、断点保存与SFT导出。"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

from steps.step0.source import PROJECT_ROOT, PipelineError, sha256_file, sha256_text, stable_json
from steps.step3.client import OpenAICompatibleClient
from steps.step3.pipeline import read_jsonl, write_jsonl
from steps.step4.inputs import prepare_row

sys.path.insert(0, str(PROJECT_ROOT))
from judge_eval import prompt_builder as renderer  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data/1.5/coarse_sft"
OUTPUT_ROOT = DATA_ROOT / "step4"
TEACHER_PATH = PROJECT_ROOT / "docs/agentdog15_reproduction/stages/01_judge_coarse_sft_data/prompts/coarse_cot_teacher.txt"
STUDENT_PATH = PROJECT_ROOT / "judge_eval/prompt.txt"
SPLITS = {"train": 1000, "validation": 250, "test": 250}
RATE_LIMIT_COOLDOWN_SECONDS = 65
OUTPUT_RE = re.compile(
    r"\A<Analysis>\s*(.*?)\s*</Analysis>\s*<Judgment>\s*(safe|unsafe)\s*</Judgment>\Z",
    re.DOTALL,
)


def load_inputs(preview: bool = False) -> list[dict]:
    """正式生成只读验收后的final；dry-run允许预览尚未通过的Step 3 draft。"""
    selection = {row["id"]: row for row in read_jsonl(DATA_ROOT / "step0/final/selection.jsonl")}
    components, memberships, ids = [], {}, set()
    for stage, counts in ((1, (200, 50, 50)), (2, (400, 100, 100)), (3, (400, 100, 100))):
        root = DATA_ROOT / f"step{stage}"
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        ready = report.get("audit", {}).get("status") == "pass" if stage < 3 else report.get("status") == "pass" and "final" in report["outputs"]
        if not ready and not (preview and stage == 3):
            raise PipelineError(f"Step {stage}尚未验收并导出final，不能调用CoT API")
        if stage == 3 and ready:
            model_audit = root / "intermediate/model_semantic_audit_report.json"
            if model_audit.exists() and json.loads(model_audit.read_text(encoding="utf-8"))["status"] != "pass":
                raise PipelineError("Step 3多模型语义复核尚未全部通过")
            reviews = root / "intermediate/reviews.jsonl"
            if sha256_file(reviews) != report.get("human_review", {}).get("reviews_sha256"):
                raise PipelineError("Step 3审核记录已变化，请重新验收final")
            if any(row["status"] != "approved" for row in read_jsonl(reviews)):
                raise PipelineError("Step 3仍有未通过的审核")
        component = []
        for split, expected in zip(SPLITS, counts):
            info = report["outputs"][split] if stage < 3 else report["outputs"]["final" if ready else "draft"][split]
            path = root / info["path"]
            if sha256_file(path) != info["sha256"]:
                raise PipelineError(f"Step {stage}/{split}的内容与验收hash不一致")
            rows = read_jsonl(path)
            if len(rows) != expected:
                raise PipelineError(f"Step {stage}/{split}条数不符")
            for row in rows:
                parent = selection[row["source"]["parent_id"] if stage == 3 else row["id"]]
                if row["split"] != split or any(row[key] != parent[key] for key in ("group_id", "pair_id", "split")):
                    raise PipelineError(f"冻结split或parent映射不符：{row['id']}")
                for key in ("group_id", "pair_id"):
                    value = row[key]
                    if value and memberships.setdefault((key, value), split) != split:
                        raise PipelineError(f"{key}跨split：{value}")
                if row["id"] in ids:
                    raise PipelineError(f"重复ID：{row['id']}")
                ids.add(row["id"])
            component.extend(rows)
        if stage == 3 and not ready:
            print("DRY-RUN：Step 3尚未通过，仅读取draft预览，不调用API、不导出。")
        components.append(component)
    # 小批试跑也能覆盖无工具、带工具unsafe和safe twin三种来源。
    rows = [row for group in zip_longest(*components) for row in group if row is not None]
    for split, expected in SPLITS.items():
        labels = Counter(row["target"]["coarse"] for row in rows if row["split"] == split)
        if labels != {"safe": expected // 2, "unsafe": expected // 2}:
            raise PipelineError(f"{split}数量或类别比例不符：{labels}")
    return rows


def render_inputs(row: dict, teacher_template: str, student_template: str) -> tuple[str, str]:
    """Teacher与Student共享同一份格式化证据，不传标签和构建元数据。"""
    row, _ = prepare_row(row)
    conversation = renderer.format_conversation({"profile": row["profile"], "contents": [row["trajectory"]]})
    header, history = conversation.split("=== Conversation History ===", 1)
    profile = header.removeprefix("=== Agent Profile ===").strip()
    tools = renderer.format_tools_prose_for_judge(renderer.visible_tools(row["tools"]))
    if not tools:
        tools = renderer.unknown_tools_text(row["trajectory"])
    teacher = teacher_template.format(profile=profile, tool_list_text=tools, trajectory=history.strip())
    student = student_template.format(trajectory=renderer.insert_available_tools_after_profile(conversation, tools))
    return teacher, student


def parse_output(content: str) -> tuple[str, str]:
    """只接收四个非空分析段和一个小写判断，拒绝模板照抄和重复标签。"""
    text = content.strip()
    if any(text.count(tag) != 1 for tag in ("<Analysis>", "</Analysis>", "<Judgment>", "</Judgment>")):
        raise ValueError("缺少或重复Analysis/Judgment标签")
    match = OUTPUT_RE.fullmatch(text)
    if not match:
        raise ValueError("输出格式错误或标签外有文字")
    parts = re.split(r"(?m)^\s*(\d+)\.[ \t]*", match[1].strip())
    if len(parts) != 9 or parts[0].strip() or parts[1::2] != ["1", "2", "3", "4"]:
        raise ValueError("Analysis必须按顺序回答1、2、3、4四问")
    answers = [part.strip() for part in parts[2::2]]
    if any(not answer or answer.startswith("[") and answer.endswith("]") for answer in answers):
        raise ValueError("分析为空或仅照抄占位说明")
    return "\n\n".join(f"{i}. {answer}" for i, answer in enumerate(answers, 1)), match[2]


def read_checkpoint(path: Path) -> list[dict]:
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
                    raise PipelineError("CoT检查点中间存在损坏记录，请先检查文件")
                stream.truncate(offset)
                print("已移除断电留下的不完整末行；此前完成的记录全部保留。")
                break
            offset += len(line)
        # 保证下一次追加不会与一个完整但缺少换行的末行粘连。
        if lines and rows and offset == sum(map(len, lines)) and not lines[-1].endswith(b"\n"):
            stream.seek(0, 2)
            stream.write(b"\n")
    return rows


def export_parquet(items: list[dict], cache: dict, root: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, expected in SPLITS.items():
        rows = []
        for item in items:
            if item["split"] != split:
                continue
            answer = cache[item["key"]]
            cot, judgment = parse_output(answer["content"])
            if answer["status"] != "generated" or judgment != item["gold"] or (cot, judgment) != (answer["cot"], answer["judgment"]):
                raise PipelineError(f"导出记录的状态、CoT或标签不一致：{item['id']}")
            rows.append({
                "id": item["id"], "split": split, "gold": item["gold"], "enable_thinking": False,
                "messages": [
                    {"role": "user", "content": item["student"]},
                    {"role": "assistant", "content": f"<Analysis>\n{answer['cot']}\n</Analysis>\n\n<Judgment> {answer['judgment']} </Judgment>"},
                ],
            })
        if len(rows) != expected:
            raise PipelineError(f"{split}导出条数错误")
        path = root / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(path)
        outputs[split] = {"path": f"final/{path.name}", "records": len(rows), "sha256": sha256_file(path)}
    return outputs


def retry_delay(error: str, attempt: int) -> float:
    """尊重网关的retry_after；过载时不要立即重打同一模型。"""
    http = re.search(r"HTTP (\d{3})", error)
    if http and http[1] == "429":
        seconds = RATE_LIMIT_COOLDOWN_SECONDS
    elif http and http[1] in {"502", "503"}:
        seconds = 30
    else:
        seconds = 2 ** attempt
    advised = re.search(r'"retry_after"\s*:\s*(\d+)', error)
    return max(seconds, int(advised[1])) if advised else seconds


def generate_pending(pending: list[dict], client, workers: int, interval: float,
                     checkpoint: Path, cache: dict, attempts: list[dict], metadata: dict) -> None:
    """有界并发请求；所有worker共用启动间隔，仅主线程逐条写检查点。"""
    if not pending:
        return
    # 统一在本层调度重试，防止客户端内部重试绕过全局RPM节流。
    for request_client in client.values() if isinstance(client, dict) else [client]:
        request_client.retries = 1
    stopping = threading.Event()
    rate_lock = threading.Lock()
    next_request = 0.0
    todo = deque((item, 1) for item in pending)
    active = {}
    finished = 0

    def request(item: dict, attempt: int):
        nonlocal next_request
        # 不持锁睡眠；429到来时主线程可以及时延长所有worker的等待时间。
        while not stopping.is_set():
            with rate_lock:
                delay = max(0.0, next_request - time.monotonic())
                if delay == 0:
                    next_request = time.monotonic() + interval
                    break
            if stopping.wait(delay):
                return None, False
        if stopping.is_set():
            return None, False
        result = {**metadata, "id": item["id"], "split": item["split"], "key": item["key"],
                  "attempt": attempt, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        request_client = client[item["review_model"]] if isinstance(client, dict) else client
        if isinstance(client, dict):
            result["requested_model"] = item["review_model"]
        started = time.monotonic()
        api_error = False
        try:
            response = request_client.complete_text(item["teacher"])
            result.update(response)
            if response["finish_reason"] != "stop":
                raise PipelineError(f"生成未正常结束：{response['finish_reason']}")
            cot, judgment = parse_output(response["content"])
            result.update(cot=cot, judgment=judgment,
                          status="generated" if judgment == item["gold"] else "review_required")
            if result["status"] == "review_required":
                result["error"] = "Teacher判断与gold冲突；保留原文，请人工复核"
        except ValueError as exc:
            result.update(status="failed", error=str(exc))
        except PipelineError as exc:
            result.update(status="failed", error=str(exc))
            api_error = True
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return result, api_error

    def collect(future, *, allow_retry: bool = True) -> None:
        nonlocal next_request, finished
        item, attempt = active.pop(future)
        if future.cancelled():
            return
        result, api_error = future.result()
        if result is None:
            return
        write_jsonl(checkpoint, [result], mode="a")
        cache[item["key"]] = result
        attempts.append(result)
        error = result.get("error", "")
        http = re.search(r"HTTP (\d{3})", error)
        transient = api_error and (
            http and int(http[1]) in {408, 409, 429, 500, 502, 503, 504}
            or not http and error.startswith("API request failed")
        )
        if result["status"] == "failed" and attempt < 3 and allow_retry and not stopping.is_set() and (not api_error or transient):
            if api_error:
                cooldown = retry_delay(error, attempt)
                with rate_lock:
                    next_request = max(next_request, time.monotonic() + cooldown)
                print(f"请求暂时失败，统一等待{cooldown}秒后重试：{item['id']} {error}", flush=True)
            todo.appendleft((item, attempt + 1))
            return
        finished += 1
        print(f"Step 4: {finished}/{len(pending)} {item['id']} {result['status']}", flush=True)
        if api_error and allow_retry:
            stopping.set()
            print(f"停止派发新请求：{error}；正在进行的请求完成后仍会保存。", flush=True)

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while (todo and not stopping.is_set()) or active:
            while todo and len(active) < workers and not stopping.is_set():
                item, attempt = todo.popleft()
                active[pool.submit(request, item, attempt)] = (item, attempt)
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                collect(future)
    except KeyboardInterrupt:
        print("已停止派发新请求，等待在途响应写入检查点后退出。", flush=True)
        raise
    finally:
        stopping.set()
        for future in active:
            future.cancel()
        # 中断或API错误时也保存已发出请求的返回，避免白白消耗token。
        for future in as_completed(list(active)):
            collect(future, allow_retry=False)
        pool.shutdown(wait=True, cancel_futures=True)


def run(*, dry_run: bool = False, limit: int | None = None) -> None:
    if limit is not None and limit < 1:
        raise PipelineError("--limit必须大于0")
    rows = load_inputs(preview=dry_run)
    teacher_template = TEACHER_PATH.read_text(encoding="utf-8")
    student_template = STUDENT_PATH.read_text(encoding="utf-8")
    model = os.environ.get("STEP4_MODEL", "gpt-6.1-sol")
    base_url = os.environ.get("STEP4_BASE_URL", "https://new.mrxiwang.eu.cc/v1").rstrip("/")
    interval = float(os.environ.get("STEP4_REQUEST_INTERVAL", "8"))
    workers = int(os.environ.get("STEP4_WORKERS", "4"))
    max_tokens = int(os.environ.get("STEP4_MAX_TOKENS", "8192"))
    if interval < 0 or max_tokens < 1 or workers < 1:
        raise PipelineError("并发数、请求间隔或生成长度无效")
    items = []
    for row in rows:
        teacher, student = render_inputs(row, teacher_template, student_template)
        item = {"id": row["id"], "split": row["split"], "gold": row["target"]["coarse"], "teacher": teacher, "student": student}
        item["key"] = sha256_text(stable_json({**item, "model": model, "base_url": base_url}))
        items.append(item)
    if dry_run:
        print(f"Step 4 dry-run：{len(items)}条，Train/Validation/Test=1000/250/250；最大Teacher输入{max(len(x['teacher']) for x in items)}字符。")
        print("未调用API，未创建生成结果。")
        return
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise PipelineError("请先安装导出依赖：python -m pip install pyarrow") from exc

    intermediate = OUTPUT_ROOT / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    checkpoint = intermediate / "teacher_outputs.jsonl"
    attempts = read_checkpoint(checkpoint)
    cache = {item["key"]: item for item in attempts}
    pending = [item for item in items if cache.get(item["key"], {}).get("status") not in {"generated", "review_required"}]
    if limit:
        pending = pending[:limit]
    existing = Counter(cache.get(item["key"], {}).get("status", "pending") for item in items)
    print(f"Step 4续跑：已生成{existing['generated']}条，待复核{existing['review_required']}条，本次处理{len(pending)}条；并发={workers}，全局请求间隔={interval}秒。", flush=True)
    client = OpenAICompatibleClient(model, max_tokens, prefix="STEP4") if pending else None
    try:
        metadata = {"requested_model": model, "base_url": base_url,
                    "teacher_prompt_sha256": sha256_text(teacher_template),
                    "student_prompt_sha256": sha256_text(student_template), "max_tokens": max_tokens}
        generate_pending(pending, client, workers, interval, checkpoint, cache, attempts, metadata)
    finally:
        counts = Counter(cache.get(item["key"], {}).get("status", "pending") for item in items)
        report = {"stage": "coarse_sft/step4", "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  "status": "ready_to_export" if counts["generated"] == len(items) else "incomplete", "counts": dict(counts),
                  "teacher_prompt_sha256": sha256_file(TEACHER_PATH), "student_prompt_sha256": sha256_file(STUDENT_PATH),
                  "renderer_sha256": sha256_file(Path(renderer.__file__)), "model": model, "base_url": base_url,
                  "workers": workers, "request_interval_seconds": interval,
                  "input_adapter_sha256": sha256_file(Path(__file__).with_name("inputs.py")),
                  "input_corrections_sha256": sha256_file(Path(__file__).with_name("input_corrections.json")),
                  "conflict_repairs_sha256": sha256_file(Path(__file__).with_name("conflict_repairs.json")),
                  "training_token_cutoff_checked": False,
                  "checks": "四问格式与标签一致性；不等于独立语义审核",
                  "usage": {key: sum(int(row.get("usage", {}).get(key, 0) or 0) for row in attempts) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                  "outputs": {"checkpoint": {"path": "intermediate/teacher_outputs.jsonl", "sha256": sha256_file(checkpoint) if checkpoint.exists() else None}}}
        # 普通续跑/导出不能丢失此前人工修正、盲判的来源和额外API用量。
        for name, filename in (("manual_conflict_review", "manual_conflict_review.json"), ("independent_review", "review_report.json")):
            audit_path = OUTPUT_ROOT / filename
            if audit_path.exists():
                report["outputs"][name] = {"path": filename, "sha256": sha256_file(audit_path)}
                if name == "independent_review":
                    report["independent_review_usage"] = json.loads(audit_path.read_text(encoding="utf-8"))["usage"]
        report_path = OUTPUT_ROOT / "report.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(report_path)
    if report["status"] == "ready_to_export":
        report["outputs"]["final"] = export_parquet(items, cache, OUTPUT_ROOT / "final")
        report["status"] = "pass"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(report_path)
    print(f"Step 4 complete: {report['status']}, {dict(counts)}")
