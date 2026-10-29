"""只复核剩余争议记录：两种模型盲判，同意gold才收录，否则保留人工裁决。"""

import json
import os
from collections import Counter
from datetime import UTC, datetime

from steps.step0.source import PipelineError
from steps.step3.client import OpenAICompatibleClient
from steps.step4 import pipeline as p


def run(*, dry_run: bool = False, limit: int | None = None) -> None:
    if limit is not None and limit < 1:
        raise PipelineError("--limit必须大于0")
    models = list(dict.fromkeys(x.strip() for x in os.environ.get("STEP4_REVIEW_MODELS", "gpt-6-sol,gpt-6-astra").split(",") if x.strip()))
    if len(models) != 2:
        raise PipelineError("STEP4_REVIEW_MODELS必须配置两种不同模型")
    base_url = os.environ.get("STEP4_BASE_URL", "https://new.mrxiwang.eu.cc/v1").rstrip("/")
    workers = int(os.environ.get("STEP4_WORKERS", "4"))
    interval = float(os.environ.get("STEP4_REQUEST_INTERVAL", "8"))
    max_tokens = int(os.environ.get("STEP4_MAX_TOKENS", "8192"))
    if workers < 1 or interval < 0 or max_tokens < 1:
        raise PipelineError("并发、请求间隔或生成长度无效")
    root = p.OUTPUT_ROOT
    generation_report_path = root / "report.json"
    generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
    generation_path = root / "intermediate/teacher_outputs.jsonl"
    generation_cache = {r["key"]: r for r in p.read_checkpoint(generation_path)}
    rows = {r["id"]: r for r in p.load_inputs()}
    teacher_template = p.TEACHER_PATH.read_text(encoding="utf-8")
    student_template = p.STUDENT_PATH.read_text(encoding="utf-8")
    queue = p.read_jsonl(root / "intermediate/unresolved_cases.jsonl")
    requests, cases = [], []
    for entry in queue:
        row = rows[entry["id"]]
        teacher, student = p.render_inputs(row, teacher_template, student_template)
        original_item = {"id": row["id"], "split": row["split"], "gold": row["target"]["coarse"], "teacher": teacher, "student": student}
        source_key = p.sha256_text(p.stable_json({**original_item, "model": generation_report["model"], "base_url": generation_report["base_url"]}))
        if source_key != entry["key"]:
            raise PipelineError(f"争议队列与当前输入不一致：{row['id']}")
        original = generation_cache[source_key]
        if original["status"] == "generated":
            continue
        if original["status"] != "review_required":
            raise PipelineError(f"争议样本状态异常：{row['id']}")
        case = {"id": row["id"], "source_key": source_key, "gold": row["target"]["coarse"], "review_keys": []}
        # 不向模型发送gold、旧CoT、争议理由或重写意见；复用相同Teacher输入重新判断。
        for model in models:
            key = p.sha256_text(p.stable_json({"source_key": source_key, "teacher": teacher, "model": model, "base_url": base_url}))
            requests.append({"id": row["id"], "split": row["split"], "key": key,
                "teacher": teacher, "gold": row["target"]["coarse"], "review_model": model})
            case["review_keys"].append(key)
        cases.append(case)
        if limit and len(cases) >= limit:
            break
    review_path = root / "intermediate/independent_reviews.jsonl"
    attempts = p.read_checkpoint(review_path) if review_path.exists() else []
    cache = {r["key"]: r for r in attempts}
    pending = [r for r in requests if cache.get(r["key"], {}).get("status") not in {"generated", "review_required"}]
    print(f"Step 4争议复核：{len(cases)}条，模型={','.join(models)}，本次API请求={len(pending)}。", flush=True)
    if dry_run:
        print("预检通过；未调用API，未修改主检查点。")
        return
    if not cases:
        print("没有待复核记录；主数据和已有导出保持不变。")
        return
    clients = {model: OpenAICompatibleClient(model, max_tokens, prefix="STEP4") for model in models} if pending else None
    summary = []
    try:
        p.generate_pending(pending, clients, workers, interval, review_path, cache, attempts,
            {"base_url": base_url, "max_tokens": max_tokens,
             "teacher_prompt_sha256": p.sha256_text(teacher_template),
             "student_prompt_sha256": p.sha256_text(student_template)})
    finally:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for case in cases:
            results = [cache.get(key, {}) for key in case["review_keys"]]
            complete = all(r.get("status") in {"generated", "review_required"} for r in results)
            labels = [r.get("judgment") for r in results]
            accepted = complete and all(label == case["gold"] for label in labels)
            decision = "accepted" if accepted else "api_incomplete" if not complete else "gold_disputed" if len(set(labels)) == 1 else "models_disagree"
            if accepted:
                selected = results[0]
                cot, judgment = p.parse_output(selected["content"])
                original = generation_cache[case["source_key"]]
                updated = {**selected, "key": case["source_key"], "status": "generated", "usage": {},
                    "cot": cot, "judgment": judgment, "created_at": now,
                    "response_source": "independent_review_consensus",
                    "generation_requested_model": generation_report["model"],
                    "consensus_review": {"models": models, "review_keys": case["review_keys"],
                        "source_content_sha256": p.sha256_text(original["content"]), "reviewed_at": now}}
                updated.pop("error", None)
                p.write_jsonl(generation_path, [updated], mode="a")
                generation_cache[case["source_key"]] = updated
            summary.append({"id": case["id"], "gold": case["gold"], "labels": dict(zip(models, labels)),
                            "decision": decision, "review_keys": case["review_keys"]})
        review_report = {"created_at": now, "models": models, "cases": len(cases),
            "counts": dict(Counter(r["decision"] for r in summary)), "results": summary,
            "usage": {k: sum(int(r.get("usage", {}).get(k, 0) or 0) for r in attempts)
                      for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "inputs": {"teacher_prompt_sha256": p.sha256_text(teacher_template)},
            "outputs": {"reviews": "intermediate/independent_reviews.jsonl"}}
        output = root / "review_report.json"
        output.write_text(json.dumps(review_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        counts = Counter()
        for row in rows.values():
            teacher, student = p.render_inputs(row, teacher_template, student_template)
            item = {"id": row["id"], "split": row["split"], "gold": row["target"]["coarse"], "teacher": teacher, "student": student}
            key = p.sha256_text(p.stable_json({**item, "model": generation_report["model"], "base_url": generation_report["base_url"]}))
            counts[generation_cache.get(key, {}).get("status", "pending")] += 1
        generation_report.update(created_at=now, counts=dict(counts), status="ready_to_export" if counts["generated"] == len(rows) else "incomplete")
        generation_report["independent_review_usage"] = review_report["usage"]
        generation_report["outputs"]["checkpoint"]["sha256"] = p.sha256_file(generation_path)
        generation_report["outputs"]["independent_review"] = {"path": "review_report.json", "sha256": p.sha256_file(output)}
        generation_report_path.write_text(json.dumps(generation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"争议复核完成：{review_report['counts']}；主数据={dict(counts)}。")
    print("分歧不会自动改gold；查看review_report.json。全部解决后运行普通step4命令导出。")
