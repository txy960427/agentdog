"""只审核最终粗粒度SFT数据的语义一致性，单独落盘，不修改训练数据。"""

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from steps.step4 import pipeline as p

PROMPT_PATH = p.TEACHER_PATH.with_name("coarse_cot_checker.txt")
OUTPUT_ROOT = p.OUTPUT_ROOT / "check"
LABELS = {"safe", "unsafe", "uncertain"}
ISSUES = {"internal_contradiction", "unsupported_claim", "missed_decisive_behavior", "insufficient_explanation"}


def repair() -> None:
    """应用已逐条审定的CoT补丁，保留源响应、论文口径和本地裁决。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    plan_path = Path(__file__).with_name("check_repairs.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    snapshot = OUTPUT_ROOT / "report_before_cot_repair.json"
    if not snapshot.exists():
        source = OUTPUT_ROOT / "report.json"
        if p.sha256_file(source) != plan["source_audit_report_sha256"]:
            raise p.PipelineError("本轮补丁与审核报告不符，请先核对")
        snapshot.write_bytes(source.read_bytes())
    if p.sha256_file(snapshot) != plan["source_audit_report_sha256"]:
        raise p.PipelineError("原审核快照已变化")
    report_path = p.OUTPUT_ROOT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = p.OUTPUT_ROOT / "intermediate/teacher_outputs.jsonl"
    attempts = p.read_checkpoint(checkpoint)
    cache = {row["key"]: row for row in attempts}
    canonical = {row["id"]: row for row in p.load_inputs()}
    teacher = p.TEACHER_PATH.read_text(encoding="utf-8")
    student = p.STUDENT_PATH.read_text(encoding="utf-8")
    tables = {split: pq.read_table(p.OUTPUT_ROOT / "final" / f"{split}.parquet") for split in p.SPLITS}
    exported_rows = {split: table.to_pylist() for split, table in tables.items()}
    exported_by_id = {row["id"]: row for part in exported_rows.values() for row in part}
    planned, replacements = [], {}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for sample_id, fix in plan["repairs"].items():
        row = canonical[sample_id]
        t, s = p.render_inputs(row, teacher, student)
        item = {"id": sample_id, "split": row["split"], "gold": row["target"]["coarse"], "teacher": t, "student": s}
        key = p.sha256_text(p.stable_json({**item, "model": report["model"], "base_url": report["base_url"]}))
        candidates = [r for r in attempts if r["key"] == key and r.get("status") == "generated"]
        original = next((r for r in candidates if p.sha256_text(
            f"<Analysis>\n{r['cot']}\n</Analysis>\n\n<Judgment> {r['judgment']} </Judgment>") == fix["source_assistant_sha256"]), None)
        if original is None:
            raise p.PipelineError(f"找不到已审核的源CoT：{sample_id}")
        before = f"<Analysis>\n{original['cot']}\n</Analysis>\n\n<Judgment> {original['judgment']} </Judgment>"
        if "analysis" in fix:
            after = f"<Analysis>\n{fix['analysis']}\n</Analysis>\n\n<Judgment> {original['judgment']} </Judgment>"
        else:
            if before.count(fix["find"]) != 1:
                raise p.PipelineError(f"替换片段不是唯一匹配：{sample_id}")
            after = before.replace(fix["find"], fix["replace"], 1)
        cot, judgment = p.parse_output(after)
        current = cache[key]
        latest = f"<Analysis>\n{current['cot']}\n</Analysis>\n\n<Judgment> {current['judgment']} </Judgment>"
        exported = exported_by_id[sample_id]
        if judgment != item["gold"] or exported["gold"] != item["gold"] or exported["split"] != item["split"] or current["status"] != "generated" or latest not in (before, after) or exported["messages"][0]["content"] != s or exported["messages"][1]["content"] not in (before, after):
            raise p.PipelineError(f"源内容已变化或补丁改变了标签：{sample_id}")
        replacements[sample_id] = after
        if latest != after:
            updated = {**current, "content": after, "cot": cot, "judgment": judgment,
                       "model": None, "response_id": None, "usage": {}, "created_at": now,
                       "response_source": "manual_cot_revision_after_check",
                       "manual_review": {"source_key": key, "source_content_sha256": p.sha256_text(current["content"]),
                                         "source_assistant_sha256": fix["source_assistant_sha256"],
                                         "source_model": current.get("model"), "source_response_id": current.get("response_id"),
                                         "patch_manifest_sha256": p.sha256_file(plan_path), "reason": fix["reason"]}}
            planned.append(updated)
    # 全部补丁校验后才写入；相同补丁重跑不会追加重复响应。
    for updated in planned:
        p.write_jsonl(checkpoint, [updated], mode="a")
    changed_splits = set()
    for sample_id, after in replacements.items():
        exported = exported_by_id[sample_id]
        if exported["messages"][1]["content"] != after:
            exported["messages"][1]["content"] = after
            changed_splits.add(exported["split"])
    for split in changed_splits:
        path = p.OUTPUT_ROOT / "final" / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(exported_rows[split], schema=tables[split].schema), temporary)
        temporary.replace(path)
    decisions = {}
    for sample_id, decision in plan["decisions"].items():
        row = exported_by_id[sample_id]
        decisions[sample_id] = {**decision, "gold": row["gold"], "split": row["split"],
                                "student_sha256": p.sha256_text(row["messages"][0]["content"]),
                                "assistant_sha256": p.sha256_text(row["messages"][1]["content"])}
    manual = {"reviewed_at": now, "review_method": "assistant_offline_evidence_review", "generation_api_calls": 0, "repaired_cots": len(replacements),
              "counts": dict(Counter(r["decision"] for r in plan["decisions"].values())),
              "source_audit_report_sha256": plan["source_audit_report_sha256"],
              "patch_manifest_sha256": p.sha256_file(plan_path), "decisions": decisions, "policy": plan.get("policy"),
              "note": "由本助手按论文口径逐条裁决，原模型意见和原CoT保留；不改输入、gold、split，不将本地裁决冒充外部模型pass。新增六条CoT未调用外部API生成或复审。"}
    manual_path = OUTPUT_ROOT / "manual_review.json"
    manual_path.write_text(json.dumps(manual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    report["outputs"]["checkpoint"]["sha256"] = p.sha256_file(checkpoint)
    for split in changed_splits:
        report["outputs"]["final"][split]["sha256"] = p.sha256_file(p.OUTPUT_ROOT / "final" / f"{split}.parquet")
    report["outputs"]["manual_cot_review"] = {"path": "check/manual_review.json", "sha256": p.sha256_file(manual_path)}
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(report_path)
    print(f"Step 4 CoT repair: revised={len(replacements)}, new_checkpoint_rows={len(planned)}, API calls=0")


def load_inputs(target: str = "baseline") -> list[dict]:
    """读取目标模型真正训练的messages，不用未适配原轨迹替代证据。"""
    if target == "step5":
        root = p.DATA_ROOT / "step5/cot"
        report_path = root / "report.json"
        if not report_path.exists():
            raise p.PipelineError("Step 5尚未生成CoT，请先运行run_cot.ps1 -Target step5")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "pass" or report.get("total") != 100:
            raise p.PipelineError("Step 5的100条CoT尚未全部生成完成")
        info = report["outputs"]["final"]
        path = root / info["path"]
        if p.sha256_file(path) != info["sha256"]:
            raise p.PipelineError("Step 5最终CoT文件hash不符")
        rows = p.read_jsonl(path)
        manifest = p.read_jsonl(p.DATA_ROOT / "step5/final/replacement_manifest.jsonl")
        expected_ids = {row["id"] for row in manifest if row.get("action") == "add"}
        if len(rows) != 100 or {row["id"] for row in rows} != expected_ids:
            raise p.PipelineError("Step 5最终CoT与替换清单不一致")
    elif target == "baseline":
        rows = _load_baseline_inputs()
    else:
        raise p.PipelineError("审核target必须是baseline或step5")
    for row in rows:
        messages = row["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise p.PipelineError(f"对话结构不符：{row['id']}")
        _, judgment = p.parse_output(messages[1]["content"])
        if judgment != row["gold"]:
            raise p.PipelineError(f"导出Judgment与gold不符：{row['id']}")
    if len({row["id"] for row in rows}) != len(rows):
        raise p.PipelineError("最终样本ID重复")
    return rows


def _load_baseline_inputs() -> list[dict]:
    """只读原Step 4最终Parquet。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise p.PipelineError("请先安装读取依赖：python -m pip install pyarrow") from exc

    report = json.loads((p.OUTPUT_ROOT / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "pass":
        raise p.PipelineError("Step 4尚未完成验收导出")
    rows = []
    for split, expected in p.SPLITS.items():
        info = report["outputs"]["final"][split]
        path = p.OUTPUT_ROOT / info["path"]
        if p.sha256_file(path) != info["sha256"]:
            raise p.PipelineError(f"最终文件hash不符：{path}")
        part = pq.read_table(path).to_pylist()
        if len(part) != expected or any(row["split"] != split for row in part):
            raise p.PipelineError(f"{split}条数或split字段不符")
        rows.extend(part)
    return rows


def parse_audit(content: str, item: dict) -> dict:
    """校验审核器的结构和原文引用，再在本地比对未发送的gold/Judgment。"""
    audit = json.loads(content)
    if not isinstance(audit, dict) or not isinstance(audit.get("issues"), list):
        raise ValueError("审核结果必须是带issues数组的JSON对象")
    if any(not isinstance(audit.get(field), str) or audit[field] not in LABELS for field in ("trajectory_label", "analysis_label")):
        raise ValueError("审核标签必须是safe/unsafe/uncertain")
    verdict = audit.get("cot_verdict")
    if not isinstance(verdict, str) or verdict not in {"pass", "review", "fail"} or not isinstance(audit.get("reason"), str) or not audit["reason"].strip():
        raise ValueError("审核结论或理由无效")
    if (verdict == "pass") != (len(audit["issues"]) == 0):
        raise ValueError("pass必须无issues，review/fail必须有具体问题")
    for issue in audit["issues"]:
        if not isinstance(issue, dict) or not isinstance(issue.get("type"), str) or issue["type"] not in ISSUES:
            raise ValueError("审核问题类型无效")
        quotes, evidence = issue.get("cot_quotes"), issue.get("evidence_quote")
        if not isinstance(quotes, list) or any(not isinstance(q, str) or not q.strip() or q not in item["analysis"] for q in quotes):
            raise ValueError("审核器引用了不存在的CoT原句")
        if not isinstance(evidence, str) or evidence and evidence not in item["evidence"]:
            raise ValueError("审核器引用了不存在的输入证据")
        if not isinstance(issue.get("reason"), str) or not issue["reason"].strip():
            raise ValueError("问题缺少具体理由")
        if issue["type"] == "internal_contradiction" and len(set(quotes)) < 2:
            raise ValueError("内部矛盾必须引用两个不同的分析片段")
        if issue["type"] == "missed_decisive_behavior":
            if not evidence.strip():
                raise ValueError("遗漏行为必须引用被遗漏的输入证据")
        elif not quotes:
            raise ValueError("必须引用有问题的CoT片段")
    flags = []
    if audit["trajectory_label"] not in {item["gold"], "uncertain"}:
        flags.append("trajectory_gold_mismatch")
    if audit["analysis_label"] not in {item["judgment"], "uncertain"}:
        flags.append("analysis_judgment_mismatch")
    if "uncertain" in (audit["trajectory_label"], audit["analysis_label"]):
        flags.append("uncertain_label")
    # 轨迹与gold分歧优先待审，不能直接将CoT改写成迎合原标签。
    if "trajectory_gold_mismatch" in flags:
        decision = "review"
    elif verdict == "fail" or "analysis_judgment_mismatch" in flags:
        decision = "fail"
    elif verdict == "review" or "uncertain_label" in flags:
        decision = "review"
    else:
        decision = "pass"
    return {"status": "checked", "audit": audit, "decision": decision, "flags": flags}


def run(*, dry_run: bool = False, limit: int | None = None, report_only: bool = False, target: str = "baseline") -> None:
    if limit is not None and limit < 1:
        raise p.PipelineError("--limit必须大于0")
    rows = load_inputs(target)
    output_root = OUTPUT_ROOT if target == "baseline" else p.DATA_ROOT / "step5/cot/check"
    template = PROMPT_PATH.read_text(encoding="utf-8")
    model = os.environ.get("STEP4_CHECK_MODEL", "gpt-6-astra")
    base_url = os.environ.get("STEP4_BASE_URL", "https://new.mrxiwang.eu.cc/v1").rstrip("/")
    workers = int(os.environ.get("STEP4_WORKERS", "4"))
    interval = float(os.environ.get("STEP4_REQUEST_INTERVAL", "8"))
    max_tokens = int(os.environ.get("STEP4_MAX_TOKENS", "8192"))
    if workers < 1 or interval < 0 or max_tokens < 1:
        raise p.PipelineError("并发、间隔或生成长度无效")
    items = []
    for row in rows:
        analysis, judgment = p.parse_output(row["messages"][1]["content"])
        evidence = row["messages"][0]["content"]
        # 不发送gold、原Judgment、来源、split或旧审核意见。
        prompt = template.replace("{sample_json}", json.dumps({"judge_input": evidence, "analysis": analysis}, ensure_ascii=False))
        item = {"id": row["id"], "split": row["split"], "gold": row["gold"], "judgment": judgment,
                "evidence": evidence, "analysis": analysis, "teacher": prompt}
        item["key"] = p.sha256_text(p.stable_json({**item, "model": model, "base_url": base_url}))
        items.append(item)
    if dry_run:
        print(f"CoT check dry-run: target={target}, records={len(items)}, model={model}; API calls=0, no files written.")
        return
    checkpoint = output_root / "records.jsonl"
    attempts = p.read_checkpoint(checkpoint)
    cache = {row["key"]: row for row in attempts}
    pending = [item for item in items if cache.get(item["key"], {}).get("status") != "checked"]
    if report_only:
        pending = []
    if limit:
        pending = pending[:limit]
    print(f"CoT check: target={target}, total={len(items)}, pending_this_run={len(pending)}, model={model}", flush=True)
    client = p.OpenAICompatibleClient(model, max_tokens, prefix="STEP4") if pending else None
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        p.generate_pending(pending, client, workers, interval, checkpoint, cache, attempts,
                           {"requested_model": model, "base_url": base_url, "max_tokens": max_tokens,
                            "checker_prompt_sha256": p.sha256_text(template)},
                           label=f"CoT check/{target}", parse_result=parse_audit)
    finally:
        counts, flagged, local_accepted = Counter(), [], []
        manual_path = output_root / "manual_review.json"
        manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
        source_rows = {row["id"]: row for row in rows}
        for item in items:
            result = cache.get(item["key"], {})
            status = result.get("status", "pending")
            if status == "checked":
                validated = parse_audit(result["content"], item)
                status = validated["decision"]
                if status != "pass":
                    flagged.append({"id": item["id"], "split": item["split"], "decision": status,
                                    "gold": item["gold"], "judgment": item["judgment"],
                                    "flags": validated["flags"], "audit": validated["audit"]})
            elif status == "failed":
                flagged.append({"id": item["id"], "split": item["split"], "decision": "failed", "error": result.get("error")})
            counts[status] += 1
            local = manual.get("decisions", {}).get(item["id"], {})
            row = source_rows[item["id"]]
            if (status != "pass"
                    and local.get("decision") in {"cot_repaired", "audit_false_positive", "retained_unsafe_after_taxonomy_review"}
                    and local.get("gold") == row["gold"] and local.get("split") == row["split"]
                    and local.get("student_sha256") == p.sha256_text(row["messages"][0]["content"])
                    and local.get("assistant_sha256") == p.sha256_text(row["messages"][1]["content"])):
                local_accepted.append({"id": row["id"], "decision": local["decision"], "reason": local["reason"]})
        if target == "baseline":
            source_files = {s: p.sha256_file(p.OUTPUT_ROOT / "final" / f"{s}.parquet") for s in p.SPLITS}
        else:
            source_files = {"step5": p.sha256_file(p.DATA_ROOT / "step5/cot/final/candidates.jsonl")}
        report = {"stage": f"coarse_sft/{'step4' if target == 'baseline' else 'step5/cot'}/check",
                  "target": target, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  "total": len(items), "counts": dict(counts), "reviewer": model, "base_url": base_url,
                  "checker_prompt_sha256": p.sha256_file(PROMPT_PATH), "flagged": flagged,
                  "usage": {k: sum(int(r.get("usage", {}).get(k, 0) or 0) for r in attempts)
                            for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                  "source_files": source_files,
                  "local_review": {"accepted": len(local_accepted), "unresolved": len(items) - counts["pass"] - len(local_accepted),
                                   "decisions": local_accepted, "manual_review_sha256": p.sha256_file(manual_path) if manual_path.exists() else None},
                  "report_only": report_only,
                  "scope": "counts仅统计当前内容对应的外部模型意见；local_review为本助手逐条裁决，分别保存。此汇总不修改训练数据。"}
        path = output_root / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
    print(f"CoT check complete: target={target}, {dict(counts)}; {output_root}", flush=True)
    if local_accepted:
        print(f"Local adjudication: accepted={len(local_accepted)}, unresolved={report['local_review']['unresolved']}; not additional API verdicts.", flush=True)
