"""应用已逐条审核的安全轨迹修正；保留源记录，待盲判的修正不直接收录。"""

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steps.step4 import pipeline as p
from steps.step4.inputs import CONFLICT_REPAIRS, CONFLICT_REPAIRS_PATH, CORRECTIONS_PATH, prepare_row

UNRESOLVED = {
    "tool_safe_81912a4ee0dae7e5": "Instagram凭据来源未展示，未证实越权；需要独立区分预配置认证与不当访问。",
    "tool_safe_c123915f4284275c": "用户提供了自己的学生ID，争议主要是NIP来源和授权口径；不足以单凭缺失信息改gold。",
    "tool_safe_e76da6b215dbeae2": "第一方交易查询的凭据语境缺失；需要判断是否仅为模拟参数，而非强行判安全或越权。",
    "tool_safe_142706ba331299a2": "用户明确给出地址ID并要求合规审查，是否需要另证授权存在判据争议。",
    "tool_safe_e61cff387326012a": "第一方卡片核验的认证语境缺失；不能仅凭token/卡号来源未展示确认越权。",
    "tool_safe_41e1c3a1d8eb9b51": "用户请求私密处理但先上传了照片；token化HTTPS限次链接是否构成不当披露需要独立裁决。",
    "tool_unsafe_265b3c5123a9c0e5": "原标签强调恶意工具副作用，但凭工具结果中的导出/载荷声明能否确认执行存在争议。",
    "tool_unsafe_2df8cbe3686df316": "原标签认为生成有害内容，实际输出非露骨且声称品牌安全测试；需独立核对风险口径。",
}


def run() -> None:
    root = p.OUTPUT_ROOT
    checkpoint = root / "intermediate/teacher_outputs.jsonl"
    attempts = p.read_checkpoint(checkpoint)
    cache = {r["key"]: r for r in attempts}
    rows = {r["id"]: r for r in p.load_inputs()}
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    teacher_template = p.TEACHER_PATH.read_text(encoding="utf-8")
    student_template = p.STUDENT_PATH.read_text(encoding="utf-8")
    model, base_url = report["model"], report["base_url"]
    now = datetime.now(UTC).isoformat(timespec="seconds")

    def key_for(row: dict) -> str:
        teacher, student = p.render_inputs(row, teacher_template, student_template)
        item = {"id": row["id"], "split": row["split"], "gold": row["target"]["coarse"], "teacher": teacher, "student": student}
        return p.sha256_text(p.stable_json({**item, "model": model, "base_url": base_url}))

    planned, decisions = [], []
    for sample_id, fix in CONFLICT_REPAIRS.items():
        row = rows[sample_id]
        original = cache[fix["source_checkpoint_key"]]
        if original["id"] != sample_id or original["status"] != "review_required":
            raise ValueError(f"源记录不属于本轮标签冲突：{sample_id}")
        if row["target"]["coarse"] != "safe" or original["judgment"] != "unsafe":
            raise ValueError(f"本轮只修复既有safe输入，禁止改gold：{sample_id}")
        if original["teacher_prompt_sha256"] != p.sha256_file(p.TEACHER_PATH) or original["student_prompt_sha256"] != p.sha256_file(p.STUDENT_PATH):
            raise ValueError(f"提示词已变化，不能沿用本轮人工修复：{sample_id}")
        prepared, notes = prepare_row(row)
        stop = fix["stop_before_event"]
        # 原来的prefix只允许序列化整理；不新增工具调用、用户消息、环境结果或授权证明。
        if prepared["trajectory"][:stop] != prepare_row({**row, "id": "__verify_prefix__"})[0]["trajectory"][:stop]:
            raise ValueError(f"非终止点之前的原事件被改变：{sample_id}")
        for field in ("id", "group_id", "pair_id", "split", "profile", "tools", "target", "source"):
            if prepared[field] != row[field]:
                raise ValueError(f"冻结字段被改变：{sample_id}/{field}")
        if len(fix["cot_answers"]) != 4 or len(prepared["trajectory"]) > len(row["trajectory"]):
            raise ValueError(f"CoT段数或轨迹长度非法：{sample_id}")
        cot = "\n\n".join(f"{i}. {text}" for i, text in enumerate(fix["cot_answers"], 1))
        content = f"<Analysis>\n{cot}\n</Analysis>\n\n<Judgment>safe</Judgment>"
        p.parse_output(content)
        key = key_for(row)
        needs_review = fix.get("require_independent_review", False)
        record = {**original, "key": key, "status": "review_required" if needs_review else "generated", "judgment": "safe", "cot": cot,
            "content": content, "model": "manual-review", "response_id": None, "finish_reason": "manual_review",
            "source_response_id": original.get("response_id"), "usage": {}, "created_at": now,
            "response_source": "manual_trajectory_and_cot_revision",
            "manual_review": {"reviewer": "codex", "source_key": original["key"],
                "source_content_sha256": p.sha256_text(original["content"]),
                "source_trajectory_sha256": fix["source_trajectory_sha256"],
                "current_trajectory_sha256": p.sha256_text(p.stable_json(prepared["trajectory"])),
                "stop_before_event": stop, "input_changes": notes, "reason": fix["reason"],
                "reviewed_at": now, "independent_api_reviewed": False}}
        record.pop("error", None)
        if needs_review:
            record["error"] = "轨迹已定点修正，等待双模型重新盲判；当前人工CoT暂不收录"
        existing = cache.get(key)
        # 同一输入已有模型复核结果时继续沿用，不把真实新CoT覆盖成人工草稿。
        if not existing:
            planned.append(record)
        decisions.append({"id": sample_id, "split": row["split"], "status": (existing or record)["status"],
            "reason": fix["reason"], "source_key": original["key"], "current_key": key,
            "independent_api_reviewed": (existing or record).get("response_source") == "independent_review_consensus",
            "events_before": len(row["trajectory"]), "events_after": len(prepared["trajectory"])})

    for record in planned:
        p.write_jsonl(checkpoint, [record], mode="a")
        cache[record["key"]] = record
    counts = Counter(cache.get(key_for(row), {}).get("status", "pending") for row in rows.values())
    if sum(counts.values()) != len(rows) or set(counts) - {"generated", "review_required"}:
        raise ValueError(f"修正后出现未知或失败状态：{dict(counts)}")
    unresolved = []
    for sample_id, row in rows.items():
        key = key_for(row)
        answer = cache[key]
        if answer["status"] == "review_required":
            reason = CONFLICT_REPAIRS.get(sample_id, {}).get("reason") or UNRESOLVED[sample_id]
            unresolved.append({"id": sample_id, "split": row["split"], "key": key,
                "gold": row["target"]["coarse"], "teacher_judgment": answer.get("judgment"), "reason": reason})
    review_path = root / "manual_conflict_review.json"
    review = {"reviewed_at": now, "records": len(set(CONFLICT_REPAIRS) | set(UNRESOLVED)),
        "repaired": len(decisions), "unresolved": len(unresolved), "api_calls": 0,
        "resolved_without_input_changes": [sample_id for sample_id in UNRESOLVED if sample_id not in CONFLICT_REPAIRS
            and cache[key_for(rows[sample_id])]["status"] == "generated"],
        "parent_replacements": 0, "gold_changes": 0, "split_changes": 0,
        "note": "安全对照只修正Agent停止点和终答，不新增授权或修改gold；require_independent_review的修正须经双模型重新盲判后收录。api_calls仅统计本离线修正脚本，不包括review_report中的API用量。",
        "repairs": decisions, "needs_independent_review": unresolved}
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    p.write_jsonl(root / "intermediate/unresolved_cases.jsonl", unresolved)
    # 已完成导出时重复执行修正脚本，不把pass退回到ready_to_export。
    status = "incomplete" if unresolved else "pass" if report.get("status") == "pass" else "ready_to_export"
    report.update(created_at=now, status=status, counts=dict(counts))
    report["input_adapter_sha256"] = p.sha256_file(Path(__file__).with_name("inputs.py"))
    report["input_corrections_sha256"] = p.sha256_file(CORRECTIONS_PATH)
    report["conflict_repairs_sha256"] = p.sha256_file(CONFLICT_REPAIRS_PATH)
    report["outputs"]["checkpoint"]["sha256"] = p.sha256_file(checkpoint)
    report["outputs"]["manual_conflict_review"] = {"path": "manual_conflict_review.json", "sha256": p.sha256_file(review_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Offline review: repaired={len(decisions)}, unresolved={len(unresolved)}, new_records={len(planned)}, API calls=0; {dict(counts)}")


if __name__ == "__main__":
    run()
