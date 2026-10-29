"""离线复验全量SFT：来源、冻结划分、修正血缘、CoT及最终Parquet，不调用API。"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from steps.step0.source import PipelineError, normalize_text, read_rows
from steps.step1 import pipeline as step1
from steps.step2 import pipeline as step2
from steps.step3 import verify as step3
from steps.step4 import pipeline as p
from steps.step4.inputs import prepare_row


def run() -> None:
    import pyarrow.parquet as pq

    # 已清理的生成草稿不参与复验，Step 3直接核验final与保留的审核血缘。
    step3.run()
    root = p.OUTPUT_ROOT
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    rows = p.load_inputs()
    errors, manifest_files = [], 0

    def check(condition, message):
        if not condition:
            errors.append(message)

    def manifests(value, base):
        nonlocal manifest_files
        if not isinstance(value, dict):
            return
        if "path" in value and "sha256" in value:
            path = base / value["path"]
            check(path.is_file() and p.sha256_file(path) == value["sha256"], f"清单hash不符：{path}")
            manifest_files += 1
        else:
            for child in value.values():
                manifests(child, base)

    for stage in range(5):
        base = p.DATA_ROOT / f"step{stage}"
        summary = json.loads((base / "report.json").read_text(encoding="utf-8"))
        manifests(summary["outputs"], base)
    check(report["status"] == "pass" and report["counts"] == {"generated": 1500}, "Step 4尚未全部通过")
    selection = p.read_jsonl(p.DATA_ROOT / "step0/final/selection.jsonl")
    candidates = {r["id"]: r for r in p.read_jsonl(p.DATA_ROOT / "step0/intermediate/candidates.jsonl")}
    check(len(selection) == len({r["id"] for r in selection}) == 900, "冻结parent数量或ID错误")
    check(all(not candidates[r["id"]]["benchmark_overlap"] for r in selection), "选中已标记的benchmark近重复")
    source_report = json.loads((p.DATA_ROOT / "step0/report.json").read_text(encoding="utf-8"))["source"]
    frozen = {r["id"]: r for r in rows if not r["id"].startswith("tool_safe_")}
    # 900条parent逐条重做转换并比较；不仅检查最终文件之间是否自洽。
    for stage, name, module, path in (("step1", "binary", step1, step1.BINARY_PATH), ("step2", "fine", step2, step2.FINE_PATH)):
        check(p.sha256_file(path) == source_report[name]["sha256"], f"v1.0 {name}源文件变化")
        raw = read_rows(path)
        for chosen in selection:
            if chosen["stage"] == stage:
                converted = module.convert(chosen, raw[int(chosen["source_row_id"])])
                if stage == "step2":
                    converted = converted[0]
                check(converted == frozen[chosen["id"]], f"原始来源转换不一致：{chosen['id']}")

    teacher = p.TEACHER_PATH.read_text(encoding="utf-8")
    student = p.STUDENT_PATH.read_text(encoding="utf-8")
    for name, path in (("teacher_prompt", p.TEACHER_PATH), ("student_prompt", p.STUDENT_PATH),
                       ("renderer", p.PROJECT_ROOT / "judge_eval/prompt_builder.py"),
                       ("input_adapter", Path(p.__file__).with_name("inputs.py")),
                       ("input_corrections", Path(p.__file__).with_name("input_corrections.json")),
                       ("conflict_repairs", Path(p.__file__).with_name("conflict_repairs.json"))):
        check(p.sha256_file(path) == report[f"{name}_sha256"], f"提示词或适配代码变化：{name}")
    attempts = p.read_checkpoint(root / "intermediate/teacher_outputs.jsonl")
    cache = {r["key"]: r for r in attempts}
    exported = {r["id"]: r for split in p.SPLITS for r in pq.read_table(root / "final" / f"{split}.parquet").to_pylist()}
    check(set(exported) == {r["id"] for r in rows}, "导出ID未覆盖全部输入")
    seen_prompts, user_texts = defaultdict(list), defaultdict(list)
    labels, components, sources = Counter(), Counter(), Counter()
    group_splits, pair_splits = defaultdict(set), defaultdict(set)
    for row in rows:
        sample_id = row["id"]
        prepared, _ = prepare_row(row)
        for field in ("id", "source", "group_id", "pair_id", "split", "profile", "tools", "target"):
            check(prepared[field] == row[field], f"修正改变冻结字段：{sample_id}/{field}")
        t, s = p.render_inputs(row, teacher, student)
        item = {"id": sample_id, "split": row["split"], "gold": row["target"]["coarse"], "teacher": t, "student": s}
        key = p.sha256_text(p.stable_json({**item, "model": report["model"], "base_url": report["base_url"]}))
        answer = cache.get(key, {})
        check(answer.get("status") == "generated", f"当前输入缺少有效CoT：{sample_id}")
        cot, judgment = p.parse_output(answer["content"])
        check((cot, judgment) == (answer["cot"], answer["judgment"]) and judgment == item["gold"], f"CoT/标签不一致：{sample_id}")
        final = exported[sample_id]
        expected = f"<Analysis>\n{cot}\n</Analysis>\n\n<Judgment> {judgment} </Judgment>"
        check(final["messages"] == [{"role": "user", "content": s}, {"role": "assistant", "content": expected}], f"导出对话内容不符：{sample_id}")
        check(final["gold"] == judgment and final["split"] == row["split"] and final["enable_thinking"] is False, f"导出元数据不符：{sample_id}")
        labels[(row["split"], judgment)] += 1
        components[sample_id.rsplit("_", 1)[0]] += 1
        sources[answer.get("response_source", "api_generation")] += 1
        seen_prompts[p.sha256_text(s)].append(sample_id)
        user_texts[normalize_text("\n".join(e["content"] for e in row["trajectory"] if e["role"] == "user"))].append((sample_id, row["split"]))
        group_splits[row["group_id"]].add(row["split"])
        if row["pair_id"]:
            pair_splits[row["pair_id"]].add(row["split"])
    check(all(len(ids) == 1 for ids in seen_prompts.values()), "存在完全重复的Student输入")
    check(all(len(splits) == 1 for splits in [*group_splits.values(), *pair_splits.values()]), "group或pair跨split")
    check(components == {"binary_safe": 150, "binary_unsafe": 150, "tool_unsafe": 600, "tool_safe": 600}, "数据组成不符")
    for split, count in p.SPLITS.items():
        check(len(pq.read_table(root / "final" / f"{split}.parquet")) == count, f"{split}行数错误")
        check(labels[(split, "safe")] == labels[(split, "unsafe")] == count // 2, f"{split}标签不平衡")
    shared = [dict(text=text, records=ids) for text, ids in user_texts.items() if len({split for _, split in ids}) > 1]
    audit = {"status": "failed" if errors else "pass", "records": len(rows), "errors": errors,
             "manifest_files": manifest_files, "reconstructed_parents": len(selection),
             "components": dict(components), "splits": {split: {label: labels[(split, label)] for label in ("safe", "unsafe")} for split in p.SPLITS},
             "unique_student_inputs": len(seen_prompts), "cross_split_groups": sum(len(x) > 1 for x in group_splits.values()),
             "cross_split_pairs": sum(len(x) > 1 for x in pair_splits.values()),
             "cross_split_shared_user_texts": shared, "cot_sources": dict(sources), "api_calls": 0,
             "scope": "全量确定性核验；不等于1500条均经独立人工语义重标。通用拒绝句跨split复用已在Step 0记录，完整轨迹和工具不同。目标tokenizer长度另查。"}
    (root / "final_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Step 4 verify: {audit['status']}, records={len(rows)}, errors={len(errors)}, API calls=0")
    if errors:
        raise PipelineError("最终数据复验失败，查看step4/final_audit.json")
