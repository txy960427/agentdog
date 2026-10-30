"""为未采样的v1.0轨迹预生成粗粒度CoT，独立保存，供后续候选筛选。"""

import json
import os
from collections import Counter
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

from steps.step0 import pipeline as selection
from steps.step0.source import read_rows
from steps.step1 import pipeline as binary
from steps.step2 import pipeline as fine
from steps.step4 import pipeline as cot

OUTPUT_ROOT = cot.DATA_ROOT / "cot_pool"


def load_inputs() -> tuple[list[dict], dict]:
    """复用冻结的去重索引和转换器；不重新采样、分组或分配split。"""
    root = cot.DATA_ROOT / "step0"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["audit"]["status"] != "pass":
        raise cot.PipelineError("Step 0尚未通过验收")
    hashes = {"step0_report": cot.sha256_file(root / "report.json")}
    for name in ("candidates", "selection"):
        info = report["outputs"][name]
        path = root / info["path"]
        hashes[name] = cot.sha256_file(path)
        if hashes[name] != info["sha256"]:
            raise cot.PipelineError(f"Step 0的{name}已变化")
    chosen = cot.read_jsonl(root / report["outputs"]["selection"]["path"])
    chosen_ids = {row["id"] for row in chosen}
    candidates = cot.read_jsonl(root / report["outputs"]["candidates"]["path"])
    groups = {row["group_id"]: row["split"] for row in chosen}
    for row in candidates:
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise cot.PipelineError(f"group跨split：{row['group_id']}")
    raw = {}
    for stage, name, path in (("step1", "binary", binary.BINARY_PATH), ("step2", "fine", fine.FINE_PATH)):
        hashes[name] = cot.sha256_file(path)
        if hashes[name] != report["source"][name]["sha256"]:
            raise cot.PipelineError(f"v1.0 {name}源文件变化")
        raw[stage] = read_rows(path)
    components = [[], [], []]
    for candidate in candidates:
        if candidate["id"] in chosen_ids or not candidate["eligible"] or candidate["benchmark_overlap"]:
            continue
        manifest = selection.selection_row(candidate)
        source = raw[candidate["stage"]][int(candidate["source_row_id"])]
        if candidate["stage"] == "step1":
            row = binary.convert(manifest, source)
            component = 0 if row["target"]["coarse"] == "safe" else 1
        else:
            row = fine.convert(manifest, source)[0]
            component = 2
        components[component].append(row)
    # 仅交错请求顺序，让开头的小批量覆盖三种来源；不改变数量和真实分布。
    rows = [row for batch in zip_longest(*components) for row in batch if row is not None]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise cot.PipelineError("剩余候选为空或ID重复")
    return rows, hashes


def write_atomic(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = cot.write_jsonl(temporary, rows)
    temporary.replace(path)
    return count


def run(*, dry_run: bool = False, limit: int | None = None, split: str = "all", selection: str = "all") -> None:
    if limit is not None and limit < 1:
        raise cot.PipelineError("--limit必须大于0")
    if split not in ("all", "train", "validation", "test"):
        raise cot.PipelineError("split必须是all/train/validation/test")
    if selection not in ("all", "step5"):
        raise cot.PipelineError("selection必须是all或step5")
    rows, source_hashes = load_inputs()
    teacher = cot.TEACHER_PATH.read_text(encoding="utf-8")
    student = cot.STUDENT_PATH.read_text(encoding="utf-8")
    items, seen = [], set()
    for row in rows:
        t, s = cot.render_inputs(row, teacher, student)
        item = {"id": row["id"], "split": row["split"], "gold": row["target"]["coarse"], "teacher": t, "student": s}
        fingerprint = cot.sha256_text(s)
        if fingerprint in seen:
            raise cot.PipelineError(f"候选输入重复，请先检查：{row['id']}")
        seen.add(fingerprint)
        # key只依赖输入和gold；换模型/接口时复用已完成结果，来源记录在每条响应内。
        item["key"] = cot.sha256_text(cot.stable_json(item))
        items.append(item)
    requested = [item for item in items if split == "all" or item["split"] == split]
    output_root = OUTPUT_ROOT
    if selection == "step5":
        manifest_path = cot.DATA_ROOT / "step5/final/replacement_manifest.jsonl"
        manifest = cot.read_jsonl(manifest_path)
        selected_ids = {row["id"] for row in manifest if row.get("action") == "add"}
        available_ids = {item["id"] for item in items}
        if len(selected_ids) != 100 or not selected_ids <= available_ids:
            raise cot.PipelineError("Step 5替换清单不是100条，或ID不属于当前CoT候选池")
        requested = [item for item in requested if item["id"] in selected_ids]
        source_hashes["step5_manifest"] = cot.sha256_file(manifest_path)
        output_root = cot.DATA_ROOT / "step5/cot"
    requested_ids = {item["id"] for item in requested}
    requested_rows = [row for row in rows if row["id"] in requested_ids]
    output_rows = requested_rows if selection == "step5" else rows
    distribution = {s: dict(Counter(row["target"]["coarse"] for row in output_rows if row["split"] == s)) for s in cot.SPLITS}
    print(f"CoT pool: total={len(items)}, selection={selection}, selected_split={split}, selected={len(requested)}", flush=True)
    print(json.dumps(distribution, ensure_ascii=False), flush=True)
    if dry_run:
        print("Dry-run: API calls=0; no files written.")
        return

    model = os.environ.get("STEP4_MODEL", "gpt-6.1-sol")
    base_url = os.environ.get("STEP4_BASE_URL", "https://new.mrxiwang.eu.cc/v1").rstrip("/")
    workers = int(os.environ.get("STEP4_WORKERS", "4"))
    interval = float(os.environ.get("STEP4_REQUEST_INTERVAL", "8"))
    max_tokens = int(os.environ.get("STEP4_MAX_TOKENS", "8192"))
    if workers < 1 or interval < 0 or max_tokens < 1:
        raise cot.PipelineError("并发、间隔或生成长度无效")
    checkpoint = output_root / "intermediate/teacher_outputs.jsonl"
    legacy_seed_records = 0
    if selection == "step5":
        allowed_keys = {item["key"] for item in requested}
        existing_keys = {row.get("key") for row in cot.read_checkpoint(checkpoint)}
        legacy = cot.read_checkpoint(OUTPUT_ROOT / "intermediate/teacher_outputs.jsonl")
        seed = [row for row in legacy if row.get("key") in allowed_keys and row.get("key") not in existing_keys]
        if seed:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            cot.write_jsonl(checkpoint, seed, mode="a")
            legacy_seed_records = len(seed)
    attempts = cot.read_checkpoint(checkpoint)
    cache = {row["key"]: row for row in attempts}
    pending = [item for item in requested if cache.get(item["key"], {}).get("status") not in {"generated", "review_required"}]
    if limit:
        pending = pending[:limit]
    print(f"CoT pool: pending_this_run={len(pending)}, model={model}, workers={workers}, interval={interval}s", flush=True)
    # 首次缺少key时在写文件前报错；已完成的离线重导出不要求key。
    client = cot.OpenAICompatibleClient(model, max_tokens, prefix="STEP4") if pending else None
    write_atomic(output_root / "intermediate/inputs.jsonl", output_rows)
    metadata = {"requested_model": model, "base_url": base_url, "max_tokens": max_tokens,
                "teacher_prompt_sha256": cot.sha256_text(teacher), "student_prompt_sha256": cot.sha256_text(student)}
    try:
        cot.generate_pending(pending, client, workers, interval, checkpoint, cache, attempts, metadata, label="CoT pool")
    finally:
        # 中断也导出已完成部分。冲突/失败保留在checkpoint，不伪装成可用训练数据。
        completed = []
        exported_items = requested if selection == "step5" else items
        for item in exported_items:
            answer = cache.get(item["key"], {})
            if answer.get("status") != "generated":
                continue
            analysis, judgment = cot.parse_output(answer["content"])
            if judgment != item["gold"] or (analysis, judgment) != (answer["cot"], answer["judgment"]):
                raise cot.PipelineError(f"检查点CoT/标签不一致：{item['id']}")
            completed.append({"id": item["id"], "split": item["split"], "gold": item["gold"], "enable_thinking": False,
                              "messages": [{"role": "user", "content": item["student"]}, {"role": "assistant", "content":
                                           f"<Analysis>\n{analysis}\n</Analysis>\n\n<Judgment> {judgment} </Judgment>"}]})
        final = output_root / "final/candidates.jsonl"
        write_atomic(final, completed)
        counts = Counter(cache.get(item["key"], {}).get("status", "pending") for item in exported_items)
        report = {"stage": "coarse_sft/step5/cot" if selection == "step5" else "coarse_sft/cot_pool",
                  "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  "status": "pass" if counts["generated"] == len(exported_items) else "incomplete", "total": len(exported_items),
                  "selected_split": split, "selection": selection, "counts": dict(counts), "distribution": distribution,
                  "source_hashes": source_hashes, **metadata, "workers": workers, "request_interval_seconds": interval,
                  "legacy_seed_records": legacy_seed_records,
                  "renderer_sha256": cot.sha256_file(Path(cot.renderer.__file__)),
                  "input_adapter_sha256": cot.sha256_file(Path(cot.__file__).with_name("inputs.py")),
                  "usage": {key: sum(int(row.get("usage", {}).get(key, 0) or 0) for row in attempts)
                            for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                  "outputs": {"inputs": {"path": "intermediate/inputs.jsonl", "records": len(output_rows),
                                         "sha256": cot.sha256_file(output_root / "intermediate/inputs.jsonl")},
                              "checkpoint": {"path": "intermediate/teacher_outputs.jsonl",
                                             "sha256": cot.sha256_file(checkpoint) if checkpoint.exists() else None},
                              "final": {"path": "final/candidates.jsonl", "records": len(completed), "sha256": cot.sha256_file(final)}},
                  "checks": "仅格式和gold一致性通过；未做独立语义审核或影响函数筛选；训练只能读取train split"}
        path = output_root / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
    print(f"CoT pool complete: {dict(counts)}; {final}", flush=True)
