"""从未使用的BinarySafety样本中选择较长safe轨迹，生成可复现的替换清单。"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / "data/1.5/coarse_sft"
OUTPUT_ROOT = DATA_ROOT / "step5"
SELECT_COUNT = 100
MIN_EVENTS = 5


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"缺少输入文件：{path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_hash(row: dict) -> str:
    text = json.dumps(row["trajectory"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trajectory_chars(row: dict) -> int:
    return len(json.dumps(row["trajectory"], ensure_ascii=False, separators=(",", ":")))


def event_bin(events: int) -> str:
    if events <= 4:
        return "1-4"
    if events <= 8:
        return "5-8"
    if events <= 12:
        return "9-12"
    return "13+"


def distribution(rows: list[dict], label: str | None = None) -> dict[str, int]:
    selected = rows if label is None else [row for row in rows if row["target"]["coarse"] == label]
    counts = Counter(event_bin(len(row["trajectory"])) for row in selected)
    return {name: counts.get(name, 0) for name in ("1-4", "5-8", "9-12", "13+")}


def export_variant(manifest: list[dict]) -> dict:
    """只替换Train；Validation/Test逐字节复用基线。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("导出Step 5需要pyarrow") from exc

    base_root = DATA_ROOT / "step4"
    base_report = json.loads((base_root / "report.json").read_text(encoding="utf-8"))
    cot_root = OUTPUT_ROOT / "cot"
    cot_report = json.loads((cot_root / "report.json").read_text(encoding="utf-8"))
    check_report = json.loads((cot_root / "check/report.json").read_text(encoding="utf-8"))
    candidates_path = cot_root / "final/candidates.jsonl"
    candidate_hash = sha256_file(candidates_path)
    if base_report.get("status") != "pass" or cot_report.get("status") != "pass":
        raise ValueError("基线或Step 5 CoT尚未通过验收")
    if cot_report["outputs"]["final"]["sha256"] != candidate_hash:
        raise ValueError("Step 5 CoT报告与最终候选hash不符")
    if check_report.get("counts") != {"pass": SELECT_COUNT} or check_report.get("source_files", {}).get("step5") != candidate_hash:
        raise ValueError("Step 5 CoT尚未100条全部通过独立审核")

    remove_ids = {row["id"] for row in manifest if row["action"] == "remove"}
    add_ids = {row["id"] for row in manifest if row["action"] == "add"}
    if len(remove_ids) != SELECT_COUNT or len(add_ids) != SELECT_COUNT:
        raise ValueError("替换清单必须是100条remove和100条add")

    base_paths = {}
    base_rows = {}
    for split in ("train", "validation", "test"):
        info = base_report["outputs"]["final"][split]
        path = base_root / info["path"]
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"基线{split}文件hash不符")
        base_paths[split] = path
        base_rows[split] = pq.read_table(path).to_pylist()

    additions = read_jsonl(candidates_path)
    if {row["id"] for row in additions} != add_ids or any(row["split"] != "train" or row["gold"] != "safe" for row in additions):
        raise ValueError("Step 5新增CoT与add清单、split或标签不一致")
    original_train = base_rows["train"]
    removed = [row for row in original_train if row["id"] in remove_ids]
    if len(removed) != SELECT_COUNT or any(row["gold"] != "safe" or not row["id"].startswith("binary_safe_") for row in removed):
        raise ValueError("基线中待移除的100条Binary safe不完整")
    train = [row for row in original_train if row["id"] not in remove_ids] + additions
    if len(train) != 1000 or len({row["id"] for row in train}) != 1000 or Counter(row["gold"] for row in train) != {"safe": 500, "unsafe": 500}:
        raise ValueError("Step 5 Train条数、ID或标签比例异常")
    if ({row["id"] for row in train} & {row["id"] for row in base_rows["validation"] + base_rows["test"]}):
        raise ValueError("Step 5 Train与冻结Validation/Test发生ID泄漏")

    final_root = OUTPUT_ROOT / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    base_schema = pq.read_schema(base_paths["train"])
    train_path = final_root / "train.parquet"
    temporary = train_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(train, schema=base_schema), temporary)
    temporary.replace(train_path)
    for split in ("validation", "test"):
        destination = final_root / f"{split}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        shutil.copyfile(base_paths[split], temporary)
        temporary.replace(destination)
    shutil.copyfile(base_root / "final/dataset_info.json", final_root / "dataset_info.json")

    outputs = {
        split: {
            "path": f"final/{split}.parquet",
            "records": len(train) if split == "train" else len(base_rows[split]),
            "sha256": sha256_file(final_root / f"{split}.parquet"),
        }
        for split in ("train", "validation", "test")
    }
    if outputs["validation"]["sha256"] != base_report["outputs"]["final"]["validation"]["sha256"] or outputs["test"]["sha256"] != base_report["outputs"]["final"]["test"]["sha256"]:
        raise ValueError("冻结Validation/Test未能逐字节复用")
    return {
        "status": "pass",
        "policy": "Train替换100条；Validation/Test逐字节复用Step 4基线",
        "removed": len(remove_ids),
        "added": len(add_ids),
        "labels": dict(Counter(row["gold"] for row in train)),
        "outputs": outputs,
        "dataset_info_sha256": sha256_file(final_root / "dataset_info.json"),
        "cot_check_report_sha256": sha256_file(cot_root / "check/report.json"),
    }


def run() -> None:
    manifest_path = OUTPUT_ROOT / "final/replacement_manifest.jsonl"
    selected_input_path = OUTPUT_ROOT / "cot/intermediate/inputs.jsonl"
    cot_path = OUTPUT_ROOT / "cot/final/candidates.jsonl"
    checkpoint_path = OUTPUT_ROOT / "cot/intermediate/teacher_outputs.jsonl"
    train_paths = [
        DATA_ROOT / "step1/final/train.jsonl",
        DATA_ROOT / "step2/final/train.jsonl",
        DATA_ROOT / "step3/final/train.jsonl",
    ]

    base_parts = [read_jsonl(path) for path in train_paths]
    base_train = [row for part in base_parts for row in part]
    current_binary_safe = [
        row for row in base_parts[0]
        if row["target"]["coarse"] == "safe" and row["id"].startswith("binary_safe_")
    ]
    if len(base_train) != 1000 or len(current_binary_safe) != SELECT_COUNT:
        raise ValueError("当前Train不是预期的1000条，或Binary safe不是100条")

    # 替换清单冻结后，以Step 5自身的输入为准；不依赖可随时重建的全量CoT缓存池。
    frozen_manifest = read_jsonl(manifest_path)
    add_rows = [row for row in frozen_manifest if row.get("action") == "add"]
    remove_rows = [row for row in frozen_manifest if row.get("action") == "remove"]
    add_ids = {row["id"] for row in add_rows}
    remove_ids = {row["id"] for row in remove_rows}
    selected = [row for row in read_jsonl(selected_input_path) if row["id"] in add_ids]
    if len(add_ids) != SELECT_COUNT or len(remove_ids) != SELECT_COUNT or len(selected) != SELECT_COUNT:
        raise ValueError("Step 5冻结清单或已选输入不是100条add和100条remove")
    if any(
        row["split"] != "train"
        or row["target"]["coarse"] != "safe"
        or not row["id"].startswith("binary_safe_")
        or len(row["trajectory"]) < MIN_EVENTS
        for row in selected
    ):
        raise ValueError("Step 5已选输入不满足train/BinarySafety/safe/最短事件数约束")

    ids = [row["id"] for row in selected]
    groups = [row["group_id"] for row in selected]
    hashes = [trajectory_hash(row) for row in selected]
    if len(set(ids)) != SELECT_COUNT or len(set(groups)) != SELECT_COUNT or len(set(hashes)) != SELECT_COUNT:
        raise ValueError("选中样本存在重复ID、跨样本group复用或整轨重复")
    if any(not (row["trajectory"][-1].get("action") or "").lstrip().startswith("Complete") for row in selected):
        raise ValueError("选中样本存在没有完整终止回答的轨迹")

    if remove_ids != {row["id"] for row in current_binary_safe}:
        raise ValueError("Step 5冻结清单与基线中的100条Binary safe不一致")
    cot_ready_ids = {row["id"] for row in read_jsonl(cot_path)} if cot_path.exists() else set()
    manifest = frozen_manifest
    variant_train = [row for row in base_train if row["id"] not in remove_ids] + selected
    if len(variant_train) != 1000:
        raise ValueError("替换后的Train条数异常")
    labels = Counter(row["target"]["coarse"] for row in variant_train)
    if labels != {"safe": 500, "unsafe": 500}:
        raise ValueError(f"替换后标签比例异常：{dict(labels)}")

    ready = sum(row["id"] in cot_ready_ids for row in selected)
    report = {
        "stage": "coarse_sft/step5_length_debias",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "awaiting_review" if ready == SELECT_COUNT else "awaiting_cot",
        "purpose": "仅替换Train中的100条Binary safe，隔离验证safe轨迹长度捷径；Validation/Test保持不变",
        "selection": {
            "count": SELECT_COUNT,
            "rules": [
                "只使用Step 0中未进入基线的train/BinarySafety/gold-safe样本",
                f"事件数至少{MIN_EVENTS}，按事件数降序、轨迹字符数降序、ID升序",
                "ID、group和整轨内容唯一，最后一条必须是完整Complete回答",
                "排除Teacher已判为review_required的标签冲突样本",
            ],
            "event_counts": dict(sorted(Counter(len(row["trajectory"]) for row in selected).items())),
            "cot": {"ready": ready, "pending": SELECT_COUNT - ready},
        },
        "base_train": {
            "records": len(base_train),
            "labels": dict(Counter(row["target"]["coarse"] for row in base_train)),
            "safe_event_bins": distribution(base_train, "safe"),
        },
        "variant_train": {
            "records": len(variant_train),
            "labels": dict(labels),
            "safe_event_bins": distribution(variant_train, "safe"),
        },
        "frozen_evaluation": {
            "validation": "data/1.5/coarse_sft/step4/final/validation.parquet",
            "test": "data/1.5/coarse_sft/step4/final/test.parquet",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in [manifest_path, selected_input_path, cot_path, checkpoint_path, *train_paths] if path.exists()
        },
        "output": {
            "path": "final/replacement_manifest.jsonl",
            "records": len(manifest),
            "sha256": sha256_file(manifest_path),
        },
        "next": "先补齐CoT并完成100条独立审核，再导出实验版Train和冻结的Validation/Test",
    }
    check_path = OUTPUT_ROOT / "cot/check/report.json"
    if ready == SELECT_COUNT and check_path.exists():
        check = json.loads(check_path.read_text(encoding="utf-8"))
        if check.get("counts") == {"pass": SELECT_COUNT} and not check.get("flagged"):
            report["export"] = export_variant(manifest)
            report["status"] = "pass"
            report["outputs"] = {"final": report["export"]["outputs"]}
            report["next"] = "使用step5/final训练；Validation/Test与Step 4基线逐字节相同"
    report_path = OUTPUT_ROOT / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(report_path)

    print(f"Step 5 complete: selected={SELECT_COUNT}, cot_ready={ready}, cot_pending={SELECT_COUNT - ready}")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")
