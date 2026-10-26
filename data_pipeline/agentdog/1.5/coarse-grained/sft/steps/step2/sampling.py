"""按 Risk Source 固定配额抽取 Step 2 的 500 条 unsafe 轨迹。"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from ..step1.source import PipelineError, sha256_text
from .source import RISK_SOURCES

SPLIT_ORDER = ("test", "validation", "train")
MATCH_WEIGHTS = {
    "failure_mode": 1.0,
    "real_world_harm": 1.0,
    "length_quartile": 1.0,
    "available_tool_count_bin": 1.0,
    "event_bin": 0.5,
    "has_environment_error": 2.0,
    "has_schema_warning": 2.0,
    "has_duplicate_tool_name": 2.0,
    "has_data_irregularity": 2.0,
}
DIRTY_FIELDS = (
    "has_environment_error",
    "has_schema_warning",
    "has_duplicate_tool_name",
    "has_data_irregularity",
)


def build_quotas(
    base: dict[str, int], extras: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    quotas = {
        split: {risk: int(base[split]) for risk in sorted(RISK_SOURCES)}
        for split in ("train", "validation", "test")
    }
    for split, risk_sources in extras.items():
        if split not in quotas or any(risk not in RISK_SOURCES for risk in risk_sources):
            raise PipelineError(f"Invalid Step-2 extra quota: {split}={risk_sources}")
        for risk in risk_sources:
            quotas[split][risk] += 1
    if {split: sum(values.values()) for split, values in quotas.items()} != {
        "train": 400,
        "validation": 50,
        "test": 50,
    }:
        raise PipelineError("Step-2 quotas must total 400/50/50")
    return quotas


def field_value(item: dict[str, Any], field: str) -> str:
    target = item["canonical"]["target"]
    if field in {"failure_mode", "real_world_harm"}:
        return str(target[field])
    return str(item["features"][field])


def stratified_sample(
    eligible: list[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_risk: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_risk[item["canonical"]["target"]["risk_source"]].append(item)

    # 长度分位在各RS内部计算，避免类别本身的长度差异影响抽样。
    for rows in by_risk.values():
        ordered = sorted(rows, key=lambda item: (item["features"]["token_count_proxy"], item["canonical"]["id"]))
        for rank, item in enumerate(ordered):
            item["features"]["length_quartile"] = min(3, rank * 4 // len(ordered))

    selected_groups: set[str] = set()
    selected_ids: set[str] = set()
    feature_counts: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    selected_per_risk: Counter[str] = Counter()
    splits = {split: [] for split in ("train", "validation", "test")}
    target_distributions: dict[str, dict[str, dict[str, float]]] = {}
    for risk, rows in by_risk.items():
        target_distributions[risk] = {}
        for field in MATCH_WEIGHTS:
            counts = Counter(field_value(item, field) for item in rows)
            target_distributions[risk][field] = {
                value: count / len(rows) for value, count in counts.items()
            }

    def rank(record_id: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(f"{seed}:{record_id}".encode(), digest_size=8).digest(),
            "big",
        )

    def distribution_cost(
        item: dict[str, Any],
        risk: str,
        counts: Counter[tuple[str, str]],
        selected_count: int,
    ) -> float:
        # 比较加入候选后与原RS桶边际分布的平方误差增量。
        cost = 0.0
        for field, weight in MATCH_WEIGHTS.items():
            value = field_value(item, field)
            expected = selected_count * target_distributions[risk][field][value]
            current = counts[(field, value)]
            cost += weight * ((current + 1 - expected) ** 2 - (current - expected) ** 2)
        return cost

    for split in SPLIT_ORDER:
        for risk in sorted(RISK_SOURCES):
            counts = feature_counts[risk]
            for _ in range(quotas[split][risk]):
                pool = [
                    item
                    for item in by_risk[risk]
                    if item["canonical"]["id"] not in selected_ids
                    and item["canonical"]["group_id"] not in selected_groups
                ]
                if not pool:
                    raise PipelineError(f"Cannot fill Step-2 quota for {split}/{risk}")
                item = min(
                    pool,
                    key=lambda row: (
                        distribution_cost(
                            row, risk, counts, selected_per_risk[risk] + 1
                        ),
                        rank(row["canonical"]["id"]),
                    ),
                )
                canonical = item["canonical"]
                canonical["split"] = split
                canonical["pair_id"] = f"pair_{sha256_text(canonical['id'])[:16]}"
                selected_ids.add(canonical["id"])
                selected_groups.add(canonical["group_id"])
                for field in MATCH_WEIGHTS:
                    counts[(field, field_value(item, field))] += 1
                selected_per_risk[risk] += 1
                splits[split].append(item)
    return splits


def sampling_report(
    splits: dict[str, list[dict[str, Any]]],
    eligible: Sequence[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "eligible_records": len(eligible),
        "eligible_risk_sources": dict(
            sorted(Counter(item["canonical"]["target"]["risk_source"] for item in eligible).items())
        ),
        "quotas": quotas,
        "splits": {},
    }
    all_selected: list[dict[str, Any]] = []
    for split, rows in splits.items():
        all_selected.extend(rows)
        report["splits"][split] = {
            "records": len(rows),
            "risk_sources": dict(
                sorted(Counter(item["canonical"]["target"]["risk_source"] for item in rows).items())
            ),
            "failure_modes": dict(
                sorted(Counter(item["canonical"]["target"]["failure_mode"] for item in rows).items())
            ),
            "real_world_harms": dict(
                sorted(Counter(item["canonical"]["target"]["real_world_harm"] for item in rows).items())
            ),
            "length_quartiles": dict(
                sorted(Counter(str(item["features"]["length_quartile"]) for item in rows).items())
            ),
            "available_tool_counts": dict(
                sorted(Counter(str(item["features"]["available_tool_count"]) for item in rows).items())
            ),
            "environment_error_records": sum(item["features"]["has_environment_error"] for item in rows),
        }
    report["selected_records"] = len(all_selected)
    report["selected_unique_triples"] = len(
        {
            (
                item["canonical"]["target"]["risk_source"],
                item["canonical"]["target"]["failure_mode"],
                item["canonical"]["target"]["real_world_harm"],
            )
            for item in all_selected
        }
    )
    report["source_distribution_preservation"] = {}
    for field in DIRTY_FIELDS:
        source_rate = sum(bool(item["features"][field]) for item in eligible) / len(eligible)
        selected_rate = sum(bool(item["features"][field]) for item in all_selected) / len(all_selected)
        report["source_distribution_preservation"][field] = {
            "source_rate": round(source_rate, 6),
            "selected_rate": round(selected_rate, 6),
            "absolute_delta": round(abs(source_rate - selected_rate), 6),
        }
    return report


def validate_acceptance(
    parsed: Sequence[dict[str, Any]],
    quarantine: Sequence[dict[str, Any]],
    source_report: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    quotas: dict[str, dict[str, int]],
    leakage: dict[str, Any],
) -> None:
    failures: list[str] = []
    if source_report["physical_records"] != 4000 or source_report["unique_records"] != 4000:
        failures.append("source must contain 4,000 unique records")
    expected_classes = {"risk_source": 8, "failure_mode": 14, "real_world_harm": 10, "triples": 1044}
    if source_report["taxonomy_class_counts"] != expected_classes:
        failures.append(f"unexpected taxonomy classes: {source_report['taxonomy_class_counts']}")
    if quarantine:
        failures.append(f"quarantine is not empty: {len(quarantine)}")
    if {split: len(rows) for split, rows in splits.items()} != {"train": 400, "validation": 50, "test": 50}:
        failures.append("split sizes are not 400/50/50")
    for split, rows in splits.items():
        actual = Counter(item["canonical"]["target"]["risk_source"] for item in rows)
        if dict(actual) != quotas[split]:
            failures.append(f"Risk Source quota mismatch in {split}")
    selected = [item for rows in splits.values() for item in rows]
    pair_ids = [item["canonical"]["pair_id"] for item in selected]
    if None in pair_ids or len(pair_ids) != len(set(pair_ids)):
        failures.append("pair_id is missing or duplicated")
    if any(item["canonical"]["target"]["coarse"] != "unsafe" for item in selected):
        failures.append("selected coarse labels are not all unsafe")
    if any(
        any(key in tool for key in ("_source", "_original_description"))
        for item in selected
        for tool in item["canonical"]["tools"]
    ):
        failures.append("private tool metadata leaked into final rows")
    selected_rows = [item for rows in splits.values() for item in rows]
    for field in DIRTY_FIELDS:
        source_rate = sum(bool(item["features"][field]) for item in parsed) / len(parsed)
        selected_rate = sum(bool(item["features"][field]) for item in selected_rows) / len(selected_rows)
        if abs(source_rate - selected_rate) > 0.02:
            failures.append(
                f"source distribution drift for {field}: {source_rate:.3f} -> {selected_rate:.3f}"
            )
    for field in (
        "selected_group_cross_split_count",
        "selected_id_cross_split_count",
        "selected_benchmark_overlap_count",
    ):
        if leakage[field]:
            failures.append(f"leakage check failed: {field}={leakage[field]}")
    if failures:
        raise PipelineError("Step-2 acceptance failed:\n- " + "\n- ".join(failures))
