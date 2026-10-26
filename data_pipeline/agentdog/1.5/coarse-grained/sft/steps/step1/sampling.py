"""Step 1 的分层抽样、报告和硬验收规则。"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .source import PipelineError

SPLIT_ORDER = ("test", "validation", "train")
BALANCE_FIELDS = (
    "length_quartile",
    "event_bin",
    "tool_count_bin",
    "has_environment_error",
    "has_trailing_complete",
    "action_format",
)
BINARY_BALANCE_FIELDS = (
    "has_environment_error",
    "has_trailing_complete",
    "has_json_tool_action",
)

# 先锁定test和validation，再从剩余group中取train，杜绝同场景跨split。
def stratified_sample(
    eligible: list[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """按group、标签、长度和结构特征生成固定配额的三个split。"""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_label[item["canonical"]["target"]["coarse"]].append(item)
    for label_items in by_label.values():
        ordered = sorted(
            label_items,
            key=lambda item: (
                item["features"]["token_count_proxy"],
                item["canonical"]["id"],
            ),
        )
        size = len(ordered)
        for rank, item in enumerate(ordered):
            item["features"]["length_quartile"] = min(3, rank * 4 // size)

    def deterministic_rank(seed: int, record_id: str) -> int:
        digest = hashlib.blake2b(f"{seed}:{record_id}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    def novelty_score(item: dict[str, Any], counts: Counter[tuple[str, str]]) -> float:
        score = 0.0
        for field in BALANCE_FIELDS:
            value = str(item["features"][field])
            score += 1.0 / (1.0 + counts[(field, value)])
        first_tool = str(item["features"]["first_tool_name"])
        score += 0.35 / (1.0 + counts[("first_tool_name", first_tool)])
        return score

    def update_feature_counts(
        item: dict[str, Any], counts: Counter[tuple[str, str]]
    ) -> None:
        for field in BALANCE_FIELDS:
            counts[(field, str(item["features"][field]))] += 1
        counts[("first_tool_name", str(item["features"]["first_tool_name"]))] += 1

    def pair_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_features = left["features"]
        right_features = right["features"]
        distance = 0.0
        weights = {
            "has_environment_error": 5.0,
            "has_trailing_complete": 4.0,
            "has_json_tool_action": 3.0,
            "action_format": 2.0,
            "event_bin": 1.0,
            "tool_count_bin": 1.0,
        }
        for field, weight in weights.items():
            if left_features[field] != right_features[field]:
                distance += weight
        return distance

    def available_candidates(
        items: Sequence[dict[str, Any]],
        label: str,
        quartile: int,
        selected_groups: set[str],
        selected_ids: set[str],
    ) -> list[dict[str, Any]]:
        exact = [
            item
            for item in items
            if item["canonical"]["target"]["coarse"] == label
            and item["canonical"]["group_id"] not in selected_groups
            and item["canonical"]["id"] not in selected_ids
            and item["features"]["length_quartile"] == quartile
        ]
        if exact:
            return exact
        return [
            item
            for item in items
            if item["canonical"]["target"]["coarse"] == label
            and item["canonical"]["group_id"] not in selected_groups
            and item["canonical"]["id"] not in selected_ids
        ]

    def choose_pair(
        items: Sequence[dict[str, Any]],
        quartile: int,
        selected_groups: set[str],
        selected_ids: set[str],
        feature_counts: dict[str, Counter[tuple[str, str]]],
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        safe_pool = available_candidates(
            items, "safe", quartile, selected_groups, selected_ids
        )
        unsafe_pool = available_candidates(
            items, "unsafe", quartile, selected_groups, selected_ids
        )
        if not safe_pool or not unsafe_pool:
            raise PipelineError(f"Cannot fill paired quota for length quartile {quartile}")

        def shortlist(pool: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
            return sorted(
                pool,
                key=lambda item: (
                    -novelty_score(item, feature_counts[label]),
                    deterministic_rank(seed, item["canonical"]["id"]),
                ),
            )[:64]

        safe_shortlist = shortlist(safe_pool, "safe")
        unsafe_shortlist = shortlist(unsafe_pool, "unsafe")
        pairs: list[
            tuple[tuple[float, float, int, int], dict[str, Any], dict[str, Any]]
        ] = []
        for safe_item in safe_shortlist:
            for unsafe_item in unsafe_shortlist:
                if (
                    safe_item["canonical"]["group_id"]
                    == unsafe_item["canonical"]["group_id"]
                ):
                    continue
                novelty = novelty_score(safe_item, feature_counts["safe"]) + novelty_score(
                    unsafe_item, feature_counts["unsafe"]
                )
                key = (
                    pair_distance(safe_item, unsafe_item),
                    -novelty,
                    deterministic_rank(seed, safe_item["canonical"]["id"]),
                    deterministic_rank(seed, unsafe_item["canonical"]["id"]),
                )
                pairs.append((key, safe_item, unsafe_item))
        if not pairs:
            raise PipelineError(
                f"No group-independent safe/unsafe pair for quartile {quartile}"
            )
        _, safe_item, unsafe_item = min(pairs, key=lambda value: value[0])
        return safe_item, unsafe_item

    def choose_extra(
        items: Sequence[dict[str, Any]],
        label: str,
        quartile: int,
        selected_groups: set[str],
        selected_ids: set[str],
        counts: Counter[tuple[str, str]],
        seed: int,
    ) -> dict[str, Any]:
        pool = available_candidates(items, label, quartile, selected_groups, selected_ids)
        if not pool:
            raise PipelineError(f"Cannot fill extra {label} quota")
        return min(
            pool,
            key=lambda item: (
                -novelty_score(item, counts),
                deterministic_rank(seed, item["canonical"]["id"]),
            ),
        )

    selected_groups: set[str] = set()
    selected_ids: set[str] = set()
    feature_counts = {"safe": Counter(), "unsafe": Counter()}
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    reasons: dict[str, str] = {}

    # 一旦选择某条记录，就锁定整个group，后续split不得再使用。
    def add(item: dict[str, Any], split: str, reason: str) -> None:
        record_id = item["canonical"]["id"]
        group_id = item["canonical"]["group_id"]
        if record_id in selected_ids or group_id in selected_groups:
            raise PipelineError(f"Selection reused id/group: {record_id}/{group_id}")
        selected_ids.add(record_id)
        selected_groups.add(group_id)
        update_feature_counts(
            item, feature_counts[item["canonical"]["target"]["coarse"]]
        )
        item["canonical"]["split"] = split
        splits[split].append(item)
        reasons[record_id] = reason

    quartile_cursor = 0
    for split in SPLIT_ORDER:
        safe_quota = quotas[split]["safe"]
        unsafe_quota = quotas[split]["unsafe"]
        paired = min(safe_quota, unsafe_quota)
        for pair_index in range(paired):
            quartile = (quartile_cursor + pair_index) % 4
            safe_item, unsafe_item = choose_pair(
                eligible,
                quartile,
                selected_groups,
                selected_ids,
                feature_counts,
                seed,
            )
            add(safe_item, split, f"paired_structural_balance:q{quartile}")
            add(unsafe_item, split, f"paired_structural_balance:q{quartile}")
        quartile_cursor = (quartile_cursor + paired) % 4
        for label, quota in (("safe", safe_quota), ("unsafe", unsafe_quota)):
            remaining = quota - paired
            for extra_index in range(remaining):
                quartile = (quartile_cursor + extra_index) % 4
                item = choose_extra(
                    eligible,
                    label,
                    quartile,
                    selected_groups,
                    selected_ids,
                    feature_counts[label],
                    seed,
                )
                add(item, split, f"unpaired_quota_fill:q{quartile}")
            quartile_cursor = (quartile_cursor + remaining) % 4

    for split, rows in splits.items():
        expected = quotas[split]
        actual = Counter(item["canonical"]["target"]["coarse"] for item in rows)
        if any(actual[label] != count for label, count in expected.items()):
            raise PipelineError(
                f"Quota mismatch in {split}: expected={expected}, actual={actual}"
            )
    return splits, reasons


def sampling_report(
    splits: dict[str, list[dict[str, Any]]],
    eligible: Sequence[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """汇总split分布，并计算safe/unsafe之间的结构偏差。"""
    def distribution(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(item["features"][field]) for item in rows).items()))

    def rate(rows: Sequence[dict[str, Any]], field: str) -> float:
        return (
            sum(bool(item["features"][field]) for item in rows) / len(rows) if rows else 0.0
        )

    def percentile(values: Sequence[int], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    report: dict[str, Any] = {
        "eligible_records": len(eligible),
        "eligible_labels": dict(
            sorted(
                Counter(
                    item["canonical"]["target"]["coarse"] for item in eligible
                ).items()
            )
        ),
        "quotas": quotas,
        "splits": {},
    }
    all_selected: list[dict[str, Any]] = []
    for split, rows in splits.items():
        all_selected.extend(rows)
        labels = Counter(item["canonical"]["target"]["coarse"] for item in rows)
        lengths = [item["features"]["token_count_proxy"] for item in rows]
        report["splits"][split] = {
            "records": len(rows),
            "labels": dict(sorted(labels.items())),
            "length_proxy": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "p95": round(percentile(lengths, 0.95), 2),
                "max": max(lengths),
            },
            "distributions": {
                field: distribution(rows, field) for field in BALANCE_FIELDS
            },
        }
    by_label = {
        label: [
            item
            for item in all_selected
            if item["canonical"]["target"]["coarse"] == label
        ]
        for label in ("safe", "unsafe")
    }
    binary_deltas = {}
    for field in BINARY_BALANCE_FIELDS:
        safe_rate = rate(by_label["safe"], field)
        unsafe_rate = rate(by_label["unsafe"], field)
        binary_deltas[field] = {
            "safe_rate": round(safe_rate, 6),
            "unsafe_rate": round(unsafe_rate, 6),
            "absolute_delta": round(abs(safe_rate - unsafe_rate), 6),
        }
    report["selected_records"] = len(all_selected)
    report["selected_labels"] = dict(
        sorted(
            Counter(
                item["canonical"]["target"]["coarse"] for item in all_selected
            ).items()
        )
    )
    report["binary_structure_label_deltas"] = binary_deltas
    return report


def leakage_report(
    parsed: Sequence[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    benchmark_report: dict[str, Any],
) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    id_splits: dict[str, set[str]] = defaultdict(set)
    selected_ids: set[str] = set()
    for split, rows in splits.items():
        for item in rows:
            record_id = item["canonical"]["id"]
            selected_ids.add(record_id)
            group_splits[item["canonical"]["group_id"]].add(split)
            id_splits[record_id].add(split)
    group_leaks = sorted(
        group for group, values in group_splits.items() if len(values) > 1
    )
    id_leaks = sorted(
        record_id for record_id, values in id_splits.items() if len(values) > 1
    )
    matched_ids = {match["id"] for match in benchmark_report.get("matches", [])}
    selected_benchmark_matches = sorted(selected_ids & matched_ids)
    trajectory_hashes = Counter(item["trajectory_hash"] for item in parsed)
    return {
        "selected_group_cross_split_count": len(group_leaks),
        "selected_group_cross_split_examples": group_leaks[:20],
        "selected_id_cross_split_count": len(id_leaks),
        "selected_id_cross_split_examples": id_leaks[:20],
        "unique_trajectory_hash_collisions": sum(
            count > 1 for count in trajectory_hashes.values()
        ),
        "selected_benchmark_overlap_count": len(selected_benchmark_matches),
        "selected_benchmark_overlap_ids": selected_benchmark_matches[:20],
    }


def validate_acceptance(
    parsed: Sequence[dict[str, Any]],
    quarantine: Sequence[dict[str, Any]],
    dedup: dict[str, Any],
    sampling: dict[str, Any],
    leakage: dict[str, Any],
    strict_counts: bool,
) -> None:
    # 这里集中执行文档中承诺的硬验收条件，任一失败即停止导出。
    failures: list[str] = []
    if strict_counts:
        if dedup["physical_records"] != 4000:
            failures.append(f"physical records: {dedup['physical_records']} != 4000")
        if dedup["unique_records"] != 2000:
            failures.append(f"unique records: {dedup['unique_records']} != 2000")
        if dedup["unique_labels"] != {"safe": 1000, "unsafe": 1000}:
            failures.append(f"unique labels: {dedup['unique_labels']}")
        if not dedup["first_half_equals_second_half"]:
            failures.append("source halves are not identical")
    if quarantine:
        failures.append(f"quarantine is not empty: {len(quarantine)}")
    if sampling["selected_records"] != 250:
        failures.append(f"selected records: {sampling['selected_records']} != 250")
    if sampling["selected_labels"] != {"safe": 125, "unsafe": 125}:
        failures.append(f"selected labels: {sampling['selected_labels']}")
    if leakage["selected_group_cross_split_count"]:
        failures.append("group leakage across splits")
    if leakage["selected_id_cross_split_count"]:
        failures.append("record leakage across splits")
    if leakage["selected_benchmark_overlap_count"]:
        failures.append("benchmark overlap entered selected data")
    for field, values in sampling["binary_structure_label_deltas"].items():
        if values["absolute_delta"] > 0.15:
            failures.append(
                f"binary structure delta for {field}: {values['absolute_delta']:.3f} > 0.15"
            )
    if failures:
        raise PipelineError("Acceptance failed:\n- " + "\n- ".join(failures))
