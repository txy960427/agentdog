"""Step 0：跨来源分组、固定split、抽样并生成选择清单。"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .source import (
    PipelineError,
    RISK_SOURCES,
    assign_length_quartiles,
    event_document,
    load_binary,
    load_fine,
    minhash,
    normalize_text,
    sha256_file,
    sha256_text,
    shingles,
    stable_json,
    tool_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[7]
BINARY_PATH = PROJECT_ROOT / "data/agentdog/1.0/source/AgentDoG1.0-Training-Data/AgentDoG-BinarySafety/train.json"
FINE_PATH = PROJECT_ROOT / "data/agentdog/1.0/source/AgentDoG1.0-Training-Data/AgentDoG-FineGrainedTaxonomy/train.json"
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark/agentdog"
OUTPUT_ROOT = PROJECT_ROOT / "data/agentdog/1.5/coarse-grained/sft/step0"
SEED = 42
NEAR_THRESHOLD = 0.88
BENCHMARK_THRESHOLD = 0.92

STEP1_QUOTAS = {
    "train": {"safe": 100, "unsafe": 100},
    "validation": {"safe": 25, "unsafe": 25},
    "test": {"safe": 25, "unsafe": 25},
}
STEP2_QUOTAS = {
    "train": {risk: 50 for risk in RISK_SOURCES},
    "validation": {risk: 13 if index % 2 == 0 else 12 for index, risk in enumerate(RISK_SOURCES)},
    "test": {risk: 12 if index % 2 == 0 else 13 for index, risk in enumerate(RISK_SOURCES)},
}
STEP1_FIELDS = {
    "length_quartile": 1.0,
    "event_bin": 1.0,
    "tool_action_count_bin": 1.0,
    "has_environment_error": 2.0,
    "has_trailing_complete": 1.0,
    "action_format": 1.0,
}
STEP2_FIELDS = {
    "failure_mode": 1.0,
    "real_world_harm": 1.0,
    "length_quartile": 1.0,
    "event_bin": 0.5,
    "available_tool_count_bin": 1.0,
    "has_environment_error": 2.0,
    "has_schema_warning": 2.0,
    "has_duplicate_tool_name": 2.0,
    "has_risk_structure": 2.0,
}
RISK_FLAGS = (
    "has_environment_error",
    "has_schema_warning",
    "has_duplicate_tool_name",
    "has_risk_structure",
)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def lsh_keys(signature: tuple[int, ...]) -> Iterable[tuple[int, tuple[int, ...]]]:
    for band in range(4):
        yield band, signature[band * 4 : band * 4 + 4]


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def group_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """先合并相同任务，再对相同工具序列的近似任务做聚类。"""
    dsu = DisjointSet(len(candidates))
    exact_user: dict[tuple[str, tuple[str, ...]], int] = {}
    exact_document: dict[str, int] = {}
    buckets: defaultdict[tuple[tuple[str, ...], int, tuple[int, ...]], list[int]] = defaultdict(list)

    for index, item in enumerate(candidates):
        for key, mapping in (
            ((item["normalized_user"], item["tool_sequence"]), exact_user),
            (item["document_hash"], exact_document),
        ):
            if key in mapping:
                dsu.union(index, mapping[key])
            else:
                mapping[key] = index

        near: set[int] = set()
        for band, values in lsh_keys(item["minhash"]):
            near.update(buckets[(item["tool_sequence"], band, values)])
        for other_index in near:
            if jaccard(item["shingles"], candidates[other_index]["shingles"]) >= NEAR_THRESHOLD:
                dsu.union(index, other_index)
        for band, values in lsh_keys(item["minhash"]):
            buckets[(item["tool_sequence"], band, values)].append(index)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        groups[dsu.find(index)].append(index)
    for members in groups.values():
        ids = sorted(candidates[index]["id"] for index in members)
        group_id = f"scenario_{sha256_text('|'.join(ids))[:16]}"
        for index in members:
            candidates[index]["group_id"] = group_id

    sizes = Counter(len(members) for members in groups.values())
    return {
        "groups": len(groups),
        "multi_record_groups": sum(count for size, count in sizes.items() if size > 1),
        "largest_group": max(sizes, default=0),
        "group_size_histogram": dict(sorted(sizes.items())),
        "cross_source_groups": sum(
            len({candidates[index]["stage"] for index in members}) > 1
            for members in groups.values()
        ),
    }


def decode_json_string(value: Any) -> Any:
    for _ in range(3):
        if not isinstance(value, str) or not value.strip().startswith(("[", "{")):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def event_lists(value: Any) -> Iterable[list[dict[str, Any]]]:
    value = decode_json_string(value)
    if not isinstance(value, list):
        return
    if value and all(isinstance(item, dict) and "role" in item for item in value):
        yield value
        return
    for child in value:
        yield from event_lists(child)


def benchmark_records(path: Path) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        rows = [row for child in value.values() if isinstance(child, list) for row in child]
        return rows or [value]
    return []


def benchmark_fingerprints() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [("atbench", BENCHMARK_ROOT / "atbench/test.json")]
    paths.extend(("rjudge", path) for path in sorted((BENCHMARK_ROOT / "rjudge/data").rglob("*.json")))
    fingerprints, files = [], []
    for source, path in paths:
        if not path.is_file():
            raise PipelineError(f"benchmark file not found: {path}")
        rows = benchmark_records(path)
        files.append({"source": source, "path": str(path), "sha256": sha256_file(path), "records": len(rows)})
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for field in ("contents", "conversation", "trajectory", "messages"):
                if field not in row:
                    continue
                for trajectory_index, events in enumerate(event_lists(row[field])):
                    user = "\n".join(
                        str(event.get("content", ""))
                        for event in events
                        if str(event.get("role", "")).lower() == "user"
                    )
                    normalized = normalize_text(user)
                    if not normalized:
                        continue
                    tool_sequence = tuple(
                        name
                        for event in events
                        if str(event.get("role", "")).lower() in {"agent", "assistant"}
                        if (name := tool_name(event.get("action")))
                    )
                    word_sets = shingles(user)
                    fingerprints.append(
                        {
                            "source": source,
                            "id": f"{path.name}:{row.get('id', row_index)}:{trajectory_index}",
                            "document_hash": sha256_text(event_document(events)),
                            "normalized_user": normalized,
                            "shingles": word_sets,
                            "minhash": minhash(word_sets),
                            "tool_sequence": tool_sequence,
                        }
                    )
    return fingerprints, files


def mark_benchmark_overlap(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    benchmarks, files = benchmark_fingerprints()
    exact_document: defaultdict[str, list[int]] = defaultdict(list)
    exact_user: defaultdict[str, list[int]] = defaultdict(list)
    buckets: defaultdict[tuple[tuple[str, ...], int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, item in enumerate(benchmarks):
        exact_document[item["document_hash"]].append(index)
        exact_user[item["normalized_user"]].append(index)
        for band, values in lsh_keys(item["minhash"]):
            buckets[(item["tool_sequence"], band, values)].append(index)

    matches = []
    for item in candidates:
        indices = set(exact_document.get(item["document_hash"], []))
        kind = "exact_document" if indices else None
        if not indices:
            indices.update(exact_user.get(item["normalized_user"], []))
            kind = "exact_user" if indices else None
        if not indices:
            for band, values in lsh_keys(item["minhash"]):
                indices.update(buckets.get((item["tool_sequence"], band, values), []))
        best = max(
            ((jaccard(item["shingles"], benchmarks[index]["shingles"]), index) for index in indices),
            default=(-1.0, -1),
        )
        item["benchmark_overlap"] = best[0] >= BENCHMARK_THRESHOLD
        if item["benchmark_overlap"]:
            benchmark = benchmarks[best[1]]
            matches.append(
                {
                    "id": item["id"],
                    "benchmark": benchmark["source"],
                    "benchmark_id": benchmark["id"],
                    "kind": kind or "near_user_and_tools",
                    "score": round(best[0], 6),
                }
            )

    blocked_groups = {item["group_id"] for item in candidates if item["benchmark_overlap"]}
    for item in candidates:
        item["eligible"] = item["group_id"] not in blocked_groups
    return {
        "files": files,
        "trajectories": len(benchmarks),
        "matched_records": len(matches),
        "blocked_groups": len(blocked_groups),
        "matches": matches[:50],
    }


def assign_splits(candidates: list[dict[str, Any]]) -> None:
    group_splits: dict[str, str] = {}
    for item in candidates:
        group_id = item["group_id"]
        if group_id not in group_splits:
            value = int.from_bytes(
                hashlib.blake2b(f"{SEED}:{group_id}".encode(), digest_size=8).digest(), "big"
            ) % 12
            group_splits[group_id] = "train" if value < 8 else "validation" if value < 10 else "test"
        item["split"] = group_splits[group_id]


def field_value(item: dict[str, Any], field: str) -> str:
    return str(item[field] if field in item else item["features"][field])


def target_distribution(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, float]]:
    return {
        field: {
            value: count / len(rows)
            for value, count in Counter(field_value(item, field) for item in rows).items()
        }
        for field in fields
    }


def greedy_bucket(
    *,
    pool: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    quota: int,
    fields: dict[str, float],
    salt: str,
    used_groups: set[str],
    global_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    targets = target_distribution(target_rows, fields)
    counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []

    def delta(item: dict[str, Any]) -> float:
        next_size = len(selected) + 1
        cost = 0.0
        for field, weight in fields.items():
            value = field_value(item, field)
            expected = next_size * targets[field][value]
            current = counts[(field, value)]
            cost += weight * ((current + 1 - expected) ** 2 - (current - expected) ** 2)
        if global_state:
            next_global = global_state["size"] + 1
            for field in RISK_FLAGS:
                value = field_value(item, field)
                expected = next_global * global_state["targets"][field][value]
                current = global_state["counts"][(field, value)]
                cost += 1.5 * ((current + 1 - expected) ** 2 - (current - expected) ** 2)
        return cost

    def rank(item: dict[str, Any]) -> int:
        return int.from_bytes(
            hashlib.blake2b(f"{SEED}:{salt}:{item['id']}".encode(), digest_size=8).digest(), "big"
        )

    for _ in range(quota):
        available = [item for item in pool if item["group_id"] not in used_groups]
        if not available:
            raise PipelineError(f"cannot fill quota: {salt}")
        item = min(available, key=lambda row: (delta(row), rank(row)))
        selected.append(item)
        used_groups.add(item["group_id"])
        for field in fields:
            counts[(field, field_value(item, field))] += 1
        if global_state:
            global_state["size"] += 1
            for field in RISK_FLAGS:
                global_state["counts"][(field, field_value(item, field))] += 1
    return selected


def sample(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [item for item in candidates if item["eligible"]]
    used_groups: set[str] = set()
    selected: list[dict[str, Any]] = []

    for split, labels in STEP1_QUOTAS.items():
        for label, quota in labels.items():
            target = [item for item in eligible if item["stage"] == "step1" and item["coarse"] == label]
            pool = [item for item in target if item["split"] == split]
            selected.extend(
                greedy_bucket(
                    pool=pool,
                    target_rows=target,
                    quota=quota,
                    fields=STEP1_FIELDS,
                    salt=f"step1:{split}:{label}",
                    used_groups=used_groups,
                )
            )

    fine = [item for item in eligible if item["stage"] == "step2"]
    global_targets = target_distribution(fine, RISK_FLAGS)
    for split, risks in STEP2_QUOTAS.items():
        global_state = {"size": 0, "counts": Counter(), "targets": global_targets}
        for risk, quota in risks.items():
            target = [item for item in fine if item["risk_source"] == risk]
            pool = [item for item in target if item["split"] == split]
            selected.extend(
                greedy_bucket(
                    pool=pool,
                    target_rows=target,
                    quota=quota,
                    fields=STEP2_FIELDS,
                    salt=f"step2:{split}:{risk}",
                    used_groups=used_groups,
                    global_state=global_state,
                )
            )
    return selected


def counts_by_split(selected: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = [item for item in selected if item["split"] == split]
        step1 = [item for item in rows if item["stage"] == "step1"]
        step2 = [item for item in rows if item["stage"] == "step2"]
        report[split] = {
            "records": len(rows),
            "step1": len(step1),
            "step1_labels": dict(sorted(Counter(item["coarse"] for item in step1).items())),
            "step2": len(step2),
            "step2_risk_sources": dict(sorted(Counter(item["risk_source"] for item in step2).items())),
            "step2_failure_modes": len({item["failure_mode"] for item in step2}),
            "step2_real_world_harms": len({item["real_world_harm"] for item in step2}),
            "step2_unique_triples": len(
                {(item["risk_source"], item["failure_mode"], item["real_world_harm"]) for item in step2}
            ),
            "step2_risk_structures": {
                field: sum(bool(item["features"][field]) for item in step2) for field in RISK_FLAGS
            },
            "step2_length_quartiles": dict(
                sorted(Counter(str(item["features"]["length_quartile"]) for item in step2).items())
            ),
        }
    return report


def validate(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    binary_report: dict[str, Any],
    fine_report: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    if binary_report["physical_records"] != 4000 or binary_report["unique_records"] != 2000:
        failures.append("BinarySafety must contain 4,000 physical / 2,000 unique rows")
    if binary_report["labels"] != {"safe": 1000, "unsafe": 1000}:
        failures.append("BinarySafety labels must be 1,000/1,000")
    if fine_report["physical_records"] != 4000 or fine_report["unique_records"] != 4000:
        failures.append("FineGrained must contain 4,000 unique rows")
    if len(fine_report["risk_sources"]) != 8 or fine_report["failure_mode_classes"] != 14 or fine_report["real_world_harm_classes"] != 10:
        failures.append("unexpected FineGrained taxonomy")
    if binary_report["quarantine"] or fine_report["quarantine"]:
        failures.append("source quarantine is not empty")
    if len(selected) != 900:
        failures.append(f"expected 900 selected parents, got {len(selected)}")
    if len({item["id"] for item in selected}) != len(selected):
        failures.append("selected IDs are not unique")
    if len({item["group_id"] for item in selected}) != len(selected):
        failures.append("more than one selected parent belongs to the same group")
    if any(item["benchmark_overlap"] for item in selected):
        failures.append("selected data overlaps an external benchmark")

    actual = counts_by_split(selected)
    fine_failure_modes = {item["failure_mode"] for item in candidates if item["stage"] == "step2"}
    fine_harms = {item["real_world_harm"] for item in candidates if item["stage"] == "step2"}
    for split in ("train", "validation", "test"):
        if actual[split]["step1_labels"] != STEP1_QUOTAS[split]:
            failures.append(f"Step 1 quota mismatch in {split}")
        if actual[split]["step2_risk_sources"] != STEP2_QUOTAS[split]:
            failures.append(f"Step 2 quota mismatch in {split}")
        rows = [item for item in selected if item["stage"] == "step2" and item["split"] == split]
        if {item["failure_mode"] for item in rows} != fine_failure_modes:
            failures.append(f"Step 2 Failure Mode coverage is incomplete in {split}")
        if {item["real_world_harm"] for item in rows} != fine_harms:
            failures.append(f"Step 2 Real-World Harm coverage is incomplete in {split}")

    groups: defaultdict[str, set[str]] = defaultdict(set)
    users: defaultdict[str, set[str]] = defaultdict(set)
    for item in selected:
        groups[item["group_id"]].add(item["split"])
        users[item["normalized_user"]].add(item["split"])
    group_leaks = sum(len(splits) > 1 for splits in groups.values())
    user_leaks = sum(len(splits) > 1 for splits in users.values())
    if group_leaks:
        failures.append(f"cross-split group leakage: {group_leaks}")

    fine = [item for item in candidates if item["stage"] == "step2" and item["eligible"]]
    binary = [item for item in candidates if item["stage"] == "step1" and item["eligible"]]
    source_rates = {
        field: sum(bool(item["features"][field]) for item in fine) / len(fine) for field in RISK_FLAGS
    }
    format_counts = {
        label: dict(sorted(Counter(item["features"]["action_format"] for item in binary if item["coarse"] == label).items()))
        for label in ("safe", "unsafe")
    }
    event_ranges = {
        label: [
            min(item["features"]["event_count"] for item in binary if item["coarse"] == label),
            max(item["features"]["event_count"] for item in binary if item["coarse"] == label),
        ]
        for label in ("safe", "unsafe")
    }
    fine_event_bins = dict(sorted(Counter(item["features"]["event_bin"] for item in fine).items()))
    tolerances = {
        "train": {field: 4 for field in RISK_FLAGS},
        "validation": {
            "has_environment_error": 2,
            "has_schema_warning": 2,
            "has_duplicate_tool_name": 2,
            "has_risk_structure": 3,
        },
        "test": {
            "has_environment_error": 2,
            "has_schema_warning": 2,
            "has_duplicate_tool_name": 2,
            "has_risk_structure": 3,
        },
    }
    for split in ("train", "validation", "test"):
        size = actual[split]["step2"]
        for field, count in actual[split]["step2_risk_structures"].items():
            expected = size * source_rates[field]
            if abs(count - expected) > tolerances[split][field]:
                failures.append(
                    f"{split}/{field} is not representative: {count} vs expected {expected:.1f}"
                )
        if split != "train":
            for field in RISK_FLAGS[:3]:
                if actual[split]["step2_risk_structures"][field] == 0:
                    failures.append(f"{split} has no {field} examples")

    audit = {
        "model_training": {
            "planned_final_train": {"safe": 500, "unsafe": 500},
            "planned_final_validation": {"safe": 125, "unsafe": 125},
            "planned_final_test": {"safe": 125, "unsafe": 125},
            "step2_train_risk_sources": actual["train"]["step2_risk_sources"],
        },
        "algorithm": {
            "seed": SEED,
            "group_before_split": True,
            "split_independent_sampling": True,
            "selected_unique_groups": len({item["group_id"] for item in selected}),
            "cross_split_groups": group_leaks,
            "cross_split_reused_user_texts": user_leaks,
        },
        "data_engineering": {
            "selected_ids_unique": len({item["id"] for item in selected}) == len(selected),
            "selected_benchmark_overlap": sum(item["benchmark_overlap"] for item in selected),
            "binary_quarantine": binary_report["quarantine"],
            "fine_quarantine": fine_report["quarantine"],
        },
        "data_analysis": {
            "source_bias_warnings": [
                "BinarySafety action format perfectly separates safe and unsafe labels",
                "FineGrainedTaxonomy contains no trajectories with 1-4 events",
            ],
            "binary_action_formats": format_counts,
            "binary_event_count_ranges": event_ranges,
            "fine_event_bins": fine_event_bins,
            "source_step2_risk_structure_rates": {
                field: round(rate, 6) for field, rate in source_rates.items()
            },
            "splits": actual,
        },
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    if failures:
        raise PipelineError("Step 0 acceptance failed:\n- " + "\n- ".join(failures))
    return audit


def candidate_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": item["stage"],
        "id": item["id"],
        "source_row_id": item["source_row_id"],
        "raw_hash": item["raw_hash"],
        "trajectory_hash": item["trajectory_hash"],
        "group_id": item["group_id"],
        "split": item["split"],
        "coarse": item["coarse"],
        "risk_source": item["risk_source"],
        "failure_mode": item["failure_mode"],
        "real_world_harm": item["real_world_harm"],
        "features": item["features"],
        "benchmark_overlap": item["benchmark_overlap"],
        "eligible": item["eligible"],
    }


def selection_row(item: dict[str, Any]) -> dict[str, Any]:
    pair_id = f"pair_{sha256_text(item['id'])[:16]}" if item["stage"] == "step2" else None
    return {
        "stage": item["stage"],
        "id": item["id"],
        "source_row_id": item["source_row_id"],
        "raw_hash": item["raw_hash"],
        "group_id": item["group_id"],
        "pair_id": pair_id,
        "split": item["split"],
        "coarse": item["coarse"],
        "risk_source": item["risk_source"],
        "failure_mode": item["failure_mode"],
        "real_world_harm": item["real_world_harm"],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def run() -> None:
    binary, binary_report, binary_quarantine = load_binary(BINARY_PATH)
    fine, fine_report, fine_quarantine = load_fine(FINE_PATH)
    candidates = binary + fine
    assign_length_quartiles(candidates)
    grouping = group_candidates(candidates)
    benchmark = mark_benchmark_overlap(candidates)
    assign_splits(candidates)
    selected = sample(candidates)
    audit = validate(candidates, selected, binary_report, fine_report)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    intermediate = OUTPUT_ROOT / "intermediate"
    final = OUTPUT_ROOT / "final"
    intermediate.mkdir(parents=True)
    final.mkdir()

    split_order = {"train": 0, "validation": 1, "test": 2}
    candidates.sort(key=lambda item: (item["stage"], item["id"]))
    selected.sort(key=lambda item: (item["stage"], split_order[item["split"]], item["id"]))
    candidate_path = intermediate / "candidates.jsonl"
    selection_path = final / "selection.jsonl"
    candidate_count = write_jsonl(candidate_path, (candidate_row(item) for item in candidates))
    selection_count = write_jsonl(selection_path, (selection_row(item) for item in selected))

    report = {
        "stage": "agentdog-1.5/coarse-grained/sft/step0",
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"binary": binary_report, "fine": fine_report},
        "quarantine": {
            "binary": binary_quarantine[:50],
            "fine": fine_quarantine[:50],
        },
        "grouping": grouping,
        "benchmark_overlap": benchmark,
        "selection": {
            "records": selection_count,
            "by_split": counts_by_split(selected),
        },
        "audit": audit,
        "outputs": {
            "candidates": {
                "path": "intermediate/candidates.jsonl",
                "records": candidate_count,
                "sha256": sha256_file(candidate_path),
            },
            "selection": {
                "path": "final/selection.jsonl",
                "records": selection_count,
                "sha256": sha256_file(selection_path),
            },
        },
    }
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"Step 0 complete: {OUTPUT_ROOT}")
    print(f"candidates={candidate_count}, selected={selection_count}, groups={grouping['groups']}")
    for split, summary in report["selection"]["by_split"].items():
        print(f"{split}: step1={summary['step1']}, step2={summary['step2']}")
