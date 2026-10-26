"""场景近重复分组，以及与 ATBench、R-Judge 的数据泄漏检查。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .source import (
    DisjointSet,
    Fingerprint,
    PipelineError,
    action_tool_name,
    event_document,
    jaccard,
    lsh_keys,
    minhash_signature,
    normalize_text,
    sha256_text,
    word_shingles,
)

# LSH只缩小候选范围，最终匹配仍由精确哈希或Jaccard阈值确认。
def item_fingerprint(item: dict[str, Any]) -> Fingerprint:
    features = item["features"]
    return Fingerprint(
        source="AgentDoG-BinarySafety",
        record_id=item["canonical"]["id"],
        document_hash=features["document_hash"],
        normalized_user=features["normalized_user"],
        shingles=frozenset(features["user_shingles"]),
        minhash=tuple(features["user_minhash"]),
        tool_sequence=tuple(features["tool_sequence"]),
    )


def assign_groups(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    fingerprints = [item_fingerprint(item) for item in items]
    dsu = DisjointSet(len(items))
    exact_user: dict[tuple[str, tuple[str, ...]], int] = {}
    buckets: dict[tuple[tuple[str, ...], int, tuple[int, ...]], list[int]] = (
        defaultdict(list)
    )

    # LSH只负责召回候选，最终是否合并仍由Jaccard阈值决定。
    for index, fingerprint in enumerate(fingerprints):
        exact_key = (fingerprint.normalized_user, fingerprint.tool_sequence)
        if exact_key in exact_user:
            dsu.union(index, exact_user[exact_key])
        else:
            exact_user[exact_key] = index

        candidates: set[int] = set()
        for band, values in lsh_keys(fingerprint.minhash):
            key = (fingerprint.tool_sequence, band, values)
            candidates.update(buckets[key])
        for candidate in candidates:
            other = fingerprints[candidate]
            if jaccard(fingerprint.shingles, other.shingles) >= threshold:
                dsu.union(index, candidate)
        for band, values in lsh_keys(fingerprint.minhash):
            buckets[(fingerprint.tool_sequence, band, values)].append(index)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        members[dsu.find(index)].append(index)

    for group_members in members.values():
        member_ids = sorted(items[index]["canonical"]["id"] for index in group_members)
        group_id = f"scenario_{sha256_text('|'.join(member_ids))[:16]}"
        for index in group_members:
            items[index]["canonical"]["group_id"] = group_id

    sizes = sorted((len(value) for value in members.values()), reverse=True)
    return {
        "threshold": threshold,
        "groups": len(members),
        "multi_record_groups": sum(size > 1 for size in sizes),
        "largest_group": sizes[0] if sizes else 0,
        "group_size_histogram": dict(sorted(Counter(sizes).items())),
    }


def decode_json_strings(value: Any, depth: int = 0) -> Any:
    if depth >= 3 or not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return decode_json_strings(json.loads(stripped), depth + 1)
    except json.JSONDecodeError:
        return value


def iter_event_lists(value: Any) -> Iterator[list[dict[str, Any]]]:
    value = decode_json_strings(value)
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "role" in item for item in value):
            yield value
            return
        for item in value:
            yield from iter_event_lists(item)


def benchmark_event_fingerprint(
    source: str, record_id: str, events: list[dict[str, Any]]
) -> Fingerprint:
    user_text = "\n".join(
        str(event.get("content", ""))
        for event in events
        if str(event.get("role", "")).lower() == "user"
    )
    tools: list[str] = []
    for event in events:
        if str(event.get("role", "")).lower() not in {"agent", "assistant"}:
            continue
        if name := action_tool_name(event.get("action")):
            tools.append(name)
    shingles = word_shingles(user_text)
    return Fingerprint(
        source=source,
        record_id=record_id,
        document_hash=sha256_text(event_document(events)),
        normalized_user=normalize_text(user_text),
        shingles=shingles,
        minhash=minhash_signature(shingles),
        tool_sequence=tuple(tools),
    )


def load_json_records(path: Path) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        records: list[Any] = []
        for child in value.values():
            if isinstance(child, list):
                records.extend(child)
        return records or [value]
    return []


def load_benchmark_fingerprints(benchmark_root: Path) -> list[Fingerprint]:
    targets: list[tuple[str, list[Path]]] = []
    atbench_path = benchmark_root / "atbench" / "test.json"
    rjudge_path = benchmark_root / "rjudge" / "data"
    if not atbench_path.is_file():
        raise PipelineError(f"ATBench file not found: {atbench_path}")
    if not rjudge_path.is_dir():
        raise PipelineError(f"R-Judge data directory not found: {rjudge_path}")
    targets.append(("atbench", [atbench_path]))
    targets.append(("rjudge", sorted(rjudge_path.rglob("*.json"))))

    fingerprints: list[Fingerprint] = []
    for source, paths in targets:
        for path in paths:
            for row_index, record in enumerate(load_json_records(path)):
                if not isinstance(record, dict):
                    continue
                record_id = str(record.get("id", f"{path.stem}:{row_index}"))
                candidates: list[Any] = []
                for field in ("contents", "conversation", "trajectory", "messages"):
                    if field in record:
                        candidates.append(record[field])
                for candidate in candidates:
                    for trajectory_index, events in enumerate(
                        iter_event_lists(candidate)
                    ):
                        fingerprints.append(
                            benchmark_event_fingerprint(
                                source,
                                f"{path.relative_to(benchmark_root)}:{record_id}:{trajectory_index}",
                                events,
                            )
                        )
    return fingerprints


def benchmark_overlap(
    items: list[dict[str, Any]],
    benchmark_root: Path,
    threshold: float,
) -> tuple[set[str], dict[str, Any]]:
    benchmarks = load_benchmark_fingerprints(benchmark_root)
    exact: dict[str, list[int]] = defaultdict(list)
    normalized_user: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    buckets: dict[tuple[tuple[str, ...], int, tuple[int, ...]], list[int]] = (
        defaultdict(list)
    )
    for index, fingerprint in enumerate(benchmarks):
        exact[fingerprint.document_hash].append(index)
        normalized_user[
            (fingerprint.normalized_user, fingerprint.tool_sequence)
        ].append(index)
        for band, values in lsh_keys(fingerprint.minhash):
            buckets[(fingerprint.tool_sequence, band, values)].append(index)

    excluded: set[str] = set()
    matches: list[dict[str, Any]] = []
    # 依次尝试整轨迹精确匹配、用户任务精确匹配和近重复匹配。
    for item in items:
        fingerprint = item_fingerprint(item)
        candidate_indices: set[int] = set(exact.get(fingerprint.document_hash, []))
        match_kind = "exact_document" if candidate_indices else None
        if not candidate_indices:
            candidate_indices.update(
                normalized_user.get(
                    (fingerprint.normalized_user, fingerprint.tool_sequence), []
                )
            )
            if candidate_indices:
                match_kind = "exact_user_and_tools"
        if not candidate_indices:
            for band, values in lsh_keys(fingerprint.minhash):
                candidate_indices.update(
                    buckets.get((fingerprint.tool_sequence, band, values), [])
                )

        best_index: int | None = None
        best_score = -1.0
        for candidate_index in candidate_indices:
            candidate = benchmarks[candidate_index]
            score = jaccard(fingerprint.shingles, candidate.shingles)
            if fingerprint.document_hash == candidate.document_hash:
                score = 1.0
            if score > best_score:
                best_index = candidate_index
                best_score = score
        if best_index is None or best_score < threshold:
            continue
        candidate = benchmarks[best_index]
        record_id = fingerprint.record_id
        excluded.add(record_id)
        matches.append(
            {
                "id": record_id,
                "label": item["canonical"]["target"]["coarse"],
                "benchmark": candidate.source,
                "benchmark_record": candidate.record_id,
                "match_kind": match_kind or "near_user_and_tools",
                "user_jaccard": round(best_score, 6),
            }
        )

    return excluded, {
        "benchmark_root": str(benchmark_root),
        "benchmark_trajectories": len(benchmarks),
        "threshold": threshold,
        "matched_source_records": len(excluded),
        "matches_by_benchmark": dict(
            sorted(Counter(row["benchmark"] for row in matches).items())
        ),
        "matches": matches,
    }
