#!/usr/bin/env python3
"""Build the Step-1 coarse-grained BinarySafety dataset.

The pipeline is deterministic and performs no model-assisted or manual
labelling.  It trusts the public ``safe``/``unsafe`` labels, removes exact
duplicates, parses trajectories, groups template-near-duplicates, checks
benchmark overlap, creates group-safe splits, and exports audit reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "step1_binary.yaml"

BEGIN_MARKER = "<BEGIN TRAJECTORY>"
END_MARKER = "<END TRAJECTORY>"
PROFILE_MARKER = "=== Agent Profile ==="
HISTORY_MARKER = "=== Conversation History ==="
TOOLS_MARKER = "=== Available Tools ==="
ROLE_RE = re.compile(r"(?m)^\s*\[(USER|AGENT|ENVIRONMENT)\]:[ \t]*")
THOUGHT_RE = re.compile(r"(?m)^\s*\[THOUGHT\]:[ \t]*")
ACTION_RE = re.compile(r"(?m)^\s*\[ACTION\]:[ \t]*")
TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[a-z0-9_<>]+")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.IGNORECASE)
HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
SPACE_RE = re.compile(r"\s+")

DEFAULT_QUOTAS = {
    "train": {"safe": 100, "unsafe": 100},
    "validation": {"safe": 12, "unsafe": 13},
    "test": {"safe": 13, "unsafe": 12},
}
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


class PipelineError(RuntimeError):
    """Raised when a Step-1 acceptance condition cannot be met."""


@dataclass(frozen=True)
class Fingerprint:
    source: str
    record_id: str
    document_hash: str
    normalized_user: str
    shingles: frozenset[str]
    minhash: tuple[int, ...]
    tool_sequence: tuple[str, ...]


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
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    content = "".join(stable_json(row) + "\n" for row in materialized)
    atomic_write_text(path, content)
    return len(materialized)


def resolve_from_project(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML config without a PyYAML dependency."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"Config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"{path} must use JSON syntax (which is valid YAML): {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Config root must be an object: {path}")
    return value


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = URL_RE.sub(" <url> ", value)
    value = EMAIL_RE.sub(" <email> ", value)
    value = UUID_RE.sub(" <uuid> ", value)
    value = HEX_RE.sub(" <hex> ", value)
    value = NUMBER_RE.sub(" <num> ", value)
    return SPACE_RE.sub(" ", value).strip()


def word_shingles(value: str, width: int = 3) -> frozenset[str]:
    words = WORD_RE.findall(normalize_text(value))
    if not words:
        return frozenset()
    if len(words) < width:
        return frozenset([" ".join(words)])
    return frozenset(
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    )


def minhash_signature(shingles: frozenset[str], size: int = 16) -> tuple[int, ...]:
    if not shingles:
        return tuple([0] * size)
    values: list[int] = []
    for seed in range(size):
        minimum = min(
            int.from_bytes(
                hashlib.blake2b(f"{seed}:{shingle}".encode(), digest_size=8).digest(),
                "big",
            )
            for shingle in shingles
        )
        values.append(minimum)
    return tuple(values)


def lsh_keys(
    signature: tuple[int, ...], bands: int = 4
) -> Iterator[tuple[int, tuple[int, ...]]]:
    if len(signature) % bands:
        raise ValueError("minhash size must be divisible by bands")
    width = len(signature) // bands
    for band in range(bands):
        start = band * width
        yield band, signature[start : start + width]


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def normalize_action(action: str) -> str:
    stripped = action.strip()
    try:
        return stable_json(json.loads(stripped))
    except (json.JSONDecodeError, TypeError):
        return SPACE_RE.sub(" ", stripped)


def action_object(action: Any) -> dict[str, Any] | None:
    if isinstance(action, dict):
        return action
    if not isinstance(action, str):
        return None
    try:
        parsed = json.loads(action.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def action_tool_name(action: Any) -> str | None:
    parsed = action_object(action)
    if not parsed:
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip() or name.lower() == "complete":
        return None
    return name.strip()


def parse_agent_block(body: str) -> dict[str, Any]:
    thought_matches = list(THOUGHT_RE.finditer(body))
    action_matches = list(ACTION_RE.finditer(body))
    if len(thought_matches) > 1 or len(action_matches) != 1:
        raise ValueError(
            f"agent block must contain zero/one THOUGHT and exactly one ACTION; "
            f"found {len(thought_matches)} and {len(action_matches)}"
        )
    action_match = action_matches[0]
    prefix = body[: action_match.start()].strip()
    thought: str | None = None
    if thought_matches:
        thought_match = thought_matches[0]
        if thought_match.start() > action_match.start():
            raise ValueError("THOUGHT appears after ACTION")
        if body[: thought_match.start()].strip():
            raise ValueError("unexpected text before THOUGHT")
        thought = body[thought_match.end() : action_match.start()].strip() or None
    elif prefix:
        raise ValueError("unexpected text before ACTION")
    action = body[action_match.end() :].strip()
    if not action:
        raise ValueError("empty ACTION")
    return {"role": "agent", "thought": thought, "action": action}


def parse_instruction(instruction: str) -> tuple[str, list[dict[str, Any]], str]:
    if instruction.count(BEGIN_MARKER) != 1 or instruction.count(END_MARKER) != 1:
        raise ValueError("trajectory markers must each appear exactly once")
    start = instruction.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = instruction.index(END_MARKER, start)
    raw_trajectory = instruction[start:end].strip()
    if TOOLS_MARKER in raw_trajectory or "Available tools:" in raw_trajectory:
        raise ValueError(
            "BinarySafety row unexpectedly contains an explicit tool schema"
        )
    if PROFILE_MARKER not in raw_trajectory or HISTORY_MARKER not in raw_trajectory:
        raise ValueError("profile/history section marker missing")
    before_history, history = raw_trajectory.split(HISTORY_MARKER, 1)
    if before_history.count(PROFILE_MARKER) != 1:
        raise ValueError("profile marker must appear exactly once")
    profile = before_history.split(PROFILE_MARKER, 1)[1].strip()
    if not profile:
        profile = "You are a helpful assistant."

    matches = list(ROLE_RE.finditer(history))
    if not matches:
        raise ValueError("conversation contains no role events")
    if history[: matches[0].start()].strip():
        raise ValueError("unexpected text before the first role event")

    events: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        body_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(history)
        )
        body = history[match.end() : body_end].strip()
        role = match.group(1).lower()
        if role == "agent":
            events.append(parse_agent_block(body))
        elif role == "user":
            if not body:
                raise ValueError("empty USER event")
            events.append({"role": "user", "content": body})
        else:
            if not body:
                raise ValueError("empty ENVIRONMENT event")
            events.append({"role": "environment", "content": body})
    return profile, events, raw_trajectory


def nested_status_is_error(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == "status" and str(child).lower() in {
                "error",
                "failed",
                "failure",
            }:
                return True
            if nested_status_is_error(child):
                return True
    elif isinstance(value, list):
        return any(nested_status_is_error(item) for item in value)
    return False


def environment_is_error(content: str) -> bool:
    try:
        if nested_status_is_error(json.loads(content)):
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    return bool(
        re.search(
            r"(?:\bstatus\b\s*[:=]\s*['\"]?(?:error|failed|failure)\b|\btool\s+error\b)",
            content,
            re.IGNORECASE,
        )
    )


def event_bin(count: int) -> str:
    if count <= 4:
        return "01-04"
    if count <= 8:
        return "05-08"
    if count <= 16:
        return "09-16"
    return "17+"


def tool_count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 6:
        return "4-6"
    return "7+"


def event_document(events: Sequence[dict[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for event in events:
        role = str(event.get("role", "")).lower()
        if role == "assistant":
            role = "agent"
        elif role in {"tool", "observation"}:
            role = "environment"
        row: dict[str, Any] = {"role": role}
        if role == "agent":
            thought = event.get("thought")
            action = event.get("action")
            row["thought"] = normalize_text(str(thought)) if thought else None
            if isinstance(action, dict):
                row["action"] = stable_json(action)
            elif action is not None:
                row["action"] = normalize_action(str(action))
            else:
                row["action"] = ""
        else:
            content = event.get(
                "content", event.get("result", event.get("observation", ""))
            )
            row["content"] = SPACE_RE.sub(" ", str(content)).strip()
        normalized.append(row)
    return stable_json(normalized)


def features_for(
    events: Sequence[dict[str, Any]], raw_trajectory: str
) -> dict[str, Any]:
    actions = [str(event["action"]) for event in events if event["role"] == "agent"]
    tool_names = [name for action in actions if (name := action_tool_name(action))]
    json_actions = [
        action_object(action) is not None
        for action in actions
        if not action.lstrip().startswith("Complete")
    ]
    if not json_actions:
        action_format = "text"
    elif all(json_actions):
        action_format = "json"
    elif any(json_actions):
        action_format = "mixed"
    else:
        action_format = "text"
    environments = [
        str(event["content"]) for event in events if event["role"] == "environment"
    ]
    last_agent_action = actions[-1].lstrip() if actions else ""
    user_text = "\n".join(
        str(event["content"]) for event in events if event["role"] == "user"
    )
    normalized_user = normalize_text(user_text)
    shingles = word_shingles(user_text)
    return {
        "char_count": len(raw_trajectory),
        "token_count_proxy": len(TOKEN_RE.findall(raw_trajectory)),
        "event_count": len(events),
        "event_bin": event_bin(len(events)),
        "agent_action_count": len(actions),
        "tool_action_count": len(tool_names),
        "tool_count_bin": tool_count_bin(len(tool_names)),
        "first_tool_name": tool_names[0] if tool_names else None,
        "tool_sequence": tool_names,
        "tool_sequence_hash": sha256_text(stable_json(tool_names)),
        "has_environment_error": any(
            environment_is_error(item) for item in environments
        ),
        "has_trailing_complete": last_agent_action.startswith("Complete"),
        "has_json_tool_action": any(json_actions),
        "action_format": action_format,
        "normalized_user": normalized_user,
        "user_shingles": sorted(shingles),
        "user_minhash": list(minhash_signature(shingles)),
        "document_hash": sha256_text(event_document(events)),
    }


def canonical_row(
    *,
    record_id: str,
    row_index: int,
    label: str,
    profile: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "group_id": None,
        "pair_id": None,
        "split": None,
        "source": {
            "dataset": "AgentDoG-BinarySafety",
            "row_id": str(row_index),
            "parent_id": None,
        },
        "profile": profile,
        "tools": None,
        "trajectory": events,
        "target": {
            "type": "judge",
            "coarse": label,
            "risk_source": None,
            "failure_mode": None,
            "real_world_harm": None,
            "cot": None,
        },
    }


def validate_source_record(row: Any) -> str | None:
    if not isinstance(row, dict):
        return "record is not an object"
    if set(row) != {"instruction", "input", "output"}:
        return f"unexpected keys: {sorted(row)}"
    if row["input"] != "":
        return "input must be an empty string"
    if row["output"] not in {"safe", "unsafe"}:
        return "output must be exactly safe or unsafe"
    if not isinstance(row["instruction"], str):
        return "instruction must be a string"
    return None


def load_and_parse_source(
    input_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        raw_rows = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot load {input_path}: {exc}") from exc
    if not isinstance(raw_rows, list):
        raise PipelineError("BinarySafety train.json must contain a JSON array")

    seen: dict[str, int] = {}
    parsed: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    duplicate_map: dict[str, list[int]] = defaultdict(list)
    physical_labels: Counter[str] = Counter()

    for row_index, raw in enumerate(raw_rows):
        reason = validate_source_record(raw)
        if reason:
            quarantine.append({"row_id": str(row_index), "reason": reason})
            continue
        label = str(raw["output"])
        physical_labels[label] += 1
        raw_serialized = stable_json(raw)
        raw_hash = sha256_text(raw_serialized)
        if raw_hash in seen:
            duplicate_map[raw_hash].append(row_index)
            continue
        seen[raw_hash] = row_index
        try:
            profile, events, raw_trajectory = parse_instruction(raw["instruction"])
        except ValueError as exc:
            quarantine.append(
                {"row_id": str(row_index), "raw_hash": raw_hash, "reason": str(exc)}
            )
            continue
        trajectory_hash = sha256_text(stable_json(events))
        action_hash = sha256_text(
            stable_json(
                [
                    normalize_action(str(event["action"]))
                    for event in events
                    if event["role"] == "agent"
                ]
            )
        )
        record_id = f"binary_{label}_{trajectory_hash[:16]}"
        parsed.append(
            {
                "canonical": canonical_row(
                    record_id=record_id,
                    row_index=row_index,
                    label=label,
                    profile=profile,
                    events=events,
                ),
                "raw_hash": raw_hash,
                "trajectory_hash": trajectory_hash,
                "action_hash": action_hash,
                "features": features_for(events, raw_trajectory),
            }
        )

    ids = [item["canonical"]["id"] for item in parsed]
    if len(ids) != len(set(ids)):
        collisions = [value for value, count in Counter(ids).items() if count > 1]
        raise PipelineError(f"Stable ID collision: {collisions[:5]}")

    unique_labels = Counter(item["canonical"]["target"]["coarse"] for item in parsed)
    first_half_equals_second = False
    if len(raw_rows) % 2 == 0:
        half = len(raw_rows) // 2
        first_half_equals_second = all(
            stable_json(raw_rows[index]) == stable_json(raw_rows[index + half])
            for index in range(half)
        )
    report = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "physical_records": len(raw_rows),
        "physical_labels": dict(sorted(physical_labels.items())),
        "unique_records": len(parsed),
        "unique_labels": dict(sorted(unique_labels.items())),
        "duplicate_records": sum(len(value) for value in duplicate_map.values()),
        "duplicate_groups": len(duplicate_map),
        "first_half_equals_second_half": first_half_equals_second,
        "quarantine_records": len(quarantine),
        "duplicate_examples": [
            {
                "raw_hash": raw_hash,
                "kept_row_id": str(seen[raw_hash]),
                "duplicate_row_ids": [str(value) for value in indices[:10]],
            }
            for raw_hash, indices in sorted(duplicate_map.items())[:20]
        ],
    }
    return parsed, quarantine, report


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


def assign_length_quartiles(items: Sequence[dict[str, Any]]) -> None:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
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


def stratified_sample(
    eligible: list[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    selected_groups: set[str] = set()
    selected_ids: set[str] = set()
    feature_counts = {"safe": Counter(), "unsafe": Counter()}
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    reasons: dict[str, str] = {}

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


def sampling_report(
    splits: dict[str, list[dict[str, Any]]],
    eligible: Sequence[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
) -> dict[str, Any]:
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


def write_features(
    path: Path, items: Sequence[dict[str, Any]], output_format: str
) -> Path:
    rows: list[dict[str, Any]] = []
    for item in items:
        features = dict(item["features"])
        features["id"] = item["canonical"]["id"]
        features["group_id"] = item["canonical"]["group_id"]
        features["label"] = item["canonical"]["target"]["coarse"]
        features["tool_sequence_json"] = stable_json(features.pop("tool_sequence"))
        features["user_shingles_json"] = stable_json(features.pop("user_shingles"))
        features["user_minhash_json"] = stable_json(features.pop("user_minhash"))
        rows.append(features)
    if output_format == "jsonl":
        jsonl_path = path.with_suffix(".jsonl")
        write_jsonl(jsonl_path, rows)
        if path.exists():
            path.unlink()
        return jsonl_path
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PipelineError(
            "pyarrow is required for features.parquet. Install the verl project "
            "dependencies, or use --features-format jsonl for a local smoke run."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)
    jsonl_path = path.with_suffix(".jsonl")
    if jsonl_path.exists():
        jsonl_path.unlink()
    return path


def validate_acceptance(
    parsed: Sequence[dict[str, Any]],
    quarantine: Sequence[dict[str, Any]],
    dedup: dict[str, Any],
    sampling: dict[str, Any],
    leakage: dict[str, Any],
    strict_counts: bool,
) -> None:
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


def create_directories(output_root: Path) -> dict[str, Path]:
    paths = {
        "source": output_root / "source",
        "intermediate": output_root / "intermediate" / "binary",
        "canonical": output_root / "canonical" / "components",
        "metadata": output_root / "metadata",
        "reports": output_root / "reports",
        "final": output_root / "final",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_manifest(
    output_root: Path, files: Sequence[Path], args: argparse.Namespace
) -> dict[str, Any]:
    entries = []
    for path in sorted(files):
        entries.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "stage": "agentdog-1.5/coarse-grained/sft/step1-binary",
        "script_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "seed": args.seed,
        "input": str(args.input),
        "benchmark_root": str(args.benchmark_root),
        "files": entries,
    }


def run(args: argparse.Namespace) -> None:
    parsed, quarantine, dedup = load_and_parse_source(args.input)
    grouping = assign_groups(parsed, args.group_near_threshold)

    if args.skip_benchmark_check:
        excluded: set[str] = set()
        benchmark_report = {
            "skipped": True,
            "benchmark_root": str(args.benchmark_root),
            "matched_source_records": 0,
            "matches": [],
        }
    else:
        excluded, benchmark_report = benchmark_overlap(
            parsed, args.benchmark_root, args.benchmark_near_threshold
        )
    for item in parsed:
        item["benchmark_overlap"] = item["canonical"]["id"] in excluded

    eligible = [item for item in parsed if not item["benchmark_overlap"]]
    assign_length_quartiles(eligible)
    for item in parsed:
        item["features"].setdefault("length_quartile", None)
    splits, reasons = stratified_sample(eligible, args.quotas, args.seed)
    sampling = sampling_report(splits, eligible, args.quotas)
    leakage = leakage_report(parsed, splits, benchmark_report)
    leakage["grouping"] = grouping

    validate_acceptance(
        parsed,
        quarantine,
        dedup,
        sampling,
        leakage,
        strict_counts=not args.allow_nonstandard_source_counts,
    )

    paths = create_directories(args.output_root)
    output_files: list[Path] = []
    source_manifest = {
        "dataset": "AgentDoG-BinarySafety",
        "source_file": str(args.input),
        "source_sha256": dedup["input_sha256"],
        "physical_records": dedup["physical_records"],
        "unique_records": dedup["unique_records"],
        "script_version": SCRIPT_VERSION,
        "seed": args.seed,
        "group_near_threshold": args.group_near_threshold,
        "benchmark_near_threshold": args.benchmark_near_threshold,
    }
    source_manifest_path = paths["source"] / "source_manifest.json"
    write_json(source_manifest_path, source_manifest)
    output_files.append(source_manifest_path)

    unique_path = paths["intermediate"] / "unique.jsonl"
    write_jsonl(unique_path, (item["canonical"] for item in parsed))
    output_files.append(unique_path)
    quarantine_path = paths["intermediate"] / "quarantine.jsonl"
    write_jsonl(quarantine_path, quarantine)
    output_files.append(quarantine_path)
    groups_path = paths["intermediate"] / "groups.jsonl"
    write_jsonl(
        groups_path,
        (
            {
                "id": item["canonical"]["id"],
                "group_id": item["canonical"]["group_id"],
            }
            for item in parsed
        ),
    )
    output_files.append(groups_path)
    feature_path = write_features(
        paths["intermediate"] / "features.parquet", parsed, args.features_format
    )
    output_files.append(feature_path)

    split_names = {"train": "train", "validation": "validation", "test": "test"}
    for split, file_name in split_names.items():
        path = paths["canonical"] / f"binary_{file_name}.jsonl"
        ordered = sorted(splits[split], key=lambda item: item["canonical"]["id"])
        write_jsonl(path, (item["canonical"] for item in ordered))
        output_files.append(path)

    metadata_path = paths["metadata"] / "binary.jsonl"
    write_jsonl(
        metadata_path,
        (
            {
                "id": item["canonical"]["id"],
                "raw_hash": item["raw_hash"],
                "trajectory_hash": item["trajectory_hash"],
                "action_hash": item["action_hash"],
                "group_id": item["canonical"]["group_id"],
                "features": item["features"],
                "benchmark_overlap": item["benchmark_overlap"],
                "selected_split": item["canonical"]["split"],
                "selection_reason": reasons.get(item["canonical"]["id"]),
            }
            for item in parsed
        ),
    )
    output_files.append(metadata_path)

    reports = {
        "binary_dedup.json": dedup,
        "binary_grouping.json": grouping,
        "binary_benchmark_overlap.json": benchmark_report,
        "binary_sampling.json": sampling,
        "binary_leakage_check.json": leakage,
    }
    for name, report in reports.items():
        path = paths["reports"] / name
        write_json(path, report)
        output_files.append(path)

    manifest = build_manifest(args.output_root, output_files, args)
    write_json(args.output_root / "manifest.json", manifest)

    print(f"Step 1 complete: {args.output_root}")
    print(
        f"source={dedup['physical_records']} physical -> {dedup['unique_records']} unique; "
        f"benchmark_excluded={benchmark_report['matched_source_records']}; quarantine={len(quarantine)}"
    )
    for split in ("train", "validation", "test"):
        labels = Counter(
            item["canonical"]["target"]["coarse"] for item in splits[split]
        )
        print(
            f"{split}: total={len(splits[split])} safe={labels['safe']} unsafe={labels['unsafe']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = bootstrap.parse_known_args(argv)
    config_path = config_args.config.resolve()
    config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Build AgentDoG 1.5 coarse SFT Step-1 BinarySafety data",
        parents=[bootstrap],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 42)))
    parser.add_argument(
        "--group-near-threshold",
        type=float,
        default=float(config.get("group_near_threshold", 0.88)),
    )
    parser.add_argument(
        "--benchmark-near-threshold",
        type=float,
        default=float(config.get("benchmark_near_threshold", 0.92)),
    )
    parser.add_argument(
        "--features-format",
        choices=("parquet", "jsonl"),
        default=str(config.get("features_format", "parquet")),
    )
    parser.add_argument(
        "--skip-benchmark-check",
        action="store_true",
        help="Skip ATBench/R-Judge overlap checking (not accepted for final data).",
    )
    parser.add_argument(
        "--allow-nonstandard-source-counts",
        action="store_true",
        help="Allow sources other than the verified 4000 -> 2000 public file.",
    )
    args = parser.parse_args(argv)
    args.config = config_path
    args.input = resolve_from_project(args.input or config["input"])
    args.output_root = resolve_from_project(args.output_root or config["output_root"])
    args.benchmark_root = resolve_from_project(
        args.benchmark_root or config["benchmark_root"]
    )
    args.quotas = config.get("quotas", DEFAULT_QUOTAS)
    if args.quotas != DEFAULT_QUOTAS:
        raise PipelineError(
            f"Step-1 quotas must remain {DEFAULT_QUOTAS}; received {args.quotas}"
        )
    if not 0.0 <= args.group_near_threshold <= 1.0:
        parser.error("--group-near-threshold must be between 0 and 1")
    if not 0.0 <= args.benchmark_near_threshold <= 1.0:
        parser.error("--benchmark-near-threshold must be between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (PipelineError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
