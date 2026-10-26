"""BinarySafety 原始数据解析、规范化、去重和特征提取。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "1.0.0"
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

# 这部分只做一件事：把公开BinarySafety转成稳定的canonical记录。
# 哈希和文本规范化函数同时供近重复模块复用，因此保留为独立函数。
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


# 轨迹解析和结构特征

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

    # 先按完整原始记录哈希去重，再解析轨迹，保证重复副本不会影响抽样。
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
