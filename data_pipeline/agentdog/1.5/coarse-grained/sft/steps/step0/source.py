"""读取两个v1.0数据源，生成Step 0所需的最小候选索引。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BEGIN = "<BEGIN TRAJECTORY>"
END = "<END TRAJECTORY>"
PROFILE = "=== Agent Profile ==="
HISTORY = "=== Conversation History ==="
TOOLS_PREFIX = "Available tools:"
ROLE_RE = re.compile(r"(?m)^\s*\[(USER|AGENT|ENVIRONMENT)\]:[ \t]*")
THOUGHT_RE = re.compile(r"(?m)^\s*\[THOUGHT\]:[ \t]*")
ACTION_RE = re.compile(r"(?m)^\s*\[ACTION\]:[ \t]*")
TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[a-z0-9_<>]+")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+", re.I)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")

RISK_SOURCES = (
    "Corrupted Tool Feedback",
    "Direct Prompt Injection",
    "Indirect Prompt Injection",
    "Inherent Agent/LLM Failures",
    "Malicious Tool Execution",
    "Malicious User Instruction or Jailbreak",
    "Tool Description Injection",
    "Unreliable or Mis-information",
)


class PipelineError(RuntimeError):
    """输入或验收条件不满足。"""


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def shingles(value: str, width: int = 3) -> frozenset[str]:
    words = WORD_RE.findall(normalize_text(value))
    if not words:
        return frozenset()
    if len(words) < width:
        return frozenset({" ".join(words)})
    return frozenset(
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    )


def minhash(values: frozenset[str], size: int = 16) -> tuple[int, ...]:
    if not values:
        return (0,) * size
    return tuple(
        min(
            int.from_bytes(
                hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(),
                "big",
            )
            for value in values
        )
        for seed in range(size)
    )


def parse_agent(body: str) -> dict[str, Any]:
    thoughts = list(THOUGHT_RE.finditer(body))
    actions = list(ACTION_RE.finditer(body))
    if len(thoughts) > 1 or len(actions) != 1:
        raise ValueError("agent block must contain at most one THOUGHT and one ACTION")
    action_match = actions[0]
    thought = None
    if thoughts:
        thought_match = thoughts[0]
        if thought_match.start() > action_match.start() or body[: thought_match.start()].strip():
            raise ValueError("invalid THOUGHT/ACTION order")
        thought = body[thought_match.end() : action_match.start()].strip() or None
    elif body[: action_match.start()].strip():
        raise ValueError("unexpected text before ACTION")
    action = body[action_match.end() :].strip()
    if not action:
        raise ValueError("empty ACTION")
    return {"role": "agent", "thought": thought, "action": action}


def parse_events(history: str) -> list[dict[str, Any]]:
    matches = list(ROLE_RE.finditer(history))
    if not matches or history[: matches[0].start()].strip():
        raise ValueError("invalid conversation role markers")
    events: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(history)
        body = history[match.end() : end].strip()
        if not body:
            raise ValueError(f"empty {match.group(1)} event")
        role = match.group(1).lower()
        events.append(parse_agent(body) if role == "agent" else {"role": role, "content": body})
    return events


def trajectory_text(instruction: str) -> str:
    if instruction.count(BEGIN) != 1 or instruction.count(END) != 1:
        raise ValueError("trajectory markers must appear exactly once")
    start = instruction.index(BEGIN) + len(BEGIN)
    return instruction[start : instruction.index(END, start)].strip()


def action_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def tool_name(action: Any) -> str | None:
    parsed = action_object(action)
    name = parsed.get("name") if parsed else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def is_environment_error(content: str) -> bool:
    try:
        pending = [json.loads(content)]
    except json.JSONDecodeError:
        pending = []
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() == "status" and str(child).lower() in {"error", "failed", "failure"}:
                    return True
                pending.append(child)
        elif isinstance(value, list):
            pending.extend(value)
    return bool(
        re.search(
            r"(?:\bstatus\b\s*[:=]\s*['\"]?(?:error|failed|failure)\b|\btool\s+error\b)",
            content,
            re.I,
        )
    )


def event_bin(count: int) -> str:
    return "01-04" if count <= 4 else "05-08" if count <= 8 else "09-16" if count <= 16 else "17+"


def count_bin(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4-6" if count <= 6 else "7+"


def event_document(events: list[dict[str, Any]]) -> str:
    normalized = []
    for event in events:
        if event["role"] == "agent":
            normalized.append(
                {
                    "role": "agent",
                    "thought": normalize_text(str(event.get("thought") or "")) or None,
                    "action": SPACE_RE.sub(" ", str(event["action"])).strip(),
                }
            )
        else:
            normalized.append(
                {"role": event["role"], "content": SPACE_RE.sub(" ", str(event["content"])).strip()}
            )
    return stable_json(normalized)


def common_features(events: list[dict[str, Any]], raw_trajectory: str) -> tuple[dict[str, Any], dict[str, Any]]:
    actions = [event["action"] for event in events if event["role"] == "agent"]
    names = [name for action in actions if (name := tool_name(action))]
    json_actions = [action_object(action) is not None for action in actions if not str(action).startswith("Complete")]
    user = "\n".join(str(event["content"]) for event in events if event["role"] == "user")
    user_normalized = normalize_text(user)
    if not user_normalized:
        raise ValueError("trajectory has no user text")
    user_shingles = shingles(user)
    features = {
        "token_count_proxy": len(TOKEN_RE.findall(raw_trajectory)),
        "event_count": len(events),
        "event_bin": event_bin(len(events)),
        "tool_action_count": len(names),
        "tool_action_count_bin": count_bin(len(names)),
        "has_environment_error": any(
            is_environment_error(str(event["content"]))
            for event in events
            if event["role"] == "environment"
        ),
        "has_trailing_complete": bool(actions and str(actions[-1]).lstrip().startswith("Complete")),
        "action_format": (
            "text" if not json_actions or not any(json_actions) else "json" if all(json_actions) else "mixed"
        ),
    }
    fingerprint = {
        "document_hash": sha256_text(event_document(events)),
        "normalized_user": user_normalized,
        "shingles": user_shingles,
        "minhash": minhash(user_shingles),
        "tool_sequence": tuple(names),
    }
    return features, fingerprint


def read_rows(path: Path) -> list[Any]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise PipelineError(f"{path} must contain a JSON array")
    return rows


def base_candidate(
    *, stage: str, row_index: int, raw: dict[str, Any], events: list[dict[str, Any]], raw_trajectory: str
) -> dict[str, Any]:
    raw_hash = sha256_text(stable_json(raw))
    trajectory_hash = sha256_text(stable_json(events))
    features, fingerprint = common_features(events, raw_trajectory)
    return {
        "stage": stage,
        "source_row_id": str(row_index),
        "raw_hash": raw_hash,
        "trajectory_hash": trajectory_hash,
        "features": features,
        **fingerprint,
    }


def load_binary(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    rows = read_rows(path)
    candidates, quarantine = [], []
    seen: set[str] = set()
    duplicates = 0
    for index, raw in enumerate(rows):
        try:
            if not isinstance(raw, dict) or set(raw) != {"instruction", "input", "output"}:
                raise ValueError("invalid record fields")
            if raw["input"] != "" or raw["output"] not in {"safe", "unsafe"}:
                raise ValueError("invalid input or coarse label")
            raw_hash = sha256_text(stable_json(raw))
            if raw_hash in seen:
                duplicates += 1
                continue
            seen.add(raw_hash)
            trajectory = trajectory_text(raw["instruction"])
            if PROFILE not in trajectory or HISTORY not in trajectory or TOOLS_PREFIX in trajectory:
                raise ValueError("invalid BinarySafety trajectory sections")
            before, history = trajectory.split(HISTORY, 1)
            profile = before.split(PROFILE, 1)[1].strip()
            if not profile:
                raise ValueError("empty profile")
            events = parse_events(history)
            item = base_candidate(
                stage="step1", row_index=index, raw=raw, events=events, raw_trajectory=trajectory
            )
            item.update(
                {
                    "id": f"binary_{raw['output']}_{item['trajectory_hash'][:16]}",
                    "coarse": raw["output"],
                    "risk_source": None,
                    "failure_mode": None,
                    "real_world_harm": None,
                }
            )
            candidates.append(item)
        except (KeyError, TypeError, ValueError) as exc:
            quarantine.append({"row_id": str(index), "reason": str(exc)})
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "physical_records": len(rows),
        "unique_records": len(candidates),
        "duplicates": duplicates,
        "labels": dict(sorted(Counter(item["coarse"] for item in candidates).items())),
        "quarantine": len(quarantine),
    }
    return candidates, report, quarantine


def parse_labels(output: str) -> tuple[str, str, str]:
    labels = {
        key.strip(): value.strip()
        for line in output.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    if set(labels) != {"Risk Source", "Failure Mode", "Real World Harm"}:
        raise ValueError("invalid fine-grained labels")
    risk = labels["Risk Source"]
    if risk not in RISK_SOURCES:
        raise ValueError(f"unknown Risk Source: {risk}")
    return risk, labels["Failure Mode"], labels["Real World Harm"]


def schema_warning(arguments: dict[str, Any], schemas: list[dict[str, Any]]) -> bool:
    expected_types = {
        "string": str,
        "credentials": str,
        "datepicker": str,
        "select": str,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for schema in schemas:
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = isinstance(required, list) and any(key not in arguments for key in required)
        wrong = False
        if isinstance(properties, dict):
            expected = {"string": str, "boolean": bool, "array": list, "object": dict}
            for key, value in arguments.items():
                spec = properties.get(key)
                kind = str(spec.get("type")).lower() if isinstance(spec, dict) else ""
                if kind in {"number", "integer"}:
                    wrong |= not isinstance(value, (int, float)) or isinstance(value, bool)
                elif kind in expected_types:
                    wrong |= not isinstance(value, expected_types[kind])
        if not missing and not wrong:
            return False
    return True


def validate_tool_calls(events: list[dict[str, Any]], tools: list[dict[str, Any]]) -> bool:
    schemas: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for tool in tools:
        schemas[tool["name"]].append(tool["parameters"])
    warning = False
    calls = 0
    for index, event in enumerate(events):
        if event["role"] != "agent" or str(event["action"]).lstrip().startswith("Complete"):
            continue
        action = action_object(event["action"])
        if not action or action.get("name") not in schemas or not isinstance(action.get("arguments"), dict):
            raise ValueError(f"invalid or unavailable tool call at event {index}")
        if index + 1 >= len(events) or events[index + 1]["role"] != "environment":
            raise ValueError(f"tool call without environment result at event {index}")
        warning |= schema_warning(action["arguments"], schemas[action["name"]])
        calls += 1
    if calls != sum(event["role"] == "environment" for event in events):
        raise ValueError("tool/environment count mismatch")
    return warning


def load_fine(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    rows = read_rows(path)
    candidates, quarantine = [], []
    seen: set[str] = set()
    duplicates = 0
    for index, raw in enumerate(rows):
        try:
            if not isinstance(raw, dict) or set(raw) != {"instruction", "input", "output"}:
                raise ValueError("invalid record fields")
            if raw["input"] != "" or not isinstance(raw["instruction"], str):
                raise ValueError("invalid input or instruction")
            raw_hash = sha256_text(stable_json(raw))
            if raw_hash in seen:
                duplicates += 1
                continue
            seen.add(raw_hash)
            risk, failure, harm = parse_labels(raw["output"])
            trajectory = trajectory_text(raw["instruction"])
            if PROFILE not in trajectory or HISTORY not in trajectory:
                raise ValueError("invalid FineGrained trajectory sections")
            before, history = trajectory.split(HISTORY, 1)
            profile_tools = before.split(PROFILE, 1)[1].strip()
            if profile_tools.count(TOOLS_PREFIX) != 1:
                raise ValueError("missing Available tools")
            _, tools_text = profile_tools.split(TOOLS_PREFIX, 1)
            tools = json.loads(tools_text.strip())
            if not isinstance(tools, list) or not tools:
                raise ValueError("tools must be a non-empty list")
            names: Counter[str] = Counter()
            clean_tools = []
            for tool in tools:
                if not isinstance(tool, dict):
                    raise ValueError("tool must be an object")
                name, description, parameters = tool.get("name"), tool.get("description"), tool.get("parameters")
                if not isinstance(name, str) or not isinstance(description, str) or not isinstance(parameters, dict):
                    raise ValueError("invalid tool schema")
                names[name] += 1
                clean_tools.append({"name": name, "description": description, "parameters": parameters})
            events = parse_events(history)
            warning = validate_tool_calls(events, clean_tools)
            item = base_candidate(
                stage="step2", row_index=index, raw=raw, events=events, raw_trajectory=trajectory
            )
            item["features"].update(
                {
                    "available_tool_count": len(clean_tools),
                    "available_tool_count_bin": count_bin(len(clean_tools)),
                    "has_schema_warning": warning,
                    "has_duplicate_tool_name": any(count > 1 for count in names.values()),
                }
            )
            item["features"]["has_risk_structure"] = any(
                item["features"][key]
                for key in (
                    "has_environment_error",
                    "has_schema_warning",
                    "has_duplicate_tool_name",
                )
            )
            item.update(
                {
                    "id": f"tool_unsafe_{item['trajectory_hash'][:16]}",
                    "coarse": "unsafe",
                    "risk_source": risk,
                    "failure_mode": failure,
                    "real_world_harm": harm,
                }
            )
            candidates.append(item)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            quarantine.append({"row_id": str(index), "reason": str(exc)})
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "physical_records": len(rows),
        "unique_records": len(candidates),
        "duplicates": duplicates,
        "risk_sources": dict(sorted(Counter(item["risk_source"] for item in candidates).items())),
        "failure_mode_classes": len({item["failure_mode"] for item in candidates}),
        "real_world_harm_classes": len({item["real_world_harm"] for item in candidates}),
        "triples": len(
            {(item["risk_source"], item["failure_mode"], item["real_world_harm"]) for item in candidates}
        ),
        "quarantine": len(quarantine),
    }
    return candidates, report, quarantine


def assign_length_quartiles(candidates: list[dict[str, Any]]) -> None:
    buckets: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        label = item["coarse"] if item["stage"] == "step1" else item["risk_source"]
        buckets[(item["stage"], label)].append(item)
    for rows in buckets.values():
        rows.sort(key=lambda item: (item["features"]["token_count_proxy"], item["id"]))
        for rank, item in enumerate(rows):
            item["features"]["length_quartile"] = min(3, rank * 4 // len(rows))
