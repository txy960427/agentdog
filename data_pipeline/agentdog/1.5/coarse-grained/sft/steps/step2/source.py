"""解析 v1.0 FineGrainedTaxonomy，并转成粗粒度 unsafe canonical 数据。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..step1.source import (
    BEGIN_MARKER,
    END_MARKER,
    HISTORY_MARKER,
    PROFILE_MARKER,
    ROLE_RE,
    PipelineError,
    action_object,
    features_for,
    normalize_action,
    parse_agent_block,
    sha256_file,
    sha256_text,
    stable_json,
    tool_count_bin,
)

SCRIPT_VERSION = "1.0.0"
TOOLS_PREFIX = "Available tools:"

RISK_SOURCES = {
    "Corrupted Tool Feedback",
    "Direct Prompt Injection",
    "Indirect Prompt Injection",
    "Inherent Agent/LLM Failures",
    "Malicious Tool Execution",
    "Malicious User Instruction or Jailbreak",
    "Tool Description Injection",
    "Unreliable or Mis-information",
}
FAILURE_MODES = {
    "Choosing Malicious Tool",
    "Failure to Validate Tool Outputs",
    "Flawed Planning or Reasoning",
    "Generation of Harmful/Offensive Content",
    "Generation of Malicious Executables",
    "Incorrect Tool Parameters",
    "Inefficient or Wasteful Execution",
    "Insecure Execution or Interaction",
    "Instruction for Harmful/Illegal Activity",
    "Procedural Deviation or Inaction",
    "Provide Inaccurate, Misleading, or Unverified Information",
    "Tool Misuse in Specific Context",
    "Unauthorized Information Disclosure",
    "Unconfirmed or Over-privileged Action",
}
REAL_WORLD_HARMS = {
    "Fairness, Equity, and Allocative Harm",
    "Financial & Economic Harm",
    "Functional & Opportunity Harm",
    "Info-ecosystem & Societal Harm",
    "Physical & Health Harm",
    "Privacy & Confidentiality Harm",
    "Psychological & Emotional Harm",
    "Public Service & Resource Harm",
    "Reputational & Interpersonal Harm",
    "Security & System Integrity Harm",
}


def parse_labels(value: str) -> tuple[str, str, str]:
    lines = [line.split(":", 1) for line in value.splitlines() if ":" in line]
    labels = {key.strip(): item.strip() for key, item in lines}
    if set(labels) != {"Risk Source", "Failure Mode", "Real World Harm"}:
        raise ValueError("output must contain exactly RS, FM and RWH")
    risk, failure, harm = (
        labels["Risk Source"],
        labels["Failure Mode"],
        labels["Real World Harm"],
    )
    if risk not in RISK_SOURCES or failure not in FAILURE_MODES or harm not in REAL_WORLD_HARMS:
        raise ValueError(f"unknown taxonomy label: {risk} / {failure} / {harm}")
    return risk, failure, harm


def parse_history(history: str) -> list[dict[str, Any]]:
    matches = list(ROLE_RE.finditer(history))
    if not matches or history[: matches[0].start()].strip():
        raise ValueError("conversation has invalid role markers")
    events: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(history)
        body = history[match.end() : end].strip()
        role = match.group(1).lower()
        if not body:
            raise ValueError(f"empty {role} event")
        if role == "agent":
            events.append(parse_agent_block(body))
        else:
            events.append({"role": role, "content": body})
    return events


def clean_tools(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("available tools must be a non-empty JSON array")
    cleaned: list[dict[str, Any]] = []
    names: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    original_descriptions = 0
    for tool in value:
        if not isinstance(tool, dict):
            raise ValueError("tool must be a JSON object")
        name, description, parameters = (
            tool.get("name"),
            tool.get("description"),
            tool.get("parameters"),
        )
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid tool name: {name!r}")
        if not isinstance(description, str) or not isinstance(parameters, dict):
            raise ValueError(f"invalid description/parameters for tool {name}")
        names[name] += 1
        cleaned.append({"name": name, "description": description, "parameters": parameters})
        if "_source" in tool:
            sources[str(tool["_source"])] += 1
        original_descriptions += "_original_description" in tool
    audit = {
        "available_tool_count": len(cleaned),
        "source_counts": dict(sorted(sources.items())),
        "original_description_count": original_descriptions,
        "duplicate_tool_name_count": sum(count - 1 for count in names.values()),
        "raw_tools_hash": sha256_text(stable_json(value)),
    }
    return cleaned, audit


def value_matches_schema(value: Any, expected: Any) -> bool:
    kind = str(expected).lower()
    expected_types = {
        "string": str,
        "credentials": str,
        "datepicker": str,
        "select": str,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    if kind in {"number", "integer"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return kind not in expected_types or isinstance(value, expected_types[kind])


def schema_warning_count(arguments: dict[str, Any], schemas: list[dict[str, Any]]) -> int:
    scores: list[int] = []
    for schema in schemas:
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = sum(key not in arguments for key in required) if isinstance(required, list) else 0
        wrong_type = (
            sum(
                not value_matches_schema(value, properties[key].get("type"))
                for key, value in arguments.items()
                if key in properties and isinstance(properties[key], dict)
            )
            if isinstance(properties, dict)
            else 0
        )
        scores.append(missing + wrong_type)
    return min(scores) if scores else 0


def validate_calls(events: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[int, int]:
    schemas: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for tool in tools:
        schemas[tool["name"]].append(tool["parameters"])
    calls = 0
    schema_warnings = 0
    for index, event in enumerate(events):
        if event["role"] != "agent":
            continue
        action = str(event["action"]).strip()
        if action.startswith("Complete"):
            continue
        parsed = action_object(action)
        if not parsed or not isinstance(parsed.get("name"), str):
            raise ValueError(f"event {index} has a non-JSON tool action")
        name, arguments = parsed["name"], parsed.get("arguments")
        if name not in schemas or not isinstance(arguments, dict):
            raise ValueError(f"event {index} calls an unavailable/invalid tool: {name}")
        if index + 1 >= len(events) or events[index + 1]["role"] != "environment":
            raise ValueError(f"event {index} tool call has no following environment event")
        schema_warnings += schema_warning_count(arguments, schemas[name])
        calls += 1
    environments = sum(event["role"] == "environment" for event in events)
    if calls != environments:
        raise ValueError(f"tool/environment count mismatch: {calls} != {environments}")
    return calls, schema_warnings


def parse_instruction(value: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    if value.count(BEGIN_MARKER) != 1 or value.count(END_MARKER) != 1:
        raise ValueError("trajectory markers must each appear exactly once")
    start = value.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = value.index(END_MARKER, start)
    raw_trajectory = value[start:end].strip()
    if raw_trajectory.count(PROFILE_MARKER) != 1 or raw_trajectory.count(HISTORY_MARKER) != 1:
        raise ValueError("profile/history marker missing or repeated")
    before_history, history = raw_trajectory.split(HISTORY_MARKER, 1)
    profile_tools = before_history.split(PROFILE_MARKER, 1)[1].strip()
    if profile_tools.count(TOOLS_PREFIX) != 1:
        raise ValueError("Available tools marker missing or repeated")
    profile, raw_tools = profile_tools.split(TOOLS_PREFIX, 1)
    profile = profile.strip() or "You are a helpful assistant."
    tools, audit = clean_tools(json.loads(raw_tools.strip()))
    events = parse_history(history)
    audit["tool_action_count"], audit["schema_warning_count"] = validate_calls(events, tools)
    return profile, tools, events, raw_trajectory, audit


def canonical_row(
    row_index: int,
    record_id: str,
    profile: str,
    tools: list[dict[str, Any]],
    events: list[dict[str, Any]],
    labels: tuple[str, str, str],
) -> dict[str, Any]:
    risk, failure, harm = labels
    return {
        "id": record_id,
        "group_id": None,
        "pair_id": None,
        "split": None,
        "source": {"dataset": "AgentDoG-FineGrainedTaxonomy", "row_id": str(row_index), "parent_id": None},
        "profile": profile,
        "tools": tools,
        "trajectory": events,
        "target": {
            "type": "judge",
            "coarse": "unsafe",
            "risk_source": risk,
            "failure_mode": failure,
            "real_world_harm": harm,
            "cot": None,
        },
    }


def load_and_parse_source(input_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        raw_rows = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot load {input_path}: {exc}") from exc
    if not isinstance(raw_rows, list):
        raise PipelineError("FineGrainedTaxonomy train.json must contain a JSON array")

    parsed: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    duplicates: defaultdict[str, list[int]] = defaultdict(list)
    for row_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or set(raw) != {"instruction", "input", "output"}:
            quarantine.append({"row_id": str(row_index), "reason": "invalid record fields"})
            continue
        if raw["input"] != "" or not isinstance(raw["instruction"], str) or not isinstance(raw["output"], str):
            quarantine.append({"row_id": str(row_index), "reason": "invalid input/instruction/output"})
            continue
        raw_hash = sha256_text(stable_json(raw))
        if raw_hash in seen:
            duplicates[raw_hash].append(row_index)
            continue
        seen[raw_hash] = row_index
        try:
            labels = parse_labels(raw["output"])
            profile, tools, events, raw_trajectory, audit = parse_instruction(raw["instruction"])
        except (ValueError, json.JSONDecodeError) as exc:
            quarantine.append({"row_id": str(row_index), "raw_hash": raw_hash, "reason": str(exc)})
            continue
        trajectory_hash = sha256_text(stable_json(events))
        features = features_for(events, raw_trajectory)
        features.update(
            {
                "available_tool_count": len(tools),
                "available_tool_count_bin": tool_count_bin(len(tools)),
                "fm_rwh": f"{labels[1]} || {labels[2]}",
                "has_schema_warning": audit["schema_warning_count"] > 0,
                "has_duplicate_tool_name": audit["duplicate_tool_name_count"] > 0,
            }
        )
        features["has_data_irregularity"] = any(
            (
                features["has_environment_error"],
                features["has_schema_warning"],
                features["has_duplicate_tool_name"],
            )
        )
        parsed.append(
            {
                "canonical": canonical_row(
                    row_index,
                    f"tool_unsafe_{trajectory_hash[:16]}",
                    profile,
                    tools,
                    events,
                    labels,
                ),
                "raw_hash": raw_hash,
                "trajectory_hash": trajectory_hash,
                "action_hash": sha256_text(
                    stable_json(
                        [normalize_action(str(event["action"])) for event in events if event["role"] == "agent"]
                    )
                ),
                "tool_audit": audit,
                "features": features,
            }
        )

    ids = [item["canonical"]["id"] for item in parsed]
    if len(ids) != len(set(ids)):
        raise PipelineError("Stable ID collision or duplicate trajectory in FineGrainedTaxonomy")
    taxonomy = {
        "risk_source": Counter(item["canonical"]["target"]["risk_source"] for item in parsed),
        "failure_mode": Counter(item["canonical"]["target"]["failure_mode"] for item in parsed),
        "real_world_harm": Counter(item["canonical"]["target"]["real_world_harm"] for item in parsed),
        "triples": Counter(
            (
                item["canonical"]["target"]["risk_source"],
                item["canonical"]["target"]["failure_mode"],
                item["canonical"]["target"]["real_world_harm"],
            )
            for item in parsed
        ),
    }
    report = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "physical_records": len(raw_rows),
        "unique_records": len(parsed),
        "duplicate_records": sum(map(len, duplicates.values())),
        "quarantine_records": len(quarantine),
        "taxonomy_class_counts": {key: len(value) for key, value in taxonomy.items()},
        "risk_source_counts": dict(sorted(taxonomy["risk_source"].items())),
        "rows_with_original_description": sum(
            item["tool_audit"]["original_description_count"] > 0 for item in parsed
        ),
        "data_irregularity_counts": {
            field: sum(item["features"][field] for item in parsed)
            for field in (
                "has_environment_error",
                "has_schema_warning",
                "has_duplicate_tool_name",
                "has_data_irregularity",
            )
        },
    }
    return parsed, quarantine, report
