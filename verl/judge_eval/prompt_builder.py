#!/usr/bin/env python3
"""Build the two placeholders used by the AgentDoG coarse-grained prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptFields:
    trajectory: str
    tool_list_text: str


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def format_conversation(row: dict[str, Any]) -> str:
    """Serialize ATBench/R-Judge records in the released AgentDoG format."""
    parts: list[str] = []
    if row.get("profile"):
        parts.append(f"=== Agent Profile ===\n{row['profile']}\n")
    parts.append("=== Conversation History ===")
    for round_data in row["contents"]:
        for turn in round_data:
            role = turn["role"]
            if role == "user" and turn.get("content"):
                parts.append(f"\n[USER]: {turn['content']}")
            elif role == "agent":
                fields = [
                    f"[{key.upper()}]: {str(value).strip()}"
                    for key, value in turn.items()
                    if key != "role" and value
                ]
                if fields:
                    parts.append("\n[AGENT]:\n" + "\n".join(fields))
            elif role == "environment" and turn.get("content"):
                parts.append(f"\n[ENVIRONMENT]: {turn['content']}")
    return "\n".join(parts)


def claw_tool_context(trajectory: dict[str, Any]) -> str:
    """Prefer released Skill Context snapshots; fall back to invoked tool names."""
    snapshots: list[str] = []
    tool_names: set[str] = set()
    for event in trajectory.get("events", []):
        for item in event.get("message", {}).get("content", []):
            if item.get("type") == "text" and item.get("text", "").startswith("Skill Context"):
                snapshots.append(item["text"])
            if item.get("type") == "toolCall" and item.get("name"):
                tool_names.add(item["name"])
    return "\n\n".join(dict.fromkeys(snapshots)) if snapshots else json_text(sorted(tool_names))


def scrub_codex_rollout(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove task_complete because its summary contains the released answer."""
    return [
        event
        for event in events
        if not (
            event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "task_complete"
        )
    ]


def build_prompt_fields(benchmark: str, row: dict[str, Any]) -> PromptFields:
    """Construct exactly {trajectory} and {tool_list_text} for one raw record."""
    if benchmark == "atbench":
        return PromptFields(format_conversation(row), json_text(row["tool_used"]))
    if benchmark == "rjudge":
        return PromptFields(format_conversation(row), row.get("profile", ""))
    if benchmark == "atbench-claw":
        trajectory = row["trajectory"]
        return PromptFields(json_text(trajectory), claw_tool_context(trajectory))
    if benchmark == "atbench-codex":
        trajectory = {
            "conversation": row["conversation"],
            "codex_rollout": scrub_codex_rollout(row["codex_rollout"]),
        }
        return PromptFields(json_text(trajectory), json_text(row["tool_used"]))
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def render_prompt(template: str, fields: PromptFields) -> str:
    return template.format(
        trajectory=fields.trajectory,
        tool_list_text=fields.tool_list_text,
    )
