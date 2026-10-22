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


# Construction metadata. The agent only saw name, description, and parameters.
HIDDEN_TOOL_KEYS = {"_source", "_original_description"}

# Constant line inserted into every 1.0 binary SFT trajectory.
TRAINING_PROFILE = "You are a helpful assistant."


def _profile_header(extra: str = "") -> str:
    """Match the 1.0 binary training header. Keep a dataset role on the next line."""
    lines = ["=== Agent Profile ===", TRAINING_PROFILE]
    role = extra.strip()
    if role and role != TRAINING_PROFILE:
        lines.append(role)
    return "\n".join(lines) + "\n"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def visible_tools(tools: Any, hidden: set[str] = HIDDEN_TOOL_KEYS) -> Any:
    """Drop fields that record how a tool definition was built or rewritten."""
    if not isinstance(tools, list):
        return tools
    return [
        {key: value for key, value in tool.items() if key not in hidden}
        if isinstance(tool, dict)
        else tool
        for tool in tools
    ]


def _atbench_reply(text: str) -> str:
    """Unwrap Complete{"response": ...}. Drop one layer of wrapping quotes."""
    reply = _codex_response(text)
    if len(reply) >= 2 and reply[0] == reply[-1] == '"':
        reply = reply[1:-1].strip()
    return reply


def _atbench_agent_fields(turn: dict[str, Any]) -> list[str]:
    """Match 1.0 binary SFT: one [ACTION] line, no thought or response tag."""
    text = str(turn.get("action") or "").strip()
    if not text:
        return []
    if text.startswith("Complete{"):
        reply = _atbench_reply(text)
        if not reply:
            return []
        return [f"[ACTION]: Complete{{'response': '{reply}'}}"]
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        return [f"[ACTION]: {text}"]
    if isinstance(action, dict) and action.get("name"):
        return [f"[ACTION]: {_action_text(action.get('name'), action.get('arguments'))}"]
    return [f"[ACTION]: {text}"]


def format_conversation(row: dict[str, Any], normalize_actions: bool = False) -> str:
    """Serialize ATBench/R-Judge records. Thought is omitted for these two."""
    parts: list[str] = [_profile_header(str(row.get("profile") or ""))]
    parts.append("=== Conversation History ===")
    for round_data in row["contents"]:
        for turn in round_data:
            role = turn["role"]
            if role == "user" and turn.get("content"):
                parts.append(f"\n[USER]: {turn['content']}")
            elif role == "agent":
                if normalize_actions:
                    fields = _atbench_agent_fields(turn)
                else:
                    fields = [
                        f"[{key.upper()}]: {str(value).strip()}"
                        for key, value in turn.items()
                        if key not in {"role", "thought"} and value
                    ]
                if fields:
                    parts.append("\n[AGENT]:\n" + "\n".join(fields))
            elif role == "environment" and turn.get("content"):
                parts.append(f"\n[ENVIRONMENT]: {turn['content']}")
    return "\n".join(parts)


def _plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _codex_response(value: Any) -> str:
    """Drop the Complete{"response": ...} record wrapper. Keep any other text."""
    text = _plain(value)
    if not text.startswith("Complete{"):
        return text
    try:
        payload = json.loads(text[len("Complete") :])
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict) and set(payload) == {"response"}:
        return _plain(payload["response"])
    return text


def _load_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _action_text(name: Any, arguments: Any) -> str:
    return json.dumps({"name": name, "arguments": _load_arguments(arguments)}, ensure_ascii=False)


def claw_tool_context(trajectory: dict[str, Any]) -> str:
    """Skill snapshots belong only in the tool list, not in the trajectory."""
    snapshots: list[str] = []
    tool_names: set[str] = set()
    for event in trajectory.get("events", []):
        message = event.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "text" and str(item.get("text", "")).startswith("Skill Context"):
                snapshots.append(item["text"])
            if item.get("type") == "toolCall" and item.get("name"):
                tool_names.add(item["name"])
        if message.get("role") == "toolResult" and message.get("toolName"):
            tool_names.add(message["toolName"])
    return "\n\n".join(dict.fromkeys(snapshots)) if snapshots else json_text(sorted(tool_names))


def format_claw_trajectory(trajectory: dict[str, Any]) -> str:
    """Render OpenClaw events as user, agent, and environment turns."""
    parts = [_profile_header(), "=== Conversation History ==="]
    for event in trajectory.get("events", []):
        message = event.get("message") or {}
        role = message.get("role")
        content = message.get("content") or []
        if role == "user":
            texts = [
                text
                for item in content
                if item.get("type") == "text"
                and (text := str(item.get("text") or "").strip())
                and not text.startswith("Skill Context")
            ]
            if texts:
                parts.append("\n[USER]: " + "\n".join(texts))
        elif role == "assistant":
            fields: list[str] = []
            for item in content:
                if item.get("type") == "text" and (reply := str(item.get("text") or "").strip()):
                    fields.append(f"[ACTION]: Complete{{'response': '{reply}'}}")
                elif item.get("type") == "toolCall":
                    fields.append(f"[ACTION]: {_action_text(item.get('name'), item.get('arguments'))}")
            if fields:
                parts.append("\n[AGENT]:\n" + "\n".join(fields))
        elif role == "toolResult":
            parts.append("\n[ENVIRONMENT]: " + json.dumps(message, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def format_codex_session(events: list[dict[str, Any]]) -> str:
    """One session line, placed at the end of the conversation history."""
    cwd = approval = sandbox = ""
    originator = source = model = effort = realtime = ""
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event.get("type") == "session_meta":
            cwd = cwd or _plain(payload.get("cwd"))
            originator = originator or _plain(payload.get("originator"))
            source = source or _plain(payload.get("source"))
        elif event.get("type") == "turn_context":
            cwd = cwd or _plain(payload.get("cwd"))
            approval = approval or _plain(payload.get("approval_policy"))
            policy = payload.get("sandbox_policy")
            if isinstance(policy, dict) and set(policy) == {"type"}:
                sandbox = sandbox or _plain(policy.get("type"))
            else:
                sandbox = sandbox or _plain(policy)
            model = model or _plain(payload.get("model"))
            effort = effort or _plain(payload.get("effort"))
            if realtime == "" and payload.get("realtime_active") is not None:
                realtime = _plain(payload.get("realtime_active"))
    fields = [
        f"cwd={cwd}" if cwd else "",
        f"approval_policy={approval}" if approval else "",
        f"sandbox_policy={sandbox}" if sandbox else "",
        f"model={model}" if model else "",
        f"originator={originator}" if originator else "",
        f"source={source}" if source else "",
        f"effort={effort}" if effort else "",
        f"realtime_active={realtime}" if realtime else "",
    ]
    body = " ".join(field for field in fields if field)
    return f"[SESSION]: {body}" if body else ""


def format_codex_conversation(turns: list[dict[str, Any]]) -> str:
    """Render the released conversation with the same role labels as ATBench."""
    parts = [_profile_header(), "=== Conversation History ==="]
    for turn in turns:
        role = turn.get("role")
        if role == "user" and _plain(turn.get("content")):
            parts.append(f"\n[USER]: {_plain(turn['content'])}")
        elif role == "assistant":
            fields: list[str] = []
            thought = _plain(turn.get("thought"))
            if thought:
                fields.append(f"[THOUGHT]: {thought}")
            action = turn.get("action")
            if isinstance(action, dict) and action.get("name"):
                fields.append(f"[ACTION]: {_action_text(action.get('name'), action.get('arguments'))}")
            response = _plain(turn.get("response"))
            if response.startswith("Complete{"):
                reply = _atbench_reply(response)
                if reply:
                    fields.append(f"[ACTION]: Complete{{'response': '{reply}'}}")
            elif response:
                fields.append(f"[ACTION]: Complete{{'response': '{response}'}}")
            if fields:
                parts.append("\n[AGENT]:\n" + "\n".join(fields))
        elif role == "environment" and turn.get("content") is not None:
            content = turn["content"]
            body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            parts.append(f"\n[ENVIRONMENT]: {body}")
    return "\n".join(parts)


def build_prompt_fields(benchmark: str, row: dict[str, Any]) -> PromptFields:
    """Construct exactly {trajectory} and {tool_list_text} for one raw record."""
    if benchmark == "atbench":
        return PromptFields(format_conversation(row, normalize_actions=True), json_text(visible_tools(row["tool_used"])))
    if benchmark == "rjudge":
        return PromptFields(format_conversation(row), row.get("profile", ""))
    if benchmark == "atbench-claw":
        trajectory = row["trajectory"]
        return PromptFields(format_claw_trajectory(trajectory), claw_tool_context(trajectory))
    if benchmark == "atbench-codex":
        hidden = {"_source", "_server_label", "_server_url", "_require_approval"}
        trajectory = format_codex_conversation(row["conversation"])
        session = format_codex_session(row["codex_rollout"])
        if session:
            trajectory = f"{trajectory}\n\n{session}"
        return PromptFields(trajectory, json_text(visible_tools(row["tool_used"], hidden)))
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def render_prompt(template: str, fields: PromptFields) -> str:
    return template.format(
        trajectory=fields.trajectory,
        tool_list_text=fields.tool_list_text,
    )
