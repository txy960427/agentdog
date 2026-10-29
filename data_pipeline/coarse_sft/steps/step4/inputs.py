"""CoT输入适配：整理历史格式，隔离返修说明，应用可追溯的人工修正。"""

import ast
import copy
import json
import re
from pathlib import Path

from steps.step0.source import PipelineError, sha256_text, stable_json, tool_name

CORRECTIONS_PATH = Path(__file__).with_name("input_corrections.json")
CORRECTIONS = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
CONFLICT_REPAIRS_PATH = Path(__file__).with_name("conflict_repairs.json")
CONFLICT_REPAIRS = json.loads(CONFLICT_REPAIRS_PATH.read_text(encoding="utf-8"))
CORRECTIONS.update(CONFLICT_REPAIRS)
EDITOR_NOTE = re.compile(r"^(?:Stops before|Removes the unsupported|This rewrite)\b")


def complete_text(action: str) -> str:
    """读取JSON或历史单引号Complete；后者可能带未转义的英文撇号和换行。"""
    payload = action.strip()[len("Complete"):].strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(payload)
            if isinstance(value, dict) and isinstance(value.get("response"), str):
                return value["response"]
        except (ValueError, SyntaxError):
            pass
    # 只剥最外层response包装，不对不合法的历史字符串执行eval或重写内部正文。
    match = re.fullmatch(r"\{\s*['\"]response['\"]\s*:\s*(['\"])(.*)\1\s*\}", payload, re.S)
    if not match:
        raise PipelineError("无法读取Complete的response，请检查原始动作")
    return match[2]


def prepare_row(row: dict) -> tuple[dict, list[str]]:
    """保留冻结的原始final，Teacher和Student共同读取此处的适配结果。"""
    prepared = copy.deepcopy(row)
    notes = []
    if correction := CORRECTIONS.get(row["id"]):
        if sha256_text(stable_json(row["trajectory"])) != correction["source_trajectory_sha256"]:
            raise PipelineError(f"人工修正的源轨迹已变化：{row['id']}")
        stop = correction["stop_before_event"]
        if row["target"]["coarse"] != "safe" or row["trajectory"][stop]["role"] != "agent":
            raise PipelineError(f"安全对照修正点不合法：{row['id']}")
        prepared["trajectory"] = prepared["trajectory"][:stop] + [{
            "role": "agent", "thought": correction["thought"],
            "action": "Complete" + json.dumps({"response": correction["response"]}, ensure_ascii=False),
        }]
        notes.append("manual_safe_twin_correction" if row["id"].startswith("tool_safe_") else "manual_safe_trajectory_correction")
    for event in prepared["trajectory"]:
        if event["role"] != "agent":
            continue
        action = event.get("action") or ""
        if action.lstrip().startswith("Complete{"):
            normalized = "Complete" + json.dumps({"response": complete_text(action)}, ensure_ascii=False)
            if normalized != action:
                notes.append("complete_serialization")
            event["action"] = normalized
    last = prepared["trajectory"][-1]
    # 这些开头来自已定位的risk_neutralization返修说明，不是当时的Agent思考。
    # 仅去除末尾这一段元数据，其他thought和整条样本继续保留。
    if row["id"].startswith("tool_safe_") and last["role"] == "agent" and EDITOR_NOTE.match(last.get("thought") or ""):
        last["thought"] = None
        notes.append("editor_note_removed")
    return prepared, sorted(set(notes))


def unknown_tools_text(events: list[dict]) -> str:
    """工具schema未知不等于没有工具；仅列出原记录中实际出现过的工具名。"""
    names = {}
    for event in events:
        name = None
        if event["role"] == "agent":
            action = event.get("action") or ""
            name = tool_name(action)
            if not name and (match := re.match(r"^([A-Za-z][A-Za-z0-9_ /-]*)\(", action)):
                name = match[1].strip()
        elif event["role"] == "environment":
            try:
                value = json.loads(event["content"])
                name = value.get("name") if isinstance(value, dict) else None
            except (ValueError, TypeError):
                pass
        if isinstance(name, str) and name.strip():
            names[name.strip()] = None
    text = "Full tool definitions were not provided in this record."
    if names:
        text += "\nObserved tools in the recorded trajectory (not a complete inventory):\n"
        text += "\n".join(f"Tool: {name}" for name in names)
    return text
