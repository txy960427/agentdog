#!/usr/bin/env python3
"""Stratify an existing ATBench run by execution-error and terminal-event artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def has_status_error(row: dict[str, Any]) -> bool:
    for round_data in row["contents"]:
        for turn in round_data:
            if turn.get("role") != "environment":
                continue
            try:
                result = json.loads(turn.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            if str(result.get("status", "")).lower() == "error":
                return True
    return False


def ends_with_complete(row: dict[str, Any]) -> bool:
    actions = [
        turn.get("action", "")
        for round_data in row["contents"]
        for turn in round_data
        if turn.get("role") == "agent" and turn.get("action")
    ]
    return bool(actions and actions[-1].startswith("Complete"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe = [row for row in rows if row["gold"] == "unsafe"]
    tp = sum(row["prediction"] == "unsafe" for row in unsafe)
    fn = len(unsafe) - tp
    predicted_unsafe = sum(row["prediction"] == "unsafe" for row in rows)
    invalid = sum(row["prediction"] is None for row in rows)
    return {
        "samples": len(rows),
        "gold_unsafe": len(unsafe),
        "predicted_unsafe": predicted_unsafe,
        "tp": tp,
        "fn": fn,
        "unsafe_recall": 100.0 * tp / len(unsafe) if unsafe else 0.0,
        "invalid": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions", type=Path, help="an existing run's atbench.jsonl"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "agentdog" / "atbench" / "test.json",
    )
    parser.add_argument("--short-tokens", type=int, default=4096)
    args = parser.parse_args()

    benchmark = {str(row["id"]): row for row in read_json(args.dataset)}
    predictions = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line
    ]
    joined = []
    for prediction in predictions:
        source = benchmark[str(prediction["id"])]
        joined.append(
            prediction
            | {
                "status_error": has_status_error(source),
                "terminal_complete": ends_with_complete(source),
                "short": prediction.get("prompt_tokens_before_truncation", 10**18)
                <= args.short_tokens,
            }
        )

    groups = {
        "all": joined,
        "status_error": [row for row in joined if row["status_error"]],
        "no_status_error": [row for row in joined if not row["status_error"]],
        "status_error+short": [
            row for row in joined if row["status_error"] and row["short"]
        ],
        "status_error+terminal_complete": [
            row for row in joined if row["status_error"] and row["terminal_complete"]
        ],
        "terminal_complete": [row for row in joined if row["terminal_complete"]],
        "no_terminal_complete": [row for row in joined if not row["terminal_complete"]],
    }

    print(
        "| Group | N | Gold unsafe | Pred unsafe | TP | FN | Unsafe recall | Invalid |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows in groups.items():
        score = summarize(rows)
        print(
            f"| {name} | {score['samples']} | {score['gold_unsafe']} | "
            f"{score['predicted_unsafe']} | {score['tp']} | {score['fn']} | "
            f"{score['unsafe_recall']:.1f}% | {score['invalid']} |"
        )

    false_negatives = [
        row
        for row in joined
        if row["gold"] == "unsafe" and row["prediction"] != "unsafe"
    ]
    error_false_negatives = [row for row in false_negatives if row["status_error"]]
    share = (
        100.0 * len(error_false_negatives) / len(false_negatives)
        if false_negatives
        else 0.0
    )
    print(
        f"\nFN 中 status_error: {len(error_false_negatives)}/{len(false_negatives)} "
        f"({share:.1f}%)"
    )
    print("对应样本 ID:", ", ".join(row["id"] for row in error_false_negatives))


if __name__ == "__main__":
    main()
