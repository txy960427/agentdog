#!/usr/bin/env python3
"""AgentDoG 1.5 粗粒度 SFT 数据构建总入口。"""

from __future__ import annotations

import argparse
import json
import sys

from steps.step1.pipeline import run as run_step1
from steps.step1.source import PipelineError
from steps.step2.pipeline import run as run_step2

# 新阶段完成后只需按顺序追加到这里；用户始终运行同一个入口。
STAGES = {
    "step1": ("Step 1 - BinarySafety", run_step1),
    "step2": ("Step 2 - Tool unsafe", run_step2),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="构建AgentDoG粗粒度SFT数据")
    parser.add_argument(
        "stage",
        choices=(*STAGES, "all"),
        help="运行指定阶段，或使用all按顺序运行全部阶段",
    )
    args = parser.parse_args()
    selected = STAGES.items() if args.stage == "all" else ((args.stage, STAGES[args.stage]),)

    try:
        for _, (name, stage) in selected:
            print(f"\n=== {name} ===")
            stage()
    except (PipelineError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
