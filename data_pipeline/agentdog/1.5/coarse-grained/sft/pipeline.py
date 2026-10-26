#!/usr/bin/env python3
"""AgentDoG 1.5 粗粒度 SFT 数据构建总入口。"""

from __future__ import annotations

import argparse
import sys

from steps.step0.pipeline import run as run_step0
from steps.step0.source import PipelineError

STAGES = {
    "step0": ("Step 0 - Global selection", run_step0),
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
    except (PipelineError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
