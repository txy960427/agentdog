#!/usr/bin/env python3
"""AgentDoG 1.5 粗粒度 SFT 数据构建总入口。"""

from __future__ import annotations

import argparse
import sys

from steps.step0.pipeline import run as run_step0
from steps.step0.source import PipelineError
from steps.step1.pipeline import run as run_step1
from steps.step2.pipeline import run as run_step2
from steps.step3.pipeline import run as run_step3

STAGES = {
    "step0": ("Step 0 - Global selection", run_step0),
    "step1": ("Step 1 - BinarySafety conversion", run_step1),
    "step2": ("Step 2 - FineGrained unsafe conversion", run_step2),
    "step3": ("Step 3 - Safe twin generation", run_step3),
}
ALL_STAGES = ("step0", "step1", "step2")


def main() -> int:
    parser = argparse.ArgumentParser(description="构建AgentDoG粗粒度SFT数据")
    parser.add_argument(
        "stage",
        choices=(*STAGES, "all"),
        help="运行指定阶段，或使用all按顺序运行全部阶段",
    )
    args = parser.parse_args()
    # all只运行不产生API费用的确定性阶段；Step 3必须显式执行。
    selected = (
        ((stage, STAGES[stage]) for stage in ALL_STAGES)
        if args.stage == "all"
        else ((args.stage, STAGES[args.stage]),)
    )

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
