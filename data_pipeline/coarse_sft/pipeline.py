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
from steps.step4.pipeline import run as run_step4
from steps.step4.review import run as review_step4
from steps.step4.verify import run as verify_step4

STAGES = {
    "step0": ("Step 0 - Global selection", run_step0),
    "step1": ("Step 1 - BinarySafety conversion", run_step1),
    "step2": ("Step 2 - FineGrained unsafe conversion", run_step2),
    "step3": ("Step 3 - Safe twin generation", run_step3),
    "step4": ("Step 4 - CoT generation and SFT export", run_step4),
}
ALL_STAGES = ("step0", "step1", "step2")


def main() -> int:
    parser = argparse.ArgumentParser(description="构建AgentDoG粗粒度SFT数据")
    parser.add_argument(
        "stage",
        choices=(*STAGES, "all"),
        help="运行指定阶段；all只运行不调用API的step0–2",
    )
    parser.add_argument("--dry-run", action="store_true", help="Step 4：仅检查输入和渲染，不调用API")
    parser.add_argument("--limit", type=int, help="Step 4：本次最多处理多少条，省略则处理所有未完成记录")
    parser.add_argument("--review", action="store_true", help="Step 4：仅对争议队列做双模型盲判，不重跑其他数据")
    parser.add_argument("--verify", action="store_true", help="Step 4：离线复验全部来源、划分、CoT和Parquet，不调用API")
    args = parser.parse_args()
    if args.stage != "step4" and (args.dry_run or args.limit is not None or args.review or args.verify):
        parser.error("--dry-run、--limit、--review和--verify仅用于step4")
    if args.verify and (args.dry_run or args.limit is not None or args.review):
        parser.error("--verify不能与其他Step 4参数混用")
    # all只运行不产生API费用的确定性阶段；Step 3/4必须显式执行。
    selected = (
        ((stage, STAGES[stage]) for stage in ALL_STAGES)
        if args.stage == "all"
        else ((args.stage, STAGES[args.stage]),)
    )

    try:
        for stage_name, (name, stage) in selected:
            print(f"\n=== {name} ===")
            if stage_name == "step4":
                if args.verify:
                    verify_step4()
                else:
                    (review_step4 if args.review else stage)(dry_run=args.dry_run, limit=args.limit)
            else:
                stage()
    except (PipelineError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
