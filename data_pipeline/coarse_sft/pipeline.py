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
from steps.step4.cot_pool import run as run_cot_pool
from steps.step4.check import run as check_step4
from steps.step4.check import repair as repair_step4_cot
from steps.step5.pipeline import run as run_step5

STAGES = {
    "step0": ("Step 0 - Global selection", run_step0),
    "step1": ("Step 1 - BinarySafety conversion", run_step1),
    "step2": ("Step 2 - FineGrained unsafe conversion", run_step2),
    "step3": ("Step 3 - Safe twin generation", run_step3),
    "step4": ("Step 4 - CoT generation and SFT export", run_step4),
    "step5": ("Step 5 - Length-debias replacement selection", run_step5),
    "cot_pool": ("CoT pool - Remaining v1.0 candidates", run_cot_pool),
}
ALL_STAGES = ("step0", "step1", "step2")


def main() -> int:
    parser = argparse.ArgumentParser(description="构建AgentDoG粗粒度SFT数据")
    parser.add_argument(
        "stage",
        choices=(*STAGES, "all"),
        help="运行指定阶段；all只运行不调用API的step0–2",
    )
    parser.add_argument("--dry-run", action="store_true", help="step4/cot_pool：仅检查输入和渲染，不调用API")
    parser.add_argument("--limit", type=int, help="step4/cot_pool：本次最多处理多少条，省略则处理所有未完成记录")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all", help="cot_pool：选择处理的split，默认all")
    parser.add_argument("--selection", choices=("all", "step5"), default="all", help="cot_pool：只处理指定选择清单")
    parser.add_argument("--target", choices=("baseline", "step5"), default="baseline", help="Step 4 --check：选择审核数据目录")
    parser.add_argument("--review", action="store_true", help="Step 4：仅对争议队列做双模型盲判，不重跑其他数据")
    parser.add_argument("--verify", action="store_true", help="Step 4：离线复验全部来源、划分、CoT和Parquet，不调用API")
    parser.add_argument("--check", action="store_true", help="Step 4：调用另一个模型审核最终CoT、证据和标签，不修改数据")
    parser.add_argument("--repair-cot", action="store_true", help="Step 4：应用已审定的CoT定点修正，不调用API、不改输入或gold")
    parser.add_argument("--report-only", action="store_true", help="Step 4 --check：只更新模型记录与本地裁决的汇总，不调用API")
    args = parser.parse_args()
    if args.stage not in ("step4", "cot_pool") and (args.dry_run or args.limit is not None):
        parser.error("--dry-run和--limit仅用于step4/cot_pool")
    if args.stage != "step4" and (args.review or args.verify or args.check or args.repair_cot):
        parser.error("--review、--verify、--check和--repair-cot仅用于step4")
    if args.stage != "cot_pool" and args.split != "all":
        parser.error("--split仅用于cot_pool")
    if args.stage != "cot_pool" and args.selection != "all":
        parser.error("--selection仅用于cot_pool")
    if args.target != "baseline" and not (args.stage == "step4" and args.check):
        parser.error("--target step5仅用于step4 --check")
    if args.report_only and (not args.check or args.dry_run or args.limit is not None):
        parser.error("--report-only须与step4 --check配合，不与--dry-run/--limit混用")
    if sum((args.review, args.verify, args.check, args.repair_cot)) > 1:
        parser.error("--review、--verify、--check和--repair-cot不能混用")
    if args.repair_cot and (args.dry_run or args.limit is not None):
        parser.error("--repair-cot仅应用固定补丁，不与--dry-run/--limit混用")
    if args.verify and (args.dry_run or args.limit is not None):
        parser.error("--verify不能与其他Step 4参数混用")
    # all只运行不产生API费用的确定性阶段；Step 3/4必须显式执行。
    selected = (
        ((stage, STAGES[stage]) for stage in ALL_STAGES)
        if args.stage == "all"
        else ((args.stage, STAGES[args.stage]),)
    )

    try:
        for stage_name, (name, stage) in selected:
            if stage_name == "step4" and args.check:
                name = "Step 4 - CoT quality check"
            elif stage_name == "step4" and args.repair_cot:
                name = "Step 4 - Apply reviewed CoT corrections"
            print(f"\n=== {name} ===")
            if stage_name == "cot_pool":
                stage(dry_run=args.dry_run, limit=args.limit, split=args.split, selection=args.selection)
            elif stage_name == "step4":
                if args.verify:
                    verify_step4()
                elif args.repair_cot:
                    repair_step4_cot()
                elif args.check:
                    check_step4(dry_run=args.dry_run, limit=args.limit, report_only=args.report_only, target=args.target)
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
