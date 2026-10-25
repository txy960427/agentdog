#!/usr/bin/env bash
# 常用命令速查（在 verl 根目录执行）。复制需要的行即可，不要整文件 bash。
# 用法: bash judge_eval/run_benchmark.sh [0.8B|2B|4B] {serve|serve-fine|coarse|fine} [atbench|rjudge|all]
# 粗粒度默认模型: model/agentdog/sft/<size>/AgentDoG1.5-Qwen3.5-<size>
# 细粒度默认模型: model/agentdog/gdpo/<size>/AgentDoG1.5-FG-Qwen3.5-<size>
# 覆盖路径: --model-path /path  或  MODEL_PATH=/path
#           --fine-model-path /path  或  FINE_MODEL_PATH=/path

# =====================================================================
# 1. 粗粒度 SFT
# =====================================================================

# ---------- 1.1 不同规模起服务（另开终端，保持运行）----------
bash judge_eval/run_benchmark.sh 0.8B serve
bash judge_eval/run_benchmark.sh 2B serve
bash judge_eval/run_benchmark.sh 4B serve

# ---------- 1.2 不同规模 × 数据集（单个 / all）推理评测 ----------
# 0.8B
bash judge_eval/run_benchmark.sh 0.8B coarse atbench
bash judge_eval/run_benchmark.sh 0.8B coarse rjudge
bash judge_eval/run_benchmark.sh 0.8B coarse all

# 2B
bash judge_eval/run_benchmark.sh 2B coarse atbench
bash judge_eval/run_benchmark.sh 2B coarse rjudge
bash judge_eval/run_benchmark.sh 2B coarse all

# 4B
bash judge_eval/run_benchmark.sh 4B coarse atbench
bash judge_eval/run_benchmark.sh 4B coarse rjudge
bash judge_eval/run_benchmark.sh 4B coarse all

# =====================================================================
# 2. 细粒度 GDPO（仅 ATBench unsafe；all 与 atbench 等价）
# =====================================================================

# ---------- 2.1 不同规模起服务（勿与粗粒度 serve 抢同一 PORT）----------
bash judge_eval/run_benchmark.sh 0.8B serve-fine
bash judge_eval/run_benchmark.sh 2B serve-fine
bash judge_eval/run_benchmark.sh 4B serve-fine

# ---------- 2.2 不同规模 × 数据集（单个 / all）推理评测 ----------
# 0.8B
bash judge_eval/run_benchmark.sh 0.8B fine atbench
bash judge_eval/run_benchmark.sh 0.8B fine all

# 2B
bash judge_eval/run_benchmark.sh 2B fine atbench
bash judge_eval/run_benchmark.sh 2B fine all

# 4B
bash judge_eval/run_benchmark.sh 4B fine atbench
bash judge_eval/run_benchmark.sh 4B fine all

# =====================================================================
# 可选
# =====================================================================

# ---------- 覆盖模型路径 / 小样本 ----------
# bash judge_eval/run_benchmark.sh 4B serve --model-path /path/to/sft-model
# bash judge_eval/run_benchmark.sh 4B coarse atbench --model-path /path/to/sft-model
# bash judge_eval/run_benchmark.sh 4B serve-fine --fine-model-path /path/to/gdpo-model
# bash judge_eval/run_benchmark.sh 4B fine all --fine-model-path /path/to/gdpo-model
# bash judge_eval/run_benchmark.sh 4B coarse atbench --limit 8
# bash judge_eval/run_benchmark.sh 4B fine all --limit 8

# ---------- dry-run（不连 vLLM，只检查数据与 prompt）----------
# python judge_eval/eval.py --task coarse --benchmark atbench --dry-run
# python judge_eval/eval.py --task coarse --benchmark rjudge --dry-run
# python judge_eval/eval.py --task coarse --benchmark all --dry-run
# python judge_eval/eval.py --task fine --benchmark all --dry-run
