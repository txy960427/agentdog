#!/usr/bin/env bash
# 常用命令速查（在 verl 根目录执行）。复制需要的行即可，不要整文件 bash。
# 粗粒度默认: model/agentdog/sft/<size>/...
# 细粒度默认: model/agentdog/gdpo/<size>/...
# 覆盖路径: --model-path /path  或  MODEL_PATH=/path
#           --fine-model-path /path  或  FINE_MODEL_PATH=/path

# ---------- 粗粒度 SFT：起服务（另开终端，保持运行）----------
bash judge_eval/run_benchmark.sh 0.8B serve
bash judge_eval/run_benchmark.sh 2B serve
bash judge_eval/run_benchmark.sh 4B serve

# ---------- 粗粒度 SFT：四个数据集 / 一次跑完 ----------
bash judge_eval/run_benchmark.sh 0.8B atbench
bash judge_eval/run_benchmark.sh 0.8B atbench-claw
bash judge_eval/run_benchmark.sh 0.8B atbench-codex
bash judge_eval/run_benchmark.sh 0.8B rjudge
bash judge_eval/run_benchmark.sh 0.8B all

bash judge_eval/run_benchmark.sh 2B atbench
bash judge_eval/run_benchmark.sh 2B atbench-claw
bash judge_eval/run_benchmark.sh 2B atbench-codex
bash judge_eval/run_benchmark.sh 2B rjudge
bash judge_eval/run_benchmark.sh 2B all

bash judge_eval/run_benchmark.sh 4B atbench
bash judge_eval/run_benchmark.sh 4B atbench-claw
bash judge_eval/run_benchmark.sh 4B atbench-codex
bash judge_eval/run_benchmark.sh 4B rjudge
bash judge_eval/run_benchmark.sh 4B all

# ---------- 细粒度 GDPO：起服务（勿与粗粒度 serve 抢同一 PORT）----------
bash judge_eval/run_benchmark.sh 0.8B serve-fine
bash judge_eval/run_benchmark.sh 2B serve-fine
bash judge_eval/run_benchmark.sh 4B serve-fine

# ---------- 细粒度 GDPO：ATBench unsafe 三维准确率（表 3）----------
bash judge_eval/run_benchmark.sh 0.8B fine
bash judge_eval/run_benchmark.sh 2B fine
bash judge_eval/run_benchmark.sh 4B fine

# ---------- 可选：覆盖模型路径 / 小样本 ----------
# bash judge_eval/run_benchmark.sh 4B serve --model-path /path/to/sft-model
# bash judge_eval/run_benchmark.sh 4B atbench --model-path /path/to/sft-model
# bash judge_eval/run_benchmark.sh 4B serve-fine --fine-model-path /path/to/gdpo-model
# bash judge_eval/run_benchmark.sh 4B fine --fine-model-path /path/to/gdpo-model
# bash judge_eval/run_benchmark.sh 4B atbench --limit 8
# bash judge_eval/run_benchmark.sh 4B fine --limit 8

# ---------- dry-run（不连 vLLM，只检查数据与 prompt）----------
# python judge_eval/eval.py --benchmark atbench --dry-run
# python judge_eval/eval.py --benchmark atbench-claw --dry-run
# python judge_eval/eval.py --benchmark atbench-codex --dry-run
# python judge_eval/eval.py --benchmark rjudge --dry-run
# python judge_eval/eval.py --task fine --benchmark atbench --dry-run
