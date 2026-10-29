#!/usr/bin/env bash
set -euo pipefail

# GPU机器运行；改这里的几个变量即可。模型也可以填写本地目录。
MODEL="Qwen/Qwen3.5-0.8B"
GPU="0"
CUTOFF_LEN=0                     # 0=按全部1500条的最大完整长度自动选择
RESUME_FROM=""                  # 续训时填写原checkpoint目录，否则留空
MODE="${1:-check}"              # check / smoke / train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export USE_V1=0

ARGS=("${MODE}" --model "${MODEL}" --cutoff-len "${CUTOFF_LEN}")
if [[ -n "${RESUME_FROM}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM}")
fi
python "${SCRIPT_DIR}/run.py" "${ARGS[@]}"
