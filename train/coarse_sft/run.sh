#!/usr/bin/env bash
set -euo pipefail

# GPU机器运行；改这里的几个变量即可。模型也可以填写本地目录。
MODEL="model/Qwen3.5-0.8B"
GPU="0"
DATASET_VARIANT="step5"          # baseline / step5；当前实验使用step5
CUTOFF_LEN=0                     # 0=按全部1500条的最大完整长度自动选择
RESUME_FROM=""                  # 续训时填写原checkpoint目录，否则留空
MODE="${1:-check}"              # check / smoke / train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export USE_V1=0
# 减轻长序列训练时的显存碎片（OOM 日志里 reserved but unallocated 时有用）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=("${MODE}" --model "${MODEL}" --dataset-variant "${DATASET_VARIANT}" --cutoff-len "${CUTOFF_LEN}")
if [[ -n "${RESUME_FROM}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM}")
fi
python "${SCRIPT_DIR}/run.py" "${ARGS[@]}"
