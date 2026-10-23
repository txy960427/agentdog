#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法: bash judge_eval/run_benchmark.sh [0.8B|2B|4B] {serve|atbench|atbench-claw|atbench-codex|rjudge|all|serve-fine|fine} [额外参数]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODEL_SIZE="0.8B"
if [[ "$1" == "0.8B" || "$1" == "2B" || "$1" == "4B" ]]; then
  MODEL_SIZE="$1"
  shift
fi
if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

BENCHMARK="$1"
shift
case "$BENCHMARK" in
  serve|atbench|atbench-claw|atbench-codex|rjudge|all|serve-fine|fine) ;;
  *)
    usage
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ======================== 直接修改这里 ========================
# 尺寸由命令第一个参数选择：bash judge_eval/run_benchmark.sh 4B all
MODEL_PATH="$VERL_ROOT/model/agentdog/1.5/${MODEL_SIZE}/coarse-grained/AgentDoG1.5-Qwen3.5-${MODEL_SIZE}"
# Table 3 的细粒度模型。目录名里的 0.8B/2B/4B/8B 用来选择论文对照行。
FINE_MODEL_PATH="$VERL_ROOT/model/agentdog/1.5/${MODEL_SIZE}/fine-grained/AgentDoG1.5-FG-Qwen3.5-${MODEL_SIZE}"
CUDA_VISIBLE_DEVICES="0"
PORT="8000"

TENSOR_PARALLEL_SIZE=1
DTYPE="bfloat16"
SEED=42

GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=131072
MAX_TOKENS=2048
# 48GB 4090 holds about 3.3M KV tokens for this 0.8B model. The scheduler
# admits fewer sequences when prompts are long; 64 is the short-prompt cap.
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=32768

# Qwen3.5 is a hybrid attention model. vLLM 0.29 KV offload then requires
# prefix caching to align block sizes, and this GPU already has enough KV cache.
SWAP_SPACE=0
CPU_OFFLOAD_GB=0
ENABLE_CHUNKED_PREFILL=1
ENABLE_PREFIX_CACHING=0
ENFORCE_EAGER=0
TRUST_REMOTE_CODE=0

PROMPT_OVERFLOW="truncate"
TRUNCATION_SIDE="left"
# =============================================================

TASK_ARGS=()
RUNS_ARGS=()
TASK_NAME="coarse"
if [[ "$BENCHMARK" == "serve-fine" || "$BENCHMARK" == "fine" ]]; then
  MODEL_PATH="$FINE_MODEL_PATH"
  TASK_NAME="fine"
fi
if [[ "$BENCHMARK" == "serve-fine" ]]; then
  BENCHMARK="serve"
fi
if [[ "$BENCHMARK" == "fine" ]]; then
  BENCHMARK="atbench"
  TASK_ARGS=(--task fine)
  RUNS_ARGS=(--runs-dir "$SCRIPT_DIR/runs/fine")
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "没有找到模型目录: $MODEL_PATH" >&2
  echo "请先按照 judge_eval/README.md 下载模型，或修改脚本顶部的模型路径。" >&2
  exit 1
fi

EXTRA_ARGS=()
[[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  EXTRA_ARGS+=(--enable-chunked-prefill)
else
  EXTRA_ARGS+=(--no-enable-chunked-prefill)
fi
[[ "$ENABLE_PREFIX_CACHING" == "1" ]] && EXTRA_ARGS+=(--enable-prefix-caching)
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)
[[ "$TRUST_REMOTE_CODE" == "1" ]] && EXTRA_ARGS+=(--trust-remote-code)

export CUDA_VISIBLE_DEVICES
BASE_URL="http://127.0.0.1:${PORT}"

if [[ "$BENCHMARK" == "serve" ]]; then
  echo "vLLM server on $BASE_URL"
  echo "model=$MODEL_PATH cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "Leave this process running. Evaluate with: bash judge_eval/run_benchmark.sh ${MODEL_SIZE} all"
  SERVE_ARGS=(
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$MODEL_PATH"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --seed "$SEED"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
  )
  [[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && SERVE_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    SERVE_ARGS+=(--enable-chunked-prefill)
  else
    SERVE_ARGS+=(--no-enable-chunked-prefill)
  fi
  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    SERVE_ARGS+=(--enable-prefix-caching)
  else
    SERVE_ARGS+=(--no-enable-prefix-caching)
  fi
  [[ "$ENFORCE_EAGER" == "1" ]] && SERVE_ARGS+=(--enforce-eager)
  [[ "$TRUST_REMOTE_CODE" == "1" ]] && SERVE_ARGS+=(--trust-remote-code)
  [[ "$CPU_OFFLOAD_GB" != "0" ]] && SERVE_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
  if [[ "$SWAP_SPACE" != "0" ]]; then
    SERVE_ARGS+=(--kv-offloading-size "$((SWAP_SPACE * TENSOR_PARALLEL_SIZE))")
  fi
  exec vllm serve "$MODEL_PATH" "${SERVE_ARGS[@]}" "$@"
fi

echo "benchmark=$BENCHMARK task=$TASK_NAME model_size=$MODEL_SIZE model=$MODEL_PATH server=$BASE_URL"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "max_model_len=$MAX_MODEL_LEN max_tokens=$MAX_TOKENS overflow=$PROMPT_OVERFLOW/$TRUNCATION_SIDE"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION max_num_seqs=$MAX_NUM_SEQS"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS chunked_prefill=$ENABLE_CHUNKED_PREFILL dtype=$DTYPE"

cd "$VERL_ROOT"
exec python judge_eval/eval.py \
  --benchmark "$BENCHMARK" \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --swap-space "$SWAP_SPACE" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --prompt-overflow "$PROMPT_OVERFLOW" \
  --truncation-side "$TRUNCATION_SIDE" \
  "${TASK_ARGS[@]}" \
  "${RUNS_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
