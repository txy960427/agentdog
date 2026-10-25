#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法: bash judge_eval/run_benchmark.sh [0.8B|2B|4B] {serve|serve-fine|coarse|fine} [数据集] [选项] [额外参数]" >&2
  echo "" >&2
  echo "任务:" >&2
  echo "  serve        启动粗粒度 SFT 模型服务" >&2
  echo "  serve-fine   启动细粒度 GDPO 模型服务" >&2
  echo "  coarse       粗粒度 safe/unsafe 评测（默认数据集 all）" >&2
  echo "  fine         细粒度 RS/FM/RWH 评测（默认数据集 all）" >&2
  echo "" >&2
  echo "数据集（coarse / fine 时）:" >&2
  echo "  all          coarse=atbench+rjudge；fine=atbench" >&2
  echo "  atbench      ATBench" >&2
  echo "  rjudge       R-Judge（仅 coarse）" >&2
  echo "" >&2
  echo "选项:" >&2
  echo "  --model-path PATH       覆盖本次模型目录" >&2
  echo "  --fine-model-path PATH  覆盖细粒度（GDPO）默认路径" >&2
  echo "  --dry-run               只构建输入，不要求模型目录或 vLLM 服务" >&2
  echo "环境变量（可选）: MODEL_PATH / FINE_MODEL_PATH / SFT_MODEL_PATH / GDPO_MODEL_PATH" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODEL_SIZE="0.8B"
if [[ "$1" == "0.8B" || "$1" == "2B" || "$1" == "4B" ]]; then
  MODEL_SIZE="$1"
  shift
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
shift
case "$MODE" in
  serve|serve-fine|coarse|fine) ;;
  *)
    usage
    exit 2
    ;;
esac

BENCHMARK="all"
if [[ "$MODE" == "coarse" || "$MODE" == "fine" ]]; then
  if [[ $# -gt 0 && ( "$1" == "all" || "$1" == "atbench" || "$1" == "rjudge" ) ]]; then
    BENCHMARK="$1"
    shift
  fi
  if [[ "$MODE" == "fine" && "$BENCHMARK" == "rjudge" ]]; then
    echo "fine 任务只支持 atbench / all，不支持 rjudge" >&2
    exit 2
  fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ======================== 直接修改这里 ========================
DEFAULT_SFT_MODEL_PATH="$PROJECT_ROOT/model/agentdog/1.5/${MODEL_SIZE}/coarse-grained/AgentDoG1.5-Qwen3.5-${MODEL_SIZE}"
DEFAULT_GDPO_MODEL_PATH="$PROJECT_ROOT/model/agentdog/1.5/${MODEL_SIZE}/fine-grained/AgentDoG1.5-FG-Qwen3.5-${MODEL_SIZE}"

CUDA_VISIBLE_DEVICES="0"
PORT="8000"

TENSOR_PARALLEL_SIZE=1
DTYPE="bfloat16"
SEED="${SEED:-42}"

GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=131072
MAX_TOKENS=2048
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=32768

SWAP_SPACE=0
CPU_OFFLOAD_GB=0
ENABLE_CHUNKED_PREFILL=1
ENABLE_PREFIX_CACHING=0
ENFORCE_EAGER=0
TRUST_REMOTE_CODE=0

PROMPT_OVERFLOW="truncate"
TRUNCATION_SIDE="left"
# =============================================================

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${MODEL_PATH:-$DEFAULT_SFT_MODEL_PATH}}"
GDPO_MODEL_PATH="${GDPO_MODEL_PATH:-${FINE_MODEL_PATH:-$DEFAULT_GDPO_MODEL_PATH}}"

OVERRIDE_MODEL_PATH=""
FILTERED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      if [[ $# -lt 2 ]]; then
        echo "--model-path 需要路径参数" >&2
        exit 2
      fi
      OVERRIDE_MODEL_PATH="$2"
      shift 2
      ;;
    --model-path=*)
      OVERRIDE_MODEL_PATH="${1#*=}"
      shift
      ;;
    --fine-model-path)
      if [[ $# -lt 2 ]]; then
        echo "--fine-model-path 需要路径参数" >&2
        exit 2
      fi
      GDPO_MODEL_PATH="$2"
      shift 2
      ;;
    --fine-model-path=*)
      GDPO_MODEL_PATH="${1#*=}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${FILTERED_ARGS[@]}"

TASK_NAME="coarse"
FOLLOWUP_MODE="coarse"
MODEL_PATH="$SFT_MODEL_PATH"
TASK_ARGS=()
RUNS_ARGS=()

case "$MODE" in
  serve)
    ;;
  serve-fine)
    MODEL_PATH="$GDPO_MODEL_PATH"
    FOLLOWUP_MODE="fine"
    MODE="serve"
    ;;
  coarse)
    TASK_NAME="coarse"
    TASK_ARGS=(--task coarse)
    RUNS_ARGS=(--runs-dir "$SCRIPT_DIR/runs/coarse")
    ;;
  fine)
    TASK_NAME="fine"
    MODEL_PATH="$GDPO_MODEL_PATH"
    TASK_ARGS=(--task fine)
    RUNS_ARGS=(--runs-dir "$SCRIPT_DIR/runs/fine")
    ;;
esac

if [[ -n "$OVERRIDE_MODEL_PATH" ]]; then
  MODEL_PATH="$OVERRIDE_MODEL_PATH"
fi

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
    break
  fi
done

if [[ "$DRY_RUN" != "1" && ! -d "$MODEL_PATH" ]]; then
  echo "没有找到模型目录: $MODEL_PATH" >&2
  echo "请先按照 judge_eval/README.md 下载模型，或用 --model-path / MODEL_PATH 指定。" >&2
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

if [[ "$MODE" == "serve" ]]; then
  echo "vLLM server on $BASE_URL"
  echo "model=$MODEL_PATH cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "Leave this process running. Evaluate with: bash judge_eval/run_benchmark.sh ${MODEL_SIZE} ${FOLLOWUP_MODE} all"
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

echo "task=$TASK_NAME benchmark=$BENCHMARK model_size=$MODEL_SIZE model=$MODEL_PATH server=$BASE_URL"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "max_model_len=$MAX_MODEL_LEN max_tokens=$MAX_TOKENS overflow=$PROMPT_OVERFLOW/$TRUNCATION_SIDE"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION max_num_seqs=$MAX_NUM_SEQS"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS chunked_prefill=$ENABLE_CHUNKED_PREFILL dtype=$DTYPE"

cd "$PROJECT_ROOT"
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
