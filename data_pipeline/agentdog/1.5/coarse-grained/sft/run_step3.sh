#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# pilot生成16条；full生成600条草稿；finalize只做本地人工审核验收。
MODE="${MODE:-pilot}"
export STEP3_MODE="${MODE}"
export STEP3_BASE_URL="${STEP3_BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.deepseek.com}}"
export STEP3_GENERATOR_MODEL="${STEP3_GENERATOR_MODEL:-deepseek-v4-pro}"
export STEP3_CHECKER_MODEL="${STEP3_CHECKER_MODEL:-deepseek-v4-pro}"
export STEP3_MAX_TOKENS="${STEP3_MAX_TOKENS:-65536}"
export STEP3_GENERATOR_MAX_TOKENS="${STEP3_GENERATOR_MAX_TOKENS:-65536}"
export STEP3_CHECKER_MAX_TOKENS="${STEP3_CHECKER_MAX_TOKENS:-16384}"
export STEP3_THINKING="${STEP3_THINKING:-enabled}"
export STEP3_REASONING_EFFORT="${STEP3_REASONING_EFFORT:-high}"
export STEP3_STREAM="${STEP3_STREAM:-false}"
export STEP3_MAX_TOKENS_FIELD="${STEP3_MAX_TOKENS_FIELD:-max_tokens}"
export STEP3_WORKERS="${STEP3_WORKERS:-1}"
export STEP3_BATCH_SIZE="${BATCH_SIZE:-8}"
export STEP3_TIMEOUT="${STEP3_TIMEOUT:-900}"
export STEP3_RETRIES="${STEP3_RETRIES:-1}"
export STEP3_GENERATION_ATTEMPTS="${STEP3_GENERATION_ATTEMPTS:-1}"

if [[ "${MODE}" != "finalize" && -z "${STEP3_API_KEY:-${DEEPSEEK_API_KEY:-}}" ]]; then
  echo "ERROR: export STEP3_API_KEY before running Step 3 ${MODE}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
python3 data_pipeline/agentdog/1.5/coarse-grained/sft/pipeline.py step3
