#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE="${1:-step0}"

cd "${PROJECT_ROOT}"
python3 data_pipeline/coarse_sft/pipeline.py "${STAGE}"
