#!/usr/bin/env bash
set -euo pipefail

# RTX 4090 / cuda132_torch2130_py312；直接使用当前 Python，不使用 UV。
PYTHON_BIN="python"
INSTALL_TORCH=0
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
TORCH_VERSION="2.13.0"
VLLM_VERSION="0.29.0"
TRANSFORMERS_VERSION="5.12.1"

nvidia-smi
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --force-reinstall \
    --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION"
else
  "$PYTHON_BIN" -c "import torch; assert torch.__version__.split('+', 1)[0] == '$TORCH_VERSION', torch.__version__; assert torch.cuda.is_available(); print('复用镜像 PyTorch:', torch.__version__, 'torch CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))"
fi

"$PYTHON_BIN" -m pip install --upgrade \
  "vllm==$VLLM_VERSION" \
  "transformers==$TRANSFORMERS_VERSION" \
  "huggingface_hub[cli]"
"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" -c "import torch, vllm, transformers; print('PyTorch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('vLLM:', vllm.__version__); print('Transformers:', transformers.__version__)"
