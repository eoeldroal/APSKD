#!/bin/bash

set -euo pipefail

USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-1}
USE_VLLM=${USE_VLLM:-0}
CUDNN_VERSION=${CUDNN_VERSION:-9.16.0.29}

export MAX_JOBS=${MAX_JOBS:-48}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "Error: this installer requires Python 3.12, but found Python ${PYTHON_VERSION}."
    echo "Please activate a Python 3.12 environment and retry."
    echo "Example: conda create -n verl python=3.12 && conda activate verl"
    exit 1
fi

mkdir -p "$UV_CACHE_DIR"
echo "Using uv cache dir: $UV_CACHE_DIR"
echo "Using uv link mode: $UV_LINK_MODE"

echo "0. install uv"
python -m pip install uv

echo "1. install inference frameworks and pytorch they need"
if [ "$USE_SGLANG" -eq 1 ]; then
    python -m uv pip install "sglang[all]==0.5.9"
    python -m uv pip install torch-memory-saver
fi
if [ "$USE_VLLM" -eq 1 ]; then
    python -m uv pip install "vllm==0.19.0"
fi

echo "2. install basic packages"
python -m uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "python -m uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

python -m uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install FlashAttention and FlashInfer"
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3}
export FLASH_ATTN_VERSION
if [ -z "${FLASH_ATTN_WHEEL:-}" ]; then
    FLASH_ATTN_WHEEL=$(python - <<'PY'
import platform
import sys
import os

import torch

version = os.environ["FLASH_ATTN_VERSION"]
cuda_version = torch.version.cuda
if cuda_version is None:
    raise SystemExit("PyTorch was not built with CUDA; cannot select a flash-attn CUDA wheel.")
cuda_major = cuda_version.split(".")[0]
torch_major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
machine = platform.machine().lower()
arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)

print(
    f"flash_attn-{version}+cu{cuda_major}torch{torch_major_minor}"
    f"cxx11abi{abi}-{python_tag}-{python_tag}-linux_{arch}.whl"
)
PY
)
fi
FLASH_ATTN_WHEEL_URL=${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL//+/%2B}}
echo "Using FlashAttention wheel: ${FLASH_ATTN_WHEEL}"

if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
    if ! wget -nv -O "$FLASH_ATTN_WHEEL" "$FLASH_ATTN_WHEEL_URL"; then
        rm -f "$FLASH_ATTN_WHEEL"
        if [ "${FLASH_ATTN_SOURCE_FALLBACK:-0}" -eq 1 ]; then
            FLASH_ATTENTION_FORCE_BUILD=${FLASH_ATTENTION_FORCE_BUILD:-TRUE} \
                python -m uv pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-deps
        else
            echo "Failed to download ${FLASH_ATTN_WHEEL}."
            echo "Set FLASH_ATTN_WHEEL_URL to a compatible wheel or set FLASH_ATTN_SOURCE_FALLBACK=1 to build from source."
            exit 1
        fi
    fi
fi
if [ -f "$FLASH_ATTN_WHEEL" ]; then
    python -m uv pip install --no-deps "$FLASH_ATTN_WHEEL"
fi
python -m uv pip install flashinfer-python==0.6.3

if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    python -m uv pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch python -m uv pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    python -m uv pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi

echo "5. May need to fix opencv"
python -m uv pip install opencv-python
python -m uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "6. Install CuDNN python package (avoid being overridden)"
python -m uv pip install --no-deps --reinstall-package nvidia-cudnn-cu12 "nvidia-cudnn-cu12==${CUDNN_VERSION}"

echo "Successfully installed all packages"
