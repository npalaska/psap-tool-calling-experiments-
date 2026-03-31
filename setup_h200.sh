#!/bin/bash
# Setup script for running vLLM with GPT-OSS-120B on H200

set -e

echo "======================================================================"
echo "vLLM Setup for H200 with GPT-OSS-120B"
echo "======================================================================"

# Check NVIDIA driver
echo -e "\n[1/6] Checking NVIDIA driver..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver version: $DRIVER_VERSION"

if [[ "${DRIVER_VERSION%%.*}" -lt 550 ]]; then
    echo "⚠ WARNING: Driver version should be 550+ for H200. Current: $DRIVER_VERSION"
else
    echo "✓ Driver version OK"
fi

# Set H200-specific environment variables
echo -e "\n[2/6] Setting H200 optimizations..."
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✓ Environment variables set:"
echo "  VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "  VLLM_USE_TRITON_FLASH_ATTN=$VLLM_USE_TRITON_FLASH_ATTN"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Check Python version
echo -e "\n[3/6] Checking Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
VENV_DIR="vllm-env"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n[4/6] Creating virtual environment..."
    python3 -m venv $VENV_DIR
    echo "✓ Virtual environment created: $VENV_DIR"
else
    echo -e "\n[4/6] Virtual environment already exists: $VENV_DIR"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate
echo "✓ Virtual environment activated"

# Install vLLM
echo -e "\n[5/6] Installing vLLM..."
pip install --upgrade pip
pip install vllm
pip install openai  # For client examples
echo "✓ vLLM installed"

# Verify installation
echo -e "\n[6/6] Verifying installation..."
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

echo -e "\n======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "To start the vLLM server with GPT-OSS-120B, run:"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo "  ./launch_server.sh"
echo ""
echo "Or manually:"
echo ""
echo "  vllm serve openai/gpt-oss-120b \\"
echo "    --tensor-parallel-size 2 \\"
echo "    --dtype bfloat16 \\"
echo "    --max-model-len 32768 \\"
echo "    --gpu-memory-utilization 0.92 \\"
echo "    --enable-chunked-prefill \\"
echo "    --max-num-batched-tokens 8192 \\"
echo "    --tool-call-parser openai \\"
echo "    --enable-auto-tool-choice \\"
echo "    --port 8000"
echo ""
echo "======================================================================"
