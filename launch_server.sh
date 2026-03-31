#!/bin/bash
# Launch vLLM server with GPT-OSS-120B optimized for H200

set -e

echo "======================================================================"
echo "Launching vLLM Server: GPT-OSS-120B on H200"
echo "======================================================================"

# H200-specific optimizations
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Enable debug logging
# export VLLM_LOGGING_LEVEL=DEBUG

# Model configuration
MODEL="openai/gpt-oss-120b"
TENSOR_PARALLEL=2  # Adjust based on your GPU count
PORT=8000

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL"
echo "  Port: $PORT"
echo "  Attention Backend: $VLLM_ATTENTION_BACKEND"
echo ""
echo "Starting server..."
echo "======================================================================"
echo ""

# Launch vLLM server
vllm serve $MODEL \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --port $PORT \
  --served-model-name gpt-oss-120b

# Note: Server will run in foreground. Press Ctrl+C to stop.
# To run in background, add '&' at the end and redirect output:
# vllm serve ... > vllm_server.log 2>&1 &
