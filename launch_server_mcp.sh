#!/bin/bash
# Launch vLLM server with MCP (Model Context Protocol) support
# Based on: https://docs.vllm.ai/en/latest/examples/online_serving/openai_responses_client_with_mcp_tools/

set -e

echo "======================================================================"
echo "Launching vLLM Server with MCP Support"
echo "======================================================================"

# H200-specific optimizations
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MCP and Responses API Configuration
export VLLM_ENABLE_RESPONSES_API_STORE=1
export VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=code_interpreter,container
export VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS=1

# Optional: Enable debug logging for troubleshooting
# export VLLM_LOGGING_LEVEL=DEBUG

# Model configuration
MODEL="${MODEL:-openai/gpt-oss-120b}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
PORT="${PORT:-8000}"

# MCP Tool Server (demo mode for testing)
TOOL_SERVER="${TOOL_SERVER:-demo}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL"
echo "  Port: $PORT"
echo "  Attention Backend: $VLLM_ATTENTION_BACKEND"
echo ""
echo "MCP Configuration:"
echo "  VLLM_ENABLE_RESPONSES_API_STORE: $VLLM_ENABLE_RESPONSES_API_STORE"
echo "  VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: $VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS"
echo "  VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: $VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS"
echo "  Tool Server: $TOOL_SERVER"
echo ""
echo "Starting server..."
echo "======================================================================"
echo ""

# Launch vLLM server with MCP support
vllm serve $MODEL \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --tool-server $TOOL_SERVER \
  --port $PORT \
  --served-model-name gpt-oss-120b

# Usage Examples:
# ---------------
# Default launch:
#   ./launch_server_mcp.sh
#
# Custom model:
#   MODEL=openai/gpt-oss-20b ./launch_server_mcp.sh
#
# Different port:
#   PORT=8001 ./launch_server_mcp.sh
#
# Custom MCP tool server:
#   TOOL_SERVER=http://localhost:8080 ./launch_server_mcp.sh
#
# After launching, test with:
#   python experiment_7_vllm_mcp_native.py -m gpt-oss-120b --mcp-url http://localhost:8080
