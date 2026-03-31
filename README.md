# vLLM Tool Calling Experiments

This directory contains experiments for testing tool calling with vLLM and GPT-OSS-120B on H200 GPUs.

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
chmod +x setup_h200.sh
./setup_h200.sh

# Activate virtual environment
source vllm-env/bin/activate
```

### 2. Launch vLLM Server

```bash
# Make launch script executable
chmod +x launch_server.sh

# Start the server
./launch_server.sh
```

The server will start on `http://localhost:8000` by default.

### 3. Run Experiments

In a new terminal (with virtual environment activated):

```bash
# Make experiments executable
chmod +x experiment_*.py

# Run basic tool calling test
python experiment_1_basic.py

# Run parallel tool calling test
python experiment_2_parallel.py

# Test different tool_choice modes
python experiment_3_tool_choice.py

# Multi-turn conversation with tools
python experiment_4_multiturn.py

# Performance benchmark
python experiment_5_benchmark.py -n 50  # 50 requests
```

## Experiments Overview

### Experiment 1: Basic Tool Calling
**File**: `experiment_1_basic.py`

Tests single tool invocation with a stock price lookup function.

**What it tests**:
- Basic tool definition
- Tool call detection
- Argument parsing
- Function execution

**Expected output**: Model should call `get_stock_price` with ticker "AAPL"

---

### Experiment 2: Parallel Tool Calling
**File**: `experiment_2_parallel.py`

Tests if the model can invoke multiple tools simultaneously.

**What it tests**:
- Multiple tool definitions
- Parallel function calling
- Multi-tool orchestration

**Expected output**: Model should call multiple functions (weather, time, population) in parallel

---

### Experiment 3: Tool Choice Modes
**File**: `experiment_3_tool_choice.py`

Compares different `tool_choice` parameter values.

**What it tests**:
- `auto`: Model decides whether to use tools
- `required`: Model must use at least one tool
- `none`: Model cannot use tools

**Expected output**: Different behavior for each mode

---

### Experiment 4: Multi-Turn Conversation
**File**: `experiment_4_multiturn.py`

Simulates a realistic conversation with multiple tool calls across turns.

**What it tests**:
- Conversation state management
- Multiple sequential tool calls
- Tool result integration
- Complex task completion

**Expected output**: Model searches products, gets details, checks inventory in sequence

---

### Experiment 5: Performance Benchmark
**File**: `experiment_5_benchmark.py`

Measures performance metrics for tool calling.

**What it measures**:
- Latency (mean, median, p95, p99)
- Throughput (requests/second)
- Success rate
- Token usage

**Usage**:
```bash
python experiment_5_benchmark.py -n 100 -w 5
# -n: number of requests (default: 20)
# -w: warmup requests (default: 3)
```

---

## Server Configuration

### Basic Launch
```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

### Optimized for H200
```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --port 8000
```

### Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--tensor-parallel-size` | Number of GPUs for model parallelism | 2 (for 120B model) |
| `--dtype` | Data type for inference | `bfloat16` |
| `--max-model-len` | Maximum context length | `32768` |
| `--gpu-memory-utilization` | GPU memory usage ratio | `0.90-0.92` |
| `--tool-call-parser` | Parser for tool calls | `openai` |
| `--enable-auto-tool-choice` | Enable automatic tool selection | Required |

---

## Troubleshooting

### Issue: Tool calls not being parsed

**Symptoms**: Response has text but no `tool_calls` array

**Solutions**:
1. Verify `--tool-call-parser openai` is set
2. Check `--enable-auto-tool-choice` is enabled
3. Try using Completions API instead of Chat API
4. Validate tool definitions match OpenAI schema

**Test**:
```bash
curl http://localhost:8000/v1/models
```

---

### Issue: Out of memory errors

**Symptoms**: CUDA OOM during model loading or inference

**Solutions**:
```bash
# Reduce memory utilization
--gpu-memory-utilization 0.85

# Reduce context length
--max-model-len 16384

# Increase tensor parallelism (requires more GPUs)
--tensor-parallel-size 4
```

---

### Issue: Slow inference

**Solutions**:
```bash
# Enable chunked prefill
--enable-chunked-prefill

# Adjust batch size
--max-num-batched-tokens 16384

# Use performance mode
--performance-mode throughput
```

---

### Issue: Server won't start

**Check**:
1. Driver version: `nvidia-smi` (should be 550+)
2. vLLM installation: `python -c "import vllm; print(vllm.__version__)"`
3. CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Port availability: `lsof -i :8000`

---

## Testing Server Health

```bash
# Check server is running
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Simple completion test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

---

## Expected Performance (H200)

Based on benchmarks, you can expect:

- **Latency**: 100-500ms per request (depending on input/output length)
- **Throughput**: 2-10 requests/second (with tool calling)
- **GPU Memory**: ~120GB per GPU for 2-way tensor parallelism
- **Context Length**: Up to 32K tokens efficiently

---

## Next Steps

1. **Customize Tools**: Modify experiments to test your own functions
2. **Production Integration**: Build real applications using the patterns
3. **Performance Tuning**: Adjust parameters based on your workload
4. **Monitoring**: Add logging and metrics collection
5. **Scale Testing**: Test with concurrent requests

---

## Additional Resources

- Main guide: `../vllm-tool-calling-guide.md`
- vLLM docs: https://docs.vllm.ai/
- GPT-OSS model: https://huggingface.co/openai/gpt-oss-120b
