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