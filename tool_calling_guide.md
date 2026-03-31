# Tool Calling in vLLM: A Deep Dive

## Talk Overview
- **Duration:** 30 minutes
- **Audience:** Developers, ML Engineers, Platform Teams
- **Goal:** Understand tool calling concepts, implementation in vLLM, and performance considerations

---

## Table of Contents

1. [Introduction: What is Tool Calling?](#1-introduction-what-is-tool-calling-5-min)
2. [How Tool Calling Works](#2-how-tool-calling-works-5-min)
3. [Tool Calling in vLLM](#3-tool-calling-in-vllm-5-min)
4. [Client-Side vs Server-Side Execution](#4-client-side-vs-server-side-execution-5-min)
5. [Function Calling vs MCP](#5-function-calling-vs-mcp-5-min)
6. [Performance Considerations](#6-performance-considerations-3-min)
7. [Demo & Best Practices](#7-demo--best-practices-2-min)

---

## 1. Introduction: What is Tool Calling? (5 min)

### The Problem
LLMs are powerful but have limitations:
- ❌ Can't access real-time data (weather, stock prices)
- ❌ Can't perform calculations reliably
- ❌ Can't interact with external systems (databases, APIs)
- ❌ Knowledge cutoff date

### The Solution: Tool Calling
Give the LLM the ability to **request** external function execution.

```
User: "What's the weather in San Francisco?"

Traditional LLM:
└─► "I don't have access to real-time weather data..."

LLM with Tool Calling:
└─► Calls: get_weather(location="San Francisco")
└─► Gets result: {"temp": 68, "condition": "Sunny"}
└─► "The weather in San Francisco is 68°F and sunny!"
```

### Key Insight
The LLM doesn't execute tools - it **decides** which tool to call and with what arguments.

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Calling Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User Query ──► LLM ──► "I need to call get_weather()"    │
│                          with args: {location: "SF"}        │
│                              │                              │
│                              ▼                              │
│                    [Tool Execution]                         │
│                              │                              │
│                              ▼                              │
│   Final Response ◄── LLM ◄── Tool Result                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Real-World Use Cases
| Domain | Tool Examples |
|--------|--------------|
| **Customer Support** | Check order status, process refunds |
| **Data Analysis** | Query databases, generate charts |
| **Coding Assistants** | Run code, search documentation |
| **Personal Assistants** | Calendar access, email, smart home |
| **Enterprise** | CRM updates, ticket creation, approvals |

---

## 2. How Tool Calling Works (5 min)

### Step-by-Step Process

```
Step 1: Define Tools
────────────────────
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
}

Step 2: Send Query with Tools
─────────────────────────────
User: "What's the weather in Paris?"
+ Tools: [get_weather, get_time, ...]

Step 3: LLM Decides
───────────────────
LLM Output: {
  "tool_calls": [{
    "id": "call_123",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
    }
  }]
}

Step 4: Execute Tool
────────────────────
Result: {"temp": 22, "condition": "Cloudy"}

Step 5: Send Result Back
────────────────────────
Messages: [
  {role: "user", content: "What's the weather in Paris?"},
  {role: "assistant", tool_calls: [...]},
  {role: "tool", tool_call_id: "call_123", content: "{...}"}
]

Step 6: Final Response
──────────────────────
LLM: "The weather in Paris is 22°C and cloudy."
```

### Tool Choice Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `auto` | LLM decides whether to use tools | General queries |
| `required` | Must use at least one tool | Force tool usage |
| `none` | Cannot use tools | Text-only response |
| `{name: "func"}` | Must use specific tool | Deterministic flow |

### Parallel vs Sequential Tool Calls

**Sequential (Single Tool)**
```
Query: "What's Apple's stock price?"
└─► get_stock_price(ticker="AAPL")
```

**Parallel (Multiple Tools)**
```
Query: "Compare weather in Paris and London"
├─► get_weather(location="Paris")
└─► get_weather(location="London")
    (Both called simultaneously)
```

**⚠️ Important:** Not all models support parallel tool calls!

---

## 3. Tool Calling in vLLM (5 min)

### Enabling Tool Calling

```bash
vllm serve MODEL_NAME \
  --tool-call-parser openai \      # Parser for tool call format
  --enable-auto-tool-choice        # Let model decide tool usage
```

### Supported Tool Call Parsers

| Parser | Models | Format |
|--------|--------|--------|
| `openai` | gpt-oss, general | OpenAI-compatible |
| `hermes` | Hermes models | `<tool_call>...</tool_call>` |
| `llama` | Llama 3.1+ | Llama-specific format |
| `mistral` | Mistral models | Mistral format |
| `granite` | IBM Granite | Granite format |

### API Formats in vLLM

**1. Chat Completions API (Traditional)**
```python
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "..."}],
    tools=[...],
    tool_choice="auto"
)

# Access tool calls
tool_calls = response.choices[0].message.tool_calls
```

**2. Responses API (Newer)**
```python
response = client.responses.create(
    model="gpt-oss-120b",
    input="...",
    tools=[...],
    previous_response_id="..."  # For multi-turn
)
```

### Structured Output (Guided Generation)

vLLM can **constrain** tool call output to valid JSON:

```bash
vllm serve MODEL_NAME \
  --guided-decoding-backend outlines  # or lm-format-enforcer
```

| Without Structured Output | With Structured Output |
|---------------------------|------------------------|
| May generate invalid JSON | Always valid JSON |
| Wrong types possible | Matches schema |
| ~85-95% reliability | ~99%+ reliability |

### Stop Tokens and Tool Calling

**Critical concept for gpt-oss models:**

```
Model generates: <|call|>get_weather(...)<|call|>
                                         ↑
                                   STOP TOKEN!
                                   
Second tool call never generated.
```

This is why gpt-oss models **cannot** do native parallel tool calls.

---

## 4. Client-Side vs Server-Side Execution (5 min)

### Client-Side Tool Execution

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│   vLLM   │────▶│ Response │
│          │     │          │     │ w/ tools │
│          │◀────│          │◀────│          │
└──────────┘     └──────────┘     └──────────┘
     │
     │ Execute tool locally
     ▼
┌──────────┐
│  Tool    │
│ (Weather │
│   API)   │
└──────────┘
     │
     │ Send result back
     ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│   vLLM   │────▶│  Final   │
│ w/ result│     │          │     │ Response │
└──────────┘     └──────────┘     └──────────┘
```

**Characteristics:**
- ✅ Full control over tool execution
- ✅ Can handle authentication, rate limits
- ✅ Works with any vLLM setup
- ❌ Multiple round-trips
- ❌ Client must implement tool logic

### Server-Side Tool Execution (MCP)

```
┌──────────┐     ┌──────────────────────────┐     ┌──────────┐
│  Client  │────▶│         vLLM             │────▶│   MCP    │
│          │     │  ┌─────────────────┐     │     │  Server  │
│          │     │  │ LLM decides     │     │     │          │
│          │     │  │ tool call       │─────┼────▶│ Execute  │
│          │     │  │                 │◀────┼─────│ Return   │
│          │     │  │ Generate final  │     │     │          │
│          │◀────│  │ response        │     │     │          │
│          │     │  └─────────────────┘     │     │          │
└──────────┘     └──────────────────────────┘     └──────────┘
```

**Characteristics:**
- ✅ Single request from client
- ✅ Lower latency
- ✅ Simpler client code
- ❌ Requires MCP server setup
- ❌ Less control over execution

### Comparison Table

| Aspect | Client-Side | Server-Side (MCP) |
|--------|-------------|-------------------|
| Round trips | Multiple | Single |
| Latency | Higher | Lower |
| Client complexity | Higher | Lower |
| Security control | Full | Limited |
| Debugging | Easier | Harder |
| Setup complexity | Lower | Higher |

---

## 5. Function Calling vs MCP (5 min)

### Traditional Function Calling

**Definition:** Schema-based tool definitions passed to the model.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]
```

**Flow:**
1. Define tools as JSON schemas
2. Pass to LLM with query
3. LLM outputs tool call (JSON)
4. Client parses and executes
5. Client sends result back
6. LLM generates final response

### MCP (Model Context Protocol)

**Definition:** Standardized protocol for tool servers.

```python
tools = [{
    "type": "mcp",
    "server_label": "weather_service",
    "server_url": "http://mcp-server:8080",
    "allowed_tools": ["get_weather", "get_forecast"]
}]
```

**Key Features:**
- **Dynamic Tool Discovery:** Server advertises available tools
- **Standardized Protocol:** Any model can use any MCP server
- **Server-Side Execution:** vLLM connects directly to MCP server

### MCP Tool Filtering

```python
# Allow all tools
{"type": "mcp", "server_url": "..."}

# Explicit wildcard
{"type": "mcp", "server_url": "...", "allowed_tools": ["*"]}

# Specific tools only
{"type": "mcp", "server_url": "...", "allowed_tools": ["tool1", "tool2"]}

# Advanced filtering
{
    "type": "mcp",
    "server_url": "...",
    "allowed_tools": {
        "tool_names": ["read_file", "list_dir"],
        "read_only": True
    }
}
```

### Comparison

| Aspect | Function Calling | MCP |
|--------|------------------|-----|
| Tool definition | Static JSON schema | Dynamic from server |
| Execution | Client-side | Server-side (vLLM → MCP) |
| Discovery | Manual | Automatic |
| Standardization | Vendor-specific | Protocol standard |
| Multi-model | Requires adaptation | Universal |
| Setup complexity | Lower | Higher |
| Best for | Simple integrations | Complex tool ecosystems |

### When to Use Each

**Function Calling:**
- Simple, well-defined tools
- Full control needed
- Quick prototyping
- Single-model deployments

**MCP:**
- Complex tool ecosystems
- Multi-model environments
- Centralized tool management
- Enterprise deployments

---

## 6. Performance Considerations (3 min)

### Latency Breakdown

```
Tool Calling Request Timeline
─────────────────────────────────────────────────────────────────

│◄─────── LLM Inference ───────▶│◄─ Tool Exec ─▶│◄─ LLM ─▶│
│                               │               │         │
├───────────────────────────────┼───────────────┼─────────┤
0ms                           300ms           400ms     600ms

Components:
1. Initial LLM inference (decide tool): ~200-500ms
2. Tool execution: Variable (network, compute)
3. Final LLM response: ~100-300ms
```

### Throughput Under Concurrency

From benchmarking experiments:

| Concurrency | Latency (P50) | Latency (P99) | Throughput |
|-------------|---------------|---------------|------------|
| 1 | 350ms | 450ms | 2.8 req/s |
| 4 | 380ms | 620ms | 10.5 req/s |
| 8 | 420ms | 850ms | 18.2 req/s |
| 16 | 550ms | 1200ms | 28.1 req/s |

### Optimization Strategies

**1. Structured Output**
```bash
# Eliminates JSON parsing failures
--guided-decoding-backend outlines
```

**2. Batching**
```bash
# Increase batch size for throughput
--max-num-batched-tokens 8192
```

**3. Parallel Tool Execution**
```python
# If model returns multiple tool calls
# Execute them in parallel
with ThreadPoolExecutor() as executor:
    results = executor.map(execute_tool, tool_calls)
```

**4. Caching**
```python
# Cache common tool results
@lru_cache(maxsize=1000)
def get_weather(location: str) -> dict:
    ...
```

### Token Usage Patterns

```
Typical Tool Calling Request:
─────────────────────────────
System prompt:      ~100-500 tokens
Tool definitions:   ~200-1000 tokens  ◄── Can be significant!
User query:         ~20-100 tokens
─────────────────────────────────────
Total input:        ~320-1600 tokens

Output:
Tool call JSON:     ~50-200 tokens
Final response:     ~50-500 tokens
```

**Tip:** Keep tool descriptions concise but clear.

---

## 7. Demo & Best Practices (2 min)

### Quick Demo

```bash
# Terminal 1: Start vLLM
vllm serve openai/gpt-oss-120b \
  --tool-call-parser openai \
  --enable-auto-tool-choice

# Terminal 2: Run experiment
python experiment_1_basic.py -m gpt-oss-120b
```

### Best Practices

**1. Tool Design**
```python
# ✅ Good: Clear, specific description
{
    "name": "search_orders",
    "description": "Search customer orders by order ID, email, or date range. Returns order details including status and items.",
    "parameters": {...}
}

# ❌ Bad: Vague description
{
    "name": "search",
    "description": "Search for stuff",
    "parameters": {...}
}
```

**2. Error Handling**
```python
def execute_tool(name: str, args: dict) -> dict:
    try:
        result = TOOLS[name](**args)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**3. Validation**
```python
# Validate tool arguments before execution
from pydantic import BaseModel, ValidationError

class WeatherArgs(BaseModel):
    location: str
    unit: str = "celsius"

def get_weather(args: dict):
    validated = WeatherArgs(**args)  # Raises if invalid
    ...
```

**4. Timeouts**
```python
# Always set timeouts for tool execution
response = client.chat.completions.create(
    ...,
    timeout=60.0  # Don't hang forever
)
```

**5. Limit Tool Calls**
```python
# Prevent infinite loops
MAX_TOOL_CALLS = 10
for i in range(MAX_TOOL_CALLS):
    response = call_llm(messages)
    if not response.tool_calls:
        break
    # Execute tools...
```

---

## Key Takeaways

1. **Tool calling extends LLM capabilities** to real-world actions
2. **vLLM supports multiple formats** - use the right parser for your model
3. **Client-side vs Server-side** - choose based on control vs simplicity needs
4. **Function calling vs MCP** - MCP for complex ecosystems, functions for simple cases
5. **Structured output** dramatically improves reliability
6. **Performance varies** - benchmark your specific use case
7. **Model limitations exist** - gpt-oss can't do native parallel calls

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM MCP Example](https://docs.vllm.ai/en/latest/examples/online_serving/openai_responses_client_with_mcp_tools/)
- [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

---

## Q&A Prep

### Expected Questions

**Q: Why does gpt-oss show 90% on parallel benchmarks if it can't do parallel calls?**
A: The benchmark handler may be doing multi-turn iteration. Direct testing shows only 1 tool call per response due to `<|call|>` stop token.

**Q: Should I use MCP or function calling?**
A: Function calling for simple cases, MCP for complex tool ecosystems or multi-model environments.

**Q: How do I improve tool calling reliability?**
A: Enable structured output (`--guided-decoding-backend outlines`), write clear tool descriptions, validate arguments.

**Q: What's the performance impact of tool calling?**
A: Adds latency for tool execution + second LLM call. Throughput depends on concurrency and tool execution time.

**Q: Can I mix MCP and function tools?**
A: Yes! vLLM supports both in the same request.

---

## Appendix: Experiment Files Reference

| File | Purpose |
|------|---------|
| `experiment_1_basic.py` | Basic single tool call |
| `experiment_2_parallel.py` | Parallel tool calling test |
| `experiment_3_tool_choice.py` | Tool choice modes (auto/required/none) |
| `experiment_4_multiturn.py` | Multi-turn conversations |
| `experiment_5_benchmark.py` | Performance benchmarking with concurrency |
| `experiment_6_mcp_responses.py` | Responses API with MCP |
| `experiment_7_vllm_mcp_native.py` | Native vLLM MCP support |
| `mcp_test_server.py` | Test MCP server |
| `mcp_client_test.py` | MCP client validation |
