#!/usr/bin/env python3
"""
Experiment 7: vLLM Native MCP Tool Calling
Uses vLLM's built-in MCP support with the Responses API

Based on: https://docs.vllm.ai/en/latest/examples/online_serving/openai_responses_client_with_mcp_tools/

Setup Requirements:
1. Start vLLM with MCP enabled:
   vllm serve MODEL_NAME --tool-server demo
   
2. Set environment variables:
   export VLLM_ENABLE_RESPONSES_API_STORE=1
   export VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=code_interpreter,container
   export VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS=1

3. Start an MCP server (e.g., on port 8080):
   python mcp_test_server.py --port 8080
"""

from openai import OpenAI
import argparse
import json
from typing import Optional


def get_first_model(client: OpenAI) -> str:
    """Get the first available model from the server"""
    models = client.models.list()
    if models.data:
        return models.data[0].id
    raise RuntimeError("No models available on the server")


def test_mcp_no_filter(client: OpenAI, model: str, mcp_server_url: str):
    """Example with no allowed_tools filter - allows all tools"""
    print("\n" + "="*70)
    print("Test 1: No allowed_tools filter (allows all tools)")
    print("="*70)
    
    try:
        response = client.responses.create(
            model=model,
            input="Search the database for users and tell me what you find",
            instructions="Use the available tools to complete the task.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": mcp_server_url,
                    # No allowed_tools specified - all tools are available
                }
            ],
        )
        
        print(f"\n✓ Status: {response.status}")
        print(f"✓ Output: {response.output_text[:500] if response.output_text else 'N/A'}...")
        
        # Show any tool calls made
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'function_call':
                    print(f"\n  Tool called: {item.name}")
                    print(f"  Arguments: {item.arguments}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return False


def test_mcp_wildcard(client: OpenAI, model: str, mcp_server_url: str):
    """Example with allowed_tools=['*'] - explicitly allows all tools"""
    print("\n" + "="*70)
    print("Test 2: allowed_tools=['*'] (select all tools)")
    print("="*70)
    
    try:
        response = client.responses.create(
            model=model,
            input="Calculate 15 * 7 + 23 for me",
            instructions="Use the calculator tool to compute the result.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": mcp_server_url,
                    "allowed_tools": ["*"],  # Explicitly allow all tools
                }
            ],
        )
        
        print(f"\n✓ Status: {response.status}")
        print(f"✓ Output: {response.output_text[:500] if response.output_text else 'N/A'}...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return False


def test_mcp_specific_tools(client: OpenAI, model: str, mcp_server_url: str):
    """Example with specific allowed_tools list - filters available tools"""
    print("\n" + "="*70)
    print("Test 3: allowed_tools=['query_database', 'search'] (filter to specific tools)")
    print("="*70)
    
    try:
        response = client.responses.create(
            model=model,
            input="Search for information about 'Alice' in the system",
            instructions="Use the search tool to find information.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "test_server", 
                    "server_url": mcp_server_url,
                    # Only allow specific tools
                    "allowed_tools": ["query_database", "search"],
                }
            ],
        )
        
        print(f"\n✓ Status: {response.status}")
        print(f"✓ Output: {response.output_text[:500] if response.output_text else 'N/A'}...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return False


def test_mcp_object_format(client: OpenAI, model: str, mcp_server_url: str):
    """Example using object format for allowed_tools with more control"""
    print("\n" + "="*70)
    print("Test 4: allowed_tools with object format (advanced filtering)")
    print("="*70)
    
    try:
        response = client.responses.create(
            model=model,
            input="Read the file at /docs/readme.md and summarize it",
            instructions="Use the file system tools.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": mcp_server_url,
                    # Object format with more control
                    "allowed_tools": {
                        "tool_names": ["read_file", "list_files"],
                        "read_only": True,  # Only allow read operations
                    },
                }
            ],
        )
        
        print(f"\n✓ Status: {response.status}")
        print(f"✓ Output: {response.output_text[:500] if response.output_text else 'N/A'}...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return False


def test_mixed_tools(client: OpenAI, model: str, mcp_server_url: str):
    """Example mixing MCP tools with regular function tools"""
    print("\n" + "="*70)
    print("Test 5: Mixed MCP + Function tools")
    print("="*70)
    
    try:
        response = client.responses.create(
            model=model,
            input="Get the current time and also search the database for products",
            instructions="Use the available tools to complete both tasks.",
            tools=[
                # MCP tools
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": mcp_server_url,
                    "allowed_tools": ["query_database", "get_current_time"],
                },
                # Regular function tool
                {
                    "type": "function",
                    "function": {
                        "name": "format_output",
                        "description": "Format the output nicely",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to format"}
                            },
                            "required": ["text"]
                        }
                    }
                }
            ],
        )
        
        print(f"\n✓ Status: {response.status}")
        print(f"✓ Output: {response.output_text[:500] if response.output_text else 'N/A'}...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return False


def test_multi_turn_mcp(client: OpenAI, model: str, mcp_server_url: str):
    """Test multi-turn conversation with MCP tools"""
    print("\n" + "="*70)
    print("Test 6: Multi-turn conversation with MCP tools")
    print("="*70)
    
    try:
        # First turn
        print("\n--- Turn 1: Initial query ---")
        response1 = client.responses.create(
            model=model,
            input="List all files in the /docs directory",
            instructions="Use file system tools to explore.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": mcp_server_url,
                }
            ],
        )
        
        print(f"Status: {response1.status}")
        print(f"Output: {response1.output_text[:200] if response1.output_text else 'N/A'}...")
        
        # Second turn - continue conversation
        print("\n--- Turn 2: Follow-up ---")
        response2 = client.responses.create(
            model=model,
            input="Now read the api.md file you found",
            instructions="Continue helping with file exploration.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": mcp_server_url,
                }
            ],
            previous_response_id=response1.id,  # Continue the conversation
        )
        
        print(f"Status: {response2.status}")
        print(f"Output: {response2.output_text[:200] if response2.output_text else 'N/A'}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_responses_api(client: OpenAI) -> bool:
    """Check if Responses API is available"""
    try:
        # Check if client has responses attribute
        if not hasattr(client, 'responses'):
            print("✗ OpenAI client doesn't have 'responses' attribute")
            print("  Make sure you're using openai>=1.0.0 with Responses API support")
            return False
        return True
    except Exception as e:
        print(f"✗ Error checking Responses API: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM native MCP tool calling with Responses API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup Instructions:
  1. Start MCP test server:
     python mcp_test_server.py --port 8080

  2. Start vLLM with MCP support:
     export VLLM_ENABLE_RESPONSES_API_STORE=1
     vllm serve MODEL_NAME --tool-server demo

  3. Run this test:
     python experiment_7_vllm_mcp_native.py -m MODEL_NAME --mcp-url http://localhost:8080

Examples:
  python experiment_7_vllm_mcp_native.py -m gpt-oss-120b
  python experiment_7_vllm_mcp_native.py -m gpt-oss-120b --test specific
  python experiment_7_vllm_mcp_native.py -m gpt-oss-120b --mcp-url http://mcp-server:8080
        """
    )
    parser.add_argument("-m", "--model", type=str, default=None,
                       help="Model name (auto-detects if not specified)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL")
    parser.add_argument("--api-key", type=str, default="empty",
                       help="API key")
    parser.add_argument("--mcp-url", type=str, default="http://localhost:8080",
                       help="MCP server URL")
    parser.add_argument("--test", type=str,
                       choices=["all", "no_filter", "wildcard", "specific", "object", "mixed", "multi_turn"],
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Experiment 7: vLLM Native MCP Tool Calling")
    print("="*70)
    print(f"vLLM Server: {args.base_url}")
    print(f"MCP Server: {args.mcp_url}")
    
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    # Check Responses API availability
    if not check_responses_api(client):
        print("\n⚠ Responses API not available. Tests may fail.")
    
    # Get model
    if args.model:
        model = args.model
    else:
        try:
            model = get_first_model(client)
            print(f"Auto-detected model: {model}")
        except Exception as e:
            print(f"✗ Could not detect model: {e}")
            print("  Please specify --model explicitly")
            return
    
    print(f"Model: {model}")
    
    results = {}
    
    test_map = {
        "no_filter": lambda: test_mcp_no_filter(client, model, args.mcp_url),
        "wildcard": lambda: test_mcp_wildcard(client, model, args.mcp_url),
        "specific": lambda: test_mcp_specific_tools(client, model, args.mcp_url),
        "object": lambda: test_mcp_object_format(client, model, args.mcp_url),
        "mixed": lambda: test_mixed_tools(client, model, args.mcp_url),
        "multi_turn": lambda: test_multi_turn_mcp(client, model, args.mcp_url),
    }
    
    if args.test == "all":
        tests_to_run = list(test_map.keys())
    else:
        tests_to_run = [args.test]
    
    for test_name in tests_to_run:
        results[test_name] = test_map[test_name]()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} tests passed")
    
    print("\n" + "="*70)
    print("MCP Tool Configuration Reference:")
    print("="*70)
    print("""
  No filter (all tools):
    {"type": "mcp", "server_label": "...", "server_url": "..."}
    
  Wildcard (explicit all):
    {"type": "mcp", ..., "allowed_tools": ["*"]}
    
  Specific tools:
    {"type": "mcp", ..., "allowed_tools": ["tool1", "tool2"]}
    
  Object format (advanced):
    {"type": "mcp", ..., "allowed_tools": {"tool_names": [...], "read_only": True}}
""")


if __name__ == "__main__":
    main()
