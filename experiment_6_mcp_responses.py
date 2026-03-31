#!/usr/bin/env python3
"""
Experiment 6: MCP Tool Calling with vLLM Responses API
Tests the newer OpenAI Responses API format with MCP-style tool calls
"""

import httpx
import json
import argparse
from typing import List, Dict, Optional, Any


class ResponsesAPIClient:
    """Client for OpenAI Responses API (newer format)"""
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
    
    def create_response(
        self,
        model: str,
        input_items: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        previous_response_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Create a response using the Responses API
        
        Args:
            model: Model name
            input_items: List of input items (messages)
            tools: List of tool definitions
            tool_choice: Tool choice mode (auto, required, none)
            previous_response_id: ID of previous response for multi-turn
        """
        payload = {
            "model": model,
            "input": input_items,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        
        payload.update(kwargs)
        
        response = self.client.post(
            f"{self.base_url}/responses",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        self.client.close()


# MCP-style tool definitions
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the knowledge database for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "execute_code",
            "description": "Execute Python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write contents to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["path", "content"]
            }
        }
    }
]


# Mock tool implementations
def search_database(query: str, max_results: int = 5) -> Dict:
    """Mock database search"""
    results = [
        {"id": 1, "title": f"Result for '{query}' #1", "relevance": 0.95},
        {"id": 2, "title": f"Result for '{query}' #2", "relevance": 0.87},
        {"id": 3, "title": f"Result for '{query}' #3", "relevance": 0.72},
    ]
    return {"results": results[:max_results], "total": len(results)}


def execute_code(code: str, timeout: int = 30) -> Dict:
    """Mock code execution"""
    return {
        "status": "success",
        "output": f"Executed: {code[:50]}...",
        "execution_time": 0.5
    }


def read_file(path: str, encoding: str = "utf-8") -> Dict:
    """Mock file read"""
    return {
        "path": path,
        "content": f"Mock content of {path}",
        "size": 1024
    }


def write_file(path: str, content: str) -> Dict:
    """Mock file write"""
    return {
        "path": path,
        "bytes_written": len(content),
        "status": "success"
    }


TOOL_IMPLEMENTATIONS = {
    "search_database": search_database,
    "execute_code": execute_code,
    "read_file": read_file,
    "write_file": write_file
}


def extract_tool_calls(response: Dict) -> List[Dict]:
    """Extract tool calls from Responses API response"""
    tool_calls = []
    
    output = response.get("output", [])
    for item in output:
        if item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id", item.get("id")),
                "name": item.get("name"),
                "arguments": item.get("arguments", {})
            })
        elif item.get("type") == "tool_calls":
            for tc in item.get("tool_calls", []):
                tool_calls.append({
                    "id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "arguments": json.loads(tc.get("function", {}).get("arguments", "{}"))
                })
    
    return tool_calls


def extract_text_content(response: Dict) -> str:
    """Extract text content from Responses API response"""
    output = response.get("output", [])
    text_parts = []
    
    for item in output:
        if item.get("type") == "message":
            content = item.get("content", [])
            for c in content:
                if c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
        elif item.get("type") == "text":
            text_parts.append(item.get("text", ""))
    
    return "\n".join(text_parts)


def test_basic_responses_api(client: ResponsesAPIClient, model: str):
    """Test basic Responses API call"""
    print("\n" + "="*70)
    print("Test 1: Basic Responses API Call")
    print("="*70)
    
    try:
        response = client.create_response(
            model=model,
            input_items=[
                {"type": "message", "role": "user", "content": "Hello! What can you help me with?"}
            ]
        )
        
        print(f"\n✓ Response ID: {response.get('id', 'N/A')}")
        print(f"✓ Model: {response.get('model', 'N/A')}")
        
        text = extract_text_content(response)
        if text:
            print(f"\n✓ Response text:\n{text[:500]}...")
        
        print("\n✓ SUCCESS: Basic Responses API working")
        return True
        
    except httpx.HTTPStatusError as e:
        print(f"\n✗ HTTP Error: {e.response.status_code}")
        print(f"  Response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        return False


def test_tool_calling(client: ResponsesAPIClient, model: str):
    """Test tool calling with Responses API"""
    print("\n" + "="*70)
    print("Test 2: Tool Calling with Responses API")
    print("="*70)
    
    try:
        response = client.create_response(
            model=model,
            input_items=[
                {
                    "type": "message",
                    "role": "user", 
                    "content": "Search the database for information about machine learning algorithms"
                }
            ],
            tools=MCP_TOOLS,
            tool_choice="auto"
        )
        
        print(f"\n✓ Response ID: {response.get('id', 'N/A')}")
        
        tool_calls = extract_tool_calls(response)
        
        if tool_calls:
            print(f"\n✓ Tool calls detected: {len(tool_calls)}")
            for i, tc in enumerate(tool_calls, 1):
                print(f"\n  Tool Call #{i}:")
                print(f"    ID: {tc['id']}")
                print(f"    Function: {tc['name']}")
                print(f"    Arguments: {json.dumps(tc['arguments'], indent=6)}")
            print("\n✓ SUCCESS: Tool calling working")
            return True, response, tool_calls
        else:
            text = extract_text_content(response)
            print(f"\n⚠ No tool calls detected")
            print(f"  Text response: {text[:200]}...")
            return False, response, []
            
    except httpx.HTTPStatusError as e:
        print(f"\n✗ HTTP Error: {e.response.status_code}")
        print(f"  Response: {e.response.text[:500]}")
        return False, None, []
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, []


def test_multi_turn_with_tools(client: ResponsesAPIClient, model: str):
    """Test multi-turn conversation with tool execution"""
    print("\n" + "="*70)
    print("Test 3: Multi-Turn Conversation with Tool Execution")
    print("="*70)
    
    conversation = []
    response_id = None
    max_turns = 5
    
    initial_query = "I need to search for Python tutorials, then read a file at /docs/python_intro.md"
    print(f"\nUser: {initial_query}")
    
    conversation.append({
        "type": "message",
        "role": "user",
        "content": initial_query
    })
    
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")
        
        try:
            response = client.create_response(
                model=model,
                input_items=conversation,
                tools=MCP_TOOLS,
                tool_choice="auto",
                previous_response_id=response_id
            )
            
            response_id = response.get("id")
            tool_calls = extract_tool_calls(response)
            
            if not tool_calls:
                text = extract_text_content(response)
                print(f"\n✓ Final Response:\n{text[:500]}")
                print("\n✓ SUCCESS: Multi-turn conversation completed")
                return True
            
            print(f"Tool calls: {len(tool_calls)}")
            
            # Execute tools and add results to conversation
            for tc in tool_calls:
                func_name = tc["name"]
                args = tc["arguments"]
                
                print(f"\n  → Executing {func_name}({json.dumps(args)})")
                
                if func_name in TOOL_IMPLEMENTATIONS:
                    result = TOOL_IMPLEMENTATIONS[func_name](**args)
                    print(f"  ← Result: {json.dumps(result, indent=6)}")
                    
                    # Add tool result to conversation (Responses API format)
                    conversation.append({
                        "type": "function_call_output",
                        "call_id": tc["id"],
                        "output": json.dumps(result)
                    })
                else:
                    print(f"  ✗ Unknown function: {func_name}")
                    conversation.append({
                        "type": "function_call_output",
                        "call_id": tc["id"],
                        "output": json.dumps({"error": f"Unknown function: {func_name}"})
                    })
                    
        except httpx.HTTPStatusError as e:
            print(f"\n✗ HTTP Error: {e.response.status_code}")
            print(f"  Response: {e.response.text[:500]}")
            return False
        except Exception as e:
            print(f"\n✗ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n⚠ Max turns reached")
    return False


def test_mcp_style_tools(client: ResponsesAPIClient, model: str):
    """Test MCP-style tool definitions"""
    print("\n" + "="*70)
    print("Test 4: MCP-Style Tool Definitions")
    print("="*70)
    
    # MCP uses a specific format for tool definitions
    mcp_tools = [
        {
            "type": "mcp",
            "server_label": "filesystem",
            "server_url": "mcp://localhost:8080/filesystem",
            "allowed_tools": ["read_file", "write_file", "list_directory"]
        }
    ]
    
    print("\nNote: MCP server tools require actual MCP server support in vLLM")
    print("This test checks if vLLM accepts MCP-style tool definitions")
    
    try:
        response = client.create_response(
            model=model,
            input_items=[
                {"type": "message", "role": "user", "content": "List files in the current directory"}
            ],
            tools=mcp_tools,
            tool_choice="auto"
        )
        
        print(f"\n✓ Response received (ID: {response.get('id', 'N/A')})")
        print("✓ vLLM accepted MCP-style tool definitions")
        
        return True
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            error_detail = e.response.json().get("detail", "")
            if "mcp" in error_detail.lower() or "unknown" in error_detail.lower():
                print(f"\n⚠ vLLM does not support MCP-style tools yet")
                print(f"  Error: {error_detail[:200]}")
            else:
                print(f"\n✗ HTTP Error: {e.response.status_code}")
                print(f"  Response: {e.response.text[:500]}")
        else:
            print(f"\n✗ HTTP Error: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        return False


def check_responses_api_available(client: ResponsesAPIClient) -> bool:
    """Check if Responses API is available"""
    try:
        # Try to hit the responses endpoint with minimal payload
        response = client.client.post(
            f"{client.base_url}/responses",
            json={"model": "test", "input": []}
        )
        # Even an error response means the endpoint exists
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code in [400, 422]:  # Bad request means endpoint exists
            return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test MCP tool calling with vLLM Responses API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python experiment_6_mcp_responses.py -m gpt-oss-120b

  # With custom endpoint
  python experiment_6_mcp_responses.py -m gpt-oss-120b --base-url http://localhost:8000/v1

  # Run specific test
  python experiment_6_mcp_responses.py -m gpt-oss-120b --test basic
        """
    )
    parser.add_argument("-m", "--model", type=str, required=True,
                       help="Model name to use for inference")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                       help="Base URL for the API endpoint")
    parser.add_argument("--api-key", type=str, default="test",
                       help="API key")
    parser.add_argument("-t", "--timeout", type=float, default=60.0,
                       help="Request timeout in seconds")
    parser.add_argument("--test", type=str, choices=["all", "basic", "tools", "multi_turn", "mcp"],
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Experiment 6: MCP Tool Calling with Responses API")
    print(f"Model: {args.model}")
    print(f"Endpoint: {args.base_url}")
    print("="*70)
    
    client = ResponsesAPIClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout
    )
    
    # Check if Responses API is available
    print("\nChecking Responses API availability...")
    if not check_responses_api_available(client):
        print("\n✗ Responses API not available at this endpoint")
        print("  Make sure vLLM is running with Responses API support")
        print("  The endpoint should be: {base_url}/responses")
        return
    
    print("✓ Responses API endpoint found")
    
    results = {}
    
    try:
        if args.test in ["all", "basic"]:
            results["basic"] = test_basic_responses_api(client, args.model)
        
        if args.test in ["all", "tools"]:
            success, _, _ = test_tool_calling(client, args.model)
            results["tools"] = success
        
        if args.test in ["all", "multi_turn"]:
            results["multi_turn"] = test_multi_turn_with_tools(client, args.model)
        
        if args.test in ["all", "mcp"]:
            results["mcp"] = test_mcp_style_tools(client, args.model)
        
    finally:
        client.close()
    
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


if __name__ == "__main__":
    main()
