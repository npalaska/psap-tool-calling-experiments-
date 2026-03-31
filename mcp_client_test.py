#!/usr/bin/env python3
"""
MCP Client Test Script
Tests the MCP server directly and validates integration with vLLM
"""

import httpx
import json
import argparse
from typing import Dict, List, Any, Optional
from openai import OpenAI


class MCPClient:
    """Client for interacting with MCP server"""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
    
    def get_server_info(self) -> Dict:
        """Get server information"""
        response = self.client.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def list_tools(self) -> List[Dict]:
        """List available tools"""
        response = self.client.get(f"{self.base_url}/tools")
        response.raise_for_status()
        return response.json().get("tools", [])
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict:
        """Call a tool"""
        response = self.client.post(
            f"{self.base_url}/call",
            json={"name": name, "arguments": arguments}
        )
        response.raise_for_status()
        return response.json()
    
    def get_history(self) -> List[Dict]:
        """Get call history"""
        response = self.client.get(f"{self.base_url}/history")
        response.raise_for_status()
        return response.json().get("history", [])
    
    def rpc_call(self, method: str, params: Dict = None) -> Dict:
        """Make a JSON-RPC style call"""
        response = self.client.post(
            f"{self.base_url}/rpc",
            json={
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": 1
            }
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        self.client.close()


def convert_mcp_tools_to_openai(mcp_tools: List[Dict]) -> List[Dict]:
    """Convert MCP tool format to OpenAI tool format"""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"]
            }
        })
    return openai_tools


def test_mcp_server_directly(mcp_client: MCPClient):
    """Test MCP server directly without LLM"""
    print("\n" + "="*70)
    print("Test 1: Direct MCP Server Tests")
    print("="*70)
    
    # Test 1.1: Server info
    print("\n--- 1.1: Server Info ---")
    try:
        info = mcp_client.get_server_info()
        print(f"✓ Server: {info['server']['name']} v{info['server']['version']}")
        print(f"✓ Protocol: {info['server']['protocolVersion']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.2: List tools
    print("\n--- 1.2: List Tools ---")
    try:
        tools = mcp_client.list_tools()
        print(f"✓ Available tools: {len(tools)}")
        for tool in tools[:5]:
            print(f"  - {tool['name']}: {tool['description'][:40]}...")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.3: Call echo tool
    print("\n--- 1.3: Echo Tool ---")
    try:
        result = mcp_client.call_tool("echo", {"message": "Hello MCP!"})
        content = json.loads(result["content"][0]["text"])
        print(f"✓ Echo response: {content['message']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.4: Read file
    print("\n--- 1.4: Read File ---")
    try:
        result = mcp_client.call_tool("read_file", {"path": "/docs/readme.md"})
        content = json.loads(result["content"][0]["text"])
        print(f"✓ File content preview: {content['content'][:50]}...")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.5: Query database
    print("\n--- 1.5: Query Database ---")
    try:
        result = mcp_client.call_tool("query_database", {"table": "users", "limit": 2})
        content = json.loads(result["content"][0]["text"])
        print(f"✓ Found {content['total']} users")
        for user in content["results"]:
            print(f"  - {user['name']} ({user['email']})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.6: Calculate
    print("\n--- 1.6: Calculate ---")
    try:
        result = mcp_client.call_tool("calculate", {"expression": "2 + 2 * 3"})
        content = json.loads(result["content"][0]["text"])
        print(f"✓ 2 + 2 * 3 = {content['result']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.7: Search
    print("\n--- 1.7: Search ---")
    try:
        result = mcp_client.call_tool("search", {"query": "alice", "scope": "all"})
        content = json.loads(result["content"][0]["text"])
        print(f"✓ Search 'alice': {content['count']} results")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 1.8: JSON-RPC endpoint
    print("\n--- 1.8: JSON-RPC Endpoint ---")
    try:
        result = mcp_client.rpc_call("tools/list")
        tools_count = len(result.get("result", {}).get("tools", []))
        print(f"✓ RPC tools/list: {tools_count} tools")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n✓ All direct MCP server tests passed!")
    return True


def test_mcp_with_vllm(mcp_client: MCPClient, llm_client: OpenAI, model: str):
    """Test MCP integration with vLLM"""
    print("\n" + "="*70)
    print("Test 2: MCP + vLLM Integration")
    print("="*70)
    
    # Get tools from MCP server and convert to OpenAI format
    mcp_tools = mcp_client.list_tools()
    openai_tools = convert_mcp_tools_to_openai(mcp_tools)
    
    print(f"\n✓ Loaded {len(openai_tools)} tools from MCP server")
    
    # Test scenarios
    test_cases = [
        {
            "name": "File Reading",
            "query": "Read the file at /docs/readme.md and tell me what it contains",
            "expected_tool": "read_file"
        },
        {
            "name": "Database Query",
            "query": "Show me all users in the database",
            "expected_tool": "query_database"
        },
        {
            "name": "Calculation",
            "query": "Calculate 15 * 7 + 23",
            "expected_tool": "calculate"
        },
        {
            "name": "Search",
            "query": "Search for anything related to 'Widget' in the system",
            "expected_tool": "search"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- 2.{i}: {test['name']} ---")
        print(f"Query: {test['query']}")
        
        try:
            # Call vLLM with MCP tools
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test['query']}],
                tools=openai_tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"✓ LLM chose tool: {func_name}")
                print(f"  Arguments: {json.dumps(func_args)}")
                
                # Execute tool via MCP server
                mcp_result = mcp_client.call_tool(func_name, func_args)
                tool_output = json.loads(mcp_result["content"][0]["text"])
                
                print(f"✓ MCP result: {json.dumps(tool_output)[:100]}...")
                
                correct_tool = func_name == test['expected_tool']
                results.append({
                    "test": test['name'],
                    "passed": True,
                    "correct_tool": correct_tool,
                    "tool_used": func_name
                })
                
            else:
                print(f"✗ No tool call made")
                print(f"  Response: {message.content[:100]}...")
                results.append({
                    "test": test['name'],
                    "passed": False,
                    "reason": "No tool call"
                })
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "test": test['name'],
                "passed": False,
                "reason": str(e)
            })
    
    # Summary
    print("\n--- Integration Test Summary ---")
    passed = sum(1 for r in results if r['passed'])
    print(f"Passed: {passed}/{len(results)}")
    
    for r in results:
        status = "✓" if r['passed'] else "✗"
        extra = f" (used: {r.get('tool_used', 'N/A')})" if r['passed'] else f" ({r.get('reason', '')})"
        print(f"  {status} {r['test']}{extra}")
    
    return passed == len(results)


def test_multi_turn_with_mcp(mcp_client: MCPClient, llm_client: OpenAI, model: str):
    """Test multi-turn conversation using MCP tools"""
    print("\n" + "="*70)
    print("Test 3: Multi-Turn Conversation with MCP Tools")
    print("="*70)
    
    mcp_tools = mcp_client.list_tools()
    openai_tools = convert_mcp_tools_to_openai(mcp_tools)
    
    # Conversation scenario
    initial_query = """I need to do the following:
1. First, list all files in the /docs directory
2. Then read the api.md file
3. Finally, search for any mentions of 'API' in the database

Please do these tasks one by one."""
    
    print(f"\nUser: {initial_query}\n")
    
    messages = [{"role": "user", "content": initial_query}]
    max_turns = 10
    
    for turn in range(max_turns):
        print(f"--- Turn {turn + 1} ---")
        
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                print(f"\n✓ Final Response:\n{message.content[:300]}...")
                return True
            
            messages.append(message)
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"  → {func_name}({json.dumps(func_args)})")
                
                # Execute via MCP
                mcp_result = mcp_client.call_tool(func_name, func_args)
                tool_output = json.loads(mcp_result["content"][0]["text"])
                
                print(f"  ← {json.dumps(tool_output)[:80]}...")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_output)
                })
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    print("⚠ Max turns reached")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Test MCP server and vLLM integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test MCP server only
  python mcp_client_test.py --mcp-url http://localhost:8080

  # Test with vLLM integration
  python mcp_client_test.py --mcp-url http://localhost:8080 --vllm-url http://localhost:8000/v1 -m gpt-oss-120b

  # Run specific test
  python mcp_client_test.py --mcp-url http://localhost:8080 --test direct
        """
    )
    parser.add_argument("--mcp-url", type=str, default="http://localhost:8080",
                       help="MCP server URL")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL")
    parser.add_argument("-m", "--model", type=str, default=None,
                       help="Model name for vLLM (required for integration tests)")
    parser.add_argument("--api-key", type=str, default="test",
                       help="API key for vLLM")
    parser.add_argument("--test", type=str, 
                       choices=["all", "direct", "integration", "multi_turn"],
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print("="*70)
    print("MCP Client Test Suite")
    print("="*70)
    print(f"MCP Server: {args.mcp_url}")
    if args.model:
        print(f"vLLM Server: {args.vllm_url}")
        print(f"Model: {args.model}")
    
    mcp_client = MCPClient(args.mcp_url)
    llm_client = None
    
    if args.model:
        llm_client = OpenAI(
            base_url=args.vllm_url,
            api_key=args.api_key
        )
    
    results = {}
    
    try:
        # Test MCP server directly
        if args.test in ["all", "direct"]:
            results["direct"] = test_mcp_server_directly(mcp_client)
        
        # Test integration with vLLM
        if args.test in ["all", "integration"] and llm_client:
            results["integration"] = test_mcp_with_vllm(mcp_client, llm_client, args.model)
        elif args.test == "integration" and not llm_client:
            print("\n⚠ Skipping integration test: --model not specified")
        
        # Test multi-turn
        if args.test in ["all", "multi_turn"] and llm_client:
            results["multi_turn"] = test_multi_turn_with_mcp(mcp_client, llm_client, args.model)
        elif args.test == "multi_turn" and not llm_client:
            print("\n⚠ Skipping multi-turn test: --model not specified")
            
    finally:
        mcp_client.close()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} test suites passed")
    
    # Show MCP call history
    print("\n--- MCP Call History ---")
    history = mcp_client.get_history()
    for call in history[-10:]:
        status = "✓" if call.get("success") else "✗"
        print(f"  {status} {call['tool']}({json.dumps(call['arguments'])[:50]}...)")


if __name__ == "__main__":
    main()
