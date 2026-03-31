#!/usr/bin/env python3
"""
Experiment 3: Test different tool_choice modes
Compares auto, required, and none modes
"""

from openai import OpenAI
import json
import argparse

tools = [{
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search the knowledge database for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}]

def test_tool_choice(client: OpenAI, model: str, mode: str, user_query: str):
    """Test a specific tool_choice mode"""
    print(f"\n{'='*70}")
    print(f"Testing tool_choice='{mode}'")
    print(f"Query: {user_query}")
    print('='*70)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_query}
            ],
            tools=tools,
            tool_choice=mode
        )

        message = response.choices[0].message

        if message.tool_calls:
            print("✓ Tool calls made:")
            for tc in message.tool_calls:
                args = tc.function.arguments
                print(f"  - {tc.function.name}({args})")
        else:
            print("✗ No tool calls")
            if message.content:
                content = message.content[:200]
                print(f"Response: {content}{'...' if len(message.content) > 200 else ''}")

    except Exception as e:
        print(f"✗ ERROR: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test different tool_choice modes")
    parser.add_argument("-m", "--model", type=str, required=True,
                       help="Model name to use for inference")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                       help="Base URL for the API endpoint (default: http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default="test",
                       help="API key (default: test)")
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )

    print("="*70)
    print("Experiment 3: Tool Choice Modes")
    print(f"Model: {args.model}")
    print("="*70)

    test_cases = [
        ("auto", "Tell me about quantum computing"),
        ("required", "Tell me about quantum computing"),
        ("none", "Tell me about quantum computing"),
    ]

    for mode, query in test_cases:
        test_tool_choice(client, args.model, mode, query)

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print("- 'auto': Model decides whether to use tools")
    print("- 'required': Model MUST use at least one tool")
    print("- 'none': Model CANNOT use tools, text response only")

if __name__ == "__main__":
    main()
