#!/usr/bin/env python3
"""
Experiment 1: Basic tool calling
Tests single tool invocation
"""

from openai import OpenAI
import json
import argparse

# Define a simple tool
tools = [{
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, GOOGL"
                }
            },
            "required": ["ticker"]
        }
    }
}]

def get_stock_price(ticker: str) -> dict:
    """Mock implementation of stock price lookup"""
    mock_prices = {
        "AAPL": 178.50,
        "GOOGL": 142.30,
        "MSFT": 378.90,
        "NVDA": 875.20,
        "TSLA": 195.30
    }
    return {
        "ticker": ticker,
        "price": mock_prices.get(ticker.upper(), 100.00),
        "currency": "USD"
    }

def main():
    parser = argparse.ArgumentParser(description="Basic tool calling test")
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
    print("Experiment 1: Basic Tool Calling Test")
    print(f"Model: {args.model}")
    print("="*70)

    user_query = "What's the current price of Apple stock?"
    print(f"\nUser Query: {user_query}\n")

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "user", "content": user_query}
            ],
            tools=tools,
            tool_choice="auto"
        )

        print("Raw Response:")
        print(json.dumps(response.model_dump(), indent=2))
        print("\n" + "="*70)

        message = response.choices[0].message

        if message.tool_calls:
            print("\n✓ SUCCESS: Tool calling detected!")
            print("-"*70)

            for i, tool_call in enumerate(message.tool_calls, 1):
                print(f"\nTool Call #{i}:")
                print(f"  ID: {tool_call.id}")
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")

                args_dict = json.loads(tool_call.function.arguments)
                result = get_stock_price(**args_dict)

                print(f"\n  Execution Result:")
                print(f"    {json.dumps(result, indent=4)}")
        else:
            print("\n✗ FAILURE: No tool calls detected")
            print(f"Model Response: {message.content}")

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
