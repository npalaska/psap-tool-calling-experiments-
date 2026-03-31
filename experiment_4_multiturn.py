#!/usr/bin/env python3
"""
Experiment 4: Multi-turn conversation with tools
Simulates a realistic conversation flow with multiple tool invocations
"""

from openai import OpenAI
import json
import argparse

# Define multiple tools for a product search scenario
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the inventory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for products"
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category filter"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Unique product identifier"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check if a product is in stock and get quantity",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Unique product identifier"
                    }
                },
                "required": ["product_id"]
            }
        }
    }
]

# Mock implementations
def search_products(query: str, category: str = None) -> dict:
    """Mock product search"""
    products = {
        "laptop": [
            {"id": "LAPTOP001", "name": "MacBook Pro 16", "price": 2499},
            {"id": "LAPTOP002", "name": "ThinkPad X1", "price": 1899}
        ],
        "phone": [
            {"id": "PHONE001", "name": "iPhone 15 Pro", "price": 999},
            {"id": "PHONE002", "name": "Galaxy S24", "price": 849}
        ]
    }

    for key, items in products.items():
        if query.lower() in key:
            return {"products": items, "count": len(items)}

    return {"products": [], "count": 0}

def get_product_details(product_id: str) -> dict:
    """Mock product details"""
    details = {
        "LAPTOP001": {
            "id": "LAPTOP001",
            "name": "MacBook Pro 16",
            "price": 2499,
            "specs": "M3 Max, 36GB RAM, 1TB SSD",
            "rating": 4.8
        },
        "LAPTOP002": {
            "id": "LAPTOP002",
            "name": "ThinkPad X1",
            "price": 1899,
            "specs": "Intel i7, 32GB RAM, 512GB SSD",
            "rating": 4.6
        }
    }
    return details.get(product_id, {"error": "Product not found"})

def check_inventory(product_id: str) -> dict:
    """Mock inventory check"""
    inventory = {
        "LAPTOP001": {"in_stock": True, "quantity": 15, "warehouse": "US-West"},
        "LAPTOP002": {"in_stock": True, "quantity": 8, "warehouse": "US-East"},
        "PHONE001": {"in_stock": False, "quantity": 0, "warehouse": None}
    }
    return inventory.get(product_id, {"in_stock": False, "quantity": 0})

# Function registry
FUNCTIONS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "check_inventory": check_inventory
}

def run_conversation(client: OpenAI, model: str, user_input: str):
    """Run a complete multi-turn conversation with tool calls"""
    messages = [{"role": "user", "content": user_input}]

    iteration = 0
    max_iterations = 10

    print(f"\nUser: {user_input}\n")
    print("="*70)

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if not message.tool_calls:
                print(f"\n✓ Final Response:\n{message.content}")
                return message.content

            print(f"Tool calls requested: {len(message.tool_calls)}")
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"\n  → Calling {func_name}({json.dumps(func_args)})")

                if func_name in FUNCTIONS:
                    result = FUNCTIONS[func_name](**func_args)
                    print(f"  ← Result: {json.dumps(result, indent=6)}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                else:
                    print(f"  ✗ Unknown function: {func_name}")

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    print("\n⚠ Max iterations reached without final response")
    return None

def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversation with tools")
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
    print("Experiment 4: Multi-Turn Tool Conversation")
    print(f"Model: {args.model}")
    print("="*70)

    query = "I need a laptop for development work. Can you find options and tell me which ones are in stock?"

    result = run_conversation(client, args.model, query)

    print("\n" + "="*70)
    print("Conversation Complete")
    print("="*70)

if __name__ == "__main__":
    main()
