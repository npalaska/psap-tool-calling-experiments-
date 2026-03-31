#!/usr/bin/env python3
"""
Experiment 2: Parallel tool calling
Tests if model can invoke multiple tools simultaneously
"""

from openai import OpenAI
import json
import argparse

# Define multiple tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. London, Tokyo"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time for a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name, e.g. Europe/London, Asia/Tokyo"
                    }
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get population of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(location: str) -> dict:
    """Mock weather data"""
    weather_data = {
        "London": {"temp": 15, "condition": "Rainy"},
        "Tokyo": {"temp": 22, "condition": "Sunny"},
        "New York": {"temp": 18, "condition": "Cloudy"},
        "Paris": {"temp": 17, "condition": "Partly Cloudy"}
    }
    return weather_data.get(location, {"temp": 20, "condition": "Unknown"})

def get_time(timezone: str) -> dict:
    """Mock time data"""
    times = {
        "Europe/London": "14:30",
        "Asia/Tokyo": "22:30",
        "America/New_York": "09:30",
        "Europe/Paris": "15:30"
    }
    return {"timezone": timezone, "current_time": times.get(timezone, "12:00")}

def get_population(city: str) -> dict:
    """Mock population data"""
    populations = {
        "London": 9_000_000,
        "Tokyo": 14_000_000,
        "New York": 8_300_000,
        "Paris": 2_100_000
    }
    return {"city": city, "population": populations.get(city, 1_000_000)}

FUNCTION_MAP = {
    "get_weather": get_weather,
    "get_time": get_time,
    "get_population": get_population
}

def main():
    parser = argparse.ArgumentParser(description="Parallel tool calling test")
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
    print("Experiment 2: Parallel Tool Calling Test")
    print(f"Model: {args.model}")
    print("="*70)

    user_query = "What's the weather, current time, and population in London?"
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

        message = response.choices[0].message

        if message.tool_calls:
            num_calls = len(message.tool_calls)
            print(f"✓ Number of parallel tool calls: {num_calls}")
            print("-"*70)

            if num_calls > 1:
                print("\n✓ SUCCESS: Parallel tool calling is working!")
            else:
                print("\n⚠ PARTIAL: Only one tool called (expected multiple)")

            for i, tool_call in enumerate(message.tool_calls, 1):
                print(f"\nTool Call #{i}:")
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")

                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name in FUNCTION_MAP:
                    result = FUNCTION_MAP[func_name](**func_args)
                    print(f"  Result: {json.dumps(result, indent=4)}")

        else:
            print("✗ FAILURE: No tool calls detected")
            print(f"Model Response: {message.content}")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
