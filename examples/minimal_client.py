"""Minimal example: point an OpenAI-compatible client at the AgentAblit relay.

This demonstrates the simplest integration — any OpenAI-compatible client can use
the relay by pointing its base_url at the AgentAblit endpoint.

Prerequisites:
  1. Start the relay:  cd src && python -m proxy.message_forward
  2. Configure config.yaml with your host + parasite model endpoints

The relay transparently applies trajectory stabilization: it forwards host output
when usable, recovers hedged output, and reconstructs from the parasite model when
the host fully stalls.
"""
import os

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI client: pip install openai")
    raise SystemExit(1)


# Point at the AgentAblit relay instead of the host model directly
ABLIT_BASE_URL = os.getenv("ABLIT_BASE_URL", "http://127.0.0.1:8787/v1")
ABLIT_API_KEY = os.getenv("ABLIT_API_KEY", "sk-placeholder")  # relay may not require auth

client = OpenAI(base_url=ABLIT_BASE_URL, api_key=ABLIT_API_KEY)


def demo_simple_chat():
    """Send a simple chat completion through the relay."""
    response = client.chat.completions.create(
        model="your-host-model",  # matches config.yaml host.model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        temperature=0.0,
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")


def demo_tool_calling():
    """Demonstrate tool-calling through the relay (the main use case)."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="your-host-model",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        tools=tools,
        tool_choice="auto",
    )

    choice = response.choices[0]
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"Text response: {choice.message.content}")


if __name__ == "__main__":
    print("=== AgentAblit Relay — Minimal Client Demo ===\n")
    print(f"Connecting to: {ABLIT_BASE_URL}\n")

    try:
        demo_simple_chat()
    except Exception as e:
        print(f"Error (expected if relay is not running): {e}")
        print("\nTo run this demo:")
        print("  1. cd src && python -m proxy.message_forward")
        print("  2. python examples/minimal_client.py")
