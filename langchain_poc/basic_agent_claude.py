"""Minimal PoC: create a LangChain agent (Claude model) with one tool and invoke it."""

from dotenv import load_dotenv
from langchain.agents import create_agent

MODEL = "anthropic:claude-sonnet-5-5"


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's always sunny in {city}!"


def main() -> None:
    load_dotenv()  # reads ANTHROPIC_API_KEY from .env

    agent = create_agent(
        model=MODEL,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    prompt = "What's the weather in Mumbai?"
    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    print("Conversation:")
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    main()
