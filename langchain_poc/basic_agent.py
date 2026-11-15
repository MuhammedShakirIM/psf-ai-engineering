"""Minimal PoC: create a LangChain agent (Groq model) with one tool and invoke it."""

from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

MODEL = "openai/gpt-oss-120b"
ENV = dotenv_values(".env")  # reads keys straight from the project-root .env file


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's always sunny in {city}!"


def main() -> None:
    # Groq exposes an OpenAI-compatible endpoint, so LangChain's ChatOpenAI can talk to it.
    model = ChatOpenAI(
        model=MODEL,
        api_key=ENV["GROQ_API_KEY"], # pyright: ignore[reportArgumentType]
        base_url="https://api.groq.com/openai/v1",
    )

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    prompt = "What's the weather in Mumbai?"
    print(f"Model : groq / {MODEL}")
    print(f"Prompt: {prompt}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    print("Conversation:")
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    main()
