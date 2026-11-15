"""Minimal PoC: create a LangChain agent (Groq model) with no tools and invoke it."""

from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

MODEL = "openai/gpt-oss-120b"
ENV = dotenv_values(".env")  # reads keys straight from the project-root .env file


def main() -> None:
    # Groq exposes an OpenAI-compatible endpoint, so LangChain's ChatOpenAI can talk to it.
    model = ChatOpenAI(
        model=MODEL,
        api_key=ENV["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    agent = create_agent(
        model=model,
        system_prompt="You are a helpful assistant.",
    )

    prompt = "In three short bullet points, explain what an LLM token is."
    print(f"Model : groq / {MODEL}")
    print(f"Prompt: {prompt}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    print("Response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
