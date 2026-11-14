"""Minimal PoC: one chat call to Claude via the local Claude Subscription Adapter.

The adapter (running at localhost:9001) speaks the OpenAI Chat Completions API
and answers using your Claude Code subscription - no Anthropic API key needed.
"""

from openai import OpenAI

BASE_URL = "http://localhost:9001/v1"
MODEL = "sonnet"  # adapter alias -> claude-sonnet-5-5


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key="local")  # placeholder key

    prompt = "In three short bullet points, explain what an LLM token is."
    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")
    print("Response:")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)
    print(f"\nTokens -> input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}") # type: ignore


if __name__ == "__main__":
    main()
