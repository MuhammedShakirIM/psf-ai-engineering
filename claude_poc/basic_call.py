"""Minimal PoC: one API call to Claude Sonnet 5.5, printed to the terminal."""

import anthropic
from dotenv import load_dotenv

MODEL = "claude-sonnet-5-5"


def main() -> None:
    load_dotenv()  # reads ANTHROPIC_API_KEY from a .env file if present
    client = anthropic.Anthropic()


    prompt = "In three short bullet points, explain what an LLM token is."
    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if Sonnet 5.5 declines, the API retries on a fallback model
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        print("Claude declined this request.")
        return

    print("Response:")
    for block in response.content:
        if block.type == "text":
            print(block.text)

    

    print("\nToken usage:")
    print(f"  input tokens : {response.usage.input_tokens}")
    print(f"  output tokens: {response.usage.output_tokens}")
    print(f"  stop reason  : {response.stop_reason}")


if __name__ == "__main__":
    main()
