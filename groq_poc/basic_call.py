"""Minimal PoC: one streaming API call to Groq, printed to the terminal."""

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-120b"


def main() -> None:
    load_dotenv()  # reads GROQ_API_KEY from .env
    client = Groq()

    prompt = "In three short bullet points, explain what an LLM token is."
    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")
    print("Response:")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
