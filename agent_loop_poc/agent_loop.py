"""PoC: a hand-written agent loop with 3 tools, via the local Claude Subscription Adapter.

Each iteration prints what is sent to the model and what comes back
(a tool call, or the final answer / end of turn).
"""

import json
import random

from openai import OpenAI

BASE_URL = "http://localhost:9001/v1"
MODEL = "sonnet"  # adapter alias -> claude-sonnet-5-5
MAX_ITERATIONS = 10


# ---------------------------------------------------------------- tools (fake data)

def get_customer_by_name(name: str) -> dict:
    return {"id": random.randint(1000, 9999), "name": name, "email": f"{name.lower()}@example.com"}


def get_customer_invoices_by_customer_id(customer_id: int) -> list[dict]:
    return [
        {
            "invoice_id": f"INV-{random.randint(10000, 99999)}",
            "customer_id": customer_id,
            "date": f"2026-{random.randint(1, 11):02d}-{random.randint(1, 28):02d}",
            "amount": round(random.uniform(100, 5000), 2),
        }
        for _ in range(random.randint(2, 5))
    ]


def calculate_invoices_total(invoices: list[dict]) -> dict:
    return {"invoice_count": len(invoices), "total": round(sum(i["amount"] for i in invoices), 2)}


TOOL_FUNCTIONS = {
    "get_customer_by_name": get_customer_by_name,
    "get_customer_invoices_by_customer_id": get_customer_invoices_by_customer_id,
    "calculate_invoices_total": calculate_invoices_total,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_by_name",
            "description": "Look up a customer by name. Returns the customer's id, name and email.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Customer name, e.g. 'John'"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_invoices_by_customer_id",
            "description": "Get all invoices raised on a customer, using the customer's id.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "integer", "description": "Customer id"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_invoices_total",
            "description": "Calculate the total amount of a list of invoices. "
            "Pass the invoices exactly as returned by get_customer_invoices_by_customer_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoices": {
                        "type": "array",
                        "description": "List of invoice objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "invoice_id": {"type": "string"},
                                "customer_id": {"type": "integer"},
                                "date": {"type": "string"},
                                "amount": {"type": "number"},
                            },
                            "required": ["invoice_id", "amount"],
                        },
                    }
                },
                "required": ["invoices"],
            },
        },
    },
]


# ---------------------------------------------------------------- agent loop

def show(title: str, data) -> None:
    print(f"  {title}:")
    text = data if isinstance(data, str) else json.dumps(data, indent=2)
    for line in text.splitlines():
        print(f"    {line}")


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key="local")  # placeholder key

    query = "Give me the list of invoices raised on John and the total of the invoices."
    messages: list[dict] = [
        {"role": "system", "content": "You are a billing assistant. Use the tools to answer. "
         "Always use calculate_invoices_total for totals - do not add numbers yourself."},
        {"role": "user", "content": query},
    ]
    print(f"Model : {MODEL}")
    print(f"Query : {query}")
    print(f"Tools : {', '.join(TOOL_FUNCTIONS)}")

    sent_upto = 0  # to print only the messages that are new in each iteration
    for iteration in range(1, MAX_ITERATIONS + 1):

        # print the messages that are sent to the model in this iteration
        print(f"\n{'=' * 25} ITERATION {iteration} {'=' * 25}")
        print(f">>> SENT TO MODEL ({len(messages)} messages total, {len(messages) - sent_upto} new + tool definitions)")
        for m in messages[sent_upto:]:
            show(f"[{m['role']}]", m.get("content") or {"tool_calls": m.get("tool_calls")})
        sent_upto = len(messages)

        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)  # type: ignore
        choice = response.choices[0]
        msg = choice.message

        print(f"\n<<< RECEIVED FROM MODEL  (finish_reason = {choice.finish_reason})")
        if msg.content:
            show("text", msg.content)

        if not msg.tool_calls:
            print("\n  => END TURN: model gave its final answer, loop stops.")
            break

        print(f"  => TOOL CALL: model wants {len(msg.tool_calls)} tool(s) run")
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"\n  --- running {tc.function.name}({json.dumps(args)})")
            result = TOOL_FUNCTIONS[tc.function.name](**args)
            show("result", result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    else:
        print(f"\nStopped: hit MAX_ITERATIONS ({MAX_ITERATIONS}).")


if __name__ == "__main__":
    main()
