"""PoC: the same agent as agent_loop_poc/agent_loop.py, built with LangChain + LangGraph.

The loop is now a LangGraph graph:  START -> model -> (tool calls?) -> tools -> model ... -> END
Each iteration prints what is sent to the model and what comes back
(a tool call, or the final answer / end of turn).
"""

import json
import random
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

BASE_URL = "http://localhost:9001/v1"
MODEL = "sonnet"  # adapter alias -> claude-sonnet-5-5
MAX_ITERATIONS = 10


# ---------------------------------------------------------------- tools (fake data)

class Invoice(TypedDict):
    invoice_id: str
    customer_id: int
    date: str
    amount: float


@tool
def get_customer_by_name(name: str) -> dict:
    """Look up a customer by name. Returns the customer's id, name and email.

    Args:
        name: Customer name, e.g. 'John'
    """
    return {"id": random.randint(1000, 9999), "name": name, "email": f"{name.lower()}@example.com"}


@tool
def get_customer_invoices_by_customer_id(customer_id: int) -> list[dict]:
    """Get all invoices raised on a customer, using the customer's id.

    Args:
        customer_id: Customer id
    """
    return [
        {
            "invoice_id": f"INV-{random.randint(10000, 99999)}",
            "customer_id": customer_id,
            "date": f"2026-{random.randint(1, 11):02d}-{random.randint(1, 28):02d}",
            "amount": round(random.uniform(100, 5000), 2),
        }
        for _ in range(random.randint(2, 5))
    ]


@tool
def calculate_invoices_total(invoices: list[Invoice]) -> dict:
    """Calculate the total amount of a list of invoices.
    Pass the invoices exactly as returned by get_customer_invoices_by_customer_id.

    Args:
        invoices: List of invoice objects
    """
    return {"invoice_count": len(invoices), "total": round(sum(i["amount"] for i in invoices), 2)}


TOOLS = [get_customer_by_name, get_customer_invoices_by_customer_id, calculate_invoices_total]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------- graph state

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # add_messages = append, not overwrite
    iteration: int
    sent_upto: int  # to print only the messages that are new in each iteration


def show(title: str, data) -> None:
    print(f"  {title}:")
    text = data if isinstance(data, str) else json.dumps(data, indent=2)
    for line in text.splitlines():
        print(f"    {line}")


llm = ChatOpenAI(base_url=BASE_URL, api_key="local", model=MODEL).bind_tools(TOOLS)  # type: ignore


# ---------------------------------------------------------------- graph nodes

def model_node(state: AgentState) -> dict:
    iteration = state["iteration"] + 1
    messages = state["messages"]

    # print the messages that are sent to the model in this iteration
    print(f"\n{'=' * 25} ITERATION {iteration} {'=' * 25}")
    print(f">>> SENT TO MODEL ({len(messages)} messages total, "
          f"{len(messages) - state['sent_upto']} new + tool definitions)")
    for m in messages[state["sent_upto"]:]:
        show(f"[{m.type}]", m.content or {"tool_calls": getattr(m, "tool_calls", None)})

    response = llm.invoke(messages)

    finish_reason = response.response_metadata.get("finish_reason")
    print(f"\n<<< RECEIVED FROM MODEL  (finish_reason = {finish_reason})")
    if response.content:
        show("text", response.content)

    if response.tool_calls:  # type: ignore
        print(f"  => TOOL CALL: model wants {len(response.tool_calls)} tool(s) run")  # type: ignore
    else:
        print("\n  => END TURN: model gave its final answer, loop stops.")

    return {"messages": [response], "iteration": iteration, "sent_upto": len(messages)}


def tools_node(state: AgentState) -> dict:
    results = []
    for tc in state["messages"][-1].tool_calls:  # type: ignore
        print(f"\n  --- running {tc['name']}({json.dumps(tc['args'])})")
        result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
        show("result", result)
        results.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))
    return {"messages": results}


def route_after_model(state: AgentState) -> str:
    if not state["messages"][-1].tool_calls:  # type: ignore
        return END
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\nStopped: hit MAX_ITERATIONS ({MAX_ITERATIONS}).")
        return END
    return "tools"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def main() -> None:
    query = "Give me the list of invoices raised on John and the total of the invoices."
    print(f"Model : {MODEL}")
    print(f"Query : {query}")
    print(f"Tools : {', '.join(TOOLS_BY_NAME)}")

    agent = build_graph()
    agent.invoke(
        {
            "messages": [
                SystemMessage("You are a billing assistant. Use the tools to answer. "
                              "Always use calculate_invoices_total for totals - do not add numbers yourself."),
                HumanMessage(query),
            ],
            "iteration": 0,
            "sent_upto": 0,
        },
        {"recursion_limit": 2 * MAX_ITERATIONS + 1},
    )


if __name__ == "__main__":
    main()
