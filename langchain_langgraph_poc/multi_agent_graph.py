"""Customer invoice workflow with an outer LangGraph agent loop and a nested risk agent.

Install:
    pip install -U langchain langchain-core langchain-openai langgraph

Requires an OpenAI-compatible API at BASE_URL (default http://localhost:9001/v1)
with a model named MODEL (default sonnet). Override with environment variables.
Run:
    python multi_agent_graph.py
"""

import json
import os
import random
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

BASE_URL = os.getenv("BASE_URL", "http://localhost:9001/v1")
MODEL = os.getenv("MODEL", "sonnet")
MAX_ITERATIONS = 10


class Invoice(TypedDict):
    invoice_id: str
    customer_id: int
    date: str
    amount: float


@tool
def get_customer_by_name(name: str) -> dict:
    """Look up a customer by name. Returns the customer's id, name and email."""
    return {"id": random.randint(1000, 9999), "name": name,
            "email": f"{name.lower()}@example.com"}


@tool
def get_customer_invoices_by_customer_id(customer_id: int) -> list[dict]:
    """Get all invoices raised on a customer, using the customer's id."""
    return [
        {"invoice_id": f"INV-{random.randint(10000, 99999)}",
         "customer_id": customer_id,
         "date": f"2026-{random.randint(1, 11):02d}-{random.randint(1, 28):02d}",
         "amount": round(random.uniform(100, 5000), 2)}
        for _ in range(random.randint(2, 5))
    ]


@tool
def calculate_invoices_total(invoices: list[Invoice]) -> dict:
    """Calculate the total amount of invoices. Pass invoices exactly as returned by the invoice lookup."""
    return {"invoice_count": len(invoices),
            "total": round(sum(invoice["amount"] for invoice in invoices), 2)}


@tool
def check_payment_history(customer_id: int) -> dict:
    """Check a customer's payment history for late payments and outstanding balances."""
    return {"customer_id": customer_id, "late_payments": 2, "outstanding_amount": 1200}


@tool
def check_credit_status(customer_id: int) -> dict:
    """Check a customer's credit limit and credit risk classification."""
    return {"customer_id": customer_id, "credit_limit": 10000, "risk_level": "medium"}


TOOLS = [get_customer_by_name, get_customer_invoices_by_customer_id,
         calculate_invoices_total]

TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}

ANALYSIS_TOOLS = [check_payment_history, check_credit_status]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int
    sent_upto: int
    customer_id: int | None
    invoice_total: float | None
    invoice_count: int | None
    risk_analysis: str | None


def show(title: str, data) -> None:
    print(f"  {title}:")
    rendered = data if isinstance(data, str) else json.dumps(data, indent=2, default=str)
    for line in rendered.splitlines():
        print(f"    {line}")


def make_llm() -> ChatOpenAI:
    # return ChatOpenAI(base_url=BASE_URL, api_key="local", model=MODEL)
    return  ChatOpenAI(base_url=BASE_URL, api_key="local", model=MODEL).bind_tools(TOOLS)  # type: ignore


llm = make_llm().bind_tools(TOOLS)
analysis_agent = create_agent(
    model=make_llm(),
    tools=ANALYSIS_TOOLS,
    system_prompt=(
        "You are a customer financial risk analyst. Use both the payment history and "
        "credit status tools for the supplied customer ID. Summarize the risk and "
        "recommend next steps. Never claim to have changed a customer's account."
    ),
)


def model_node(state: AgentState) -> dict:
    iteration = state["iteration"] + 1
    messages = state["messages"]
    print(f"\n{'=' * 25} OUTER ITERATION {iteration} {'=' * 25}")
    print(f">>> SENT TO MODEL ({len(messages)} messages; "
          f"{len(messages) - state['sent_upto']} new + tool definitions)")
    for message in messages[state["sent_upto"]:]:
        show(f"[{message.type}]", message.content or
             {"tool_calls": getattr(message, "tool_calls", None)})

    response = llm.invoke(messages)
    show("<<< MODEL TEXT", response.content or "(no text)")
    if response.tool_calls:
        print("  => TOOL CALLS:", [call["name"] for call in response.tool_calls])
    else:
        print("  => OUTER AGENT FINAL ANSWER; next: nested risk agent")
    return {"messages": [response], "iteration": iteration,
            "sent_upto": len(messages)}


def tools_node(state: AgentState) -> dict:
    results = []
    updates = {}
    for call in state["messages"][-1].tool_calls: # pyright: ignore[reportAttributeAccessIssue]
        name, args = call["name"], call["args"]
        print(f"\n  --- running {name}({json.dumps(args)})")
        try:
            if name not in TOOLS_BY_NAME:
                raise ValueError(f"Unknown tool: {name}")
            result = TOOLS_BY_NAME[name].invoke(args)
            if name == "get_customer_by_name":
                updates["customer_id"] = result["id"]
            elif name == "calculate_invoices_total":
                updates["invoice_total"] = result["total"]
                updates["invoice_count"] = result["invoice_count"]
            show("result", result)
            content = json.dumps(result)
        except Exception as exc:
            content = json.dumps({"error": str(exc)})
            show("tool error", content)
        results.append(ToolMessage(content=content, tool_call_id=call["id"]))
    return {"messages": results, **updates}


def route_after_model(state: AgentState) -> str:
    if not state["messages"][-1].tool_calls: # type: ignore
        return "invoice_analysis_agent"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"Stopped: hit MAX_ITERATIONS ({MAX_ITERATIONS}).")
        return END
    return "tools"


def invoice_analysis_agent_node(state: AgentState) -> dict:
    print("\n=== NESTED INVOICE ANALYSIS AGENT STARTED ===")
    customer_id = state.get("customer_id")
    if customer_id is None:
        analysis = "Risk analysis skipped: no customer ID was retrieved."
    else:
        result = analysis_agent.invoke(
            {"messages": [HumanMessage(content=(
                f"Analyze financial risk for customer ID {customer_id}. "
                f"Invoice count: {state.get('invoice_count')}; "
                f"invoice total: {state.get('invoice_total')}. "
                "Check payment history and credit status using your tools."
            ))]},
            config={"recursion_limit": 20},
        )
        analysis = str(result["messages"][-1].content)
        for message in result["messages"]:
            if getattr(message, "tool_calls", None):
                show("nested tool requests", message.tool_calls)
            if isinstance(message, ToolMessage):
                show("nested tool result", message.content)
    show("risk analysis", analysis)
    # Keep the nested agent's output as structured outer state, not a fake user message.
    return {"risk_analysis": analysis}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("invoice_analysis_agent", invoice_analysis_agent_node)


    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model,
                                ["tools", "invoice_analysis_agent", END])
    graph.add_edge("tools", "model")
    graph.add_edge("invoice_analysis_agent", END)
    return graph.compile()


def main() -> None:
    query = "Give me the list of invoices raised on John and the total of the invoices."
    print(f"Model : {MODEL}\nQuery : {query}")
    print("Outer tools:", ", ".join(TOOLS_BY_NAME))
    print("Nested tools:", ", ".join(tool_.name for tool_ in ANALYSIS_TOOLS))

    agent = build_graph()
    
    result = agent.invoke(
        {"messages": [
            SystemMessage(content=(
                "You are a billing assistant. Use the tools to answer. "
                "Always use calculate_invoices_total for totals; do not add numbers yourself."
            )),
            HumanMessage(content=query),
        ], "iteration": 0, "sent_upto": 0, "customer_id": None,
         "invoice_total": None, "invoice_count": None, "risk_analysis": None},
        {"recursion_limit": 2 * MAX_ITERATIONS + 3},
    )
    print("\n=== OUTER AGENT ANSWER ===")
    for message in reversed(result["messages"]):
        if message.type == "ai" and not getattr(message, "tool_calls", None):
            print(message.content)
            break
    print("\n=== NESTED AGENT RISK ANALYSIS ===")
    print(result["risk_analysis"])
    print("\n=== SUMMARY ===")
    print("Outer model iterations:", result["iteration"])
    print("Customer ID:", result["customer_id"])
    print("Invoice count:", result["invoice_count"])
    print("Invoice total:", result["invoice_total"])


if __name__ == "__main__":
    main()
