"""
Step 8 — LangGraph: the agent loop as an explicit graph.

No model here on purpose. LangGraph's distinctive idea is control flow, and
wiring a model in at the same time blurs it. The nodes below fake the work so
the graph itself is the only thing on screen.

The shape is the same loop you wrote by hand in phase 2:

    START -> agent -> (tools needed?) -> tools -> agent -> ... -> END

Run it:  uv run step8_graph.py
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    """What flows through the graph.

    Two list fields, deliberately different:

      messages  carries a reducer          -> updates are MERGED
      log       carries no reducer         -> updates REPLACE

    Both are appended to by every node, with the same code shape. Printing
    them side by side at the end is the whole lesson of this step.
    """

    messages: Annotated[list, add_messages]
    log: list[str]
    step: int


# ---------------------------------------------------------------------------
# Nodes — each returns a PARTIAL state, never the whole thing
# ---------------------------------------------------------------------------

FINISH_AFTER = 3


def agent_node(state: State) -> dict:
    """Stands in for the model call."""
    step = state["step"] + 1

    if step >= FINISH_AFTER:
        reply = AIMessage(content=f"done after {step} steps")
    else:
        reply = AIMessage(
            content=f"step {step}: I need a tool",
            tool_calls=[{"name": "fake_tool", "args": {}, "id": f"call_{step}"}],
        )

    return {
        "messages": [reply],
        "log": [f"agent_node step={step}"],
        "step": step,
    }


def tools_node(state: State) -> dict:
    """Stands in for executing the tool calls in the last message."""
    last = state["messages"][-1]

    results = [
        ToolMessage(content="fake result", tool_call_id=call["id"])
        for call in last.tool_calls
    ]

    return {
        "messages": results,
        "log": [f"tools_node ran {len(results)} tool(s)"],
    }


# ---------------------------------------------------------------------------
# The conditional edge — phase 2's `if not response.tool_calls: break`
# ---------------------------------------------------------------------------

def should_continue(state: State) -> str:
    last = state["messages"][-1]

    if last.tool_calls:
        return "tools"

    return END

# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(State)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    # A graph is a declaration; compiling it produces something runnable.
    return builder.compile()


def main() -> None:
    graph = build_graph()

    print("=== MERMAID ===")
    print(graph.get_graph().draw_mermaid())

    initial: State = {
        "messages": [HumanMessage(content="find the price anomaly")],
        "log": [],
        "step": 0,
    }

    print("=== NODE BY NODE ===")
    for chunk in graph.stream(initial):
        for node_name, update in chunk.items():
            print(f"  {node_name}: {list(update)}")

    final = graph.invoke(initial)

    print("\n=== messages (reducer: add_messages) ===")
    for m in final["messages"]:
        print(f"  {type(m).__name__:<12} {str(m.content)[:45]}")

    print("\n=== log (no reducer) ===")
    for line in final["log"]:
        print(f"  {line}")

    print(f"\nmessages: {len(final['messages'])} items")
    print(f"log:      {len(final['log'])} items")
    print(f"step:     {final['step']}")


if __name__ == "__main__":
    main()
