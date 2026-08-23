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

    # TODO(1): Give `messages` the reducer that appends instead of replacing.
    #
    # `add_messages` is imported above. The syntax is Annotated[<type>, <reducer>]
    # — the same Annotated you used in step 2 to attach Field(ge=1, le=5) to a
    # tool parameter. There, the metadata told Pydantic how to validate. Here
    # it tells LangGraph how to merge.
    #
    # A reducer is a function (old_value, update) -> new_value. Without one,
    # LangGraph does what a dict does: the update wins.
    messages: ...

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
    # TODO(2): Return the name of the next step.
    #
    # Return "tools" when the last message asked for tool calls, and END when
    # it did not. END is imported above and is a sentinel, not the string
    # "END" — return the object.
    #
    # Note what this function is NOT allowed to do: it cannot change state.
    # Routing and mutation are separate jobs in LangGraph, which is why the
    # loop is inspectable at all. A node that also decided where to go next
    # would hide the control flow again.
    ...


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
