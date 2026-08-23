"""
Step 8 — LangGraph from zero. Four graphs, each showing exactly one thing.

Nothing to fill in. Run it and read the output:

    uv run step8_intro.py

Then go back to step8_graph.py — its two TODOs are demo 3 and demo 4.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage


def header(n: int, title: str, point: str) -> None:
    print(f"\n{'=' * 70}\nDEMO {n}: {title}\n{'=' * 70}\n{point}\n")


# ===========================================================================
# 1. The smallest possible graph
# ===========================================================================
# A graph needs three things: a State type, at least one node, and edges
# saying where to start and stop.
#
# A node is an ordinary function. It receives the whole state and returns a
# dict containing ONLY the fields it wants to change. LangGraph merges that
# dict into the state for you.

def demo1() -> None:
    header(1, "one node",
           "A node returns a PARTIAL dict, not the whole state.")

    class State(TypedDict):
        question: str
        answer: str

    def answer_node(state: State) -> dict:
        print(f"  node sees state: {state}")
        return {"answer": f"answer to '{state['question']}'"}   # only one field

    builder = StateGraph(State)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)

    graph = builder.compile()

    result = graph.invoke({"question": "is there an anomaly?", "answer": ""})

    print(f"\n  final state: {result}")
    print("\n  -> `question` survived even though the node never mentioned it.")


# ===========================================================================
# 2. Two nodes in sequence
# ===========================================================================
# Edges decide the order. The second node sees whatever the first one wrote.
# This is the whole 'state flows through the graph' idea.

def demo2() -> None:
    header(2, "two nodes in sequence",
           "The second node reads what the first one wrote.")

    class State(TypedDict):
        value: int

    def double(state: State) -> dict:
        new = state["value"] * 2
        print(f"  double: {state['value']} -> {new}")
        return {"value": new}

    def add_ten(state: State) -> dict:
        new = state["value"] + 10
        print(f"  add_ten: {state['value']} -> {new}")
        return {"value": new}

    builder = StateGraph(State)
    builder.add_node("double", double)
    builder.add_node("add_ten", add_ten)
    builder.add_edge(START, "double")
    builder.add_edge("double", "add_ten")     # order lives in the edges
    builder.add_edge("add_ten", END)

    result = builder.compile().invoke({"value": 5})

    print(f"\n  final: {result}")
    print("\n  -> `value` was REPLACED each time. That is the default.")


# ===========================================================================
# 3. Reducers — the one idea people get wrong
# ===========================================================================
# By default an update REPLACES the field. A reducer changes that: it is a
# function (old_value, update) -> new_value, attached to the field with
# Annotated[<type>, <reducer>].
#
# Both fields below are appended to by the same code shape. Only one has a
# reducer. Watch what happens to the other one.

def demo3() -> None:
    header(3, "reducer vs no reducer",
           "Same node code, two fields, two different outcomes.")

    def append_lists(old: list, new: list) -> list:
        """A reducer is just a merge function. This one concatenates."""
        return old + new

    class State(TypedDict):
        with_reducer: Annotated[list[str], append_lists]
        without_reducer: list[str]

    def node_a(state: State) -> dict:
        return {"with_reducer": ["a"], "without_reducer": ["a"]}

    def node_b(state: State) -> dict:
        return {"with_reducer": ["b"], "without_reducer": ["b"]}

    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    result = builder.compile().invoke({"with_reducer": [], "without_reducer": []})

    print(f"  with_reducer:    {result['with_reducer']}")
    print(f"  without_reducer: {result['without_reducer']}")
    print("\n  -> Identical node code. The reducer is the only difference.")
    print("  -> For a message history you always want the appending behaviour,")
    print("     which is what langgraph's built-in `add_messages` gives you.")

    # add_messages does the same thing, plus it dedupes by message id and
    # handles updates to an existing message.
    class MsgState(TypedDict):
        messages: Annotated[list, add_messages]

    def talk(state: MsgState) -> dict:
        return {"messages": [AIMessage(content=f"reply {len(state['messages'])}")]}

    b2 = StateGraph(MsgState)
    b2.add_node("talk", talk)
    b2.add_edge(START, "talk")
    b2.add_edge("talk", END)

    out = b2.compile().invoke({"messages": [HumanMessage(content="hello")]})

    print(f"\n  add_messages result: {[type(m).__name__ for m in out['messages']]}")


# ===========================================================================
# 4. Conditional edges — this is the loop
# ===========================================================================
# add_edge is a fixed arrow. add_conditional_edges asks a function where to
# go next. The function receives the state and returns the NAME of the next
# node (or END).
#
# It cannot change the state. Routing and mutation are separate jobs, and
# that separation is why the control flow stays readable.

def demo4() -> None:
    header(4, "conditional edge = the loop",
           "This is phase 2's `while ...: if done: break`, as a graph.")

    class State(TypedDict):
        count: int
        history: Annotated[list[str], lambda old, new: old + new]

    def work(state: State) -> dict:
        n = state["count"] + 1
        print(f"  work: count -> {n}")
        return {"count": n, "history": [f"work#{n}"]}

    def keep_going(state: State) -> str:
        decision = "work" if state["count"] < 3 else END
        print(f"    router: count={state['count']} -> {decision}")
        return decision

    builder = StateGraph(State)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_conditional_edges("work", keep_going, ["work", END])

    graph = builder.compile()

    result = graph.invoke({"count": 0, "history": []})

    print(f"\n  final: count={result['count']} history={result['history']}")

    print("\n  the graph can draw itself:\n")
    print(graph.get_graph().draw_mermaid())

    print("  -> `work` has an edge back to itself. That IS the agent loop.")
    print("  -> Nothing here is hidden: the loop is data you can print.")


if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
