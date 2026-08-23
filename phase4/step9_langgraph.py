"""
Step 9 — the retail agent, rebuilt in LangGraph.

The point of this step is what PydanticAI was doing for you invisibly.

In step 3 you saw, on the wire, that PydanticAI turned AnalysisResult into a
fake tool called `final_result` and set tool_choice: any so the model could
not end its turn with plain text. You never wrote that. Here you do: the real
tools AND the output schema go into the same bind_tools() list, and the router
decides where to go based on which one the model called.

Run it:  uv run step9_langgraph.py
"""

import os
from typing import Annotated, TypedDict

from elasticsearch import Elasticsearch
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pathlib import Path

import tools_core
# NOTE: importing step3_output also imports retail_agent, which builds the
# PydanticAI agent we do not need here. Harmless, but it is the coupling
# tools_core.py was created to avoid. Same schema on purpose: step 10 scores
# both implementations with the same golden set.
from step3_output import AnalysisResult


DB_PATH = (Path(__file__).parent / "../phase2/retail.db").resolve()

INSTRUCTIONS = (
    "You are a retail database analyst; the database is your single source of "
    "truth. Investigate with the tools, then call AnalysisResult exactly once "
    "to deliver your answer. Do not attach units, currency symbols or labels "
    "that are not present in the source data."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: AnalysisResult | None


# ---------------------------------------------------------------------------
# Tools — dependency injection by closure
# ---------------------------------------------------------------------------
# LangGraph has no `deps_type`. The idiomatic answer is a factory: build the
# resources once, close over them, hand back tools that need no context
# argument. Compare with PydanticAI, where `ctx.deps` was threaded through
# every signature by the framework.

def make_tools(db_path: Path, es: Elasticsearch):

    @tool(parse_docstring=True)
    def run_sql(query: str) -> str:
        """Execute a read-only SQL query (SELECT or WITH only).

        Database schema:
        stores       (id, name, city)
        products     (id, name, category, unit_price REAL)
        transactions (id, store_id, product_id, quantity, unit_price REAL, ts TEXT)

        Args:
            query: A read-only SELECT or WITH query.
        """
        text, _meta = tools_core.sql_query(db_path, query)
        return text

    @tool(parse_docstring=True)
    def search_reviews(
        query: str | None = None,
        store_id: int | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> str:
        """Search customer reviews. Reviews are in ENGLISH — search in English.

        Omit `query` to list every review matching the filters, which is how
        you check whether an absence is real rather than a search miss.

        Args:
            query: Optional English text to search for.
            store_id: Optional store filter.
            min_rating: Optional minimum rating, inclusive.
            max_rating: Optional maximum rating, inclusive.
        """
        text, _meta = tools_core.review_search(
            es, query, store_id, min_rating, max_rating
        )
        return text

    return [run_sql, search_reviews]


def build_model():
    return ChatAnthropic(
        model="us.anthropic.claude-sonnet-4-6",
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_BASE_URL"],
        max_tokens=4096,
    )


OUTPUT_TOOL_NAME = "AnalysisResult"


def build_graph(db_path: Path, es: Elasticsearch):
    tools = make_tools(db_path, es)
    tools_by_name = {t.name: t for t in tools}

    # The real tools and the output schema, bound together. This single line
    # is what PydanticAI's `output_type=` was doing behind the scenes.
    model = build_model().bind_tools([*tools, AnalysisResult])

    def agent_node(state: State) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(state: State) -> dict:
        last = state["messages"][-1]

        results = []

        for call in last.tool_calls:
            if call["name"] == OUTPUT_TOOL_NAME:
                continue

            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])

            results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call["id"],
                )
            )

        return {"messages": results}

    def finish_node(state: State) -> dict:
        last = state["messages"][-1]

        call = next(
            c for c in last.tool_calls if c["name"] == OUTPUT_TOOL_NAME
        )

        result = AnalysisResult.model_validate(call["args"])

        # The output tool still needs a reply, or the history is invalid for
        # any later turn. PydanticAI wrote "Final result processed." here —
        # you saw it as the unsent 5th message in step 5c.
        return {
            "result": result,
            "messages": [
                ToolMessage(
                    content="Final result processed.",
                    tool_call_id=call["id"],
                )
            ],
        }

    def route(state: State) -> str:
        last = state["messages"][-1]

        if any(c["name"] == OUTPUT_TOOL_NAME for c in last.tool_calls):
            return "finish"

        if last.tool_calls:
            return "tools"

        return END

    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, ["tools", "finish", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("finish", END)

    return builder.compile()


def main() -> None:
    es = Elasticsearch("http://localhost:9200")
    graph = build_graph(DB_PATH, es)

    print(graph.get_graph().draw_mermaid())

    question = "42 numaralı mağazada fiyat anormalliği var mı?"

    state: State = {
        "messages": [
            SystemMessage(content=INSTRUCTIONS),
            HumanMessage(content=question),
        ],
        "result": None,
    }

    print("=== NODE BY NODE ===")
    for chunk in graph.stream(state):
        for node, update in chunk.items():
            msgs = update.get("messages") or []
            kinds = [type(m).__name__ for m in msgs]
            print(f"  {node:<8} {kinds}")

    final = graph.invoke(state)

    print("\n=== RESULT ===")
    if final["result"] is None:
        print("  no structured result — the model ended with plain text")
    else:
        print(final["result"].model_dump_json(indent=2)[:1200])

    print(f"\nmessages: {len(final['messages'])}")


if __name__ == "__main__":
    main()
