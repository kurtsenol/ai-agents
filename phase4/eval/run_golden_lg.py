"""
Step 10 — run the golden set against the LangGraph implementation.

Produces the SAME RunRecord shape as run_golden.py, into a different file.
That is the whole payoff of step 6b: the record format belongs to the eval,
not to the framework, so a second implementation needs a second producer and
no changes at all to score.py.

    uv run run_golden_lg.py
    uv run score.py runs_lg.jsonl
    uv run score.py runs.jsonl runs_lg.jsonl      # pydantic-ai vs langgraph
"""

import os
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from elasticsearch import Elasticsearch
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools_core                                        # noqa: E402
from step3_output import AnalysisResult                   # noqa: E402
from run_golden import (                                  # noqa: E402
    N_RUNS,
    RESULT_PREVIEW_CHARS,
    RunRecord,
    ToolCall,
    load_golden_set,
)

EVAL_DIR = Path(__file__).parent
RUNS_FILE = EVAL_DIR / "runs_lg.jsonl"
DB_PATH = (EVAL_DIR / "../../phase2/retail.db").resolve()

OUTPUT_TOOL_NAME = "AnalysisResult"

# Same rules as retail_agent.INSTRUCTIONS, condensed. If these two drift
# apart the comparison stops being about the frameworks.
INSTRUCTIONS = (
    "You are a retail database analyst; the database is your single source of "
    "truth.\n\n"
    "A question is ambiguous when it contains a reference such as 'these "
    "transactions' or 'that store' whose target is not fixed by the "
    "conversation and could refer to more than one set of rows. Do not "
    "silently choose the widest possible scope. When a reference is "
    "ambiguous, set answer_status to 'ambiguous_question' and give a "
    "status_reason naming the reference and its plausible readings. Do not "
    "mark a question ambiguous when the user stated its scope.\n\n"
    "Do not attach units, currency symbols, labels or precision that are not "
    "present in the source data. The database stores unit_price as a bare "
    "number with no currency anywhere. Identify the source field instead: say "
    "'unit_price = 6.99', never '6.99 TL' or '$6.99'.\n\n"
    "Investigate with the tools, then call AnalysisResult exactly once."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: AnalysisResult | None


def build_graph(db_path: Path, es: Elasticsearch, sink: list[ToolCall]):
    """`sink` collects one ToolCall per invocation, for the record."""

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
        text, meta = tools_core.sql_query(db_path, query)
        sink.append(ToolCall(
            tool_name="run_sql",
            args={"query": query},
            result_preview=text[:RESULT_PREVIEW_CHARS],
            result_length=len(text),
            result_metadata=meta,
        ))
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
        text, meta = tools_core.review_search(
            es, query, store_id, min_rating, max_rating
        )
        sink.append(ToolCall(
            tool_name="search_reviews",
            args={"query": query, "store_id": store_id,
                  "min_rating": min_rating, "max_rating": max_rating},
            result_preview=text[:RESULT_PREVIEW_CHARS],
            result_length=len(text),
            result_metadata=meta,
        ))
        return text

    tools_by_name = {"run_sql": run_sql, "search_reviews": search_reviews}

    model = ChatAnthropic(
        model="us.anthropic.claude-sonnet-4-6",
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_BASE_URL"],
        max_tokens=4096,
    ).bind_tools([run_sql, search_reviews, AnalysisResult])

    def agent_node(state: State) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def tools_node(state: State) -> dict:
        last = state["messages"][-1]
        out = []

        for call in last.tool_calls:
            if call["name"] == OUTPUT_TOOL_NAME:
                continue
            result = tools_by_name[call["name"]].invoke(call["args"])
            out.append(ToolMessage(content=result, tool_call_id=call["id"]))

        return {"messages": out}

    def finish_node(state: State) -> dict:
        last = state["messages"][-1]
        call = next(c for c in last.tool_calls if c["name"] == OUTPUT_TOOL_NAME)

        return {
            "result": AnalysisResult.model_validate(call["args"]),
            "messages": [ToolMessage(content="Final result processed.",
                                     tool_call_id=call["id"])],
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


def record_run(item_id: str, question: str, es: Elasticsearch) -> RunRecord:
    sink: list[ToolCall] = []
    graph = build_graph(DB_PATH, es, sink)

    start = time.perf_counter()
    error = None
    final = None

    try:
        final = graph.invoke({
            "messages": [SystemMessage(content=INSTRUCTIONS),
                         HumanMessage(content=question)],
            "result": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    duration = time.perf_counter() - start

    # LangGraph does not track usage for you. Every AIMessage carries its own
    # usage_metadata, so a model request equals an AIMessage and the totals
    # are a sum. PydanticAI's RunUsage did this bookkeeping; here it is yours.
    requests = input_tokens = output_tokens = None
    output = None

    if final is not None:
        ai = [m for m in final["messages"] if type(m).__name__ == "AIMessage"]
        requests = len(ai)
        input_tokens = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai)
        output_tokens = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai)

        if final["result"] is not None:
            output = final["result"].model_dump(mode="json")

    return RunRecord(
        run_id=None,
        conversation_id=None,
        item_id=item_id,
        question=question,
        tool_calls=sink,
        output=output,
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration,
        ok=error is None and output is not None,
        error=error,
    )


def main() -> None:
    items = load_golden_set()
    es = Elasticsearch("http://localhost:9200")

    total = len(items) * N_RUNS
    print(f"LangGraph: {len(items)} items x {N_RUNS} runs = {total} runs\n")

    started = time.perf_counter()
    done = 0

    for item in items:
        for n in range(1, N_RUNS + 1):
            record = record_run(item["id"], item["question"], es)

            with RUNS_FILE.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")

            done += 1
            status = "OK" if record.ok else f"FAILED ({record.error})"
            print(f"[{done}/{total}] {item['id']} run {n}/{N_RUNS}: {status}")

    elapsed = time.perf_counter() - started
    print(f"\n{done} runs in {elapsed / 60:.1f} min -> {RUNS_FILE}")


if __name__ == "__main__":
    main()
