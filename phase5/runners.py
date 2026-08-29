"""One entry point per framework, so step 3 onward stops duplicating them.

Step 2 inlined these. From here they live in one place - a small rehearsal
for step 6, where tools_core.py becomes the single source for tool bodies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))

FRAMEWORKS = ("pydantic", "langgraph")


def run_pydantic(question: str) -> str:
    from pydantic_ai import Agent

    Agent.instrument_all()
    from retail_agent import agent, build_deps

    result = agent.run_sync(question, deps=build_deps())
    return str(result.output)


def run_langgraph(question: str) -> str:
    from elasticsearch import Elasticsearch
    from langchain_core.messages import HumanMessage, SystemMessage

    from step9_langgraph import DB_PATH, INSTRUCTIONS, build_graph

    graph = build_graph(DB_PATH, Elasticsearch("http://localhost:9200"))
    state = {
        "messages": [
            SystemMessage(content=INSTRUCTIONS),
            HumanMessage(content=question),
        ],
        "result": None,
    }
    final = graph.invoke(state)
    if final["result"] is None:
        return "(no structured result)"
    return final["result"].model_dump_json()


def instrument_langgraph(provider) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=provider)


def run(framework: str, question: str) -> str:
    return run_pydantic(question) if framework == "pydantic" else run_langgraph(question)
