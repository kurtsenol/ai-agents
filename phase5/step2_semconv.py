"""Step 2 - the same question, two frameworks, one collector.

    uv run step2_semconv.py pydantic
    uv run step2_semconv.py langgraph

Each run prints its span tree and every attribute key it produced, and saves
the key list to out/. Once both files exist, the script also prints a
side-by-side comparison of the two vocabularies.

The point is not that one framework is better. The point is what the two
instrumentations agree on - because a dashboard can only be built on the
part they agree on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))

from opentelemetry import trace

from otel_setup import setup_tracing
from span_dump import SpanDump

QUESTION = "42 numaralı mağazada fiyat anormalliği var mı?"
OUT_DIR = Path(__file__).parent / "out"

# TODO(2): Both runs land in the same Grafana. How do you tell them apart?
#
#   (a) a different `service.name` per framework
#         -> lives on the Resource, fixed for the whole process
#   (b) one shared `service.name`, with the framework as a span attribute
#         -> lives on a single span, set per run
#
# These are not interchangeable. Ask yourself which question you want to be
# able to answer in Grafana: "show me everything the LangGraph service did"
# or "show me all agent runs, and let me break them down by framework".
# Pick one, and wire it in below. If you pick (b), set the attribute on the
# root span in run_pydantic/run_langgraph.
def service_name(framework: str) -> str:
    ...


def run_pydantic(tracer) -> None:
    from pydantic_ai import Agent

    Agent.instrument_all()

    from retail_agent import agent, build_deps

    with tracer.start_as_current_span("step2.question") as span:
        span.set_attribute("phase5.question", QUESTION)
        result = agent.run_sync(QUESTION, deps=build_deps())

    print("=== OUTPUT ===")
    print(str(result.output)[:600])


def run_langgraph(tracer, provider) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # LangGraph has no native OTel support. This instrumentor hooks LangChain's
    # callback system and turns callbacks into spans. Note what that means:
    # the span names and attribute keys are chosen by *this package*, not by
    # LangGraph and not by OpenTelemetry.
    LangChainInstrumentor().instrument(tracer_provider=provider)

    from elasticsearch import Elasticsearch
    from langchain_core.messages import HumanMessage, SystemMessage

    from step9_langgraph import DB_PATH, INSTRUCTIONS, build_graph

    graph = build_graph(DB_PATH, Elasticsearch("http://localhost:9200"))

    state = {
        "messages": [
            SystemMessage(content=INSTRUCTIONS),
            HumanMessage(content=QUESTION),
        ],
        "result": None,
    }

    with tracer.start_as_current_span("step2.question") as span:
        span.set_attribute("phase5.question", QUESTION)
        final = graph.invoke(state)

    print("=== OUTPUT ===")
    if final["result"] is None:
        print("  no structured result - the model ended with plain text")
    else:
        print(final["result"].model_dump_json(indent=2)[:600])


def compare() -> None:
    files = {
        fw: OUT_DIR / f"step2_{fw}_keys.json"
        for fw in ("pydantic", "langgraph")
    }
    if not all(p.exists() for p in files.values()):
        missing = [fw for fw, p in files.items() if not p.exists()]
        print(f"\n(run {', '.join(missing)} too, to see the comparison)")
        return

    keys = {fw: set(json.loads(p.read_text())) for fw, p in files.items()}
    both = keys["pydantic"] & keys["langgraph"]
    only_p = keys["pydantic"] - keys["langgraph"]
    only_l = keys["langgraph"] - keys["pydantic"]

    print("\n=== ATTRIBUTE VOCABULARY ===")
    print(f"in both        : {len(both)}")
    for k in sorted(both):
        print("   ", k)
    print(f"pydantic only  : {len(only_p)}")
    for k in sorted(only_p):
        print("   ", k)
    print(f"langgraph only : {len(only_l)}")
    for k in sorted(only_l):
        print("   ", k)


def main() -> None:
    framework = sys.argv[1] if len(sys.argv) > 1 else "pydantic"
    if framework not in ("pydantic", "langgraph"):
        sys.exit("usage: uv run step2_semconv.py [pydantic|langgraph]")

    provider = setup_tracing(service_name=service_name(framework))

    dump = SpanDump()
    provider.add_span_processor(dump)

    tracer = trace.get_tracer("phase5.step2")

    if framework == "pydantic":
        run_pydantic(tracer)
    else:
        run_langgraph(tracer, provider)

    provider.force_flush()

    dump.print_tree(framework)
    dump.print_all_keys(framework)
    dump.save_keys(OUT_DIR / f"step2_{framework}_keys.json")
    compare()

    provider.shutdown()


if __name__ == "__main__":
    main()
