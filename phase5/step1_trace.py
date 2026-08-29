"""Step 1 - run the phase 4 agent once, with tracing on.

Nothing about the agent changes. We add two lines around it:
  setup_tracing()        -> where spans go
  Agent.instrument_all() -> pydantic-ai starts emitting them

Then we look at the shape of what comes out.

    cd phase5
    docker compose up -d
    uv run step1_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The phase 4 agent is a sibling package with no installable entry point,
# so we put it on the import path. Step 6 replaces this with a real shared
# module; until then, a shim is honest about being a shim.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))

from opentelemetry import trace

from otel_setup import setup_tracing
from pydantic_ai import Agent

QUESTION = "42 numaralı mağazada fiyat anormalliği var mı?"


def main() -> None:
    provider = setup_tracing(service_name="retail-agent")

    # Turns on instrumentation for every Agent in the process. The agent in
    # retail_agent.py does not pass instrument=..., so it inherits this.
    Agent.instrument_all()

    # Imported after setup so the tracer provider is already global.
    from retail_agent import agent, build_deps

    tracer = trace.get_tracer("phase5.step1")

    # Our own span, wrapping the agent run. pydantic-ai's spans will nest
    # underneath it, which is what makes the whole thing one trace with one
    # trace_id instead of several unrelated ones.
    with tracer.start_as_current_span("step1.question") as span:
        span.set_attribute("phase5.question", QUESTION)
        result = agent.run_sync(QUESTION, deps=build_deps())
        trace_id = format(span.get_span_context().trace_id, "032x")

    print("=== OUTPUT ===")
    print(result.output)
    print("\n=== USAGE ===")
    print(result.usage)
    print("\n=== TRACE ===")
    print("trace_id:", trace_id)
    print("Grafana -> Explore -> Tempo -> paste the trace_id above")

    # Spans still sitting in a queue are lost when the process exits, so we
    # drain before shutting down.
    provider.force_flush()
    provider.shutdown()


if __name__ == "__main__":
    main()
