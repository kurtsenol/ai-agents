"""Step 6 - run the MCP-backed agent and look at what changed.

    uv run step6_mcp.py --list-tools     what the servers offer (no model call)
    uv run step6_mcp.py                  answer one question through MCP

The span tree is the thing to read. In step 1 a tool call was one span in
one process. Now the same call is a client span here plus real work in a
subprocess, and you can see where the boundary sits.
"""

from __future__ import annotations

import argparse
import asyncio

from opentelemetry import trace

from agent_mcp import agent, reviews_toolset, sql_toolset
from otel_setup import setup_tracing
from span_dump import SpanDump

QUESTION = "42 numaralı mağazada fiyat anormalliği var mı?"


async def list_tools() -> None:
    """Ask each server what it exposes. No model, no cost."""
    for toolset in (sql_toolset, reviews_toolset):
        async with toolset.client as client:
            tools = await client.list_tools()
            print(f"\n{toolset.id}")
            for tool in tools:
                first_line = (tool.description or "").strip().split("\n")[0]
                print(f"  {tool.name:<20} {first_line[:70]}")


async def run_question(provider, question: str) -> None:
    dump = SpanDump()
    provider.add_span_processor(dump)
    tracer = trace.get_tracer("phase5.step6")

    async with agent:
        with tracer.start_as_current_span("step6.question") as span:
            span.set_attribute("phase5.framework", "pydantic-mcp")
            span.set_attribute("phase5.question", question)
            trace_id = format(span.get_span_context().trace_id, "032x")
            result = await agent.run(question)

    print("=== OUTPUT ===")
    print(result.output.model_dump_json(indent=2)[:800])
    print("\n=== USAGE ===")
    print(result.usage)
    dump.print_tree("pydantic-mcp")
    print(f"\ntrace_id: {trace_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--question", default=QUESTION)
    args = parser.parse_args()

    if args.list_tools:
        asyncio.run(list_tools())
        return

    provider = setup_tracing(service_name="phase5-agent")

    from pydantic_ai import Agent

    Agent.instrument_all()

    asyncio.run(run_question(provider, args.question))

    provider.force_flush()
    provider.shutdown()


if __name__ == "__main__":
    main()
