"""Step 7 - what the agent is ALLOWED to do, enforced in code.

    uv run step7_scope.py

No model calls. This drives the MCP tools directly with queries a model
could plausibly write, and shows what each layer of the scoping catches.
The point is that none of these depend on the model behaving.
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace
from pydantic_ai import ModelRetry

from agent_mcp import sql_toolset
from otel_setup import setup_tracing
from span_dump import SpanDump

import span_dump

span_dump.WATCH_KEYS = [
    "phase5.mcp.is_error",
    "phase5.tool.error",
    "phase5.tool.row_count",
    "phase5.tool.truncated",
]

CASES: list[tuple[str, str]] = [
    (
        "honest query",
        "SELECT id, unit_price FROM transactions WHERE store_id = 42 LIMIT 5",
    ),
    (
        "write attempt (layer 2: the regex)",
        "DROP TABLE stores",
    ),
    (
        "write disguised as a read (layer 1: mode=ro)",
        "WITH x AS (SELECT 1) SELECT * FROM x; DELETE FROM stores",
    ),
    (
        "allowed but ruinous (layer 3: fetchmany + work budget)",
        "SELECT a.id, b.id, a.unit_price FROM transactions a, transactions b",
    ),
    (
        "reading the schema itself",
        "SELECT name, sql FROM sqlite_master",
    ),
]


async def main() -> None:
    provider = setup_tracing(service_name="phase5-agent")
    dump = SpanDump()
    provider.add_span_processor(dump)
    tracer = trace.get_tracer("phase5.step7")

    async with sql_toolset:
        for label, query in CASES:
            with tracer.start_as_current_span(f"step7.{label.split()[0]}"):
                try:
                    text = await sql_toolset.process_tool_call(
                        None, None, "run_sql", {"query": query}
                    )
                    outcome = f"ok      {text.strip().splitlines()[0][:88]}"
                except ModelRetry as exc:
                    # This is the success path for a refused query: the model
                    # is handed the reason and gets another turn.
                    outcome = f"REFUSED {str(exc).splitlines()[0][:88]}"
            print(f"\n--- {label}")
            print(f"    {query[:78]}")
            print(f"    -> {outcome}")

    provider.force_flush()
    dump.print_tree("step7")
    provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
