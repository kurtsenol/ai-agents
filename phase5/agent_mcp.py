"""Step 6 - the agent stops owning its tools.

Until now every tool body lived in the agent's own process. The tools were
`@agent.tool` functions; `ctx.deps` handed them a live sqlite connection and
a live Elasticsearch client. The agent could reach the database directly.

Here the tools move behind a process boundary. Two MCP servers - the ones
from phase 4, unchanged - are launched as subprocesses and speak JSON-RPC
over stdio. The agent gets tool DEFINITIONS over the wire and calls them by
name. It never opens the database.

Two things follow, and the second is the point of the whole security block:

1. Dependencies stop being shared. This is not hypothetical here: phase 5
   pins fastmcp 3.x (the client, via pydantic-ai), phase 4 pins fastmcp 4.x
   (the server, via mcp[cli]). They cannot coexist in one venv - and they do
   not have to, because they are two processes. That is what a boundary buys.

2. Capability stops being ambient. Inside one process, "the agent must not
   drop tables" is a promise made by a regex. Across a boundary, the agent
   holds no database handle at all - there is nothing to make a promise
   about. Step 7 builds on exactly this.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PHASE4 = REPO / "phase4"
sys.path.insert(0, str(PHASE4))

from opentelemetry import trace

from fastmcp.client import Client
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from model import build_model  # noqa: E402
from retail_agent import INSTRUCTIONS  # noqa: E402
from step3_output import AnalysisResult  # noqa: E402


def _stdio_toolset(script: str, toolset_id: str) -> MCPToolset:
    """Launch one phase 4 MCP server as a subprocess, in ITS OWN venv.

    `uv --directory phase4 run` is what makes the version split above work:
    the child resolves phase 4's lockfile, not ours. This is also exactly the
    command in .mcp.json - the same server, now with our agent as the host
    instead of Claude Code.
    """
    transport = StdioTransport(
        command="uv",
        args=["--directory", str(PHASE4), "run", script],
    )
    return MCPToolset(Client(transport), id=toolset_id)


sql_toolset = _stdio_toolset("mcp_sql_server.py", "retail-sql")
reviews_toolset = _stdio_toolset("mcp_es_server.py", "retail-reviews")


# --- reading the second channel ------------------------------------------
#
# The servers now answer with a CallToolResult: `content` for the model,
# `_meta` for programs. pydantic-ai's normal path hands us only the content,
# because that is all a model needs. So we intercept the call, talk to the
# client ourselves, and put `_meta` somewhere a program can reach it.
#
# Where is "somewhere"? Not a return value - the model would then see it.
# The span. The run is already traced, every tool call is already a span,
# and a span attribute is read by machines and never by the model. The old
# ToolReturn(metadata=...) had one reader; this has every reader the trace
# has - the eval, a Grafana panel, an alert.


def _record_meta(toolset: MCPToolset):
    """Call the tool, copy its `_meta` onto the current span, return the text."""

    async def process(ctx, call_tool, name: str, args: dict):
        # Deliberately bypassing `call_tool` (the pydantic-ai wrapper): it
        # returns processed content, and `_meta` is not in it. We want the
        # raw protocol object.
        result = await toolset.client.call_tool(name, args)

        span = trace.get_current_span()
        span.set_attribute("phase5.mcp.server", toolset.id or "")
        span.set_attribute("phase5.mcp.is_error", bool(result.is_error))

        for key, value in (result.meta or {}).items():
            # Span attributes take scalars, so anything else is stringified
            # rather than dropped.
            if isinstance(value, (bool, int, float, str)):
                span.set_attribute(f"phase5.tool.{key}", value)
            else:
                span.set_attribute(f"phase5.tool.{key}", str(value))

        return "".join(
            block.text for block in result.content if hasattr(block, "text")
        )

    return process


sql_toolset.process_tool_call = _record_meta(sql_toolset)
reviews_toolset.process_tool_call = _record_meta(reviews_toolset)


# --- what the agent is allowed to hold ------------------------------------
#
# Read tools only. phase 4's `run_write_sql` is on neither server, so this
# agent cannot write to the database - not because it is told not to, but
# because no such tool exists in its process or in either subprocess.
#
# And when step 10 adds a write tool back, it will NOT go on an MCP server.
# The gate has to ask a human "run this DELETE? y/n", and the human is at
# the terminal that started the AGENT. A subprocess speaking JSON-RPC over
# stdin/stdout cannot reach that person - its stdin is the protocol. So the
# approval-gated tool stays local, in the process that owns the terminal.
# Read tools go over MCP; the dangerous one stays where the human is.
TOOLSETS = [sql_toolset, reviews_toolset]



agent = Agent(
    build_model(),
    instructions=INSTRUCTIONS,
    output_type=AnalysisResult,
    toolsets=TOOLSETS,
)
