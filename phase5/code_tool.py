"""The sandbox as an agent tool.

Attach alongside the MCP toolsets:

    from code_tool import code_tools
    Agent(..., toolsets=[sql_toolset, reviews_toolset, code_tools])

The docstring below is the ACI - it is what the model reads, and it has to
be honest about the limits, because a model that does not know there is no
network will waste turns discovering it. Telling it the constraints is not
a security control (the container is), it is ordinary tool design.
"""

from __future__ import annotations

from opentelemetry import metrics, trace
from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.toolsets import FunctionToolset

import sandbox

_meter = metrics.get_meter("phase5.sandbox")
sandbox_runs = _meter.create_counter(
    "agent.sandbox.runs",
    unit="{run}",
    description="Sandboxed code executions by outcome",
)

code_tools = FunctionToolset()


@code_tools.tool
def run_python(ctx: RunContext[None], code: str) -> ToolReturn[str]:
    """Run a short Python script and return whatever it prints.

    Use this for arithmetic, statistics or data shaping that is awkward in
    SQL. Print your result - nothing else is returned.

    The script runs in an isolated container with no network access, no
    access to any file from this machine, a read-only filesystem apart from
    /tmp, 256 MB of memory and a 10 second limit. Only the standard library
    is available; there is no pandas or numpy, and pip cannot reach an index.

    Args:
        code: A self-contained Python script that prints its result.
    """
    result = sandbox.run_python(code)

    span = trace.get_current_span()
    span.set_attribute("phase5.sandbox.exit_code", result.exit_code)
    span.set_attribute("phase5.sandbox.timed_out", result.timed_out)

    if result.killed_for:
        span.set_attribute("phase5.sandbox.killed_for", result.killed_for)
        sandbox_runs.add(1, {"outcome": "killed"})
        # A limit that trips is an ERROR, not an empty result - step 7's
        # lesson. A model handed "" cannot tell "no output" from "we killed
        # you", and will answer from the silence.
        return ToolReturn(
            return_value=(
                f"Execution stopped: {result.killed_for}. "
                "Rewrite the script to do less work, or process the data in "
                "smaller pieces."
            ),
            metadata={"error": True, "exit_code": result.exit_code,
                      "killed_for": result.killed_for},
        )

    if not result.ok:
        sandbox_runs.add(1, {"outcome": "error"})
        return ToolReturn(
            return_value=f"The script failed:\n{result.stderr.strip()[:1500]}",
            metadata={"error": True, "exit_code": result.exit_code},
        )

    sandbox_runs.add(1, {"outcome": "ok"})
    return ToolReturn(
        return_value=result.stdout.strip() or "The script produced no output.",
        metadata={"error": False, "exit_code": 0,
                  "output_chars": len(result.stdout)},
    )
