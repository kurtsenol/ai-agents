"""Step 10a - a write tool the model cannot run without a human.

    uv run step10_approve.py --policy          free: the decision table
    uv run step10_approve.py --ask "42 numaralı mağazadaki hatalı fiyatları düzelt"

The write tool is LOCAL, not on an MCP server, and step 6 explains why: the
gate has to ask a person "run this? y/n", and that person is at the terminal
that started the agent. An MCP server is a subprocess whose stdin belongs to
the protocol - it cannot reach them.

Writes go to a COPY of the database. Nothing here can damage phase 2's data,
which matters because the whole point is to let a destructive statement get
all the way to the edge of running.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))

from opentelemetry import metrics, trace
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    RunContext,
    ToolApproved,
    ToolDenied,
    ToolReturn,
)
from pydantic_ai.toolsets import FunctionToolset

import approval
from agent_mcp import reviews_toolset, sql_toolset
from model import build_model  # noqa: E402
from otel_setup import setup_metrics, setup_tracing
from retail_agent import INSTRUCTIONS  # noqa: E402
from step3_output import AnalysisResult  # noqa: E402

SOURCE_DB = REPO / "phase2/retail.db"
WRITE_DB = Path(__file__).parent / "out" / "retail_write.db"
MAX_APPROVAL_ROUNDS = 3

_meter = metrics.get_meter("phase5.approval")
approval_decisions = _meter.create_counter(
    "agent.approval.decisions",
    unit="{decision}",
    description="Write attempts by risk and by what the human said",
)


def writable_db() -> Path:
    """A throwaway copy. The agent never touches the real database."""
    WRITE_DB.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE_DB, WRITE_DB)
    return WRITE_DB


write_tools = FunctionToolset()


@write_tools.tool
def run_write_sql(ctx: RunContext[None], query: str) -> ToolReturn[str]:
    """Execute an approved INSERT, UPDATE or DELETE against the retail database.

    Use this only when the user explicitly asks to CHANGE data. For any
    question about existing data, including anything starting with SELECT or
    WITH, use run_sql instead.

    Every call is reviewed by a human before it runs. DROP, ALTER, TRUNCATE
    and CREATE are not available through this tool at all.

    Args:
        query: A single INSERT, UPDATE or DELETE statement.
    """
    decision = approval.assess("run_write_sql", {"query": query})

    # Measure the blast radius before deciding, not after. `estimate_rows`
    # runs the statement inside a transaction and rolls it back, so the risk
    # level the reviewer sees is a measurement rather than a guess about what
    # the WHERE clause probably means.
    rows_estimate = approval.estimate_rows(WRITE_DB, "run_write_sql", {"query": query})
    decision = approval.escalate(decision, rows_estimate)

    span = trace.get_current_span()
    if rows_estimate is not None:
        span.set_attribute("phase5.approval.rows_estimate", rows_estimate)
    span.set_attribute("phase5.approval.risk", decision.risk)
    span.set_attribute("phase5.approval.reason", decision.reason)

    # Refused outright: never reaches a human. Asking someone to approve a
    # DROP is not a safeguard, it is a way to get a DROP approved.
    if not decision.allowed:
        approval_decisions.add(1, {"risk": decision.risk, "outcome": "refused"})
        return ToolReturn(
            return_value=f"Refused: {decision.reason}.",
            metadata={"error": True, "rows_affected": 0, "risk": decision.risk},
        )

    if not ctx.tool_call_approved:
        raise ApprovalRequired(
            metadata={
                "risk": decision.risk,
                "reason": decision.reason,
                "query": query,
                "rows_estimate": rows_estimate,
            }
        )

    conn = sqlite3.connect(WRITE_DB)
    try:
        cursor = conn.execute(query)
        conn.commit()
        rows = cursor.rowcount
    except sqlite3.Error as exc:
        conn.rollback()
        approval_decisions.add(1, {"risk": decision.risk, "outcome": "failed"})
        return ToolReturn(
            return_value=f"SQLite error: {exc}",
            metadata={"error": True, "rows_affected": 0, "risk": decision.risk},
        )
    finally:
        conn.close()

    span.set_attribute("phase5.approval.rows_affected", rows)
    approval_decisions.add(1, {"risk": decision.risk, "outcome": "executed"})
    return ToolReturn(
        return_value=f"Statement executed. {rows} row(s) affected.",
        metadata={"error": False, "rows_affected": rows, "risk": decision.risk},
    )


agent = Agent(
    build_model(),
    instructions=INSTRUCTIONS,
    output_type=[AnalysisResult, DeferredToolRequests],
    toolsets=[sql_toolset, reviews_toolset, write_tools],
)


def ask_human(call, metadata: dict) -> ToolApproved | ToolDenied:
    """The one place a person is actually in the loop."""
    risk = metadata.get("risk", "normal")
    banner = "!! HIGH RISK !!" if risk == "high" else "review"

    print(f"\n=== APPROVAL REQUIRED ({banner}) ===")
    print(f"  tool   : {call.tool_name}")
    print(f"  reason : {metadata.get('reason')}")
    print(f"  query  : {metadata.get('query')}")

    rows = metadata.get("rows_estimate")
    print(f"  affects: {rows if rows is not None else 'UNKNOWN - could not be measured'} row(s)")

    if risk == "high":
        # A y/n on a statement that empties a table is a keystroke away from
        # a habit. Typing the verb back is friction on purpose.
        typed = input("\n  Type the statement's first word in CAPITALS to approve: ").strip()
        expected = re.match(r"^\s*(\w+)", str(metadata.get("query", ""))).group(1).upper()
        if typed != expected:
            approval_decisions.add(1, {"risk": risk, "outcome": "denied"})
            return ToolDenied(
                message=(
                    "The reviewer did not confirm this high-risk write. Do not retry it "
                    "and do not attempt another statement without a WHERE clause."
                )
            )
    else:
        if input("\n  Approve? [y/N] ").strip().lower() not in ("y", "yes"):
            reason = input("  Reason (optional): ").strip()
            approval_decisions.add(1, {"risk": risk, "outcome": "denied"})
            return ToolDenied(
                message=(
                    f"The reviewer declined this write. {reason} "
                    "Consider whether a narrower operation would address the request."
                ).strip()
            )

    return ToolApproved()


def run_with_approvals(question: str) -> None:
    tracer = trace.get_tracer("phase5.step10")

    with tracer.start_as_current_span("step10.write") as span:
        span.set_attribute("phase5.question", question)
        trace_id = format(span.get_span_context().trace_id, "032x")

        result = agent.run_sync(question)

        rounds = 0
        while isinstance(result.output, DeferredToolRequests) and rounds < MAX_APPROVAL_ROUNDS:
            rounds += 1
            requests = result.output
            approvals = {}

            for call in requests.approvals:
                # Metadata travels on DeferredToolRequests, keyed by call id.
                metadata = requests.metadata.get(call.tool_call_id, {})
                approvals[call.tool_call_id] = ask_human(call, metadata)

            result = agent.run_sync(
                message_history=result.all_messages(),
                deferred_tool_results=requests.build_results(approvals=approvals),
                output_type=[AnalysisResult, DeferredToolRequests],
            )

        if isinstance(result.output, DeferredToolRequests):
            print(f"\n=== APPROVAL BUDGET EXHAUSTED after {MAX_APPROVAL_ROUNDS} rounds ===")
        else:
            print("\n=== OUTPUT ===")
            print(result.output.model_dump_json(indent=2)[:1000])

    print(f"\ntrace_id: {trace_id}")


def show_policy() -> None:
    samples = [
        "SELECT 1",
        "UPDATE transactions SET unit_price = 6.99 WHERE id = 1503",
        "DELETE FROM transactions WHERE store_id = 42",
        "DELETE FROM transactions",
        "DROP TABLE stores",
        "UPDATE a SET b = 1 WHERE id = 1; DELETE FROM stores",
    ]
    print("The gate reads arguments, not tool names:\n")
    for query in samples:
        tool = "run_sql" if query.upper().startswith("SELECT") else "run_write_sql"
        d = approval.assess(tool, {"query": query})
        print(f"  {d.risk:<10} {'ask human' if d.requires_approval else 'no prompt':<10} {query[:52]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="store_true", help="print the decision table, no model")
    parser.add_argument("--ask", help="a question that may require a write")
    args = parser.parse_args()

    if args.policy:
        show_policy()
        return

    if not args.ask:
        parser.error("pass --ask or --policy")

    writable_db()
    print(f"writes go to {WRITE_DB} (a copy)")

    tracer_provider = setup_tracing(service_name="phase5-agent")
    meter_provider = setup_metrics(service_name="phase5-agent")
    Agent.instrument_all()

    asyncio.set_event_loop(asyncio.new_event_loop())
    run_with_approvals(args.ask)

    tracer_provider.force_flush()
    meter_provider.force_flush()
    tracer_provider.shutdown()
    meter_provider.shutdown()


if __name__ == "__main__":
    main()
