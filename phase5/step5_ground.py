"""Step 5 - groundedness, checked where the evidence still exists.

    uv run step5_ground.py --items g01,g03
    uv run step5_ground.py --items g03 --no-judge

Runs the agent through pydantic-ai's event stream, which hands us each tool
result IN FULL - before any of the 2000/2048-character ceilings that make the
recorded copies unusable for this check. Layer 1 (regex, free) runs on every
run; layer 2 (a model) runs only on what layer 1 left open.

The verdict is written back as span attributes and a metric, so a drop in
groundedness shows up on the same dashboard as cost and latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))
sys.path.insert(0, str(REPO / "phase4/eval"))

from opentelemetry import metrics, trace

import grounding
from otel_setup import setup_metrics, setup_tracing

SERVICE_NAME = "phase5-agent"
GOLDEN_FILE = REPO / "phase4/eval/golden_set.jsonl"


async def run_capturing(question: str) -> tuple[dict | None, list[grounding.ToolOutput]]:
    """Run one question, keeping every tool result untruncated in memory."""
    from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
    from retail_agent import agent, build_deps
    from step3_output import AnalysisResult

    args_by_id: dict[str, tuple[str, dict]] = {}
    outputs: list[grounding.ToolOutput] = []
    result = None

    async with agent.run_stream_events(
        question, deps=build_deps(), output_type=AnalysisResult
    ) as events:
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                part = event.part
                args_by_id[part.tool_call_id] = (part.tool_name, part.args_as_dict())

            elif isinstance(event, FunctionToolResultEvent):
                part = event.part
                name, args = args_by_id.get(part.tool_call_id, (part.tool_name, {}))
                outputs.append(
                    grounding.ToolOutput(
                        tool_name=name, args=args, content=str(part.content)
                    )
                )

            elif type(event).__name__ == "AgentRunResultEvent":
                result = event.result.output

    output = result.model_dump(mode="json") if result is not None else None
    return output, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="g01", help="comma-separated golden ids")
    parser.add_argument("--no-judge", action="store_true", help="layer 1 only")
    args = parser.parse_args()

    tracer_provider = setup_tracing(service_name=SERVICE_NAME)
    meter_provider = setup_metrics(service_name=SERVICE_NAME)

    from pydantic_ai import Agent

    Agent.instrument_all()

    meter = metrics.get_meter("phase5.grounding")
    grounded_score = meter.create_histogram(
        "agent.groundedness",
        unit="1",
        description="Fraction of findings with no grounding violation",
    )
    violations_counter = meter.create_counter(
        "agent.grounding.violations",
        unit="{violation}",
        description="Grounding violations, by kind",
    )

    rows = {
        r["id"]: r
        for r in (json.loads(l) for l in GOLDEN_FILE.read_text().splitlines() if l.strip())
    }
    wanted = [rows[i.strip()] for i in args.items.split(",") if i.strip() in rows]

    tracer = trace.get_tracer("phase5.step5")

    for item in wanted:
        with tracer.start_as_current_span("step5.grounding") as span:
            span.set_attribute("phase5.question_id", item["id"])
            span.set_attribute("phase5.framework", "pydantic")
            trace_id = format(span.get_span_context().trace_id, "032x")

            output, outputs = asyncio.run(run_capturing(item["question"]))

            if output is None:
                print(f"{item['id']}: no structured output, skipping")
                continue

            report = grounding.check(output, outputs)

            span.set_attribute("phase5.grounded", report.grounded)
            span.set_attribute("phase5.grounding.score", report.score)
            span.set_attribute("phase5.grounding.violations", len(report.violations))
            span.set_attribute("phase5.grounding.warnings", len(report.warnings))

            grounded_score.record(report.score, {"question_id": "", "framework": "pydantic"})
            for violation in report.violations:
                violations_counter.add(1, {"kind": violation.kind, "framework": "pydantic"})

            print(f"\n{item['id']}  grounded={report.grounded}  score={report.score:.2f}")
            print(f"  tool results captured: {len(outputs)} "
                  f"({sum(len(o.content) for o in outputs)} chars, untruncated)")
            print(f"  findings checked: {report.checked_claims}")

            for violation in report.violations:
                print(f"  VIOLATION [{violation.kind}] {violation.detail}")
            for warning in report.warnings:
                print(f"  warning   [{warning.kind}] {warning.detail}")

            if report.undecidable_claims and not args.no_judge:
                from judge import judge_report

                print(f"  -> judge: {len(report.undecidable_claims)} undecidable claim(s)")
                for claim, verdict in judge_report(report, outputs):
                    print(f"     [{verdict.supported}] {claim[:60]}...")
                    print(f"       {verdict.reason}")
            elif report.undecidable_claims:
                print(f"  {len(report.undecidable_claims)} claim(s) left undecided (judge off)")

            print(f"  trace: {trace_id}")

    tracer_provider.force_flush()
    meter_provider.force_flush()
    tracer_provider.shutdown()
    meter_provider.shutdown()


if __name__ == "__main__":
    main()
