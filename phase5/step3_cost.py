"""Step 3 - cost and latency as metrics.

    uv run step3_cost.py pydantic
    uv run step3_cost.py langgraph --n 3

Runs the first N golden-set questions, and for each one emits our own
metrics: duration, tokens, derived cost, tool calls. Traces still go to
Tempo; the metrics now go to Prometheus alongside them.

Why both signals for the same run:
    trace  -> "why did THIS run take 21 seconds?"
    metric -> "what is p95 latency this week, and what did it cost?"
Neither answers the other's question.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opentelemetry import trace

from agent_metrics import record_run
from otel_setup import setup_metrics, setup_tracing
from runners import FRAMEWORKS, instrument_langgraph, run
from span_dump import SpanDump

GOLDEN = Path(__file__).resolve().parent.parent / "phase4/eval/golden_set.jsonl"
SERVICE_NAME = "phase5-agent"


def load_questions(n: int) -> list[dict]:
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("--n", type=int, default=3, help="how many golden questions")
    args = parser.parse_args()

    tracer_provider = setup_tracing(service_name=SERVICE_NAME)
    meter_provider = setup_metrics(service_name=SERVICE_NAME)

    if args.framework == "langgraph":
        instrument_langgraph(tracer_provider)

    tracer = trace.get_tracer("phase5.step3")

    for row in load_questions(args.n):
        # A fresh dump per question, so the spans we hand to record_run belong
        # to exactly one run.
        dump = SpanDump()
        tracer_provider.add_span_processor(dump)

        started = time.perf_counter()
        with tracer.start_as_current_span("step3.question") as span:
            span.set_attribute("phase5.framework", args.framework)
            span.set_attribute("phase5.question_id", row["id"])
            trace_id = format(span.get_span_context().trace_id, "032x")
            try:
                answer = run(args.framework, row["question"])
            except Exception as exc:  # noqa: BLE001 - we want the metric either way
                span.record_exception(exc)
                answer = f"(failed: {exc})"
        duration_s = time.perf_counter() - started

        usage = record_run(
            framework=args.framework,
            question_id=row["id"],
            question=row["question"],
            trace_id=trace_id,
            duration_s=duration_s,
            spans=dump.spans,
        )

        print(
            f"{row['id']}  {duration_s:6.1f}s  "
            f"in={usage.input_tokens:<6} out={usage.output_tokens:<5} "
            f"cache_r={usage.cache_read_tokens:<6} cache_w={usage.cache_write_tokens:<6} "
            f"trace={trace_id}"
        )
        print(f"      {answer[:120]}...")

    # Metrics leave on the reader's 5s timer, so a short script must drain
    # both pipelines before exiting.
    tracer_provider.force_flush()
    meter_provider.force_flush()
    tracer_provider.shutdown()
    meter_provider.shutdown()


if __name__ == "__main__":
    main()
