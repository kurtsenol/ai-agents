"""Step 4 - run the golden set with tracing on, and keep the trace id.

    uv run eval_traced.py            all 10 items, 1 run each
    uv run eval_traced.py --runs 3   the phase 4 cadence
    uv run eval_traced.py --items g01,g03

This does NOT reimplement the phase 4 eval. It imports `record_run` from
phase4/eval/run_golden.py - the same function that produced runs.jsonl -
wraps each call in a span, and writes the same record with one field added:

    trace_id

That single field is the bridge. Everything else in phase 5 hangs off it.

Writes out/runs_traced.jsonl, which phase4/eval/score.py can score as-is
(an unknown extra field is ignored) and which eval_report.py can turn into
per-run links.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))
sys.path.insert(0, str(REPO / "phase4/eval"))

from opentelemetry import trace

from agent_metrics import record_run as record_metrics
from otel_setup import setup_metrics, setup_tracing
from span_dump import SpanDump

SERVICE_NAME = "phase5-agent"
OUT_FILE = Path(__file__).parent / "out" / "runs_traced.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="runs per golden item")
    parser.add_argument("--items", default="", help="comma-separated item ids")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    tracer_provider = setup_tracing(service_name=SERVICE_NAME)
    meter_provider = setup_metrics(service_name=SERVICE_NAME)

    from pydantic_ai import Agent

    Agent.instrument_all()

    # Imported after instrument_all() so the agent it builds is instrumented.
    from run_golden import load_golden_set, record_run

    items = load_golden_set()
    if args.items:
        wanted = {s.strip() for s in args.items.split(",")}
        items = [i for i in items if i["id"] in wanted]

    tracer = trace.get_tracer("phase5.eval")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(items) * args.runs
    done = 0

    with out_path.open("w", encoding="utf-8") as out:
        for item in items:
            for run_number in range(1, args.runs + 1):
                dump = SpanDump()
                tracer_provider.add_span_processor(dump)

                started_wall = datetime.now(timezone.utc)
                started = time.perf_counter()

                with tracer.start_as_current_span("eval.item") as span:
                    span.set_attribute("phase5.framework", "pydantic")
                    span.set_attribute("phase5.question_id", item["id"])
                    span.set_attribute("phase5.run_number", run_number)
                    trace_id = format(span.get_span_context().trace_id, "032x")

                    record = asyncio.run(record_run(item["id"], item["question"]))

                duration_s = time.perf_counter() - started

                record_metrics(
                    framework="pydantic",
                    question_id=item["id"],
                    question=item["question"],
                    trace_id=trace_id,
                    duration_s=duration_s,
                    spans=dump.spans,
                )

                # The record phase 4 already knows how to score, plus the bridge.
                row = record.model_dump(mode="json")
                row["trace_id"] = trace_id
                row["run_number"] = run_number
                row["started_at"] = started_wall.isoformat()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

                done += 1
                status = "OK" if record.ok else "FAILED"
                print(f"[{done}/{total}] {item['id']} run {run_number}: {status}  {trace_id}")

    tracer_provider.force_flush()
    meter_provider.force_flush()
    tracer_provider.shutdown()
    meter_provider.shutdown()

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
