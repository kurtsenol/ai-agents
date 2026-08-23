"""
Step 7b — run the golden set and record what happened.

Reads eval/golden_set.jsonl, runs every item N_RUNS times, and appends one
JSON line per run to eval/runs.jsonl.

This module MEASURES. It does not score and it does not present.
Scoring happens in step 7c, reading the file this produces.

Prerequisite: reseed the data first (see eval/README.md).
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from retail_agent import agent, build_deps          # noqa: E402
from step3_output import AnalysisResult             # noqa: E402


EVAL_DIR = Path(__file__).parent
GOLDEN_FILE = EVAL_DIR / "golden_set.jsonl"
RUNS_FILE = EVAL_DIR / "runs.jsonl"

N_RUNS = 3

RESULT_PREVIEW_CHARS = 2000  # Keep enough context for later re-analysis without bloating runs.jsonl.


class ToolCall(BaseModel):
    """One tool invocation: what was asked, and what came back."""

    tool_name: str
    args: dict

    result_preview: str | None = None
    result_length: int | None = None

    # Replaces the old `result_empty`. That field measured whether the tool
    # returned an empty string — which never happens, because the tools return
    # informative prose like "No rows returned." instead. The tools now report
    # machine-readable facts via ToolReturn(metadata=...), and the eval reads
    # those instead of guessing from the prose.
    result_metadata: dict | None = None


class RunRecord(BaseModel):
    """Everything one agent run produced, in a form 7c can score."""

    # identity
    run_id: str | None
    conversation_id: str | None
    item_id: str
    question: str

    # PATH
    tool_calls: list[ToolCall]

    # OUTCOME
    output: dict | None

    # COST
    requests: int | None
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float

    # status
    ok: bool
    error: str | None


def load_golden_set() -> list[dict]:
    items = []

    with GOLDEN_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def append_record(record: RunRecord) -> None:
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


async def record_run(item_id: str, question: str) -> RunRecord:
    """Run one golden item once and return its record. Prints nothing.

      FunctionToolCallEvent.part    -> ToolCallPart
                                       .tool_name, .args_as_dict(), .tool_call_id
      FunctionToolResultEvent.part  -> ToolReturnPart
                                       .content, .tool_call_id
      AgentRunResultEvent.result    -> AgentRunResult
                                       .output, .usage, .run_id, .conversation_id
    """
    deps = build_deps()
    start = time.perf_counter()

    # Calls and results arrive as separate events, and with parallel tool use
    # they do not arrive in matching order. Something has to pair them.
    calls_by_id: dict[str, ToolCall] = {}
    call_order: list[str] = []

    result = None
    error = None

    try:
        async with agent.run_stream_events(
            question,
            deps=deps,
            output_type=AnalysisResult,
        ) as events:

            async for event in events:

                if isinstance(event, FunctionToolCallEvent):
                    part = event.part
                    tool_call_id = part.tool_call_id

                    calls_by_id[tool_call_id] = ToolCall(
                        tool_name=part.tool_name,
                        args=part.args_as_dict(),
                    )
                    call_order.append(tool_call_id)

                if isinstance(event, FunctionToolResultEvent):
                    part = event.part
                    tool_call_id = part.tool_call_id

                    tool_call = calls_by_id.get(tool_call_id)

                    if tool_call is not None:
                        content = str(part.content)

                        tool_call.result_length = len(content)
                        tool_call.result_preview = content[:RESULT_PREVIEW_CHARS]

                        tool_call.result_metadata = getattr(part, "metadata", None)

                if isinstance(event, AgentRunResultEvent):
                    result = event.result

    except Exception as exc:
        error = str(exc)

    duration_seconds = time.perf_counter() - start

    if result is not None:
        usage = result.usage

        run_id = result.run_id
        conversation_id = result.conversation_id
        requests = usage.requests
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        ok = error is None

        output = result.output.model_dump(mode="json")

    else:
        run_id = None
        conversation_id = None
        requests = None
        input_tokens = None
        output_tokens = None
        output = None
        ok = False

    return RunRecord(
        run_id=run_id,
        conversation_id=conversation_id,
        item_id=item_id,
        question=question,
        tool_calls=[calls_by_id[cid] for cid in call_order],
        output=output,
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration_seconds,
        ok=ok,
        error=error,
    )


async def main() -> None:
    items = load_golden_set()

    print(f"{len(items)} items x {N_RUNS} runs = {len(items) * N_RUNS} runs")
    print("Reseed the data first if you have not — see eval/README.md\n")

    started = time.perf_counter()
    completed = 0

    for item in items:
        item_id = item["id"]
        question = item["question"]

        for run_number in range(1, N_RUNS + 1):
            record = await record_run(item_id, question)
            append_record(record)

            completed += 1

            status = "OK" if record.ok else "FAILED"
            print(
                f"[{completed}/{len(items) * N_RUNS}] "
                f"{item_id} run {run_number}/{N_RUNS}: {status}"
            )

    elapsed = time.perf_counter() - started

    print(f"\n{completed} runs in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Wrote {RUNS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
