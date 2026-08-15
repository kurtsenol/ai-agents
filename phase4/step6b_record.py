import asyncio
import time
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import AgentRunResultEvent, FunctionToolCallEvent

from retail_agent import agent, build_deps
from step3_output import AnalysisResult


RUNS_FILE =  db_path=(Path(__file__).parent / "runs.jsonl")   


class RunRecord(BaseModel):
    run_id: str | None
    conversation_id: str | None
    question: str

    tool_calls: list[str]

    requests: int | None
    tool_calls_count: int
    input_tokens: int | None
    output_tokens: int | None

    duration_seconds: float

    ok: bool
    error: str | None


async def record_run(question: str) -> RunRecord:
    deps = build_deps()

    start = time.perf_counter()

    tool_calls: list[str] = []
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
                    tool_calls.append(event.part.tool_name)

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

    else:
        run_id = None
        conversation_id = None
        requests = None
        input_tokens = None
        output_tokens = None
        ok = False

    record = RunRecord(
        run_id=run_id,
        conversation_id=conversation_id,
        question=question,
        tool_calls=tool_calls,
        requests=requests,
        tool_calls_count=len(tool_calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration_seconds,
        ok=ok,
        error=error,
    )

    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")

    return record


def load_records() -> list[RunRecord]:
    records = []

    with RUNS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(RunRecord.model_validate_json(line))

    return records


async def main():
    question = "42 numaralı mağazada fiyat anormalliği var mı?"

    for _ in range(3):
        await record_run(question)

    records = load_records()

    print(
        f"{'Run':<12} "
        f"{'OK':<5} "
        f"{'Requests':<10} "
        f"{'Tool calls':<12} "
        f"{'Input':<10} "
        f"{'Output':<10} "
        f"{'Duration':<12}"
    )

    print("-" * 75)

    for record in records[-3:]:
        print(
            f"{str(record.run_id)[:10]:<12} "
            f"{str(record.ok):<5} "
            f"{str(record.requests):<10} "
            f"{record.tool_calls_count:<12} "
            f"{str(record.input_tokens):<10} "
            f"{str(record.output_tokens):<10} "
            f"{record.duration_seconds:<12.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())