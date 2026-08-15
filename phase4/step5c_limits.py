from pydantic_ai import UsageLimits, UsageLimitExceeded, capture_run_messages
from pydantic_ai.messages import ModelRequest, ModelResponse

from retail_agent import agent, build_deps
from step3_output import AnalysisResult


QUESTION = "42 numaralı mağazada fiyat anormalliği var mı?"


def print_messages(label: str, messages, *, exception=None, result=None) -> None:
    requests = [m for m in messages if isinstance(m, ModelRequest)]
    responses = [m for m in messages if isinstance(m, ModelResponse)]

    print(f"\n--- {label} ---")

    if exception is not None:
        print("Outcome: LIMIT EXCEEDED")
        print(f"Exception: {exception}")
    else:
        print("Outcome: COMPLETED")

    print(f"ModelRequest count: {len(requests)}")
    print(f"ModelResponse count: {len(responses)}")

    if messages:
        last = messages[-1]
        print(f"Last message type: {type(last).__name__}")
        print(
            "Last message parts:",
            [type(part).__name__ for part in last.parts],
        )
    else:
        print("Last message type: <none>")
        print("Last message parts: <none>")

    if result is not None:
        print("\nResult:")
        print(result.output)

        print("\nUsage:")
        print(result.usage)


def run_a(deps) -> None:
    result = None
    exception = None

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(
                QUESTION,
                deps=deps,
                output_type=AnalysisResult,
                usage_limits=UsageLimits(request_limit=1),
            )
        except UsageLimitExceeded as e:
            exception = e

    print_messages(
        "A) request_limit=1",
        messages,
        exception=exception,
        result=result,
    )


def run_b(deps) -> None:
    result = None
    exception = None

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(
                QUESTION,
                deps=deps,
                output_type=AnalysisResult,
                usage_limits=UsageLimits(tool_calls_limit=1),
            )
        except UsageLimitExceeded as e:
            exception = e

    print_messages(
        "B) tool_calls_limit=1",
        messages,
        exception=exception,
        result=result,
    )


def run_c(deps) -> None:
    result = None
    exception = None

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(
                QUESTION,
                deps=deps,
                output_type=AnalysisResult,
                usage_limits=UsageLimits(
                    request_limit=10,
                    total_tokens_limit=200_000,
                ),
            )
        except UsageLimitExceeded as e:
            exception = e

    print_messages(
        "C) request_limit=10, total_tokens_limit=200_000",
        messages,
        exception=exception,
        result=result,
    )


def main() -> None:
    deps = build_deps()

    run_a(deps)
    run_b(deps)
    run_c(deps)


if __name__ == "__main__":
    main()