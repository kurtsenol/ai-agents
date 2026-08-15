
"""
LABORATORY:

This file is an isolated experiment for observing PydanticAI's
output validation and retry mechanism.

This is not intended to define permanent application behavior.

DO NOT MODIFY retail_agent.
"""

from pydantic_ai import (
    Agent,
    ModelRetry,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.messages import ModelRequest

from model import build_model


# ---------------------------------------------------------------------------
# 1. Create a separate agent for this experiment.
# ---------------------------------------------------------------------------

lab_agent = Agent(
    build_model(),
    output_type=str,
)


# ---------------------------------------------------------------------------
# 2. The output validator always raises ModelRetry.
# ---------------------------------------------------------------------------

@lab_agent.output_validator
def always_retry(output: str) -> str:
    raise ModelRetry(
        "LAB_RETRY: output validator deliberately rejected the output"
    )


# ---------------------------------------------------------------------------
# 3. Capture all messages from the run, including messages produced
#    before an exception terminates the run.
# ---------------------------------------------------------------------------

with capture_run_messages() as messages:
    try:
        result = lab_agent.run_sync(
            "What is the capital of Turkey?",
            retries={"output": 1},
        )

    except UnexpectedModelBehavior as exc:
        print("=== UNEXPECTED MODEL BEHAVIOR ===")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception message: {exc}")

    else:
        print("=== RUN SUCCEEDED ===")
        print(f"Result: {result.output}")

    request_count = sum(
        isinstance(message, ModelRequest)
        for message in messages
    )

    print(f"Request count: {request_count}")

    print("\n=== CAPTURED MESSAGES ===")
    for i, message in enumerate(messages, start=1):
        print(
            f"\n--- Message {i}: {type(message).__name__} ---"
        )
        print(message)