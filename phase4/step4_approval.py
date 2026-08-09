from retail_agent import agent, build_deps
from step3_output import AnalysisResult

from pydantic_ai import (
    DeferredToolRequests,
    ToolApproved,
    ToolDenied,
)


MESSAGE = "42 numaralı mağazadaki hatalı fiyatlı işlemleri düzelt."


def print_tool_calls(requests: DeferredToolRequests) -> None:
    print("\n=== PENDING TOOL CALLS ===")

    for approval in requests.approvals:
        print(f"\nTool: {approval.tool_name}")
        print(f"Call ID: {approval.tool_call_id}")
        print(f"Args: {approval.args_as_dict()}")


def run_approval_flow() -> None:
    MAX_ITERATIONS = 5

    deps = build_deps()

    print("=== INITIAL REQUEST ===")
    print(MESSAGE)

    result = agent.run_sync(
        MESSAGE,
        deps=deps,
        output_type=[AnalysisResult, DeferredToolRequests],
    )

    print("\n=== FIRST RESULT ===")
    print("Output type:", type(result.output).__name__)

    iteration = 0

    while (
        iteration < MAX_ITERATIONS
        and isinstance(result.output, DeferredToolRequests)
    ):
        print("\n=== APPROVAL ROUND ===")
        print(f"Iteration {iteration + 1} of {MAX_ITERATIONS}")
        iteration += 1

        requests = result.output
        print_tool_calls(requests)

        approvals = {}

        for call in requests.approvals:

            args = call.args_as_dict()
            query = args.get("query", "")

            print("\n" + "=" * 80)
            print("SQL REQUIRES APPROVAL")
            print("=" * 80)
            print(query)

            # Metadata is stored on DeferredToolRequests,
            # keyed by tool_call_id.
            metadata = requests.metadata.get(call.tool_call_id, {})

            risk = metadata.get("risk")
            statement_type = metadata.get("statement_type", "WRITE")
            tool_reason = metadata.get(
                "reason",
                "No reason provided.",
            )

            print(f"Risk: {risk}")
            print(f"Reason: {tool_reason}")

            if risk == "high":
                confirmation = input(
                    f"\nHIGH-RISK {statement_type}: this may affect every row.\n"
                    f"Type '{statement_type} ALL' to approve: "
                )

                approved = (
                    confirmation.strip().upper()
                    == f"{statement_type} ALL"
                )

            else:
                confirmation = input(
                    "\nApprove this write? [y/N]: "
                )

                approved = confirmation.strip().lower() == "y"

            if approved:
                approvals[call.tool_call_id] = ToolApproved()

            else:
                denial_reason = input(
                    "\nWhy was this operation rejected? "
                    "(optional): "
                ).strip()

                denial_message = (
                    "The requested write operation was rejected by the "
                    "human reviewer. Do not execute the rejected SQL. "
                )

                if denial_reason:
                    denial_message += (
                        f"The reviewer gave this reason: "
                        f"{denial_reason} "
                    )

                if risk == "high":
                    denial_message += (
                        "Because the rejected operation was high risk, "
                        "do not broaden the operation or attempt another "
                        "high-risk write without explicit approval."
                    )
                else:
                    denial_message += (
                        "Consider whether a safer, more targeted operation "
                        "can address the request."
                    )

                approvals[call.tool_call_id] = ToolDenied(
                    message=denial_message
                )

        deferred_tool_results = requests.build_results(
            approvals=approvals,
        )

        print("\n=== CONTINUING AGENT RUN ===")

        # The continuation becomes the new state.
        result = agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=deferred_tool_results,
            deps=deps,
            output_type=[AnalysisResult, DeferredToolRequests],
        )

    # The loop can stop for two different reasons:
    #
    # 1. The agent produced a final AnalysisResult.
    # 2. MAX_ITERATIONS was reached while the agent still wants
    #    another approval.
    #
    # Only the second case is a real budget exhaustion.
    if (
        iteration >= MAX_ITERATIONS
        and isinstance(result.output, DeferredToolRequests)
    ):
        print("\n=== APPROVAL BUDGET EXHAUSTED ===")
        print(
            f"The agent did not reach a final result within "
            f"{MAX_ITERATIONS} approval rounds."
        )

    print("\n=== FINAL OUTPUT ===")
    print(result.output)

    print("\n=== FINAL OUTPUT TYPE ===")
    print(type(result.output).__name__)

    print("\n=== MESSAGE HISTORY ===")

    for msg in result.all_messages():
        print(f"\n--- {msg.kind} ---")

        for part in msg.parts:
            print(f"{type(part).__name__}: {part}")


if __name__ == "__main__":
    run_approval_flow()