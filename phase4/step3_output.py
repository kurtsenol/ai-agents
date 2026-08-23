from retail_agent import agent, build_deps

from datetime import datetime
from pydantic import BaseModel, Field, model_validator

from typing import Annotated, Literal


# Whether the question that was asked could be answered at all.
# This is independent of whether findings were produced: an analysis can
# report real findings while still failing to answer what was asked.
AnswerStatus = Literal[
    "answered",
    "not_in_data",
    "ambiguous_question",
    "inconclusive",
]

class ReviewEvidence(BaseModel):
    text: str = Field(
        description=(
            "The review text exactly as returned by Elasticsearch. "
            "Preserve the original text verbatim; do not translate, paraphrase, "
            "summarize, or modify it."
        )
    )
    ts: datetime = Field(
        description="The review timestamp exactly as returned by Elasticsearch."
    )
    rating: Annotated[int, Field(ge=1, le=5)] = Field(
        description="The review rating exactly as returned by Elasticsearch."
    )


class Finding(BaseModel):
    claim: str = Field(
        description="A concise statement of the finding supported by the evidence."
    )
    transaction_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Transaction IDs returned by the database query that directly support "
            "this finding. Do not invent or infer transaction IDs."
        ),
    )
    review_evidence: list[ReviewEvidence] = Field(
        default_factory=list,
        description="Review evidence supporting this finding, when applicable.",
    )


class AnalysisResult(BaseModel):
    """Call this tool when the analysis is complete and you are ready to provide
    the final answer. Only report findings that are directly supported by the
    evidence returned by the available tools. Findings and answering the
    question are separate obligations: report supported findings when useful,
    and use answer_status to state whether the question itself was answered.
    If answer_status is not "answered", provide a specific status_reason
    explaining what prevented a definite answer.
    """

    summary: str = Field(
        description="A concise human-readable summary of the overall analysis."
    )

    answer_status: AnswerStatus = Field(
        description=(
            "State whether the question itself was answered. Use "
            "'answered' when the evidence establishes the answer, including "
            "when the answer is that there are no matching findings. Use "
            "'not_in_data' when the database or index does not contain the "
            "information needed to answer the question at all. Use "
            "'ambiguous_question' when you cannot determine what the user "
            "is asking. Use 'inconclusive' when you investigated the question "
            "but the available evidence does not establish the answer either "
            "way. In particular, finding no matches in a search is not enough "
            "to conclude 'not_in_data' or 'answered': use 'inconclusive' when "
            "the search itself may have been incomplete or inappropriate."
        )
    )

    status_reason: str | None = Field(
        default=None,
        description=(
            "Required when answer_status is not 'answered'. Explain "
            "specifically what prevented the question from being answered: "
            "what information is absent, what part of the question is "
            "ambiguous, or why the available investigation could not establish "
            "the answer. Do not merely repeat the status value."
        ),
    )

    findings: list[Finding] = Field(
        default_factory=list,
        description=(
            "Evidence-backed findings produced during the analysis. Findings "
            "are independent of answer_status: report relevant findings even "
            "when they do not answer the user's question. For example, a "
            "'not_in_data' answer may still contain findings about related "
            "information that is present in the database or index. Include "
            "only claims directly supported by tool evidence."
        ),
    )

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.answer_status != "answered" and not self.status_reason:
            raise ValueError(
                "When answer_status is not 'answered', provide a specific "
                "status_reason explaining what prevented the question from "
                "being answered. Do not merely repeat the status value."
            )

        return self

if __name__ == "__main__":

    result = agent.run_sync(
        "42 numaralı mağazada fiyat anormalliği var mı?",
        deps=build_deps(),
        output_type = AnalysisResult
    )

    print("=== OUTPUT ===")
    print(result.output.model_dump_json(indent=2))

    print("\n=== USAGE ===")
    print(result.usage)

    for msg in result.all_messages():
        for part in msg.parts:
            if type(part).__name__ == "ToolCallPart":
                print("TOOL CALL")
                print("  name:", part.tool_name)
                print("  args:", part.args_as_dict())

            elif type(part).__name__ == "ToolReturnPart":
                print("TOOL RETURN")
                print("  name:", part.tool_name)
                print("  content:", str(part.content)[:200])

