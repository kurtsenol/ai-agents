from retail_agent import agent, build_deps

from datetime import datetime
from pydantic import BaseModel, Field, model_validator

from typing import Annotated

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
    evidence returned by the available tools. If the evidence is insufficient
    to support any finding, leave findings empty and explain why in
    insufficient_evidence_reason.
    """
    
    summary: str = Field(
        description="A concise human-readable summary of the overall analysis."
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description=(
            "Findings that are directly supported by the available evidence. "
            "Leave empty when there is not enough evidence to support a finding."
        ),
    )
    insufficient_evidence_reason: str | None = Field(
        default=None,
        description=(
            "Required when findings is empty. Explain specifically why the "
            "available evidence is insufficient to support a finding."
        ),
    )

    @model_validator(mode="after")
    def validate_evidence(self):
        if not self.findings and not self.insufficient_evidence_reason:
            raise ValueError( "Findings is empty, so you must provide "
                              "insufficient_evidence_reason explaining what you searched for "
                              "and why the available evidence was not conclusive." )
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

