"""Phase 1 deliverable, part 2: structured outputs.

Instead of parsing JSON out of free text (fragile), you give the API a
schema and it GUARANTEES the response validates against it. With the
Python SDK the cleanest path is `client.messages.parse()` + a Pydantic
model — the SDK converts the model to a JSON Schema, sends it as
`output_config.format`, and validates the response for you.

This replaces the old "prefill the assistant message with '{'" trick,
which modern Claude models no longer support.

Run:  uv run extract.py
"""

from pydantic import BaseModel, Field

import anthropic

MODEL = "claude-opus-4-8"


# The Pydantic model IS the contract. Field descriptions are sent to the
# model as part of the schema — write them like documentation.
class JobPosting(BaseModel):
    title: str
    company: str
    seniority: str = Field(description="junior, mid, senior, staff, or unknown")
    remote: bool
    skills: list[str] = Field(description="Technical skills mentioned, normalized to lowercase")
    salary_range: str | None = Field(description="As written in the text, or null if absent")


RAW_TEXT = """
We're hiring! Acme Analytics is looking for a Senior AI Engineer to build
LLM-powered features on our data platform. You'll work with Python,
Elasticsearch, and Airflow, and design RAG pipelines end to end.
Fully remote (EU timezones). Compensation: €75k–€95k depending on experience.
"""


def main() -> None:
    client = anthropic.Anthropic()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Extract the job posting details:\n\n{RAW_TEXT}",
            }
        ],
        output_format=JobPosting,
    )

    # parsed_output is a validated JobPosting instance — no json.loads,
    # no regex, no "hope the model didn't add prose around the JSON".
    job = response.parsed_output
    print(f"title:        {job.title}")
    print(f"company:      {job.company}")
    print(f"seniority:    {job.seniority}")
    print(f"remote:       {job.remote}")
    print(f"skills:       {', '.join(job.skills)}")
    print(f"salary_range: {job.salary_range}")


if __name__ == "__main__":
    main()
