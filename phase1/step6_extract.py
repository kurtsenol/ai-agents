import os

import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)
MODEL = "us.anthropic.claude-sonnet-4-6"


class ComplaintRecord(BaseModel):
    station_ref: str | None = Field(description="Station name/number if mentioned, else null")
    category: str = Field(description="One of: fuel_quality, staff, pricing, facilities, other")
    sentiment: str = Field(description="positive, neutral, or negative")
    summary_en: str = Field(description="One-sentence English summary")
    urgent: bool = Field(description="True if the issue needs immediate attention")


COMMENT = """
İzmir'deki 4512 nolu istasyonda yakıt aldıktan sonra aracım sarsılmaya başladı.
Pompacı ilgisizdi, kimse yardımcı olmadı. Bir daha buradan yakıt almam!
"""

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=[{
        "name": "record_complaint",
        "description": "Save a structured complaint record extracted from a customer comment.",
        # Pydantic generates the JSON Schema — one source of truth, no duplication:
        "input_schema": ComplaintRecord.model_json_schema(),
    }],
    # THE trick: don't let the model choose — force this exact tool.
    tool_choice={"type": "tool", "name": "record_complaint"},
    messages=[{"role": "user", "content": f"Extract a complaint record:\n{COMMENT}"}],
)

tool_use = next(b for b in response.content if b.type == "tool_use")
# Validate the model's arguments through the SAME Pydantic model — typed object out:
record = ComplaintRecord.model_validate(tool_use.input)

print(type(record))
print(record.model_dump_json(indent=2))
print("\nurgent?", record.urgent, "| category:", record.category)
