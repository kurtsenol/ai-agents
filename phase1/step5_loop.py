import ast
import os

import anthropic

from chat import _safe_eval

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)
MODEL = "us.anthropic.claude-sonnet-4-6"

SYSTEM = "You are a terse assistant. Answer everything less than 20 words."

# A tool is just a description + a JSON Schema for its arguments.
# The model reads the description to decide WHEN to call it.
TOOLS = [{
    "name": "calculator",
    "description": "Evaluate an arithmetic expression, e.g. '2 * (3 + 4)'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression"},
        },
        "required": ["expression"],
    },
}]


history = []

while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break

    history.append({"role": "user", "content": question})

    while True:

        r1 = client.messages.create(
            model=MODEL, max_tokens=500, tools=TOOLS, system=SYSTEM, messages=history,
            )

        for b in r1.content:
            if b.type == "text":
                print("claude>", b.text)
        print("stop_reason:", r1.stop_reason)

        history.append({"role": "assistant", "content": r1.content})

        if r1.stop_reason != "tool_use":
            break

        tool_use = [b for b in r1.content if b.type == "tool_use" ]

        print("tool_use:", tool_use)
        content_list =  []

        for t in tool_use:
            content_list.append({"type": "tool_result",
                                  "tool_use_id": t.id,
                                  "content": str(_safe_eval(ast.parse(t.input["expression"], mode="eval")))
                                    })

        history.append({"role": "user", "content": content_list})
