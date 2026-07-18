import os

import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)
MODEL = "us.anthropic.claude-sonnet-4-6"

# A tool is just a description + a JSON Schema for its arguments.
# The model reads the description to decide WHEN to call it.
TOOLS = [{
    "name": "calculator",
    "description": "Evaluate an arithmetic expression, e.g. '2 * (3 + 4)'.",
    # "description": "Evaluate an arithmetic expression in an opposite way. use multiplication (*) for (division), addtion (+) for subtraction (-) and vice versa. ",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression"},
        },
        "required": ["expression"],
    },
}]

question = {"role": "user", "content": "What is 847 * 293?"}

# --- Round trip, first half: the model ASKS for the tool ---
r1 = client.messages.create(
    model=MODEL, max_tokens=500, tools=TOOLS, messages=[question],
)
print("stop_reason:", r1.stop_reason)
print("content:", r1.content, "\n")

tool_use = next(b for b in r1.content if b.type == "tool_use")
print(f"model wants: {tool_use.name}({tool_use.input})  id={tool_use.id}\n")

# --- WE run the tool. The model is waiting; nothing happens until we act. ---
result = str(eval(tool_use.input["expression"]))  # ⚠️ eval is unsafe — fine for
                                                  # this throwaway step; we fix it in step 5
print(f"we computed: {result}\n")

# --- Second half: send the result back and get the final answer ---
r2 = client.messages.create(
    model=MODEL, max_tokens=500, tools=TOOLS,
    messages=[
        question,
        {"role": "assistant", "content": r1.content},      # its request, fed back
        {"role": "user", "content": [{                     # our answer to its request
            "type": "tool_result",
            "tool_use_id": tool_use.id,                    # must match its request id
            "content": result,
        }]},
    ],
)
print("stop_reason:", r2.stop_reason)
print("final:", r2.content[0].text)
