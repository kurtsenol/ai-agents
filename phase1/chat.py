"""Phase 1 deliverable: a chatbot on the raw Claude API with one tool call.

Concepts demonstrated (map these to your roadmap checklist):
- The Messages API is STATELESS: you send the full conversation history
  on every request. "Memory" is just a Python list you keep appending to.
- Tool calling: you describe a tool with a JSON Schema; the model doesn't
  run anything itself — it returns a `tool_use` block asking YOU to run it,
  and you send the result back in a `tool_result` block.
- The context window: every turn, the whole history is re-sent and re-read
  by the model. Watch `usage` grow as the conversation gets longer.

Run:  uv run chat.py
"""

import ast
import operator

import anthropic

MODEL = "claude-opus-4-8"

# The system prompt sets behavior for the whole conversation. It is sent
# separately from `messages` and applies to every turn.
SYSTEM = (
    "You are a concise assistant helping a data scientist learn about LLMs. "
    "Use the calculator tool for any arithmetic instead of computing it yourself."
)

# A tool definition is just metadata: name, description, and a JSON Schema
# for the input. The description is what the model reads to decide WHEN to
# call the tool — write it like documentation for the model.
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression, e.g. '2 * (3 + 4)'. "
            "Call this for any calculation instead of doing math yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A pure arithmetic expression using + - * / ** ( )",
                }
            },
            "required": ["expression"],
        },
    }
]

# Safe expression evaluator — never eval() raw model output. The model's
# tool input is untrusted, exactly like user input.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def run_tool(name: str, tool_input: dict) -> str:
    """Execute a tool the model asked for and return the result as a string."""
    if name == "calculator":
        try:
            result = _safe_eval(ast.parse(tool_input["expression"], mode="eval"))
            return str(result)
        except Exception as e:
            # Return errors as text — the model reads them and can retry
            # or explain the problem to the user.
            return f"Error: {e}"
    return f"Error: unknown tool {name!r}"


def main() -> None:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    history: list[dict] = []  # the entire conversation lives in this list

    print("Chat with Claude (Ctrl-C or 'quit' to exit)\n")
    while True:
        user_input = input("you> ").strip()
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})

        # Inner loop: keep calling the API until the model stops asking
        # for tools. This is the seed of the "agent loop" you'll build
        # from scratch in Phase 2.
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                tools=TOOLS,
                messages=history,
            )

            # Always append the assistant's full content (including any
            # tool_use blocks) — the API rejects a tool_result whose
            # matching tool_use is missing from history.
            history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break  # the model produced a final answer

            # The model asked for one or more tools: run them all and send
            # every result back in a SINGLE user message.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({block.input})")
                    output = run_tool(block.name, block.input)
                    print(f"  [tool] -> {output}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,  # must match the request
                            "content": output,
                        }
                    )
            history.append({"role": "user", "content": tool_results})

        # Print the model's text blocks (content can also hold other types).
        for block in response.content:
            if block.type == "text":
                print(f"claude> {block.text}")

        # Token accounting: input_tokens grows with history length — this is
        # the context window in action, and why context engineering matters.
        u = response.usage
        print(f"  [usage] in={u.input_tokens} out={u.output_tokens}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
