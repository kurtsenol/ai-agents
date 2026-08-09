
import ast
import json
import re
import sys
from pathlib import Path


def extract_json_data(raw: str):
    """
    Extract the json_data={...} portion from an Anthropic
    'Request options:' debug line.
    """

    marker = "'json_data': "
    start = raw.find(marker)

    if start == -1:
        return None

    start += len(marker)

    # Find the matching closing brace using Python's AST parser.
    # This handles nested dictionaries/lists correctly.
    text = raw[start:]

    try:
        tree = ast.parse(text, mode="eval")
        return ast.literal_eval(tree.body)
    except (SyntaxError, ValueError):
        # The remaining text may contain non-literal objects.
        # In that case, extract the json_data substring manually.
        pass

    # Fallback: locate balanced {...}
    if not text.startswith("{"):
        return None

    depth = 0
    in_string = False
    quote = None
    escaped = False

    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char in ("'", '"'):
            if not in_string:
                in_string = True
                quote = char
            elif quote == char:
                in_string = False

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    candidate = text[: i + 1]

                    # Replace objects that cannot be parsed.
                    candidate = re.sub(
                        r"<anthropic\.Omit object at 0x[0-9a-fA-F]+>",
                        "None",
                        candidate,
                    )

                    try:
                        return ast.literal_eval(candidate)
                    except (SyntaxError, ValueError):
                        return None

    return None


def format_log(path: str):
    text = Path(path).read_text(encoding="utf-8")

    request_count = 0

    for line in text.splitlines():

        if "Request options: " not in line:
            continue

        request_count += 1

        raw = line.split("Request options: ", 1)[1]

        data = extract_json_data(raw)

        print("=" * 100)
        print(f"REQUEST #{request_count}")
        print("=" * 100)

        if data is None:
            print("Could not extract json_data.")
            continue

        print(json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ))

        print()

    if request_count == 0:
        print("No 'Request options:' entries found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run format_debug.py debug.log")
        sys.exit(1)

    format_log(sys.argv[1])
