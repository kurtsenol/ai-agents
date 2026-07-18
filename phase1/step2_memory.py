import os

import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)
MODEL = "us.anthropic.claude-sonnet-4-6"

# --- Call 1: introduce yourself ---
r1 = client.messages.create(
    model=MODEL, max_tokens=200,
    messages=[{"role": "user", "content": "My name is Senol. Say hi."}],
)
print("call 1:", r1.content[0].text)

# --- Call 2: NO history — just the new question ---
r2 = client.messages.create(
    model=MODEL, max_tokens=200,
    messages=[{"role": "user", "content": "What is my name?"}],
)
print("\ncall 2 (no history):", r2.content[0].text)

# --- Call 3: same question, WITH history ---
history = [
    {"role": "user", "content": "My name is Senol. Say hi."},
    {"role": "assistant", "content": r1.content[0].text},  # what IT said, fed back
    {"role": "user", "content": "What is my name?"},
]
r3 = client.messages.create(model=MODEL, max_tokens=200, messages=history)
print("\ncall 3 (with history):", r3.content[0].text)

print(f"\ncall 2 input tokens: {r2.usage.input_tokens}")
print(f"call 3 input tokens: {r3.usage.input_tokens}")
