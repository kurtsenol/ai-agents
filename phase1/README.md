# Phase 1 — LLM Fundamentals Deliverable

A chatbot on the raw Claude API (no framework) with **one tool call** and
**structured outputs** — the Phase 1 deliverable from `../AI_AGENTS_ROADMAP.md`.

## Setup

1. Get an API key from https://platform.claude.com/ (Settings → API keys).
2. Export it (add to `~/.zshrc` to persist):

   ```sh
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

Dependencies are already installed in the project venv (`.venv/`, managed by uv).

## Run

```sh
uv run chat.py      # multi-turn chatbot with a calculator tool
uv run extract.py   # structured outputs: text -> validated Pydantic object
```

Try in the chat: `what is 1234 * 5678 + 9?` — watch the `[tool]` lines show
the model requesting the calculator and your code executing it.

## What each file teaches

| File | Concepts |
|---|---|
| `chat.py` | Stateless Messages API, conversation history as a list, system prompts, tool schemas, the `tool_use` → `tool_result` round-trip, token usage / context window growth |
| `extract.py` | Structured outputs via `messages.parse()` + Pydantic — schema-guaranteed JSON instead of parsing free text |

## Study prompts (do these before Phase 2)

- In `chat.py`, print `history` after a tool call. Identify the four message
  shapes: user text, assistant with `tool_use`, user with `tool_result`,
  assistant final text.
- Ask a question needing **two** calculations at once — Claude may emit two
  `tool_use` blocks in one response (parallel tool calls). Note how the code
  returns all results in a single user message.
- Watch `input_tokens` grow each turn. What would you do when it approaches
  the context window? (That's context engineering — and Phase 3's chunking.)

The inner `while stop_reason == "tool_use"` loop in `chat.py` is the embryo of
the full agent loop you'll build from scratch in Phase 2.
