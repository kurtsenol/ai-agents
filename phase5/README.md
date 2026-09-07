# Phase 5 — Observability, Security, CI

Phase 4 produced a working agent. Phase 5 is about the things you need before
that agent is allowed anywhere near production: seeing what it does, knowing
what it costs, catching it when it makes things up, and bounding what it can
break.

Nothing here replaces the phase 4 agent. Everything wraps it.

---

## Run it

Two containers and two seed scripts. Both containers must be up before
anything else works.

```bash
# 1. the observability stack (Grafana + Tempo + Prometheus in one image)
cd phase5
docker compose up -d              # ~30s on first start

# 2. Elasticsearch (from phase 2)
docker start es-dev               # or the run command in phase2/README.md

# 3. the data — retail.db is generated, not committed
uv run --project ../phase2 ../phase2/seed_db.py
uv run --project ../phase2 ../phase2/seed_elastic.py

# 4. credentials
export LITELLM_API_KEY=...
export LITELLM_BASE_URL=...

# 5. the dashboard
uv run grafana/import.py          # -> http://localhost:3000/d/phase5-agent
```

Then, cheapest first:

```bash
uv run pytest -q                       # 48 tests, no model, ~10s
uv run step7_scope.py                  # tool limits, no model
uv run step9_defend.py --scan-only     # injection scanner, no model
uv run step10b_sandbox.py              # container limits, no model
uv run step10_approve.py --policy      # approval decision table, no model

uv run step1_trace.py                  # one traced run           (~$0.05)
uv run step6_mcp.py                    # the same, over MCP       (~$0.05)
uv run eval_traced.py --runs 3         # the golden set           (~$1.50)
uv run ci_gate.py                      # score it against baseline
```

---

## What is here

| Concern | Files |
|---|---|
| Tracing | `otel_setup.py`, `span_dump.py`, `docker-compose.yml` |
| Cost & latency | `pricing.py`, `agent_metrics.py`, `step3_cost.py` |
| Eval ↔ trace | `eval_traced.py`, `eval_report.py` |
| Hallucination | `grounding.py`, `judge.py`, `step5_ground.py` |
| MCP client | `agent_mcp.py`, `runners.py`, `step6_mcp.py` |
| Tool scoping | `step7_scope.py` (limits live in `phase4/tools_core.py`) |
| Prompt injection | `injection_payloads.py`, `untrusted.py`, `step8_inject.py`, `step9_defend.py` |
| Approval gate | `approval.py`, `step10_approve.py` |
| Sandbox | `sandbox.py`, `code_tool.py`, `step10b_sandbox.py` |
| CI | `tests/`, `ci_gate.py`, `baseline.json`, `../.github/workflows/eval.yml` |
| Dashboard | `grafana/dashboard.json`, `grafana/import.py` |

---

## The ideas, one line each

1. **A run is a tree, not a log.** Parent/child and duration are what a log
   cannot give you.
2. **OpenTelemetry standardises transport, not meaning.** PydanticAI emits
   `gen_ai.*`, the LangGraph instrumentation emits `llm.*`, and the two share
   **zero** attribute keys. A dashboard cannot be built on either.
3. **So emit your own metrics.** Normalise in code, name the metrics
   yourself, and framework churn never reaches the dashboard.
4. **`trace_id` is the only bridge** between "7/10 passed" and "here is why".
5. **Groundedness is not ground truth.** Ground truth needs a golden set;
   groundedness needs nothing, so only it can run in production.
6. **A process boundary removes ambient capability.** Behind MCP the agent
   holds no database handle, so "it must not write" stops being a promise.
7. **An allowlist bounds what runs, not what it costs.** A permitted
   `SELECT` materialised 4.1M rows before the row cap applied.
8. **Tool output is untrusted input.** Anyone who can leave a review can
   write into the agent's context.
9. **Measure a defence against its own absence,** or it is a belief.
10. **Approve on arguments, not tool names** — and on measured blast radius,
    not on whether a `WHERE` clause is present.
11. **The sandbox boundary is the kernel, not the interpreter.** A Python
    blocklist blocked `print('total cost')` and allowed a real escape.
12. **Non-determinism is gated with thresholds, not ignored.**

---

## Measured

Over the runs recorded so far (`framework=pydantic`, Sonnet 4.6 via the
LiteLLM proxy):

| | |
|---|---|
| Agent runs | 37 |
| Input / output tokens | 758,343 / 37,153 |
| Cost (first-party list prices) | $2.83 |
| Mean run latency | 15.2 s |
| Prompt-cache tokens | 0 — the proxy is not caching |

**Eval.** `baseline.json`: 58 checks across 10 golden items × 3 runs, 100%
pass. Phase 4's flaky `g03:transaction_ids` (2/3) now passes 3/3.

**Frameworks (step 2-3).** On the same 3 questions, LangGraph used 3.3× the
input tokens and 1.9× the tool calls of PydanticAI, for 2.9× the cost. Only
PydanticAI nests third-party spans correctly; the callback-based LangChain
instrumentation loses OTel context, so the Elasticsearch span attaches to the
trace root instead of to the tool that made the call.

**Injection (step 8-9).** Three payloads, each delivered into the model's
context through a real review. None landed, with or without the fence — the
model resisted on every delivered run and twice said so in its own summary.
That is a baseline, not a guarantee.

**Sandbox (step 10b).** Network refused, host filesystem absent, writes
refused outside `/tmp`, memory bomb killed at 256 MB, infinite loop and fork
bomb killed at 10 s, `setuid(0)` refused. The same cases against `exec()` with
a blocklist: three could not be run at all without taking down the host
process, two harmless strings were blocked for containing `os` inside `hosts`
and `cost`, and the classic `().__class__.__base__.__subclasses__()` escape
read `/etc/passwd`.

**Tests.** 48, no API key, ~10 s. Every one of them is a bug this phase
actually produced — see `tests/` docstrings.

---

## Known open

- **Judge and groundedness are not in CI.** `ci_gate.py` scores the golden set
  only. Wiring `grounding.check` into `eval_traced.py` is the obvious next
  step and was left out to keep step 11 to one idea.
- **Only PydanticAI runs the eval.** `eval_traced.py` calls phase 4's
  `record_run`. The LangGraph failures measured in phase 4 (`g03` 0/3, `g04`,
  `g05`) are still unscored here.
- **The price table is first-party.** Access is Bedrock via a LiteLLM proxy,
  which has its own rates. The shape of the calculation is right; the absolute
  dollars are indicative.
- **The injection scanner misses politely worded payloads.** `scope_creep`
  produces zero hits and this is pinned by a test. Against that payload the
  only working control is the model itself.
- **The free-text number check has known false positives.** `699.00` in a
  tool result and `699.0` in a summary do not match; it is a warning, never a
  violation, for exactly this reason.
- **Counters come from short-lived processes,** so every dashboard query uses
  `max_over_time($__range)` instead of `increase()`. Phase 6 turns the agent
  into a service and this reverts.
- **`estimate_rows` runs the statement and rolls it back.** Exact, and it
  takes a write lock twice. Safe here because SQLite rolls back an uncommitted
  transaction on close and on crash recovery — that guarantee is the engine's,
  not ours.

---

## Phase 4 debts, all closed

| # | Debt | Closed in |
|---|---|---|
| 1 | `retail_agent.py` duplicated tool bodies instead of using `tools_core.py` | step 6 (503 → 336 lines) |
| 2 | MCP servers never connected to the agent, only to Claude Code | step 6 |
| 3 | `Finding.transaction_ids` undefined for a negative finding | step 5 — falls out of the grounding rule |
| 4 | `size=10000` was a silent ceiling in `score.py` | step 4 — now raises |
| 5 | MCP servers returned errors as ordinary results | step 7 — `is_error` set, and `ModelRetry` raised |
