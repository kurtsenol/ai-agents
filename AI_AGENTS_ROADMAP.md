# AI Agents Learning Roadmap

**Goal:** Transition to AI agent engineering with a production-oriented portfolio
**Timeline:** 16 weeks part-time (started July 2026)
**Core insight:** The gap is LLM engineering, not machine learning. Elasticsearch is a genuine RAG head start.

---

## Phase 1 — LLM Fundamentals (weeks 1–2)

**Learn:**
- [ ] Tokens, context windows, embeddings
- [ ] Prompt engineering
- [ ] Tool / function calling
- [ ] Structured outputs
- [ ] Context engineering (what goes in the window and why — this is most of the job)

**Skip:** backprop, training, CUDA, distributed training.

**Resources:**
- Andrej Karpathy — Intro to LLMs
- Anthropic — Prompt Engineering Guide
- Anthropic — *Building Effective Agents* (essay)
- OpenAI API docs

**Deliverable:** a chatbot using a raw API (no framework), with structured outputs and one tool call.

---

## Phase 2 — Build an Agent From Scratch (weeks 3–4)

No frameworks. Write the agent loop yourself:
model → tool call → execute → return result → loop until done.

- [ ] Agent loop (~100–200 lines against a raw API)
- [ ] 2–3 tools: run SQL, query Elasticsearch, execute Python
- [ ] Conversation memory
- [ ] Human-approval step for write operations

Pick up **asyncio, pydantic, httpx, uv** as you build — no dedicated Python phase (drop Poetry, use uv).

**Deliverable:** a ~200-line agent that answers questions against a real database.

---

## Phase 3 — RAG with Elasticsearch (weeks 5–7)

- [ ] Chunking strategies
- [ ] Vector + hybrid search on Elasticsearch
- [ ] Reranking
- [ ] Metadata filtering

**Start evaluation here, not later:**
- [ ] Build a 30–50 question golden set with expected answers *before* tuning anything
- [ ] Measure retrieval hit rate and groundedness on every change

**Deliverable:** document Q&A system with citations + an eval script that scores it.

---

## Phase 4 — Frameworks + MCP (weeks 8–10)

Learn in this order (fast, because you built the loop yourself in Phase 2):
1. [ ] PydanticAI or OpenAI Agents SDK
2. [ ] LangGraph (stateful / multi-agent graphs)
3. [ ] LangChain — only enough to read other people's code

**MCP (Model Context Protocol):**
- [ ] Wrap your SQL database as an MCP server
- [ ] Wrap Elasticsearch as an MCP server
- [ ] Connect them to an agent host (Claude Code, desktop clients)

**Deliverable:** Phase 2 agent rebuilt in a framework, with its tools served over MCP.

---

## Phase 5 — Evaluation, Observability, Security (weeks 11–12)

**Observability:**
- [ ] Tracing: LangSmith or OpenTelemetry + Grafana (you know the Grafana side)
- [ ] Cost + latency tracking
- [ ] Hallucination checks, agent benchmarking

**Security (table stakes for enterprise data-platform agents):**
- [ ] Prompt injection defenses
- [ ] Least-privilege tool scoping
- [ ] Sandboxed code execution
- [ ] Approval gates for destructive actions

**Deliverable:** dashboards + an eval suite that runs in CI.

---

## Phase 6 — Production (weeks 13–14)

- [ ] FastAPI service
- [ ] Docker
- [ ] Redis (caching / queues)
- [ ] Auth + rate limiting
- [ ] Streaming responses
- [ ] CI/CD

Defer Kubernetes until a job requires it — Airflow already taught you the orchestration mindset.

**Deliverable:** one agent deployed behind an authenticated API.

---

## Phase 7 — Portfolio (weeks 15–16 and ongoing)

Build in this order:

1. [ ] **Elasticsearch Investigation Agent** — "find suspicious fuel transactions" → searches, filters, summarizes, links to Kibana. *Closest to your current work, most differentiated.*
2. [ ] **Data Analyst Agent** — SQL queries → charts → explanations → PDF report.
3. [ ] **Airflow Agent** — writes, validates, deploys DAGs.
4. [ ] **Multi-Agent System** — planner + SQL/ES/Python experts + report writer. *Build last; multi-agent only pays off once single agents are solid.*

Each project ships with: README, eval results, traces, and a deployed demo if possible.

---

## Technology Picks (2026)

| Category | Choice |
|----------|--------|
| LLM APIs | Claude, GPT, Gemini — build provider-agnostic |
| Embeddings | Voyage AI, OpenAI, BGE |
| Vector store | Elasticsearch first (your edge), pgvector second — skip a third |
| Python tooling | uv, pydantic, FastAPI |
| Frameworks | PydanticAI / OpenAI Agents SDK → LangGraph; LangChain read-only |
| Observability | LangSmith or OpenTelemetry + Grafana |
| Daily habit | Use a coding agent (Claude Code, Codex CLI) — productivity tool *and* working reference implementation |

## What to Skip

TensorFlow, PyTorch internals, CNNs/RNNs/GANs, reinforcement learning, foundation-model training. Valuable, but not prerequisites for agent engineering.

---

## Week-by-Week Summary

| Weeks | Focus | Deliverable |
|-------|-------|-------------|
| 1–2 | LLM fundamentals + prompting | Raw-API chatbot with one tool call |
| 3–4 | Agent loop from scratch | ~200-line SQL/ES agent |
| 5–7 | RAG with Elasticsearch | Doc Q&A with citations + eval script |
| 8–10 | Frameworks + MCP | Framework agent with MCP tools |
| 11–12 | Evals, observability, security | Dashboards + CI eval suite |
| 13–14 | Production | Deployed authenticated agent API |
| 15–16+ | Portfolio | 3–5 polished GitHub projects |

**Target profile:** AI engineer for data platforms — agents that investigate anomalies in transactional data, generate and validate SQL, analyze Elasticsearch indices, automate reporting, and orchestrate pipelines. High-value enterprise use cases where your existing expertise is a competitive advantage.
