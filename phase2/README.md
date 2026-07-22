# Phase 2 — Agent From Scratch

A ~130-line agent loop on the raw Claude API (no framework) that investigates a
real database, searches customer reviews, and safely makes approved writes.

The agent answers open-ended questions ("anything unusual in store 42 yesterday?")
by deciding its own sequence of tool calls at runtime — the model chooses what to
query next based on what the last query returned. This is the workflow-vs-agent
distinction from Anthropic's *Building Effective Agents*: the control flow is not
hardcoded, the model drives it.

## What it does

- **Agent loop** — `model → tool call → execute → feed result back → loop until done`,
  with an iteration cap and a graceful "out of budget" summary if it hits the cap.
- **Three tools**, dispatched generically by name (no `if/elif` per tool):
  - `run_sql` — read-only SQLite (SELECT/WITH only), enforced two ways: a string
    guard *and* a read-only connection (`mode=ro`) that the engine cannot be talked out of.
  - `search_reviews` — free-text + filtered search over an Elasticsearch reviews index.
    The tool exposes a narrow parameter schema (query / store_id / rating range) and
    builds the Query DSL itself, rather than asking the model to hand-write nested JSON.
  - `run_write_sql` — INSERT/UPDATE/DELETE behind a **human approval gate**. Writes
    with no `WHERE` clause escalate to a typed confirmation (`DELETE ALL`) instead of a
    plain `y`.
- **Conversation memory** — `chat()` keeps one growing `messages` list across turns,
  so follow-up questions see the prior investigation. (Watch the input-token counter
  climb each turn — that growth is the cost of memory, printed live.)

## Setup

1. **Environment** (company LiteLLM proxy — set these in your shell):
   ```sh
   export LITELLM_API_KEY="..."
   export LITELLM_BASE_URL="..."
   ```
2. **Dependencies** (managed by uv, Python 3.13):
   ```sh
   uv sync
   ```
3. **Elasticsearch** (single-node dev container, security disabled — local only):
   ```sh
   docker run -d --name es-dev -p 9200:9200 \
     -e "discovery.type=single-node" \
     -e "xpack.security.enabled=false" \
     -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
     docker.elastic.co/elasticsearch/elasticsearch:8.15.0
   curl http://localhost:9200   # confirm it's up
   ```
4. **Seed the data** (both re-anchor timestamps to "yesterday" on each run):
   ```sh
   uv run python seed_db.py        # retail.db: stores, products, transactions + 2 planted anomalies
   uv run python seed_elastic.py   # reviews index: ~150 reviews + planted overcharge complaints
   ```

## Run

```sh
uv run python agent.py
```

Try, in order (memory carries context between them):

```
Anything unusual in store 42's transactions yesterday?
Are there any customer complaints connected to what happened at store 42 yesterday?
The 10 price-glitch transactions at store 42 yesterday were overcharged 100x. Correct their unit_price by dividing by 100.
```

The first surfaces two planted anomalies (a 30-in-30-seconds duplicate burst and a
100× price glitch) using SQL aggregation. The second chains SQL findings into a
review search across a *different* data store to corroborate them. The third triggers
the approval gate — the agent proposes an `UPDATE`, you approve it, and it verifies
its own write afterward.

## The planted anomalies (for testing)

`seed_db.py` deliberately buries two anomalies in store 42, "yesterday", so the
agent has something real to find and you know the ground truth:

1. **Duplicate burst** — the same Coffee transaction repeated 30× within one minute.
2. **Price glitch** — 10 transactions priced at exactly 100× the catalog price.

`seed_elastic.py` plants 5–8 matching 1-star "overcharged" reviews timestamped to
the same evening as the price glitch, so the two data sources tell one story.

Because the data is synthetic, it leaks synthetic patterns (e.g. transaction
timestamps are uniform across all 24 hours, which no real store looks like) — a
useful reminder for building honest eval sets in Phase 3.

## Files

| File | What it holds |
|---|---|
| `agent.py` | The agent loop, dispatch, conversation REPL, system prompt |
| `tools.py` | Tool definitions (the schemas the model sees) + implementations + `DISPATCH` |
| `seed_db.py` | Builds `retail.db` with planted SQL anomalies |
| `seed_elastic.py` | Builds the Elasticsearch `reviews` index with planted complaints |

## Known limitations

- The write gate catches the most common catastrophic pattern (missing `WHERE`) but
  is not a SQL parser — `DELETE ... WHERE 1=1` would slip past to the easy prompt.
  The read-only connection on `run_sql` is the real guarantee; the gate is friction.
- Conversation memory grows unbounded — every tool result and report is re-sent on
  every turn. Real agents trim/summarize context; out of scope here.
- Single-turn tool errors are returned to the model as strings so it can self-correct,
  but there is no retry cap on a model that loops on the same failing query beyond the
  overall iteration cap.
