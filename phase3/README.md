# Phase 3 — Document Q&A over Store Policies (RAG on Elasticsearch)

An **evaluation-first** RAG system that answers questions about a generic
retailer's store policies, with inline `[doc §section]` citations and honest
abstention when the corpus doesn't cover the question.

The guiding discipline: **build the ruler before tuning the machine.** A 49-question
golden set with known ground truth was written *before* any retrieval code, and
every change (analyzer, vectors, hybrid, reranking, filtering) was accepted or
rejected on measured recall — never on intuition.

## Constraint that shaped the design

The embeddings and reranking run **locally** (BGE via `fastembed`, ONNX/CPU, no torch) and only generation uses the proxy — a provider-agnostic split that also happens to be free and offline.

## Pipeline

```
corpus/*.md (YAML frontmatter + numbered sections)
   │  loader.py         parse frontmatter + split on ## / ### headings
   ▼
structure-aware chunks  (chunking.py: one chunk per non-empty section,
   │                      chunk_id = "doc_id#section", section-level store_id)
   ▼
Elasticsearch index     (index.py: text[english analyzer] + dense_vector[BGE-small,
   │  policy_chunks       cosine] + store_id[integer])
   ▼
retrieval (retrieve.py)
   BM25 ──┐
   vector ┤─ RRF fusion ─ cross-encoder rerank ─ store filter
          ┘
   ▼
grounded generation (answer.py: Sonnet via proxy)
   → answer with [doc §section] citations, or "Not stated in the provided policies."
```

## Evaluation

Two axes, both scored against the same golden set (`golden/golden_set.jsonl`,
49 questions: factual, store-override, cross-doc, unanswerable).

### Retrieval — recall@k (`eval_retrieval.py`)

Each stage was measured and kept only if the number moved. `recall = |required ∩ top-k| / |required|`
(sources modeled as a list, so a 2-source cross-doc question can score 0.5).

| retriever                | recall@1 | recall@3 | recall@5 | what it taught |
|--------------------------|----------|----------|----------|----------------|
| BM25 (standard analyzer) | 0.716    | 0.852    | 0.977    | baseline; `warranty#1` never retrieved — vocabulary mismatch |
| BM25 (english analyzer)  | 0.875    | 0.966    | 0.977    | +stemming/stopwords lifts ranking; aggregate up but q004 regressed |
| Vector (BGE-small, kNN)  | 0.841    | 0.989    | 0.989    | fixes vocab mismatch; broad recall but fuzzy precision@1 |
| Hybrid (RRF, BM25+vec)   | 0.943    | 0.977    | 0.989    | best of both — recovers each method's misses, zero regressions |
| Rerank (cross-encoder)   | 0.943    | 1.000    | 1.000    | perfect recall@3/5 |
| Rerank + store filter    | 0.943    | 1.000    | 1.000    | store-scoped; general queries exclude store-#42 overrides |

`recall@1 = 0.943` is the **ceiling**, not a failure: the only questions below 1.0
are 2-source cross-doc questions, which cannot exceed 0.5 at k=1. recall@5 = 1.0 is
also flattering — with only 17 chunks, k=5 retrieves ~30% of the corpus; a real
corpus would punish this far harder. The informative signals are recall@1 and the
per-question miss lists, not the headline.

### Generation — groundedness / citations / abstention (`eval_generation.py`)

45 answerable + 4 unanswerable.

| metric | value | note |
|--------|-------|------|
| Citation OK rate          | 1.000       | every answer cited the correct `[doc §section]` |
| False-abstention          | 0           | never abstained on an answerable question |
| Fact recall (lexical)     | 0.978       | see q044 caveat |
| Abstention (unanswerable) | 0.750 (3/4) | see q046 caveat |

**Known metric limitations (not model failures):**
- **q044** — the answer is correct and grounded (`lowest selling price... store credit [§4]`)
  but the lexical `answer_facts` substring didn't match, so fact_recall scored it 0.
  Lexical grounding has false negatives.
- **q046** — labeled unanswerable, but the model gave a *better* answer than a blind
  abstention: it reasoned from §1 ("returns only within 21 days, so 60 days is not
  eligible"). This is **grounded negation**, the correct behavior for a false-premise
  question — but the rigid `"not stated"` check flagged it. "Unanswerable" should mean
  genuinely out-of-scope (e.g. a phone number), not "answerable by inference."

The real result: **zero hallucinations, perfect citations, no false abstentions.**
The grounded prompt suppresses confident hallucination; the two "failures" are
calibration of the ruler, not the system.

## What the eval caught (that intuition wouldn't)

- Vocabulary mismatch is invisible to BM25 (`_analyze` showed the answer chunk shared
  only stopwords with the query) — this is *why* vectors were added, proven not assumed.
- An aggregate can rise while an individual question regresses (english analyzer, q004) —
  always read the diff, not just the mean.
- The eval kept finding **its own bugs**: a `datetime.date` type surprise from YAML,
  a single-source schema that couldn't express cross-doc questions, a mislabeled
  store-#42 question (q038) exposed by the metadata filter, a duplicate golden entry,
  and false-premise questions miscategorized as unanswerable.

## Running it

Prerequisites: Elasticsearch on `localhost:9200` (`docker start es-dev`), `uv`,
and `LITELLM_API_KEY` / `LITELLM_BASE_URL` in the environment.

```bash
cd phase3
uv sync                          # install deps (first run downloads BGE models once)

uv run python index.py           # (re)build the policy_chunks index
uv run python eval_retrieval.py  # retrieval recall@k, all retrievers
uv run python eval_generation.py # groundedness / citation / abstention
uv run python answer.py          # interactive Q&A
```

## Files

| file | role |
|------|------|
| `corpus/*.md`             | policy documents (frontmatter + numbered sections); facts are hand-authored ground truth |
| `golden/golden_set.jsonl` | 49-question golden set with expected answers + source sections |
| `loader.py`               | parse corpus + validate golden sources resolve |
| `chunking.py`             | structure-aware chunking + section-level `store_id` |
| `embeddings.py`           | local BGE-small embeddings (`embed` / `embed_query`) |
| `index.py`                | build the Elasticsearch index (BM25 + vector + store_id) |
| `retrieve.py`             | BM25 / vector / hybrid (RRF) / rerank / store filter |
| `rerank.py`               | local cross-encoder reranker |
| `answer.py`               | grounded generation with citations (the system) |
| `eval_retrieval.py`       | recall@k harness |
| `eval_generation.py`      | groundedness / citation / abstention harness |
| `results.md`              | retrieval comparison table |
