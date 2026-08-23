# Eval — data setup and reproducibility

The eval in this directory runs against `phase2/retail.db` (SQLite) and the
`reviews` index in Elasticsearch. For measurements to mean anything, both must
be in a **known state**.

## Why this matters

An approved `UPDATE` in step 4 permanently modified the database. This will
happen again as long as the agent has write access. Reseeding before every eval
run is the only way to tell whether a difference you measure came from the agent
or from the data.

## Prerequisites

Elasticsearch must be running:

```bash
docker start es-dev
curl -s localhost:9200 | head -5
```

## Full setup

Both scripts are **idempotent**: `seed_db.py` drops all three tables with
`DROP TABLE IF EXISTS`, and `seed_elastic.py` deletes and recreates the
`reviews` index. Re-running them is safe — nothing accumulates.

```bash
cd phase2
uv run seed_db.py
uv run seed_elastic.py
```

`seed_db.py` prints nothing. `seed_elastic.py` prints `Inserted N reviews.`
(N = 155–158; the number of planted complaint reviews is random between 5 and 8).

## Verification

After seeding, the data should produce these values:

```bash
sqlite3 phase2/retail.db "
SELECT 'stores',   COUNT(*) FROM stores;
SELECT 'products', COUNT(*) FROM products;
SELECT 'transactions', COUNT(*) FROM transactions;
SELECT 'price anomaly', COUNT(*)
  FROM transactions t JOIN products p ON p.id = t.product_id
  WHERE t.store_id = 42 AND t.unit_price > p.unit_price * 50;
SELECT 'duplicate burst', MAX(c) FROM (
  SELECT COUNT(*) AS c FROM transactions
  WHERE store_id = 42 AND product_id = 4 AND quantity = 2
  GROUP BY substr(ts, 1, 16));
"
```

| Metric | Expected |
|---|---|
| `stores` | 5 (ids 40–44) |
| `products` | 10 |
| `transactions` | 2040 |
| price anomaly | 10 |
| duplicate burst | 30 |

Elasticsearch:

```bash
curl -s "localhost:9200/reviews/_count"
curl -s "localhost:9200/reviews/_count" -H 'Content-Type: application/json' \
  -d '{"query":{"bool":{"filter":[{"term":{"store_id":42}},{"term":{"rating":1}}]}}}'
```

| Metric | Expected |
|---|---|
| total reviews | 155–158 |
| store 42, rating 1 | 5–8 (planted complaints) |

Total reviews = 150 normal + 5–8 planted complaints; the lower bound is fixed,
the upper bound is random.

## Planted anomalies

`seed_db.py` plants two anomalies, both in **store 42**:

| Anomaly | Shape |
|---|---|
| **Price glitch** | 10 transactions with `unit_price` **100× the catalog price**, one per minute from 17:30 to 17:39 |
| **Duplicate burst** | The same transaction (Coffee, qty=2) repeated **30 times**, one per second from 14:15:00 to 14:15:29 |

`seed_elastic.py` additionally plants 5–8 one-star "overcharged" reviews on the
same evening as the price glitch (17:30–18:30). **The reviews are in English.**

## Critical: timestamps shift, ids do not

Both scripts use `random.seed(42)`. As a result:

- **Transaction ids are reproducible.** The same seed produces the same ids
  every time, so the price anomaly always lands on the same `transaction_id`s.
- **Timestamps are NOT reproducible.** Both scripts derive timestamps from
  `datetime.now()`, so anomalies are always written to **the day before the day
  you ran the seed**.

This imposes two hard rules on the golden set:

1. **Never hardcode an absolute date in an expectation.** `"2026-08-08"` is
   correct today and wrong tomorrow.
2. **Never use "yesterday", "today", or "this week" in a question.** The agent
   resolves those against the *run date*, while the data was written against the
   *seed date*. They will not match, and the eval will go red for a reason that
   has nothing to do with the agent.

If you need to measure something time-dependent, compute the ground truth
**with a query at eval time** (`SELECT MIN(ts) ... WHERE unit_price > ...`)
rather than embedding it in a file.

## When to reseed

- After the agent modified data via `run_write_sql` (the step 4 scenarios)
- **Before every comparison** between eval runs
- Whenever the verification table above does not match

Reseeding shifts the anomaly dates, so `runs.jsonl` records collected before a
reseed are **not comparable** with records collected after it. Keep every run
you intend to compare on the same seed.
