"""
Framework-independent tool bodies.

The tools in retail_agent.py are welded to PydanticAI: they take
`ctx: RunContext[Deps]` and return `ToolReturn`. Neither type means anything
to LangGraph, and neither will mean anything to an MCP server in step 12.

So the actual work moves here as plain functions that take plain arguments
and return (text_for_the_model, metadata_for_the_program). Each framework
gets a thin adapter around these.

retail_agent.py is deliberately NOT refactored to use this yet: it is the
measured baseline for the eval, and changing it would invalidate
runs_v2/v3/v4. We will unify at step 12, where MCP forces the issue.
"""

import json
import re
import sqlite3
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search

ROW_LIMIT = 50
REVIEW_INDEX = "reviews"

SCHEMA_DOC = """
stores       (id, name, city)
products     (id, name, category, unit_price REAL)
transactions (id, store_id, product_id, quantity, unit_price REAL, ts TEXT ISO 8601)
"""


def _strip_leading_comments(sql: str) -> str:
    pattern = r"^(\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)*"
    return re.sub(pattern, "", sql, flags=re.DOTALL | re.VERBOSE)


# How much work one query may do before we stop it.
#
# SQLite has no wall-clock knob. It counts virtual-machine instructions and
# calls a handler every N of them, so a time budget has to be converted into
# an instruction budget - and the exchange rate depends on the machine. That
# makes this a MEASURED number, not a chosen one.
#
# Measured on this database (2,040 transactions), interval = 1,000:
#
#   honest, WHERE store_id=42 (421 rows)            1 callback
#   3-way join, filtered                            1
#   aggregate over the whole table                 34
#   ---------------------------------------------------
#   cartesian join, COUNT(*)                    8,327   (0.04 s)
#   cartesian join, ORDER BY                   29,135   (1.05 s)
#   cartesian join, GROUP BY                   49,943   (0.61 s)
#
# The gap between the two groups is ~250x, which is what makes a threshold
# possible at all. 20,000 sits inside that gap: ~600x headroom over the most
# expensive honest query, while stopping the two shapes that actually hurt.
#
# Note what this does NOT catch, and why that is fine: a plain cartesian
# SELECT with no ORDER BY or aggregate uses ZERO callbacks, because SQLite
# is lazy and fetchmany(51) only ever asks for 51 rows. The two limits cover
# different failure shapes - fetchmany bounds lazy queries, the instruction
# budget bounds queries that must do all the work before returning row one.
# Neither is redundant.
PROGRESS_INTERVAL = 1_000
MAX_PROGRESS_CALLS = 20_000


def sql_query(db_path: Path, query: str) -> tuple[str, dict]:
    """Read-only SELECT/WITH, with a row cap and a work cap.

    Three independent limits, and they are independent on purpose. Any one
    of them can be wrong without the other two failing:

      1. `mode=ro` on the connection      - SQLite itself refuses writes.
      2. the SELECT/WITH regex            - advisory; catches obvious intent.
      3. fetchmany + progress handler     - bounds the cost of an ALLOWED query.

    Layer 3 is the one phase 4 did not have, and the one an allowlist can
    never provide: `SELECT a.id, b.id FROM transactions a, transactions b`
    passes every check above it, and on this database materialises 4,161,600
    rows - 634 MB and 6.2 s - before the 50-row cap is applied. The cap was
    applied after the damage. Now it is applied during.
    """
    query = query.strip().rstrip(";")
    cleaned = _strip_leading_comments(query).lstrip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return (
            "Error: Only SELECT and WITH queries are allowed.",
            {"error": True, "row_count": 0, "truncated": False},
        )

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    calls = 0

    def _budget() -> int:
        # Returning non-zero aborts the statement -> sqlite3.OperationalError.
        nonlocal calls
        calls += 1
        return 1 if calls > MAX_PROGRESS_CALLS else 0

    conn.set_progress_handler(_budget, PROGRESS_INTERVAL)

    try:
        cursor = conn.execute(query)
        columns = [c[0] for c in cursor.description]

        # ROW_LIMIT + 1: one extra row is how you learn "there are more"
        # without paying to count them all.
        rows = cursor.fetchmany(ROW_LIMIT + 1)
        has_more = len(rows) > ROW_LIMIT
        rows = rows[:ROW_LIMIT]

        text = "\n".join(
            json.dumps(dict(zip(columns, r)), ensure_ascii=False) for r in rows
        )

        if has_more:
            # Deliberately no total. Reporting "and 4,161,550 more" would
            # require scanning everything, which is the cost we just refused.
            text += (
                f"\n...more rows exist; this result is capped at {ROW_LIMIT}. "
                "Narrow the query (add filters or an aggregate) rather than "
                "drawing a conclusion from a partial list."
            )

        return (
            text or "No rows returned.",
            {"error": False, "row_count": len(rows), "truncated": has_more},
        )

    except sqlite3.Error as e:
        # An aborted budget surfaces here as OperationalError('interrupted').
        # It returns an ERROR, not a partial result, and the distinction is
        # the whole point: a partial result is indistinguishable from a
        # complete one to a model reading prose, and it will answer from it.
        # An error is the only shape it cannot mistake for an answer.
        if "interrupted" in str(e).lower():
            return (
                "Error: query aborted - it would have to examine too much data. "
                "Add a WHERE filter, an aggregate, or a LIMIT and try again.",
                {"error": True, "row_count": 0, "truncated": False},
            )
        return (f"SQLite error: {e}", {"error": True, "row_count": 0, "truncated": False})

    finally:
        conn.close()


def review_search(
    es: Elasticsearch,
    query: str | None = None,
    store_id: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
) -> tuple[str, dict]:
    """Free-text review search, or enumeration when `query` is omitted."""
    must = [Q("match", text=query)] if query is not None else [Q("match_all")]

    filters = []

    if store_id is not None:
        filters.append(Q("term", store_id=store_id))

    if min_rating is not None or max_rating is not None:
        rng = {}
        if min_rating is not None:
            rng["gte"] = min_rating
        if max_rating is not None:
            rng["lte"] = max_rating
        filters.append(Q("range", rating=rng))

    search = Search(using=es, index=REVIEW_INDEX).query(
        Q("bool", must=must, filter=filters)
    )[:ROW_LIMIT]

    response = search.execute()
    total = response.hits.total.value

    hits = [
        {"store_id": h.store_id, "rating": h.rating, "text": h.text, "ts": h.ts}
        for h in response
    ]

    described = ", ".join(
        f"{k}={v}"
        for k, v in [
            ("store_id", store_id),
            ("min_rating", min_rating),
            ("max_rating", max_rating),
        ]
        if v is not None
    ) or "none"

    if total == 0:
        if query is None:
            return (
                f"No reviews matched the filters {described}. "
                "There are no reviews in the index matching these filters.",
                {"hit_count": 0, "filtered_total": 0, "truncated": False,
                 "query_used": False},
            )

        filtered = (
            Search(using=es, index=REVIEW_INDEX)
            .query(Q("bool", filter=filters))
            .execute()
            .hits.total.value
        )

        return (
            f"No reviews matched query={query!r} with filters {described}.\n"
            f"{filtered} reviews exist for these filters — "
            "try different or broader English search terms.",
            {"hit_count": 0, "filtered_total": filtered, "truncated": False,
             "query_used": True},
        )

    text = "\n".join(json.dumps(h, ensure_ascii=False) for h in hits)
    truncated = total > len(hits)

    if truncated:
        if query is None:
            text += (
                f"\nWARNING: partial listing. {len(hits)} of {total} shown. "
                "Do not conclude a topic is absent from this. Narrow the "
                "filters until all matching reviews fit, or report the "
                "question as unresolved."
            )
        else:
            text += f"\n...and {total - len(hits)} more rows."

    return (
        text,
        {"hit_count": total, "filtered_total": None, "truncated": truncated,
         "query_used": query is not None},
    )
