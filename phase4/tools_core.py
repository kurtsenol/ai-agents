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


def sql_query(db_path: Path, query: str) -> tuple[str, dict]:
    """Read-only SELECT/WITH. Returns (text, metadata)."""
    query = query.strip().rstrip(";")
    cleaned = _strip_leading_comments(query).lstrip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return (
            "Error: Only SELECT and WITH queries are allowed.",
            {"error": True, "row_count": 0, "truncated": False},
        )

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        cursor = conn.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()

        text = "\n".join(
            json.dumps(dict(zip(columns, r)), ensure_ascii=False)
            for r in rows[:ROW_LIMIT]
        )

        if len(rows) > ROW_LIMIT:
            text += f"\n...and {len(rows) - ROW_LIMIT} more rows."

        return (
            text or "No rows returned.",
            {
                "error": False,
                "row_count": len(rows),
                "truncated": len(rows) > ROW_LIMIT,
            },
        )

    except sqlite3.Error as e:
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
