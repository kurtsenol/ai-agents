from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from model import build_model 
from pydantic_ai import Agent, RunContext   

from typing import Annotated
from pydantic import Field


@dataclass
class Deps:
    db_path: Path
    es: Elasticsearch


def build_deps() -> Deps:
    deps = Deps(
        db_path=(Path(__file__).parent / "../phase2/retail.db").resolve(),
        es=Elasticsearch("http://localhost:9200")
    )
    return deps


agent = Agent(
    build_model(),
    deps_type=Deps,
    instructions=("You are a retail database analyst; the database is your single source of truth."),
)

def _strip_leading_comments(sql: str) -> str:
    """
    Remove leading whitespace and SQL comments.
    Supports:
      -- single-line comments
      /* block comments */
    """
    pattern = r"""
        ^
        (
            \s+                         |   # whitespace
            --[^\n]*(?:\n|$)            |   # -- comment
            /\*.*?\*/                   |   # /* ... */ comment
        )*
    """

    return re.sub(pattern, "", sql, flags=re.DOTALL | re.VERBOSE)


@agent.tool
def run_sql(ctx: RunContext[Deps], query: str) -> str:

    """
    Execute a read-only SQL query against a SQLite database.

    Only SELECT statements are allowed. 

    Database schema:

    stores
    - id INTEGER PRIMARY KEY
    - name TEXT
    - city TEXT

    products
    - id INTEGER PRIMARY KEY
    - name TEXT
    - category TEXT
    - unit_price REAL

    transactions
    - id INTEGER PRIMARY KEY
    - store_id INTEGER
    - product_id INTEGER
    - quantity INTEGER
    - unit_price REAL
    - ts TEXT (ISO 8601 timestamp)

    Args:
        query: A read-only SQL SELECT or WITH query to execute against the retail database.

    """

    query = query.strip().rstrip(";")

    # Remove leading comments before validation
    cleaned = _strip_leading_comments(query).lstrip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return "Error: Only SELECT and WITH queries are allowed."

    conn = sqlite3.connect(f"file:{ctx.deps.db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        cursor.execute(query)

        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

        output = "\n".join(
            json.dumps(dict(zip(columns, row)), ensure_ascii=False)
            for row in rows[:50]
        )

        if len(rows) > 50:
            output += f"\n...and {len(rows) - 50} more rows."

        return output or "No rows returned."

    except sqlite3.Error as e:
        return f"SQLite error: {e}"

    finally:
        conn.close()

@agent.tool
def search_reviews(
    ctx: RunContext[Deps],
    query: str,
    store_id: int | None = None,
    min_rating: Annotated[int | None, Field(ge=1, le=5)] = None,
    max_rating: Annotated[int | None, Field(ge=1, le=5)] = None,
) -> str:

    """ 
        Search customer reviews using free-text search and optional filters. 
        Use this tool to find reviews mentioning a specific topic, problem, or phrase.
        You can optionally restrict the search to a store and/or a rating range.
    
        Reviews in this index are written in English, so search terms must be
        in English regardless of the language of the user's question. For example,
        useful search terms include: overcharged, refund, broken, late delivery.

        Args:
            query: Text to search for in the review content.
            store_id: Optional store ID. When provided, only reviews for this
                store are returned.
            min_rating: Optional minimum review rating, inclusive.
            max_rating: Optional maximum review rating, inclusive.
    
    """


    INDEX_NAME = "reviews"

    must = [
        Q("match", text=query)
    ]

    filters = []

    if store_id is not None:
        filters.append(
            Q("term", store_id=store_id)
        )

    if min_rating is not None or max_rating is not None:
        rating_range = {}

        if min_rating is not None:
            rating_range["gte"] = min_rating

        if max_rating is not None:
            rating_range["lte"] = max_rating

        filters.append(
            Q("range", rating=rating_range)
        )

    bool_query = Q(
        "bool",
        must=must,
        filter=filters,
    )

    search = (
        Search(using=ctx.deps.es, index=INDEX_NAME)
        .query(bool_query)
        [:50]
    )

    response = search.execute()

    # Elasticsearch may return either an integer or a dict-like
    # object depending on the client configuration/version.
    total = response.hits.total.value

    reviews = [
        {
            "store_id": hit.store_id,
            "rating": hit.rating,
            "text": hit.text,
            "ts": hit.ts,
        }
        for hit in response
    ]

    # If text search found nothing, count reviews using only the filters.
    if total == 0:
        filter_only_query = Q(
            "bool",
            filter=filters,
        )

        count_search = (
            Search(using=ctx.deps.es, index=INDEX_NAME)
            .query(filter_only_query)
        )

        count_response = count_search.execute()
        filtered_total = count_response.hits.total.value

        applied_filters = []

        if store_id is not None:
            applied_filters.append(f"store_id={store_id}")
        if min_rating is not None:
            applied_filters.append(f"min_rating={min_rating}")
        if max_rating is not None:
            applied_filters.append(f"max_rating={max_rating}")

        filters_text = ", ".join(applied_filters) or "none"

        return (
            f"No reviews matched query={query!r} with filters {filters_text}.\n"
            f"{filtered_total} reviews exist for these filters — "
            "try different or broader English search terms."
        )

    output = "\n".join(
        json.dumps(review, ensure_ascii=False)
        for review in reviews
    )

    if total > len(reviews):
        output += f"\n...and {total - len(reviews)} more rows."

    return output


if __name__ == "__main__":

    deps = build_deps()

    result = agent.run_sync(
    "42 numaralı mağazada fiyat anormalliği var mı?",
    deps=deps,
)

    print("=== OUTPUT ===")
    print(result.output)

    print("\n=== USAGE ===")
    print(result.usage)

    for msg in result.all_messages():
        for part in msg.parts:
            if type(part).__name__ == "ToolCallPart":
                print("TOOL CALL")
                print("  name:", part.tool_name)
                print("  args:", part.args_as_dict())

            elif type(part).__name__ == "ToolReturnPart":
                print("TOOL RETURN")
                print("  name:", part.tool_name)
                print("  content:", str(part.content)[:200])
