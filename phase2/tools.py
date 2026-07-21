import json
import sqlite3
from pathlib import Path
import re
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "reviews"

DB_PATH = Path(__file__).parent / "retail.db"


TOOLS = [{
    "name": "run_sql",
    "description": """
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
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A single SQLite SELECT query."
            }
        },
        "required": ["query"],
    },
},
{
    "name": "search_reviews",
    "description": (
        "Search customer reviews using free-text search and optional filters. "
        "Use this tool to find reviews mentioning a specific topic, problem, "
        "or phrase. You can optionally restrict the search to a store and/or "
        "a rating range."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search query to search within review text.",
            },
            "store_id": {
                "type": "integer",
                "description": "Optional store ID to restrict the search to one store.",
            },
            "min_rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Optional minimum review rating.",
            },
            "max_rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Optional maximum review rating.",
            },
        },
        "required": ["query"],
    },
}]

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


def run_sql(query: str) -> str:
    """
    Execute a read-only SQLite SELECT query and return up to 50 rows.
    """

    query = query.strip().rstrip(";")

    # Remove leading comments before validation
    cleaned = _strip_leading_comments(query).lstrip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return "Error: Only SELECT and WITH queries are allowed."

    conn = sqlite3.connect(DB_PATH)
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

def search_reviews(
    query: str,
    store_id: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
) -> str:

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
        Search(using=es, index=INDEX_NAME)
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

    output = "\n".join(
        json.dumps(review, ensure_ascii=False)
        for review in reviews
    )

    if total > len(reviews):
        output += f"\n...and {total - len(reviews)} more rows."

    return output or "No rows returned."

DISPATCH = {
    "run_sql": run_sql,
    "search_reviews": search_reviews,
}

if __name__ == "__main__":

    # Test run_sql
    sql_query = """
        SELECT
            id,
            store_id,
            product_id,
            quantity,
            unit_price,
            ts
        FROM transactions
        WHERE store_id = 42
        LIMIT 200;
    """

    print("=== SQL RESULTS ===")
    print(run_sql(sql_query))


    # Test search_reviews
    print("\n=== REVIEW RESULTS ===")

    review_results = search_reviews(
        query="overcharged",
        store_id=42,
        max_rating=1,
    )

    print(review_results)



