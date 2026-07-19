import json
import sqlite3
from pathlib import Path
import re

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


if __name__ == "__main__":
    query = """
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

    query= """SELECT * FROM salez"""
    print(run_sql(query))


