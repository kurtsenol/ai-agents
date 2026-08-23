"""
Step 12 — the SQL tools as an MCP server.

This process exposes tools over the Model Context Protocol. Any MCP host can
connect: Claude Code, Claude Desktop, or an agent you write. The tool bodies
come from tools_core.py, unchanged — that file existed for exactly this.

Transport is stdio: the host launches this file as a SUBPROCESS and speaks
JSON-RPC over stdin/stdout.

    Which means: stdout belongs to the protocol.
    One print() and the connection is garbage.

Test it without a host:  uv run mcp_client_test.py
"""

import logging
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

import tools_core

DB_PATH = (Path(__file__).parent / "../phase2/retail.db").resolve()


# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
)

log = logging.getLogger("mcp_sql")


server = MCPServer(
    name="retail-sql",
    instructions=(
        "Read-only SQL access to a retail database (stores, products, "
        "transactions). Use run_sql for any question about transactions, "
        "prices or stores."
    ),
)


@server.tool()
def run_sql(query: str) -> str:
    """Execute a read-only SQL query against the retail database.

    Only SELECT and WITH statements are allowed. Results are capped at 50
    rows; when capped, the reply says how many rows were left out.

    Database schema:
      stores       (id, name, city)
      products     (id, name, category, unit_price REAL)
      transactions (id, store_id, product_id, quantity, unit_price REAL,
                    ts TEXT ISO 8601)

    Prices are stored as bare numbers. No currency is recorded anywhere in
    the schema.

    Args:
        query: A read-only SQL SELECT or WITH query.
    """
    text, meta = tools_core.sql_query(DB_PATH, query)

    # Goes to stderr, so it is safe. This is the operator's channel: the host
    # never sees it and the model never sees it.
    log.info("run_sql rows=%s truncated=%s", meta["row_count"], meta["truncated"])

    return text


@server.tool()
def describe_schema() -> str:
    """Return the database schema. Call this first if unsure what exists."""
    log.info("describe_schema")
    return tools_core.SCHEMA_DOC


if __name__ == "__main__":
    log.info("starting retail-sql over stdio")
    server.run(transport="stdio")
