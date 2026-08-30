"""
Step 13 — the review search tools as a second MCP server.

Two things are new relative to mcp_sql_server.py.

1. Parameter descriptions. Step 12 showed the MCP SDK does NOT parse the
   `Args:` block in a docstring — it builds the schema from type hints via
   Pydantic. So descriptions have to be attached with
   Annotated[..., Field(description=...)], the same mechanism you used in
   step 2 for ge/le constraints.

2. A long-lived resource. SQLite connections are per-call and cheap; an
   Elasticsearch client is not. MCP's answer is `lifespan`: an async context
   manager that opens resources when the process starts and closes them when
   it stops. That is the fourth answer to the same question:

       PydanticAI   deps_type=Deps  + ctx.deps
       LangGraph    closure over the resource
       MCP          lifespan  + ctx.request_context.lifespan_context

Run:  uv run mcp_es_client_test.py
"""

import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from elasticsearch import Elasticsearch
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import Field

import tools_core

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("mcp_es")

ES_URL = "http://localhost:9200"


@dataclass
class Resources:
    es: Elasticsearch


@asynccontextmanager
async def lifespan(server: MCPServer):
    """Opens once when the host starts this process, closes once when it stops."""
    log.info("lifespan: connecting to %s", ES_URL)
    es = Elasticsearch(ES_URL)

    try:
        yield Resources(es=es)
    finally:
        log.info("lifespan: closing elasticsearch client")
        es.close()


server = MCPServer(
    name="retail-reviews",
    version="0.1.0",
    instructions=(
        "Full-text search over customer reviews. The reviews are in ENGLISH: "
        "search in English regardless of the language of the question."
    ),
    lifespan=lifespan,
)


@server.tool()
def search_reviews(
    ctx: Context,
    query: Annotated[
        str | None,
        Field(
            description=(
                "Optional search text in ENGLISH. If omitted, lists every "
                "review matching the other filters. Use this to verify that "
                "an absence is real rather than a search miss."
            )
        ),
    ] = None,
    store_id: Annotated[
        int | None,
        Field(description="Optional store ID filter.")
    ] = None,
    min_rating: Annotated[
        int | None,
        Field(
            description="Optional minimum rating, inclusive. Must be between 1 and 5.",
            ge=1,
            le=5,
        )
    ] = None,
    max_rating: Annotated[
        int | None,
        Field(
            description="Optional maximum rating, inclusive. Must be between 1 and 5.",
            ge=1,
            le=5,
        )
    ] = None,
) -> CallToolResult:
    """Search customer reviews by free text, or list them when no query is given.

    A text search returning nothing does not prove a topic is absent, because
    a review worded differently would not match. Omit `query` to read every
    review matching the filters before claiming an absence.
    """
    es = ctx.request_context.lifespan_context.es

    text, meta = tools_core.review_search(
        es, query, store_id, min_rating, max_rating
    )

    log.info(
        "search_reviews query_used=%s hits=%s truncated=%s",
        meta["query_used"], meta["hit_count"], meta["truncated"],
    )

    # --- the second channel -------------------------------------------------
    # A tool result is read by two audiences. The model reads `content`; a
    # program (the eval, a dashboard) needs facts like row_count and truncated.
    # In phase 4 that was ToolReturn(metadata=...), which existed only inside
    # one Python process. Across MCP the equivalent is the protocol's own
    # `_meta` field: it travels on the wire, and the model never sees it.
    #
    # Returning CallToolResult instead of a bare str changes nothing the model
    # reads - `content` is the same text as before.
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        meta=meta,
    )


if __name__ == "__main__":
    log.info("starting retail-reviews over stdio")
    server.run(transport="stdio")
