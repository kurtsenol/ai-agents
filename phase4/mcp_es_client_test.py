"""Step 13 — inspect the reviews server from a plain client."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).parent / "mcp_es_server.py"


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"=== {init.server_info.name} v{init.server_info.version} ===")

            tools = await session.list_tools()

            for t in tools.tools:
                print(f"\n{t.name} inputSchema:")
                print(json.dumps(t.input_schema, ensure_ascii=False, indent=1))

            for label, args in [
                ("text search that matches", {"query": "overcharged", "store_id": 42}),
                ("text search that misses", {"query": "parking", "store_id": 44}),
                ("enumeration, no query", {"store_id": 44, "max_rating": 2}),
                ("invalid rating", {"store_id": 44, "min_rating": 9}),
            ]:
                print(f"\n=== {label} -> {args}")
                result = await session.call_tool("search_reviews", args)
                print(f"  is_error: {result.is_error}")
                for block in result.content:
                    print(f"  {str(block.text)[:180]}")


if __name__ == "__main__":
    asyncio.run(main())
