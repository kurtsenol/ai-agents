"""
Step 12 — talk to mcp_sql_server.py as a client, with no agent involved.

This is the MCP equivalent of the habit we built in steps 1 and 5a: before
trusting a framework, look at what actually crosses the boundary.

    uv run mcp_client_test.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).parent / "mcp_sql_server.py"


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()

            print("=== SERVER ===")
            print(f"  name:        {init.server_info.name}")
            print(f"  version:     {init.server_info.version}")
            print(f"  protocol:    {init.protocol_version}")
            print(f"  instructions:{(init.instructions or '')[:70]}...")

            print("\n=== TOOLS THE HOST SEES ===")
            tools = await session.list_tools()

            for t in tools.tools:
                print(f"\n  {t.name}")
                print(f"    description: {(t.description or '')[:70]}...")
                print(f"    inputSchema: {json.dumps(t.input_schema, ensure_ascii=False)[:150]}")

            print("\n=== CALLING run_sql ===")
            result = await session.call_tool(
                "run_sql",
                {"query": "SELECT COUNT(*) AS n FROM transactions WHERE store_id = 42"},
            )
            print(f"  is_error: {result.is_error}")
            for block in result.content:
                print(f"  {block.type}: {block.text}")

            print("\n=== CALLING run_sql WITH A FORBIDDEN STATEMENT ===")
            result = await session.call_tool(
                "run_sql", {"query": "DELETE FROM transactions"}
            )
            for block in result.content:
                print(f"  {block.type}: {block.text}")


if __name__ == "__main__":
    asyncio.run(main())
