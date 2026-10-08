"""Minimal MCP client to verify the server: lists tools, then optionally calls generate_image.

    python check_mcp.py stdio                       # spawns server.py over stdio
    python check_mcp.py http http://127.0.0.1:8000/mcp
    add --call to run a real generation, --bad to check input validation errors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

HERE = Path(__file__).resolve().parent


def open_transport(args: argparse.Namespace) -> AbstractAsyncContextManager[Any]:
    if args.transport == "http":
        return streamable_http_client(args.url)
    # stdio_client only forwards a minimal env by default; pass ours so overrides reach the server.
    server = StdioServerParameters(command=sys.executable, args=[str(HERE / "server.py")], env=dict(os.environ))
    return stdio_client(server)


def describe(result: Any) -> str:
    parts = []
    for block in result.content:
        if block.type == "text":
            parts.append(f"text: {block.text}")
        elif block.type == "image":
            parts.append(f"image: {block.mime_type}, {len(block.data) * 3 // 4 // 1024} KB")
    return f"isError={result.is_error}\n  " + "\n  ".join(parts)


async def run(args: argparse.Namespace) -> None:
    async with open_transport(args) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"tool: {tool.name}\n  {json.dumps(tool.input_schema, indent=2)}")
            if args.bad:
                bad = await session.call_tool("generate_image", {"prompt": "cat", "width": 500, "steps": 0})
                print("bad call ->", describe(bad))
            if args.call:
                ok = await session.call_tool("generate_image", {"prompt": "a lighthouse at sunset, oil painting", "seed": 1234})
                print("call ->", describe(ok))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transport", choices=["stdio", "http"])
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--call", action="store_true")
    parser.add_argument("--bad", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
