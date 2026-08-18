from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


FILESYSTEM_MCP_URL = os.getenv(
    "FILESYSTEM_MCP_URL",
    "http://filesystem-mcp:8083/mcp",
)

GITHUB_MCP_URL = os.getenv(
    "GITHUB_MCP_URL",
    "http://github-mcp:8082",
)


def _result_to_text(result) -> str:
    parts = []

    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)

        if text is not None:
            parts.append(str(text))
        else:
            parts.append(str(item))

    return "\n".join(parts).strip()


async def _call_http_tool(
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> str:
    async with httpx.AsyncClient(
        headers=headers or {},
        follow_redirects=True,
        timeout=30.0,
    ) as client:

        async with streamable_http_client(
            url,
            http_client=client,
        ) as streams:

            read_stream, write_stream, *_ = streams

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

                return _result_to_text(result)


def _safe_relative_path(path: str) -> str:
    raw = str(path or ".").strip()
    p = PurePosixPath(raw)

    if p.is_absolute() or ".." in p.parts:
        raise ValueError(
            "只能访问当前项目目录内的相对路径。"
        )

    return p.as_posix()


def mcp_filesystem(
    action: str,
    path: str = ".",
    pattern: str = "",
) -> str:
    action = str(action).strip().lower()
    safe_path = _safe_relative_path(path)

    if action == "read":
        return asyncio.run(
            _call_http_tool(
                FILESYSTEM_MCP_URL,
                "read_text_file",
                {"path": safe_path},
            )
        )

    if action == "list":
        return asyncio.run(
            _call_http_tool(
                FILESYSTEM_MCP_URL,
                "list_directory",
                {"path": safe_path},
            )
        )

    if action == "search":
        pattern = str(pattern).strip()

        if not pattern:
            raise ValueError(
                "search 操作必须提供 pattern。"
            )

        return asyncio.run(
            _call_http_tool(
                FILESYSTEM_MCP_URL,
                "search_files",
                {
                    "path": safe_path,
                    "pattern": pattern,
                },
            )
        )

    raise ValueError(
        "action 必须是 read / list / search"
    )


def _github_headers() -> dict[str, str]:
    token = os.getenv(
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "",
    ).strip()

    if not token:
        raise RuntimeError(
            "缺少 GITHUB_PERSONAL_ACCESS_TOKEN"
        )

    return {
        "Authorization": f"Bearer {token}",
        "X-MCP-Toolsets": "repos",
        "X-MCP-Readonly": "true",
    }


async def _github_search(
    query: str,
    limit: int,
) -> str:
    headers = _github_headers()

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=30.0,
    ) as client:

        async with streamable_http_client(
            GITHUB_MCP_URL,
            http_client=client,
        ) as streams:

            read_stream, write_stream, *_ = streams

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()

                available = await session.list_tools()

                search_tool = next(
                    (
                        tool
                        for tool in available.tools
                        if tool.name
                        == "search_repositories"
                    ),
                    None,
                )

                if search_tool is None:
                    raise RuntimeError(
                        "GitHub MCP 中没有找到 "
                        "search_repositories"
                    )

                arguments: dict[str, Any] = {
                    "query": query
                }

                schema = (
                    getattr(
                        search_tool,
                        "inputSchema",
                        None,
                    )
                    or getattr(
                        search_tool,
                        "input_schema",
                        None,
                    )
                    or {}
                )

                properties = schema.get(
                    "properties",
                    {},
                )

                if "perPage" in properties:
                    arguments["perPage"] = limit
                elif "per_page" in properties:
                    arguments["per_page"] = limit

                if "sort" in properties:
                    arguments["sort"] = "stars"

                if "order" in properties:
                    arguments["order"] = "desc"

                result = await session.call_tool(
                    "search_repositories",
                    arguments=arguments,
                )

                return _result_to_text(result)


def github_hot_repositories(
    keyword: str = "AI Agent",
    days: int = 30,
    min_stars: int = 100,
    limit: int = 10,
) -> str:
    keyword = str(keyword).strip() or "AI Agent"

    days = max(
        1,
        min(int(days), 365),
    )

    min_stars = max(
        0,
        int(min_stars),
    )

    limit = max(
        1,
        min(int(limit), 20),
    )

    since = (
        datetime.now(timezone.utc)
        - timedelta(days=days)
    ).date()

    query = (
        f"{keyword} "
        f"created:>={since.isoformat()} "
        f"stars:>={min_stars}"
    )

    return asyncio.run(
        _github_search(
            query=query,
            limit=limit,
        )
    )
