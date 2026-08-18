from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings


ROOT = Path(
    os.getenv("MCP_FILESYSTEM_ROOT", "/workspace")
).resolve()

mcp = MCPServer(
    "agentic-rag-filesystem"
)


def _safe_path(path: str) -> Path:
    raw = str(path or ".").strip()

    target = (
        ROOT / raw
    ).resolve()

    if (
        target != ROOT
        and ROOT not in target.parents
    ):
        raise ValueError(
            "只能访问项目目录内的文件"
        )

    return target


@mcp.tool()
def list_directory(
    path: str = ".",
) -> str:
    """列出项目目录中的文件和子目录。"""

    target = _safe_path(path)

    if not target.is_dir():
        raise ValueError(
            f"不是目录: {path}"
        )

    rows = []

    for item in sorted(
        target.iterdir()
    ):
        kind = (
            "DIR"
            if item.is_dir()
            else "FILE"
        )

        rows.append(
            f"[{kind}] {item.name}"
        )

    return "\n".join(rows)


@mcp.tool()
def read_text_file(
    path: str,
) -> str:
    """读取项目中的文本文件。"""

    target = _safe_path(path)

    if not target.is_file():
        raise ValueError(
            f"文件不存在: {path}"
        )

    text = target.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if len(text) > 200_000:
        return (
            text[:200_000]
            + "\n\n[内容过长，已截断]"
        )

    return text


@mcp.tool()
def search_files(
    path: str = ".",
    pattern: str = "",
) -> str:
    """按文件名模式搜索项目文件。"""

    pattern = str(
        pattern
    ).strip()

    if not pattern:
        raise ValueError(
            "pattern 不能为空"
        )

    target = _safe_path(path)

    if not target.is_dir():
        raise ValueError(
            f"不是目录: {path}"
        )

    matches = []

    for item in target.rglob(
        pattern
    ):
        resolved = item.resolve()

        if (
            resolved == ROOT
            or ROOT in resolved.parents
        ):
            matches.append(
                str(
                    resolved.relative_to(
                        ROOT
                    )
                )
            )

        if len(matches) >= 100:
            break

    return "\n".join(matches)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8083,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                "filesystem-mcp",
                "filesystem-mcp:*",
                "localhost",
                "localhost:*",
                "127.0.0.1",
                "127.0.0.1:*",
            ],
        ),
    )
