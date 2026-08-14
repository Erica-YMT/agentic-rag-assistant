"""用户长期记忆：保存用户明确要求记住的信息。"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "user_memory.db"
DEFAULT_USER_ID = "local-user"

REMEMBER_PATTERNS = (
    re.compile(r"^\s*请记住(?:一下)?[：:，,\s]*(.+?)\s*$", re.DOTALL),
    re.compile(r"^\s*帮我记住[：:，,\s]*(.+?)\s*$", re.DOTALL),
    re.compile(r"^\s*记住[：:，,\s]+(.+?)\s*$", re.DOTALL),
)

SENSITIVE_PATTERNS = (
    re.compile(r"\bapi[_\s-]?key\b", re.IGNORECASE),
    re.compile(r"\baccess[_\s-]?token\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"密码"),
    re.compile(r"口令"),
    re.compile(r"私钥"),
    re.compile(r"身份证"),
    re.compile(r"银行卡"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_explicit_memory(text: str) -> str | None:
    """提取“请记住……”后面的内容。"""
    value = str(text or "").strip()

    for pattern in REMEMBER_PATTERNS:
        match = pattern.match(value)

        if match:
            content = match.group(1).strip()
            content = content.rstrip("。.!！")

            return content or None

    return None


def contains_sensitive_content(content: str) -> bool:
    """避免把密码、密钥等敏感信息写入长期记忆。"""
    return any(
        pattern.search(content)
        for pattern in SENSITIVE_PATTERNS
    )


class SQLiteUserMemoryStore:
    """使用 SQLite 保存长期用户记忆。"""

    def __init__(
        self,
        db_path: str | Path | None = None,
    ) -> None:
        self.db_path = Path(
            db_path or DEFAULT_DB_PATH
        ).expanduser().resolve()

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    source_session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, normalized_content)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_user_memories_user_updated
                ON user_memories(user_id, updated_at DESC)
                """
            )

    @staticmethod
    def _normalize(content: str) -> str:
        return " ".join(
            str(content).strip().lower().split()
        )

    def save(
        self,
        content: str,
        source_session_id: str | None = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> dict[str, Any]:
        content = str(content).strip()

        if not content:
            raise ValueError("记忆内容不能为空")

        if len(content) > 1000:
            raise ValueError("单条记忆不能超过 1000 个字符")

        normalized = self._normalize(content)
        now = utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memories (
                    user_id,
                    content,
                    normalized_content,
                    source_session_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, normalized_content)
                DO UPDATE SET
                    content = excluded.content,
                    source_session_id =
                        excluded.source_session_id,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    content,
                    normalized,
                    source_session_id,
                    now,
                    now,
                ),
            )

            row = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    content,
                    source_session_id,
                    created_at,
                    updated_at
                FROM user_memories
                WHERE
                    user_id = ?
                    AND normalized_content = ?
                """,
                (user_id, normalized),
            ).fetchone()

        if row is None:
            raise RuntimeError("长期记忆保存失败")

        return dict(row)

    def list(
        self,
        user_id: str = DEFAULT_USER_ID,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    content,
                    source_session_id,
                    created_at,
                    updated_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def delete(
        self,
        memory_id: int,
        user_id: str = DEFAULT_USER_ID,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM user_memories
                WHERE id = ? AND user_id = ?
                """,
                (int(memory_id), user_id),
            )

        return cursor.rowcount > 0


class UserMemoryStore:
    """
    长期记忆统一入口。

    USER_MEMORY_BACKEND:
    - sqlite
    - postgres

    显式传入 db_path 时强制使用 SQLite，
    保留原测试和兼容行为。
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
    ) -> None:

        backend = (
            os.getenv(
                "USER_MEMORY_BACKEND",
                "sqlite",
            )
            .strip()
            .lower()
        )

        # 显式 SQLite 路径优先，
        # 保留原有兼容能力。
        if db_path is not None:
            backend = "sqlite"

        if backend == "sqlite":

            self._backend = (
                SQLiteUserMemoryStore(
                    db_path=db_path,
                )
            )

        elif backend == "postgres":

            from app.db.postgres_user_memory import (
                PostgresUserMemoryStore,
            )

            self._backend = (
                PostgresUserMemoryStore()
            )

        else:
            raise ValueError(
                "不支持的 USER_MEMORY_BACKEND："
                f"{backend!r}。"
                "只能使用 sqlite 或 postgres。"
            )

        self.backend_name = backend

        # 保留 db_path 属性，
        # PostgreSQL 时为 None。
        self.db_path = getattr(
            self._backend,
            "db_path",
            None,
        )


    @staticmethod
    def _normalize(
        content: str,
    ) -> str:
        return SQLiteUserMemoryStore._normalize(
            content
        )


    def save(
        self,
        content: str,
        source_session_id: str | None = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> dict[str, Any]:

        return self._backend.save(
            content=content,
            source_session_id=source_session_id,
            user_id=user_id,
        )


    def list(
        self,
        user_id: str = DEFAULT_USER_ID,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        return self._backend.list(
            user_id=user_id,
            limit=limit,
        )


    def delete(
        self,
        memory_id: int,
        user_id: str = DEFAULT_USER_ID,
    ) -> bool:

        return self._backend.delete(
            memory_id=memory_id,
            user_id=user_id,
        )
