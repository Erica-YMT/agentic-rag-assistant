"""SQLite-based persistent conversation memory."""

from __future__ import annotations

import os

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.memory.compaction import build_summary
from app.privacy.pii import sanitize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chat_history.db"

ALLOWED_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
}


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteMemory:
    """Store conversation messages in SQLite.

    Existing Agent calls remain compatible:

        create_session(session_id)
        add_message(session_id, role, content)
        get_messages(session_id)
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '新对话',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_created_at
                ON messages(created_at)
                """
            )

    def create_session(self, session_id: str) -> None:
        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id 不能为空")

        now = utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, '新对话', ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now, now),
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        session_id = str(session_id).strip()
        role = str(role).strip()
        content = sanitize_text(content)

        if not session_id:
            raise ValueError("session_id 不能为空")

        if role not in ALLOWED_ROLES:
            raise ValueError(f"不支持的消息角色：{role}")

        if not content.strip():
            return

        now = utc_now()

        # 第一条用户消息作为会话标题。
        title = content.strip().replace("\n", " ")[:40]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, '新对话', ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now, now),
            )

            connection.execute(
                """
                INSERT INTO messages (
                    session_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )

            if role == "user":
                connection.execute(
                    """
                    UPDATE sessions
                    SET
                        title = CASE
                            WHEN title = '新对话' THEN ?
                            ELSE title
                        END,
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (title or "新对话", now, session_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Return messages in the format expected by the model API.

        limit=None:
            返回该会话全部历史。

        limit=N:
            只返回最近 N 条消息，
            但保持正常的时间顺序。
        """

        with self._connect() as connection:

            if limit is None:

                rows = connection.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (str(session_id),),
                ).fetchall()

            else:

                limit = max(
                    1,
                    min(
                        int(limit),
                        500
                    )
                )

                rows = connection.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT
                            id,
                            role,
                            content
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (
                        str(session_id),
                        limit,
                    ),
                ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in rows
        ]

    def get_session_messages(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Return complete messages for history display."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (str(session_id),),
            ).fetchall()

        return [dict(row) for row in rows]

    def compact_session(
        self,
        session_id: str,
        keep_recent: int = 20,
        max_summary_chars: int = 5000,
    ) -> dict[str, int | bool]:
        messages = self.get_messages(session_id)
        keep_recent = max(2, int(keep_recent))
        if len(messages) <= keep_recent:
            return {"compacted": False, "removed": 0}
        old_messages = messages[:-keep_recent]
        summary = build_summary(old_messages, max_chars=max_summary_chars)
        if not summary:
            return {"compacted": False, "removed": 0}
        with self._connect() as connection:
            ids = connection.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id ASC",
                (str(session_id),),
            ).fetchall()
            remove_ids = [int(row["id"]) for row in ids[:-keep_recent]]
            connection.executemany("DELETE FROM messages WHERE id = ?", [(value,) for value in remove_ids])
            now = utc_now()
            connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, 'system', ?, ?)",
                (str(session_id), "【会话摘要】\n" + summary, now),
            )
        return {"compacted": True, "removed": len(remove_ids)}

    def list_sessions(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) AS message_count
                FROM sessions AS s
                LEFT JOIN messages AS m
                    ON m.session_id = s.session_id
                GROUP BY
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def search_messages(
        self,
        keyword: str = "",
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search message content using a simple keyword query."""
        keyword = str(keyword).strip()
        limit = max(1, min(int(limit), 500))

        conditions: list[str] = []
        parameters: list[Any] = []

        if keyword:
            conditions.append("m.content LIKE ?")
            parameters.append(f"%{keyword}%")

        if session_id:
            conditions.append("m.session_id = ?")
            parameters.append(str(session_id))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        parameters.append(limit)

        query = f"""
            SELECT
                m.id,
                m.session_id,
                s.title,
                m.role,
                m.content,
                m.created_at
            FROM messages AS m
            INNER JOIN sessions AS s
                ON s.session_id = m.session_id
            {where_clause}
            ORDER BY m.id DESC
            LIMIT ?
        """

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete one conversation and all of its messages."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM sessions
                WHERE session_id = ?
                """,
                (str(session_id),),
            )

        return cursor.rowcount > 0


class Memory:
    """
    Conversation Memory Facade.

    MULTI_USER_ISOLATION_V1：
    - PostgreSQL 正式业务要求 user_id；
    - SQLite 兼容/测试路径继续保持原接口和数据结构。
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        backend: str | None = None,
    ) -> None:
        requested_backend = (
            str(
                backend
                or os.getenv(
                    "MEMORY_BACKEND",
                    "sqlite",
                )
            )
            .strip()
            .lower()
        )

        if (
            db_path is not None
            and backend is None
        ):
            requested_backend = "sqlite"

        if requested_backend == "sqlite":
            self._backend_name = "sqlite"
            self._store = SQLiteMemory(
                db_path=db_path
            )

        elif requested_backend == "postgres":
            if db_path is not None:
                raise ValueError(
                    "PostgreSQL Memory "
                    "不支持 db_path 参数"
                )

            from app.db.postgres_memory import (
                PostgresMemory,
            )

            self._backend_name = "postgres"
            self._store = PostgresMemory()

        else:
            raise ValueError(
                "MEMORY_BACKEND 只支持："
                "sqlite / postgres"
            )

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def db_path(self):
        return getattr(
            self._store,
            "db_path",
            None,
        )

    def ensure_session_owner(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> None:
        if self._backend_name == "postgres":
            return self._store.ensure_session_owner(
                session_id=session_id,
                user_id=user_id,
            )

        self._store.create_session(session_id)
        return None

    def create_session(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> None:
        if self._backend_name == "postgres":
            return self._store.create_session(
                session_id=session_id,
                user_id=user_id,
            )

        return self._store.create_session(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: int | str | None = None,
    ) -> None:
        if self._backend_name == "postgres":
            return self._store.add_message(
                session_id=session_id,
                role=role,
                content=content,
                user_id=user_id,
            )

        return self._store.add_message(
            session_id,
            role,
            content,
        )

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
        user_id: int | str | None = None,
    ) -> list[dict[str, str]]:
        if self._backend_name == "postgres":
            return self._store.get_messages(
                session_id=session_id,
                limit=limit,
                user_id=user_id,
            )

        return self._store.get_messages(
            session_id,
            limit=limit,
        )

    def compact_session(
        self,
        session_id: str,
        keep_recent: int = 20,
        max_summary_chars: int = 5000,
        user_id: int | str | None = None,
    ) -> dict[str, int | bool]:
        if self._backend_name == "postgres":
            return self._store.compact_session(
                session_id=session_id,
                keep_recent=keep_recent,
                max_summary_chars=max_summary_chars,
                user_id=user_id,
            )
        return self._store.compact_session(
            session_id,
            keep_recent=keep_recent,
            max_summary_chars=max_summary_chars,
        )

    def get_session_messages(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        if self._backend_name == "postgres":
            return self._store.get_session_messages(
                session_id=session_id,
                user_id=user_id,
            )

        return self._store.get_session_messages(session_id)

    def list_sessions(
        self,
        limit: int = 100,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        if self._backend_name == "postgres":
            return self._store.list_sessions(
                limit=limit,
                user_id=user_id,
            )

        return self._store.list_sessions(limit=limit)

    def search_messages(
        self,
        keyword: str = "",
        session_id: str | None = None,
        limit: int = 100,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        if self._backend_name == "postgres":
            return self._store.search_messages(
                keyword=keyword,
                session_id=session_id,
                limit=limit,
                user_id=user_id,
            )

        return self._store.search_messages(
            keyword=keyword,
            session_id=session_id,
            limit=limit,
        )

    def delete_session(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> bool:
        if self._backend_name == "postgres":
            return self._store.delete_session(
                session_id=session_id,
                user_id=user_id,
            )

        return self._store.delete_session(session_id)


# MULTI_USER_ISOLATION_V1
