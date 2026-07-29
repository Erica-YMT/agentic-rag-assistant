"""SQLite-based persistent conversation memory."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
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


class Memory:
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
        content = str(content)

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

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """Return messages in the format expected by the model API."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (str(session_id),),
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
