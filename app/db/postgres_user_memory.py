"""PostgreSQL 长期用户记忆存储。"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from app.db.postgres import postgres_connection


DEFAULT_USER_ID = "local-user"


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


class PostgresUserMemoryStore:
    """
    使用 PostgreSQL 保存长期用户记忆。

    对外接口保持与 SQLite UserMemoryStore 一致：
    - save()
    - list()
    - delete()
    """

    backend_name = "postgres"

    _schema_lock = threading.Lock()
    _schema_ready = False


    def __init__(self) -> None:
        self._ensure_schema()


    @classmethod
    def _ensure_schema(cls) -> None:

        if cls._schema_ready:
            return

        with cls._schema_lock:

            if cls._schema_ready:
                return

            with postgres_connection() as connection:

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    user_memories (
                        id BIGSERIAL PRIMARY KEY,

                        user_id TEXT NOT NULL,

                        content TEXT NOT NULL,

                        normalized_content
                            TEXT NOT NULL,

                        source_session_id TEXT,

                        created_at TEXT NOT NULL,

                        updated_at TEXT NOT NULL,

                        UNIQUE (
                            user_id,
                            normalized_content
                        )
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_user_memories_user_updated
                    ON user_memories (
                        user_id,
                        updated_at DESC
                    )
                    """
                )

            cls._schema_ready = True


    @staticmethod
    def _normalize(
        content: str,
    ) -> str:

        return " ".join(
            str(content)
            .strip()
            .lower()
            .split()
        )


    def save(
        self,
        content: str,
        source_session_id: str | None = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> dict[str, Any]:

        content = str(
            content
        ).strip()

        if not content:
            raise ValueError(
                "记忆内容不能为空"
            )

        if len(content) > 1000:
            raise ValueError(
                "单条记忆不能超过 1000 个字符"
            )

        normalized = self._normalize(
            content
        )

        now = utc_now()


        with postgres_connection() as connection:

            row = connection.execute(
                """
                INSERT INTO user_memories (
                    user_id,
                    content,
                    normalized_content,
                    source_session_id,
                    created_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s
                )

                ON CONFLICT (
                    user_id,
                    normalized_content
                )

                DO UPDATE SET
                    content =
                        EXCLUDED.content,

                    source_session_id =
                        EXCLUDED.source_session_id,

                    updated_at =
                        EXCLUDED.updated_at

                RETURNING
                    id,
                    user_id,
                    content,
                    source_session_id,
                    created_at,
                    updated_at
                """,
                (
                    user_id,
                    content,
                    normalized,
                    source_session_id,
                    now,
                    now,
                ),
            ).fetchone()


        if row is None:
            raise RuntimeError(
                "长期记忆保存失败"
            )

        return dict(row)


    def list(
        self,
        user_id: str = DEFAULT_USER_ID,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        limit = max(
            1,
            min(
                int(limit),
                500,
            ),
        )


        with postgres_connection() as connection:

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

                WHERE user_id = %s

                ORDER BY
                    updated_at DESC,
                    id DESC

                LIMIT %s
                """,
                (
                    user_id,
                    limit,
                ),
            ).fetchall()


        return [
            dict(row)
            for row in rows
        ]


    def delete(
        self,
        memory_id: int,
        user_id: str = DEFAULT_USER_ID,
    ) -> bool:

        with postgres_connection() as connection:

            cursor = connection.execute(
                """
                DELETE FROM user_memories

                WHERE
                    id = %s
                    AND user_id = %s
                """,
                (
                    int(memory_id),
                    user_id,
                ),
            )


        return cursor.rowcount > 0
