"""
PostgreSQL-based persistent conversation memory.

公共接口与根目录 memory.Memory 保持一致。
"""

from __future__ import annotations

import json
import logging
import os

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from .postgres import postgres_connection
from .redis_cache import get_redis_client


logger = logging.getLogger(__name__)


ALLOWED_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
}


_schema_lock = Lock()
_schema_initialized = False


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )



# CHAT_HISTORY_REDIS_CACHE_V2
# MULTI_USER_ISOLATION_V1
_CACHE_PREFIX = "agentic-rag:chat-history:v2"


def _cache_ttl_seconds() -> int:
    """Chat History Redis TTL；PostgreSQL 始终是 source of truth。"""

    raw = str(
        os.getenv(
            "CHAT_HISTORY_CACHE_TTL_SECONDS",
            "3600",
        )
    ).strip()

    try:
        value = int(raw)
    except ValueError:
        value = 3600

    return max(
        60,
        min(
            value,
            86400,
        ),
    )


def _normalize_user_id(
    user_id: int | str | None,
) -> int:
    """Chat History 正式用户主键统一使用 users.id。"""

    if user_id is None:
        raise ValueError(
            "PostgreSQL Chat History 必须提供 user_id"
        )

    try:
        value = int(str(user_id).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            "user_id 必须是合法整数"
        ) from error

    if value <= 0:
        raise ValueError(
            "user_id 必须大于 0"
        )

    return value


def _message_cache_key(
    user_id: int,
    session_id: str,
) -> str:
    return (
        f"{_CACHE_PREFIX}:"
        f"user:{user_id}:"
        f"session:{str(session_id)}"
    )


def _cache_get_messages(
    user_id: int,
    session_id: str,
) -> list[dict[str, str]] | None:
    """Redis 命中时返回该用户该会话的完整历史。"""

    try:
        raw = (
            get_redis_client()
            .get(
                _message_cache_key(
                    user_id,
                    session_id,
                )
            )
        )

        if raw is None:
            return None

        value = json.loads(raw)

        if not isinstance(value, list):
            return None

        messages: list[dict[str, str]] = []

        for item in value:
            if not isinstance(item, dict):
                return None

            messages.append(
                {
                    "role": str(item.get("role", "")),
                    "content": str(item.get("content", "")),
                }
            )

        return messages

    except Exception as error:
        logger.warning(
            "[RedisCache] 聊天历史读取失败，"
            "自动回退 PostgreSQL：%s",
            error,
        )
        return None


def _cache_set_messages(
    user_id: int,
    session_id: str,
    messages: list[dict[str, str]],
) -> None:
    """PostgreSQL 完整历史查询成功后写 Redis。"""

    try:
        get_redis_client().set(
            _message_cache_key(
                user_id,
                session_id,
            ),
            json.dumps(
                messages,
                ensure_ascii=False,
            ),
            ex=_cache_ttl_seconds(),
        )

    except Exception as error:
        logger.warning(
            "[RedisCache] 聊天历史写缓存失败，"
            "忽略缓存错误：%s",
            error,
        )


def _cache_delete_messages(
    user_id: int,
    session_id: str,
) -> None:
    """PostgreSQL 数据变化后，只失效该用户该会话的 Cache。"""

    try:
        get_redis_client().delete(
            _message_cache_key(
                user_id,
                session_id,
            )
        )

    except Exception as error:
        logger.warning(
            "[RedisCache] 聊天历史缓存失效失败，"
            "忽略缓存错误：%s",
            error,
        )


def _ensure_schema() -> None:
    """
    每个 Python 进程只做一次 Schema 检查。

    CREATE TABLE IF NOT EXISTS 本身也是幂等的。
    """

    global _schema_initialized

    if _schema_initialized:
        return

    with _schema_lock:

        if _schema_initialized:
            return

        with postgres_connection() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id BIGINT,
                    title TEXT NOT NULL
                        DEFAULT '新对话',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    CONSTRAINT
                        fk_messages_session

                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_messages_session_id
                ON messages(session_id, id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_messages_created_at
                ON messages(created_at)
                """
            )

            # MULTI_USER_ISOLATION_V1
            # 旧数据库兼容升级；正式 NOT NULL 在最终验收后再收紧。
            connection.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN IF NOT EXISTS
                    user_id BIGINT
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_sessions_user_updated
                ON sessions (
                    user_id,
                    updated_at DESC
                )
                """
            )

            connection.execute(
                """
                DO $$
                BEGIN
                    IF
                        to_regclass('public.users')
                        IS NOT NULL
                        AND NOT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conname =
                                'fk_sessions_user'
                        )
                    THEN
                        ALTER TABLE sessions
                        ADD CONSTRAINT
                            fk_sessions_user
                        FOREIGN KEY (user_id)
                        REFERENCES users(id);
                    END IF;
                END
                $$;
                """
            )

        _schema_initialized = True


class PostgresMemory:
    """PostgreSQL 聊天历史；正式业务按 users.id 做强隔离。"""

    def __init__(self) -> None:
        _ensure_schema()

    @staticmethod
    def _session_id(
        session_id: str,
    ) -> str:
        value = str(session_id).strip()

        if not value:
            raise ValueError(
                "session_id 不能为空"
            )

        return value

    def create_session(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> None:
        """创建会话；若 session_id 已属于别人则拒绝。"""

        session_id = self._session_id(session_id)
        user_id = _normalize_user_id(user_id)
        now = utc_now()

        with postgres_connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    user_id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (
                    %s,
                    %s,
                    '新对话',
                    %s,
                    %s
                )
                ON CONFLICT(session_id)
                DO NOTHING
                """,
                (
                    session_id,
                    user_id,
                    now,
                    now,
                ),
            )

            row = connection.execute(
                """
                SELECT user_id
                FROM sessions
                WHERE session_id = %s
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError("创建会话失败")

        owner = row["user_id"]

        if (
            owner is None
            or int(owner) != user_id
        ):
            raise PermissionError(
                "无权访问该会话"
            )

    def ensure_session_owner(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> None:
        """确保 session 属于当前用户；新 session 自动建立归属。"""

        self.create_session(
            session_id=session_id,
            user_id=user_id,
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: int | str | None = None,
    ) -> None:
        session_id = self._session_id(session_id)
        user_id = _normalize_user_id(user_id)
        role = str(role).strip()
        content = str(content)

        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"不支持的消息角色：{role}"
            )

        if not content.strip():
            return

        self.ensure_session_owner(
            session_id=session_id,
            user_id=user_id,
        )

        now = utc_now()
        title = (
            content
            .strip()
            .replace("\n", " ")[:40]
        )

        with postgres_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (
                    session_id,
                    role,
                    content,
                    created_at
                )
                SELECT
                    %s,
                    %s,
                    %s,
                    %s
                WHERE EXISTS (
                    SELECT 1
                    FROM sessions
                    WHERE
                        session_id = %s
                        AND user_id = %s
                )
                """,
                (
                    session_id,
                    role,
                    content,
                    now,
                    session_id,
                    user_id,
                ),
            )

            if cursor.rowcount != 1:
                raise PermissionError(
                    "无权访问该会话"
                )

            if role == "user":
                connection.execute(
                    """
                    UPDATE sessions
                    SET
                        title = CASE
                            WHEN title = '新对话'
                            THEN %s
                            ELSE title
                        END,
                        updated_at = %s
                    WHERE
                        session_id = %s
                        AND user_id = %s
                    """,
                    (
                        title or "新对话",
                        now,
                        session_id,
                        user_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE sessions
                    SET updated_at = %s
                    WHERE
                        session_id = %s
                        AND user_id = %s
                    """,
                    (
                        now,
                        session_id,
                        user_id,
                    ),
                )

        _cache_delete_messages(
            user_id,
            session_id,
        )

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
        user_id: int | str | None = None,
    ) -> list[dict[str, str]]:
        session_id = self._session_id(session_id)
        user_id = _normalize_user_id(user_id)

        cached = _cache_get_messages(
            user_id,
            session_id,
        )

        if cached is not None:
            if limit is None:
                return cached

            limit = max(
                1,
                min(int(limit), 500),
            )
            return cached[-limit:]

        with postgres_connection() as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT
                        m.role,
                        m.content
                    FROM messages AS m
                    INNER JOIN sessions AS s
                        ON s.session_id =
                            m.session_id
                    WHERE
                        m.session_id = %s
                        AND s.user_id = %s
                    ORDER BY m.id ASC
                    """,
                    (
                        session_id,
                        user_id,
                    ),
                ).fetchall()
            else:
                limit = max(
                    1,
                    min(int(limit), 500),
                )

                rows = connection.execute(
                    """
                    SELECT
                        role,
                        content
                    FROM (
                        SELECT
                            m.id,
                            m.role,
                            m.content
                        FROM messages AS m
                        INNER JOIN sessions AS s
                            ON s.session_id =
                                m.session_id
                        WHERE
                            m.session_id = %s
                            AND s.user_id = %s
                        ORDER BY m.id DESC
                        LIMIT %s
                    ) AS recent_messages
                    ORDER BY id ASC
                    """,
                    (
                        session_id,
                        user_id,
                        limit,
                    ),
                ).fetchall()

        messages = [
            {
                "role": str(row["role"]),
                "content": str(row["content"]),
            }
            for row in rows
        ]

        if limit is None:
            _cache_set_messages(
                user_id,
                session_id,
                messages,
            )

        return messages

    def get_session_messages(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        session_id = self._session_id(session_id)
        user_id = _normalize_user_id(user_id)

        with postgres_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    m.id,
                    m.session_id,
                    m.role,
                    m.content,
                    m.created_at
                FROM messages AS m
                INNER JOIN sessions AS s
                    ON s.session_id =
                        m.session_id
                WHERE
                    m.session_id = %s
                    AND s.user_id = %s
                ORDER BY m.id ASC
                """,
                (
                    session_id,
                    user_id,
                ),
            ).fetchall()

        return [dict(row) for row in rows]

    def list_sessions(
        self,
        limit: int = 100,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        user_id = _normalize_user_id(user_id)
        limit = max(
            1,
            min(int(limit), 500),
        )

        with postgres_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id)
                        AS message_count
                FROM sessions AS s
                LEFT JOIN messages AS m
                    ON m.session_id =
                        s.session_id
                WHERE s.user_id = %s
                GROUP BY
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at
                ORDER BY
                    s.updated_at DESC
                LIMIT %s
                """,
                (
                    user_id,
                    limit,
                ),
            ).fetchall()

        return [dict(row) for row in rows]

    def search_messages(
        self,
        keyword: str = "",
        session_id: str | None = None,
        limit: int = 100,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        user_id = _normalize_user_id(user_id)
        keyword = str(keyword).strip()
        limit = max(
            1,
            min(int(limit), 500),
        )

        conditions: list[str] = [
            "s.user_id = %s"
        ]
        parameters: list[Any] = [
            user_id
        ]

        if keyword:
            conditions.append(
                "m.content ILIKE %s"
            )
            parameters.append(
                f"%{keyword}%"
            )

        if session_id:
            conditions.append(
                "m.session_id = %s"
            )
            parameters.append(
                str(session_id)
            )

        where_clause = (
            "WHERE "
            + " AND ".join(conditions)
        )
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
                ON s.session_id =
                    m.session_id
            {where_clause}
            ORDER BY m.id DESC
            LIMIT %s
        """

        with postgres_connection() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_session(
        self,
        session_id: str,
        user_id: int | str | None = None,
    ) -> bool:
        session_id = self._session_id(session_id)
        user_id = _normalize_user_id(user_id)

        with postgres_connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM sessions
                WHERE
                    session_id = %s
                    AND user_id = %s
                """,
                (
                    session_id,
                    user_id,
                ),
            )
            deleted = cursor.rowcount > 0

        _cache_delete_messages(
            user_id,
            session_id,
        )

        return deleted
