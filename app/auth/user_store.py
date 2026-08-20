"""PostgreSQL 用户存储。"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from psycopg.errors import UniqueViolation

from app.db.postgres import (
    postgres_connection,
)


_schema_lock = Lock()
_schema_ready = False

ALLOWED_ROLES = {
    "user",
    "admin",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def _ensure_schema() -> None:
    global _schema_ready

    if _schema_ready:
        return

    with _schema_lock:

        if _schema_ready:
            return

        with postgres_connection() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL,
                    email TEXT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL
                        DEFAULT 'user',
                    is_active BOOLEAN NOT NULL
                        DEFAULT TRUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    CONSTRAINT ck_users_role
                    CHECK (
                        role IN ('user', 'admin')
                    )
                )
                """
            )

            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email_lower ON users (LOWER(email)) WHERE email IS NOT NULL"
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_users_username_lower
                ON users (
                    LOWER(username)
                )
                """
            )

        _schema_ready = True


class UserStore:
    """用户数据库访问层。"""

    def __init__(self) -> None:
        _ensure_schema()

    @staticmethod
    def normalize_username(
        username: str,
    ) -> str:
        return str(
            username
        ).strip().lower()

    def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        role: str = "user",
        email: str | None = None,
    ) -> dict[str, Any] | None:

        username = (
            self.normalize_username(
                username
            )
        )

        role = str(
            role
        ).strip().lower()

        if not username:
            raise ValueError(
                "用户名不能为空"
            )

        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"不支持的角色：{role}"
            )

        now = utc_now()

        try:

            with postgres_connection() as connection:

                row = connection.execute(
                    """
                    INSERT INTO users (
                        username,
                        email,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        TRUE,
                        %s,
                        %s
                    )
                    RETURNING
                        id,
                        username,
                        email,
                        role,
                        is_active,
                        created_at,
                        updated_at
                    """,
                    (
                        username,
                        email,
                        password_hash,
                        role,
                        now,
                        now,
                    ),
                ).fetchone()

        except UniqueViolation:
            return None

        if row is None:
            raise RuntimeError(
                "创建用户失败"
            )

        return dict(row)

    def get_auth_user_by_username(
        self,
        username: str,
    ) -> dict[str, Any] | None:

        username = (
            self.normalize_username(
                username
            )
        )

        with postgres_connection() as connection:

            row = connection.execute(
                """
                SELECT
                    id,
                    username,
                    email,
                    password_hash,
                    role,
                    is_active,
                    created_at,
                    updated_at
                FROM users
                WHERE LOWER(username) = %s
                LIMIT 1
                """,
                (
                    username,
                ),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_auth_user_by_email(self, email: str) -> dict[str, Any] | None:
        with postgres_connection() as connection:
            row = connection.execute(
                "SELECT id, username, email, password_hash, role, is_active, created_at, updated_at FROM users WHERE LOWER(email) = %s LIMIT 1",
                (str(email).strip().lower(),),
            ).fetchone()
        return None if row is None else dict(row)

    def get_user_by_id(
        self,
        user_id: int,
    ) -> dict[str, Any] | None:

        with postgres_connection() as connection:

            row = connection.execute(
                """
                SELECT
                    id,
                    username,
                    role,
                    is_active,
                    created_at,
                    updated_at
                FROM users
                WHERE id = %s
                LIMIT 1
                """,
                (
                    int(user_id),
                ),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def delete_by_username(
        self,
        username: str,
    ) -> bool:
        """
        当前仅用于开发 / 自动测试清理。
        正式 API 不暴露删除用户接口。
        """

        username = (
            self.normalize_username(
                username
            )
        )

        with postgres_connection() as connection:

            cursor = connection.execute(
                """
                DELETE FROM users
                WHERE LOWER(username) = %s
                """,
                (
                    username,
                ),
            )

        return cursor.rowcount > 0
