"""
PostgreSQL connection layer.

当前阶段只负责：
1. 读取连接配置
2. 创建 PostgreSQL Connection
3. 基础健康检查

暂时不负责：
- Chat History
- User Memory
- Redis Cache
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from collections.abc import Iterator

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row


def _required_env(
    name: str,
) -> str:

    value = str(
        os.getenv(name, "")
    ).strip()

    if not value:
        raise RuntimeError(
            f"缺少数据库环境变量：{name}"
        )

    return value


def get_postgres_config() -> dict:
    """
    返回 PostgreSQL 连接参数。

    不打印 password，
    防止数据库密码进入日志。
    """

    try:
        port = int(
            os.getenv(
                "POSTGRES_PORT",
                "5432",
            )
        )

    except ValueError as error:
        raise RuntimeError(
            "POSTGRES_PORT 必须是整数"
        ) from error

    return {
        "host": _required_env(
            "POSTGRES_HOST"
        ),
        "port": port,
        "dbname": _required_env(
            "POSTGRES_DB"
        ),
        "user": _required_env(
            "POSTGRES_USER"
        ),
        "password": _required_env(
            "POSTGRES_PASSWORD"
        ),
        "connect_timeout": 5,
        "application_name":
            "agentic-rag-assistant",
    }


@contextmanager
def postgres_connection(
) -> Iterator[Connection]:
    """
    PostgreSQL Connection Context。

    正常退出：
        commit + close

    出现异常：
        rollback + close
    """

    with psycopg.connect(
        **get_postgres_config(),
        row_factory=dict_row,
    ) as connection:

        yield connection


def check_postgres() -> dict:
    """
    执行最小 SELECT，
    验证 Python → PostgreSQL 链路。
    """

    with postgres_connection() as connection:

        row = connection.execute(
            """
            SELECT
                current_database()
                    AS database,
                current_user
                    AS database_user,
                1 AS ok
            """
        ).fetchone()

    if row is None:
        raise RuntimeError(
            "PostgreSQL 健康检查没有返回结果"
        )

    return dict(row)
