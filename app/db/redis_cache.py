"""
Redis connection layer.

Redis 在本项目中只作为 Cache。

真正业务数据以后仍以 PostgreSQL 为准。
"""

from __future__ import annotations

import os
from threading import Lock

import redis
from redis import Redis


_client: Redis | None = None
_client_lock = Lock()


def get_redis_url() -> str:

    url = str(
        os.getenv(
            "REDIS_URL",
            "",
        )
    ).strip()

    if not url:
        raise RuntimeError(
            "缺少环境变量：REDIS_URL"
        )

    return url


def get_redis_client() -> Redis:
    """
    返回共享 Redis Client。

    redis-py Client 内部使用连接池，
    不为每次请求重复创建连接池。
    """

    global _client

    if _client is not None:
        return _client

    with _client_lock:

        if _client is None:

            _client = Redis.from_url(
                get_redis_url(),
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )

    return _client


def check_redis() -> bool:
    """
    验证 Python → Redis 链路。
    """

    return bool(
        get_redis_client().ping()
    )


def close_redis() -> None:
    """
    主动关闭 Redis Client。
    """

    global _client

    with _client_lock:

        if _client is not None:

            _client.close()
            _client = None
