"""Durable dead-letter queue for failed tool executions."""

from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any

from app.db.postgres import postgres_connection


logger = logging.getLogger(__name__)

_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_KEY_PARTS):
                result[key_text] = "***REDACTED***"
            else:
                result[key_text] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


class ToolDeadLetterStore:
    def __init__(self) -> None:
        self._ready = False
        self._lock = Lock()

    def _ensure_table(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                with postgres_connection() as connection:
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS tool_dead_letters (
                            id BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            tool_name TEXT NOT NULL,
                            arguments_json TEXT NOT NULL,
                            reason TEXT NOT NULL,
                            attempts INTEGER NOT NULL
                        )
                        """
                    )
                self._ready = True
            except Exception as exc:
                logger.warning("Tool DLQ 初始化失败：%s", exc)

    def enqueue(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        attempts: int,
    ) -> None:
        try:
            self._ensure_table()
            if not self._ready:
                return
            with postgres_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO tool_dead_letters (
                        tool_name, arguments_json, reason, attempts
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        str(tool_name),
                        json.dumps(_redact(arguments), ensure_ascii=False, default=str)[:20_000],
                        str(reason),
                        int(attempts),
                    ),
                )
        except Exception as exc:
            logger.warning("Tool DLQ 写入失败：%s", exc)


tool_dead_letter_store = ToolDeadLetterStore()
