"""Tool Audit 持久化。审计失败不得影响 Agent 主链路。"""

from __future__ import annotations

import json
import logging
import re
from threading import Lock
from typing import Any

from app.db.postgres import postgres_connection


logger = logging.getLogger(__name__)


SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                redacted[key_text] = "***REDACTED***"
            else:
                redacted[key_text] = _redact_value(item)
        return redacted

    if isinstance(value, list):
        return [_redact_value(item) for item in value]

    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]

    return value


def _redact_text(value: Any) -> str:
    text = str(value)
    text = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1***REDACTED***",
        text,
    )
    text = re.sub(
        r"(?i)(sk-|ghp_)[A-Za-z0-9_-]{8,}",
        "***REDACTED***",
        text,
    )
    return text


class ToolAuditStore:
    def __init__(self) -> None:
        self._ready = False
        self._ready_lock = Lock()

    def _ensure_table(self) -> None:
        if self._ready:
            return

        with self._ready_lock:
            if self._ready:
                return

            with postgres_connection() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_audit_logs (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        finished_at TIMESTAMPTZ NULL,
                        user_id BIGINT NULL,
                        session_id TEXT NULL,
                        tool_name TEXT NOT NULL,
                        role TEXT NULL,
                        risk_level TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT NULL,
                        arguments_json TEXT NOT NULL,
                        result_preview TEXT NULL,
                        elapsed_seconds DOUBLE PRECISION NULL
                    )
                    """
                )

            self._ready = True

    @staticmethod
    def _arguments_json(arguments: dict[str, Any]) -> str:
        return json.dumps(
            _redact_value(arguments),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )[:20_000]

    def start(
        self,
        *,
        user_id: int | None,
        session_id: str | None,
        tool_name: str,
        role: str | None,
        risk_level: str,
        decision: str,
        reason: str,
        arguments: dict[str, Any],
        status: str = "pending",
    ) -> int | None:
        try:
            self._ensure_table()
            with postgres_connection() as connection:
                row = connection.execute(
                    """
                    INSERT INTO tool_audit_logs (
                        user_id,
                        session_id,
                        tool_name,
                        role,
                        risk_level,
                        decision,
                        status,
                        reason,
                        arguments_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        int(user_id) if user_id is not None else None,
                        str(session_id) if session_id else None,
                        str(tool_name),
                        str(role) if role else None,
                        str(risk_level),
                        str(decision),
                        str(status),
                        str(reason),
                        self._arguments_json(arguments),
                    ),
                ).fetchone()
            return int(row["id"]) if row else None
        except Exception as exc:
            logger.warning("Tool Audit 写入开始记录失败：%s", exc)
            return None

    def finish(
        self,
        audit_id: int | None,
        *,
        status: str,
        result: Any,
        elapsed_seconds: float,
    ) -> None:
        if audit_id is None:
            return

        try:
            with postgres_connection() as connection:
                connection.execute(
                    """
                    UPDATE tool_audit_logs
                    SET
                        finished_at = NOW(),
                        status = %s,
                        result_preview = %s,
                        elapsed_seconds = %s
                    WHERE id = %s
                    """,
                    (
                        str(status),
                        _redact_text(result)[:2_000],
                        float(elapsed_seconds),
                        int(audit_id),
                    ),
                )
        except Exception as exc:
            logger.warning("Tool Audit 更新完成记录失败：%s", exc)


tool_audit_store = ToolAuditStore()
