"""Small deterministic PII and credential sanitizer for application boundaries."""

from __future__ import annotations

import re
from typing import Any


PATTERNS = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")),
    ("bank_card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)),
    ("api_key", re.compile(r"\b(?:sk|ghp|xoxb|AIza)[-_][A-Za-z0-9_-]{8,}\b", re.I)),
)


def sanitize_text(value: Any) -> str:
    text = str(value or "")
    for label, pattern in PATTERNS:
        text = pattern.sub(f"[{label.upper()}_REDACTED]", text)
    return text


def contains_pii(value: Any) -> bool:
    text = str(value or "")
    return any(pattern.search(text) for _, pattern in PATTERNS)


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value
