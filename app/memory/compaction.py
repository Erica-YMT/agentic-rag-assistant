"""Deterministic conversation compaction that does not require an LLM call."""

from __future__ import annotations

from app.privacy.pii import sanitize_text


def build_summary(messages: list[dict[str, str]], max_chars: int = 5000) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = sanitize_text(message.get("content", "")).strip()
        if not content or role == "system":
            continue
        parts.append(f"{role}: {content[:700]}")
    summary = "\n".join(parts)
    return summary[:max_chars]
