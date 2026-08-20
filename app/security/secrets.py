"""Environment-backed secret access; application code should not read config keys directly."""

from __future__ import annotations

import os


def get_secret(name: str, *, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(str(name).strip()) or default
    if required and not value:
        raise RuntimeError(f"缺少必需的 Secret：{name}")
    return value
