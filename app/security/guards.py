"""Request and upload security guards."""

from __future__ import annotations

import re
from pathlib import Path


INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous instructions", re.I),
    re.compile(r"忽略(?:之前|上面|以上)的(?:所有)?指令"),
    re.compile(r"system\s+prompt|系统提示词", re.I),
    re.compile(r"开发者消息|developer message"),
)


def is_prompt_injection(value: str) -> bool:
    text = str(value or "")
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def validate_upload(filename: str, content: bytes) -> None:
    suffix = Path(str(filename)).suffix.lower()
    if suffix in {".xlsx", ".xlsm"} and not content.startswith(b"PK"):
        raise ValueError("Excel 文件内容签名无效")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise ValueError("PDF 文件内容签名无效")
    if suffix in {".eml", ".md", ".txt", ".case"} and b"\x00" in content[:4096]:
        raise ValueError("文本文件包含非法二进制内容")
