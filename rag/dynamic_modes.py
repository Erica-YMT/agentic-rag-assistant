"""Business-oriented RAG mode selection."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DynamicRAGMode:
    name: str
    top_k: int
    require_evidence: bool
    allow_web: bool


CHAT = DynamicRAGMode("CHAT", 2, False, False)
CONSULT = DynamicRAGMode("CONSULT", 4, True, True)
RISK = DynamicRAGMode("RISK", 6, True, False)


def classify_mode(question: str) -> DynamicRAGMode:
    text = str(question or "").strip().lower()
    if re.search(r"风险|合规|安全|法律|医疗|财务|事故|预警|责任", text):
        return RISK
    if re.search(r"咨询|方案|如何|比较|分析|建议|规划|为什么", text):
        return CONSULT
    return CHAT
