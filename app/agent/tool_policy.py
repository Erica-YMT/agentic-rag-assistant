"""统一 Tool Governance 策略。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class ToolRule:
    risk_level: str
    allowed_roles: frozenset[str]


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    reason: str
    risk_level: str


TOOL_RULES: dict[str, ToolRule] = {
    "search_knowledge": ToolRule(
        risk_level="low",
        allowed_roles=frozenset({"user", "admin"}),
    ),
    "calculator": ToolRule(
        risk_level="low",
        allowed_roles=frozenset({"user", "admin"}),
    ),
    "search_web": ToolRule(
        risk_level="medium",
        allowed_roles=frozenset({"user", "admin"}),
    ),
    "mcp_filesystem": ToolRule(
        risk_level="medium",
        allowed_roles=frozenset({"user", "admin"}),
    ),
    "github_hot_repositories": ToolRule(
        risk_level="medium",
        allowed_roles=frozenset({"user", "admin"}),
    ),
}


class ToolPolicy:
    """对模型提出的工具调用做统一授权与参数边界检查。"""

    def __init__(
        self,
        rules: dict[str, ToolRule] | None = None,
        *,
        max_arguments_bytes: int = 16_384,
    ) -> None:
        self.rules = dict(rules or TOOL_RULES)
        self.max_arguments_bytes = int(max_arguments_bytes)

    def check(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        role: str | None,
    ) -> ToolDecision:
        name = str(tool_name).strip()
        normalized_role = str(role or "user").strip().lower() or "user"

        rule = self.rules.get(name)
        if rule is None:
            return ToolDecision(
                allowed=False,
                reason="工具未注册到 Tool Policy，默认拒绝。",
                risk_level="unknown",
            )

        if normalized_role not in rule.allowed_roles:
            return ToolDecision(
                allowed=False,
                reason=f"角色 {normalized_role} 无权调用该工具。",
                risk_level=rule.risk_level,
            )

        try:
            encoded = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        except Exception:
            return ToolDecision(
                allowed=False,
                reason="工具参数无法安全序列化。",
                risk_level=rule.risk_level,
            )

        if len(encoded) > self.max_arguments_bytes:
            return ToolDecision(
                allowed=False,
                reason="工具参数超过允许大小。",
                risk_level=rule.risk_level,
            )

        if name == "mcp_filesystem":
            action = str(arguments.get("action", "")).strip().lower()
            if action not in {"read", "list", "search"}:
                return ToolDecision(
                    allowed=False,
                    reason="Filesystem MCP 仅允许只读操作。",
                    risk_level=rule.risk_level,
                )

        return ToolDecision(
            allowed=True,
            reason="通过 Tool Policy。",
            risk_level=rule.risk_level,
        )
