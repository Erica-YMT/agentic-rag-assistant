"""Reusable offline harness for Agent behavior contracts."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable


@dataclass(frozen=True)
class HarnessCase:
    name: str
    question: str
    expected_tools: tuple[str, ...] = ()
    expected_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessResult:
    name: str
    passed: bool
    answer: str
    actual_tools: tuple[str, ...]
    missing_keywords: tuple[str, ...]
    error: str | None = None


class AgentHarness:
    def __init__(self, agent_factory: Callable[[], object]) -> None:
        self.agent_factory = agent_factory

    def run(self, cases: list[HarnessCase]) -> list[HarnessResult]:
        results = []
        for case in cases:
            try:
                agent = self.agent_factory()
                answer = str(agent.run(f"harness-{case.name}", case.question))
                actual = tuple(agent.get_called_tools() or ())
                missing = tuple(keyword for keyword in case.expected_keywords if keyword not in answer)
                passed = set(case.expected_tools) <= set(actual) and not missing
                results.append(HarnessResult(case.name, passed, answer, actual, missing))
            except Exception as exc:
                results.append(HarnessResult(case.name, False, "", (), (), f"{type(exc).__name__}: {exc}"))
        return results

    @staticmethod
    def report(results: list[HarnessResult]) -> dict:
        return {
            "total": len(results),
            "passed": sum(item.passed for item in results),
            "failed": sum(not item.passed for item in results),
            "cases": [asdict(item) for item in results],
        }
