from types import SimpleNamespace

from evaluation.agent_harness import AgentHarness, HarnessCase


class FakeAgent:
    def run(self, session_id, question):
        self.tools = ["calculator"] if "计算" in question else []
        return "计算结果 42" if self.tools else "你好"

    def get_called_tools(self):
        return self.tools


def test_agent_harness_contracts():
    harness = AgentHarness(FakeAgent)
    results = harness.run([
        HarnessCase("greeting", "你好", expected_keywords=("你好",)),
        HarnessCase("calc", "计算", expected_tools=("calculator",), expected_keywords=("42",)),
    ])
    assert all(item.passed for item in results)
    assert harness.report(results)["passed"] == 2
