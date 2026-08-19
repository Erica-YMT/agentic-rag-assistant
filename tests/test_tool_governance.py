from types import SimpleNamespace

from app.agent.tool_executor import ToolExecutor


class FakeAuditStore:
    def __init__(self):
        self.started = []
        self.finished = []

    def start(self, **kwargs):
        self.started.append(kwargs)
        return len(self.started)

    def finish(self, audit_id, **kwargs):
        self.finished.append(
            {
                "audit_id": audit_id,
                **kwargs,
            }
        )


def make_tool_call(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )


def test_policy_allows_registered_read_tool():
    audit = FakeAuditStore()
    executor = ToolExecutor(
        {"calculator": lambda expression: "4"},
        audit_store=audit,
    )
    executor.set_context(
        user_id=7,
        role="user",
        session_id="s-1",
    )

    result = executor.execute(
        make_tool_call(
            "calculator",
            '{"expression":"2+2"}',
        )
    )

    assert result == "4"
    assert executor.get_called_tools() == ["calculator"]
    assert executor.get_call_records()[0]["decision"] == "allow"
    assert audit.started[0]["user_id"] == 7
    assert audit.started[0]["session_id"] == "s-1"
    assert audit.finished[0]["status"] == "success"


def test_policy_denies_unknown_tool_by_default():
    audit = FakeAuditStore()
    executed = []
    executor = ToolExecutor(
        {"calculator": lambda expression: executed.append(expression)},
        audit_store=audit,
    )

    result = executor.execute(
        make_tool_call(
            "dangerous_tool",
            '{}',
        )
    )

    assert result.startswith("工具调用被策略拒绝：")
    assert executor.get_called_tools() == []
    assert executed == []
    assert executor.get_call_records()[0]["decision"] == "deny"
    assert audit.started[0]["decision"] == "deny"
    assert audit.started[0]["status"] == "blocked"


def test_filesystem_policy_is_read_only():
    audit = FakeAuditStore()
    executed = []
    executor = ToolExecutor(
        {
            "mcp_filesystem": lambda **kwargs: executed.append(kwargs),
        },
        audit_store=audit,
    )

    result = executor.execute(
        make_tool_call(
            "mcp_filesystem",
            '{"action":"delete","path":"README.md"}',
        )
    )

    assert "仅允许只读操作" in result
    assert executor.get_called_tools() == []
    assert executed == []


def test_invalid_json_is_audited_without_execution():
    audit = FakeAuditStore()
    executor = ToolExecutor(
        {"calculator": lambda expression: "never"},
        audit_store=audit,
    )

    result = executor.execute(
        make_tool_call(
            "calculator",
            '{bad-json}',
        )
    )

    assert result.startswith("工具参数不是有效的JSON：")
    assert executor.get_called_tools() == []
    assert audit.started[0]["status"] == "error"
