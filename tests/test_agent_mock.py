import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import json
from types import MethodType, SimpleNamespace

from app.agent.agent import Agent
from app.agent.tool_executor import ToolExecutor


# =========================================================
# 假 Memory
# 不写真实 SQLite
# =========================================================

class FakeMemory:
    def __init__(self):
        self.data = {}

    def add_message(
        self,
        session_id,
        role,
        content,
        user_id=None,
    ):
        key = (
            str(user_id),
            session_id,
        )

        self.data.setdefault(
            key,
            []
        ).append(
            {
                "role": role,
                "content": str(content)
            }
        )

    def get_messages(
        self,
        session_id,
        user_id=None,
    ):
        key = (
            str(user_id),
            session_id,
        )

        return list(
            self.data.get(
                key,
                []
            )
        )


# =========================================================
# 构造假的 OpenAI response
# =========================================================

def make_tool_call(
    call_id,
    name,
    arguments
):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(
                arguments,
                ensure_ascii=False
            )
        )
    )


def make_response(
    content="",
    tool_calls=None
):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or []
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message
            )
        ]
    )


# =========================================================
# 假工具
# =========================================================

def fake_search_knowledge(query):
    return (
        "FAKE_KB_RESULT："
        f"检索关键词={query}"
    )


def fake_calculator(expression):
    return (
        "FAKE_CALC_RESULT："
        f"{expression}"
    )


FAKE_TOOLS = {
    "search_knowledge": (
        fake_search_knowledge
    ),
    "calculator": fake_calculator,
}


# =========================================================
# 创建不访问真实 DB / 模型 / FAISS 的 Agent
# =========================================================

def build_agent(
    mock_completion
):
    # 不执行 Agent.__init__()
    # 因此不会创建真实 Memory
    agent = Agent.__new__(
        Agent
    )

    agent.client = None

    # Agent.__new__ 会跳过 Agent.__init__，
    # 所以 Mock 测试必须手动补齐正式 Agent
    # 现在需要的实例属性。
    agent.user_id = None

    agent.memory = (
        FakeMemory()
    )

    agent.tool_executor = (
        ToolExecutor(
            FAKE_TOOLS
        )
    )

    agent._create_completion = (
        MethodType(
            mock_completion,
            agent
        )
    )

    return agent


# =========================================================
# 测试辅助
# =========================================================

def title(text):
    print()
    print(
        "=" * 70
    )
    print(text)
    print(
        "=" * 70
    )


def success(text):
    print(
        f"✅ {text}"
    )


# =========================================================
# TEST 1
# 不调用工具
# =========================================================

def test_no_tool():

    title(
        "TEST 1：不需要工具"
    )

    model_calls = []

    def mock_completion(
        self,
        messages,
        allow_tools=True
    ):
        model_calls.append(
            allow_tools
        )

        print(
            "[MOCK MODEL] "
            f"第 {len(model_calls)} 次调用，"
            f"allow_tools={allow_tools}"
        )

        return make_response(
            content="这是 Mock 最终回答。"
        )

    agent = build_agent(
        mock_completion
    )

    answer = agent.run(
        "mock-no-tool",
        "什么是人工智能？"
    )

    assert (
        answer
        == "这是 Mock 最终回答。"
    )

    assert (
        agent.get_called_tools()
        == []
    )

    assert (
        len(model_calls)
        == 1
    )

    success(
        "不需要工具：通过"
    )


# =========================================================
# TEST 2
# 一个工具
# =========================================================

def test_one_tool():

    title(
        "TEST 2：调用一个工具"
    )

    round_number = 0

    def mock_completion(
        self,
        messages,
        allow_tools=True
    ):
        nonlocal round_number

        round_number += 1

        print(
            "[MOCK MODEL] "
            f"第 {round_number} 次调用，"
            f"allow_tools={allow_tools}"
        )

        if round_number == 1:

            return make_response(
                tool_calls=[
                    make_tool_call(
                        "call-calculator-1",
                        "calculator",
                        {
                            "expression":
                            "123 * 456"
                        }
                    )
                ]
            )

        return make_response(
            content=(
                "计算完成，"
                "这是 Mock 最终回答。"
            )
        )

    agent = build_agent(
        mock_completion
    )

    answer = agent.run(
        "mock-one-tool",
        "请计算 123 * 456"
    )

    assert (
        agent.get_called_tools()
        == ["calculator"]
    )

    assert (
        len(
            agent.get_call_records()
        )
        == 1
    )

    assert round_number == 2

    print(
        "最终回答：",
        answer
    )

    success(
        "单工具调用：通过"
    )


# =========================================================
# TEST 3
# 连续两轮不同工具
# =========================================================

def test_two_tools():

    title(
        "TEST 3：连续两轮调用两个工具"
    )

    round_number = 0

    def mock_completion(
        self,
        messages,
        allow_tools=True
    ):
        nonlocal round_number

        round_number += 1

        print(
            "[MOCK MODEL] "
            f"第 {round_number} 次调用，"
            f"allow_tools={allow_tools}"
        )

        # 第一轮：
        # 先查知识库
        if round_number == 1:

            return make_response(
                tool_calls=[
                    make_tool_call(
                        "call-search-1",
                        "search_knowledge",
                        {
                            "query":
                            "FAISS 项目作用"
                        }
                    )
                ]
            )

        # 第二轮：
        # 再调用计算器
        if round_number == 2:

            return make_response(
                tool_calls=[
                    make_tool_call(
                        "call-calculator-2",
                        "calculator",
                        {
                            "expression":
                            "256 * 128"
                        }
                    )
                ]
            )

        # 第三轮：
        # 最终回答
        return make_response(
            content=(
                "知识库查询和计算均已完成。"
            )
        )

    agent = build_agent(
        mock_completion
    )

    answer = agent.run(
        "mock-two-tools",
        (
            "先查 FAISS，"
            "然后计算 256 * 128"
        )
    )

    called_tools = (
        agent.get_called_tools()
    )

    records = (
        agent.get_call_records()
    )

    print(
        "调用工具顺序：",
        called_tools
    )

    assert called_tools == [
        "search_knowledge",
        "calculator"
    ]

    assert len(records) == 2

    assert round_number == 3

    print(
        "最终回答：",
        answer
    )

    success(
        "连续两轮 Tool Calling：通过"
    )


# =========================================================
# TEST 4
# 防止重复调用
# =========================================================

def test_duplicate_guard():

    title(
        "TEST 4：重复工具调用保护"
    )

    round_number = 0

    def mock_completion(
        self,
        messages,
        allow_tools=True
    ):
        nonlocal round_number

        round_number += 1

        print(
            "[MOCK MODEL] "
            f"第 {round_number} 次调用，"
            f"allow_tools={allow_tools}"
        )

        # 第一轮正常执行
        if round_number == 1:

            return make_response(
                tool_calls=[
                    make_tool_call(
                        "duplicate-1",
                        "calculator",
                        {
                            "expression":
                            "2 + 2"
                        }
                    )
                ]
            )

        # 第二轮故意再次请求
        # 完全相同的工具和参数
        if round_number == 2:

            return make_response(
                tool_calls=[
                    make_tool_call(
                        "duplicate-2",
                        "calculator",
                        {
                            "expression":
                            "2 + 2"
                        }
                    )
                ]
            )

        return make_response(
            content=(
                "已使用之前的计算结果回答。"
            )
        )

    agent = build_agent(
        mock_completion
    )

    agent.run(
        "mock-duplicate",
        "计算 2 + 2"
    )

    records = (
        agent.get_call_records()
    )

    # 第二次相同调用应该被 Agent
    # 阻止，所以真正执行记录只有一次。
    assert len(records) == 1

    assert (
        agent.get_called_tools()
        == ["calculator"]
    )

    success(
        "重复工具调用保护：通过"
    )


# =========================================================
# TEST 5
# 最大工具执行次数 = 5
# =========================================================

def test_tool_limit():

    title(
        "TEST 5：最大工具次数保护"
    )

    round_number = 0
    allow_tools_history = []

    def mock_completion(
        self,
        messages,
        allow_tools=True
    ):
        nonlocal round_number

        round_number += 1

        allow_tools_history.append(
            allow_tools
        )

        print(
            "[MOCK MODEL] "
            f"第 {round_number} 次调用，"
            f"allow_tools={allow_tools}"
        )

        if round_number == 1:

            calls = []

            # 故意一次要求执行 6 个不同工具
            for i in range(
                1,
                7
            ):
                calls.append(
                    make_tool_call(
                        f"limit-{i}",
                        "calculator",
                        {
                            "expression":
                            f"{i} + {i}"
                        }
                    )
                )

            return make_response(
                tool_calls=calls
            )

        # 第二轮应该已经：
        # allow_tools=False
        return make_response(
            content=(
                "达到工具上限后，"
                "根据已有结果生成最终答案。"
            )
        )

    agent = build_agent(
        mock_completion
    )

    answer = agent.run(
        "mock-limit",
        "执行多个计算"
    )

    records = (
        agent.get_call_records()
    )

    print(
        "真正执行的工具次数：",
        len(records)
    )

    print(
        "allow_tools 历史：",
        allow_tools_history
    )

    assert len(records) == 5

    assert (
        allow_tools_history
        == [True, False]
    )

    print(
        "最终回答：",
        answer
    )

    success(
        "最大工具次数保护：通过"
    )


# =========================================================
# 运行全部测试
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "🚀 开始 Agent Mock 测试"
    )
    print(
        "本测试不会调用真实模型接口。"
    )

    test_no_tool()
    test_one_tool()
    test_two_tools()
    test_duplicate_guard()
    test_tool_limit()

    print()
    print(
        "=" * 70
    )
    print(
        "🎉 全部 Mock 测试通过"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "验证完成："
    )
    print(
        "✅ 不调用工具"
    )
    print(
        "✅ 单工具调用"
    )
    print(
        "✅ 连续两轮 Tool Calling"
    )
    print(
        "✅ 重复工具调用保护"
    )
    print(
        "✅ 最大 5 次工具执行保护"
    )
