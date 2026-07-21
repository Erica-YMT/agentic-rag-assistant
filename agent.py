import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError
)

from client import client, model_name
from memory import Memory
from prompt import SYSTEM_PROMPT
from tool_executor import ToolExecutor
from tools import available_tools
from tools_config import tools


class Agent:

    def __init__(self):

        self.client = client

        self.memory = Memory()

        self.tool_executor = ToolExecutor(
            available_tools
        )


    def get_called_tools(self):
        """获取上一轮调用过的工具名称。"""

        return self.tool_executor.get_called_tools()


    def get_call_records(self):
        """获取上一轮完整的工具调用记录。"""

        return self.tool_executor.get_call_records()

    def _build_tool_fallback(
        self,
        question,
        call_records
    ):
        """
        模型最终生成失败时，
        根据工具类型生成结构化降级回答。
        """

        answer_parts = []


        for record in call_records:

            tool_name = record.get(
                "tool_name",
                ""
            )

            arguments = record.get(
                "arguments",
                {}
            )

            result = str(
                record.get(
                    "result",
                    ""
                )
            ).strip()


            # =========================
            # 计算器结果
            # =========================

            if tool_name == "calculator":

                expression = arguments.get(
                    "expression",
                    "表达式"
                )

                answer_parts.append(
                    "计算结果：\n"
                    f"{expression} = {result}"
                )


            # =========================
            # 知识库检索结果
            # =========================

            elif tool_name == "search_knowledge":

                query = arguments.get(
                    "query",
                    question
                )

                answer_parts.append(
                    "知识库检索结果：\n"
                    f"检索内容：{query}\n\n"
                    f"{result}"
                )


            # =========================
            # 其他工具
            # =========================

            else:

                answer_parts.append(
                    f"{tool_name} 执行结果：\n"
                    f"{result}"
                )


        if not answer_parts:

            return (
                "模型接口暂时不可用，"
                "本次没有获得可返回的工具结果。"
            )


        fallback_answer = "\n\n".join(
            answer_parts
        )


        return (
            "模型在整理最终回答时超时，"
            "已直接返回成功执行的结果。\n\n"
            f"{fallback_answer}"
        )

    def _create_completion(
        self,
        messages,
        max_attempts=2,
        retry_delay=1
    ):
        """
        调用大模型。

        超时、连接错误、限流或服务器错误时，
        只重试当前这一次模型请求。
        """

        last_error = None


        for attempt in range(
            1,
            max_attempts + 1
        ):

            try:

                return (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=model_name,
                        messages=messages,
                        tools=tools
                    )
                )


            except (
                APITimeoutError,
                APIConnectionError,
                RateLimitError,
                InternalServerError

            ) as exc:

                last_error = exc

                if attempt < max_attempts:

                    print(
                    f"\n模型接口暂时无响应，"
                    f"{retry_delay} 秒后重试..."
                )

                time.sleep(retry_delay)


                if last_error is not None:
                    raise last_error


        raise RuntimeError(
            "模型请求失败，但没有获得错误信息。"
        )


    def run(
        self,
        session_id,
        question
    ):

        # 清空上一轮工具记录
        self.tool_executor.reset_called_tools()


        # 保存用户消息
        self.memory.add_message(
            session_id,
            "user",
            question
        )


        history = self.memory.get_messages(
            session_id
        )


        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            *history
        ]


        max_steps = 8


        for step in range(
            max_steps
        ):

            # 每一次模型请求都有独立重试
            try:

                response = self._create_completion(
                    messages
                )


            except Exception:

                call_records = (
                    self.tool_executor
                    .get_call_records()
                )


                # 已经成功执行过工具，
                # 只是在生成最终回答时超时

                if call_records:

                    answer = (
                        self._build_tool_fallback(
                            question,
                            call_records
                        )
                    )


                    self.memory.add_message(
                        session_id,
                        "assistant",
                        answer
                    )


                    return answer


                # 第一次模型请求就失败，
                # 没有任何工具结果可以返回

                raise


            msg = (
                response
                .choices[0]
                .message
            )


            # 模型没有调用工具，返回最终答案
            if not msg.tool_calls:

                answer = (
                    msg.content
                    or ""
                )


                self.memory.add_message(
                    session_id,
                    "assistant",
                    answer
                )


                return answer


            # 保存模型的工具调用消息
            messages.append(
                msg
            )


            # 执行全部工具调用
            for tool_call in msg.tool_calls:

                result = (
                    self.tool_executor
                    .execute(
                        tool_call
                    )
                )


                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                )


        answer = (
            "Agent执行步骤过多，"
            "已停止本次任务。"
        )


        self.memory.add_message(
            session_id,
            "assistant",
            answer
        )


        return answer