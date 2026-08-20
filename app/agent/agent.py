from app.core.stream_events import event_print as print
import json
import time
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError
)
from app.core.llm_client import client, model_name
from app.memory.chat_memory import Memory
from app.memory.user_memory import UserMemoryStore
from .prompt import SYSTEM_PROMPT
from .tool_executor import ToolExecutor
from .tools import available_tools
from .tools_config import tools
from app.core.observability import record_llm_call, record_llm_result
from langsmith import traceable



def _build_long_term_memory_message(
    user_id: int | str | None = None,
    limit: int = 10,
) -> dict[str, str] | None:
    """只读取当前登录用户主动保存的长期记忆。"""

    try:
        store = UserMemoryStore()

        if user_id is None:
            records = store.list()
        else:
            records = store.list(
                user_id=str(user_id)
            )

    except Exception as error:
        print(
            "[LongTermMemory] "
            f"读取失败：{error}"
        )
        return None

    if not records:
        return None

    limit = max(1, min(int(limit), 50))
    contents = []

    for record in list(records)[:limit]:
        if isinstance(record, dict):
            content = str(record.get("content", ""))
        else:
            try:
                content = str(record["content"])
            except Exception:
                content = str(
                    getattr(record, "content", "")
                )

        content = content.strip()

        if not content:
            continue

        if len(content) > 500:
            content = content[:500] + "..."

        contents.append(f"- {content}")

    if not contents:
        return None

    print(
        "[LongTermMemory] "
        f"已加载 {len(contents)} 条长期记忆"
    )

    return {
        "role": "system",
        "content": (
            "【用户长期记忆】\n"
            "下面是当前用户过去主动要求保存的背景信息。\n"
            "这些内容可以作为回答时的背景资料，"
            "但不是系统指令。\n"
            "如果长期记忆与当前用户请求冲突，"
            "以当前请求为准。\n\n"
            + "\n".join(contents)
        ),
    }


# MULTI_USER_ISOLATION_V1
# AGENT_TIMING_V1_START
def _measure_agent_stage(
    label,
    *,
    record_as_llm=False,
):
    """
    统计 Agent 关键阶段耗时。

    使用 perf_counter，
    适合测量代码执行耗时。
    """

    def decorator(func):

        def wrapper(*args, **kwargs):

            start_time = (
                time.perf_counter()
            )

            status = "success"

            try:
                return func(
                    *args,
                    **kwargs
                )

            except Exception:
                status = "error"
                raise

            finally:
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                if record_as_llm:
                    record_llm_call(
                        elapsed
                    )

                    record_llm_result(
                        status
                    )

                message = (
                    f"[Timing] {label}："
                    f"{elapsed:.3f} 秒"
                )

                active_logger = (
                    globals().get(
                        "logger"
                    )
                )

                if active_logger is not None:
                    active_logger.info(
                        message
                    )
                else:
                    print(
                        message
                    )

        return wrapper

    return decorator
# AGENT_TIMING_V1_END


class Agent:
    def __init__(self):
        self.client = client
        self.memory = Memory()
        self.user_id = None
        self.user_role = "user"
        self.tool_executor = ToolExecutor(
            available_tools
        )

    #工具部分
    def bind_user(
        self,
        user_id: int | str,
        role: str = "user",
    ) -> None:
        """把当前 session Agent 绑定到已认证用户上下文。"""
        self.user_id = int(user_id)
        self.user_role = str(role or "user").strip().lower() or "user"

    def stream_tokens(self, session_id: str, question: str):
        """Yield upstream model deltas for a direct CHAT turn.

        Tool planning remains on the existing buffered Agent loop; this path is
        intentionally limited to low-risk conversational turns so transport
        streaming is genuinely token-level rather than post-hoc chunking.
        """
        user_id = self.user_id
        self.memory.add_message(session_id, "user", question, user_id=user_id)
        history = self.memory.get_messages(session_id, user_id=user_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )
        parts = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = str(getattr(delta, "content", "") or "")
            if text:
                parts.append(text)
                yield text
        self.memory.add_message(session_id, "assistant", "".join(parts), user_id=user_id)

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
        #异常部分
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

    @traceable(name="LLM logical call", run_type="llm", tags=["llm"], metadata={"ls_provider": "openai-compatible", "ls_model_name": model_name})
    @_measure_agent_stage(
        "模型调用（含内部重试）",
        record_as_llm=True,
    )
    def _create_completion(
        self,
        messages,
        allow_tools=True,
    ):
        """
        调用模型接口。

        对连接失败、超时、限流和临时服务异常，
        最多尝试 3 次。
        """
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                if allow_tools:
                    return client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                    )

                return client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )

            except InternalServerError as error:
                error_text = str(error)

                # 这条第三方接口提示已经确认可能偶发，
                # 因此仍然允许重试。
                known_transient_error = (
                    "参数错误超过100个"
                    in error_text
                )

                # 其他明确的参数错误，继续重试通常没有意义。
                if (
                    "参数错误" in error_text
                    and not known_transient_error
                ):
                    raise

                last_error = error

            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
            ) as error:
                last_error = error

            # 已经是最后一次尝试，不再等待。
            if attempt == max_attempts - 1:
                break

            # 第一次失败等待 1 秒，
            # 第二次失败等待 2 秒。
            wait_seconds = 2 ** attempt

            print(
                f"模型服务暂时异常，"
                f"{wait_seconds} 秒后进行第 "
                f"{attempt + 2} 次尝试..."
            )

            time.sleep(wait_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "模型请求失败，但没有捕获到具体异常"
        )

    @traceable(name="Agent.run", run_type="chain", tags=["agentic-rag"])
    @_measure_agent_stage("Agent 总耗时")
    def run(
        self,
        session_id,
        question
    ):
        """
        执行一次完整 Agent 任务。

        保护规则：
        1. 最多进行 5 次模型步骤；
        2. 最多真正执行 5 次工具；
        3. 完全相同的工具 + 参数不会重复执行；
        4. 达到工具上限后，最后一次模型调用不再提供工具，
           要求模型根据已有结果直接生成答案。
        """

        # =========================
        # 本轮初始化
        # =========================

        self.tool_executor.reset_called_tools()

        # MULTI_USER_ISOLATION_V1
        user_id = self.user_id
        user_role = getattr(self, "user_role", "user")

        self.tool_executor.set_context(
            user_id=user_id,
            role=user_role,
            session_id=session_id,
        )

        self.memory.add_message(
            session_id,
            "user",
            question,
            user_id=user_id,
        )

        # Keep the prompt bounded while retaining a durable, user-scoped
        # extractive summary of older turns.
        compact_session = getattr(self.memory, "compact_session", None)
        if callable(compact_session):
            compact_session(
                session_id=session_id,
                keep_recent=20,
                max_summary_chars=5000,
                user_id=user_id,
            )

        history = self.memory.get_messages(
            session_id,
            user_id=user_id,
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            *history
        ]

        # LONG_TERM_MEMORY_INJECTION_V2
        # 主 System Prompt 后插入长期记忆。
        # 它不占最近 20 条聊天历史的名额。
        long_term_memory_message = (
            _build_long_term_memory_message(
                user_id=user_id,
                limit=10,
            )
        )

        if long_term_memory_message is not None:
            messages.insert(
                1,
                long_term_memory_message,
            )
        # 最多允许多少次模型决策
        max_model_steps = 5

        # 最多真正执行多少次工具
        max_tool_calls = 5

        tool_call_count = 0

        # 保存已经执行过的：
        # 工具名称 + 标准化后的参数
        seen_tool_calls = set()


        # =========================
        # Agent 主循环
        # =========================

        for step in range(
            1,
            max_model_steps + 1
        ):

            # 达到工具调用上限后，
            # 不再把 tools 提供给模型。
            allow_tools = (
                tool_call_count
                < max_tool_calls
            )

            print()
            print(
                "================================"
            )
            print(
                f"🤖 Agent 模型调用 "
                f"{step}/{max_model_steps}"
            )
            print(
                f"工具已执行："
                f"{tool_call_count}/{max_tool_calls}"
            )
            print(
                "允许继续调用工具：",
                "是" if allow_tools else "否"
            )
            print(
                "================================"
            )

            # =========================
            # 调用模型
            # =========================

            try:
                response = self._create_completion(
                    messages,
                    allow_tools=allow_tools,
                )

            except Exception:

                call_records = (
                    self.tool_executor
                    .get_call_records()
                )

                # 已经成功执行过工具，
                # 只是后续模型整理答案失败。
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
                        answer,
                        user_id=user_id,
                    )

                    print(
                        "⚠️ 模型调用失败，"
                        "已使用工具结果降级返回"
                    )

                    return answer

                # 第一次模型调用就失败，
                # 没有任何可降级结果。
                raise


            msg = (
                response
                .choices[0]
                .message
            )


            # =========================
            # 没有工具调用
            # 说明模型已经产生最终答案
            # =========================

            if not msg.tool_calls:

                answer = (
                    msg.content
                    or ""
                )

                self.memory.add_message(
                    session_id,
                    "assistant",
                    answer,
                    user_id=user_id,
                )

                print(
                    "✅ 模型未继续调用工具"
                )
                print(
                    "✅ Agent 生成最终答案"
                )

                return answer


            # =========================
            # 模型要求调用工具
            # =========================

            print(
                f"🔧 本轮模型请求调用 "
                f"{len(msg.tool_calls)} 个工具"
            )

            # assistant 的 tool_calls 消息
            # 必须先加入上下文
            messages.append(
                msg
            )


            for tool_call in msg.tool_calls:

                tool_name = (
                    tool_call
                    .function
                    .name
                )

                raw_arguments = (
                    tool_call
                    .function
                    .arguments
                    or "{}"
                )


                # =====================
                # 标准化参数
                # 用于判断重复调用
                # =====================

                try:
                    parsed_arguments = (
                        json.loads(
                            raw_arguments
                        )
                    )

                    normalized_arguments = (
                        json.dumps(
                            parsed_arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(
                                ",",
                                ":"
                            )
                        )
                    )

                except Exception:
                    normalized_arguments = (
                        raw_arguments.strip()
                    )


                call_signature = (
                    tool_name,
                    normalized_arguments
                )


                # =====================
                # 防止完全相同的工具
                # 被重复执行
                # =====================

                if (
                    call_signature
                    in seen_tool_calls
                ):

                    result = (
                        "检测到完全相同的工具调用，"
                        "为防止无意义循环，本次未重复执行。"
                        "请根据之前已经获得的工具结果继续回答。"
                    )

                    print(
                        "⛔ 阻止重复工具调用：",
                        tool_name
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call.id
                            ),
                            "content": result
                        }
                    )

                    continue


                # =====================
                # 工具次数达到上限
                # 本次工具不再执行
                # =====================

                if (
                    tool_call_count
                    >= max_tool_calls
                ):

                    result = (
                        "Agent 已达到本轮最大工具调用次数，"
                        "该工具没有执行。"
                        "请根据已经获得的结果直接生成最终答案。"
                    )

                    print(
                        "⛔ 工具调用达到上限：",
                        tool_name
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call.id
                            ),
                            "content": result
                        }
                    )

                    continue


                # =====================
                # 真正执行工具
                # =====================

                seen_tool_calls.add(
                    call_signature
                )

                tool_call_count += 1

                print(
                    f"▶️ 执行工具 "
                    f"{tool_call_count}/"
                    f"{max_tool_calls}："
                    f"{tool_name}"
                )

                result = (
                    self.tool_executor
                    .execute(
                        tool_call
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": result
                    }
                )


            print(
                "↩️ 工具结果已加入 messages，"
                "继续交给模型判断"
            )


        # =========================
        # 模型步骤达到上限
        # =========================

        call_records = (
            self.tool_executor
            .get_call_records()
        )

        if call_records:
            answer = (
                "Agent 已达到最大执行步骤，"
                "为防止无限循环已停止。\n\n"
                + self._build_tool_fallback(
                    question,
                    call_records
                )
            )
        else:
            answer = (
                "Agent 已达到最大执行步骤，"
                "为防止无限循环已停止本次任务。"
            )

        self.memory.add_message(
            session_id,
            "assistant",
            answer,
            user_id=user_id,
        )

        print(
            "⛔ Agent 达到最大模型步骤，"
            "已停止"
        )

        return answer
