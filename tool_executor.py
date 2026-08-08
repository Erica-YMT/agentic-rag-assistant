import time
import json
from observability import record_tool_call, record_tool_result
from langsmith import get_current_run_tree, traceable


# TOOL_TIMING_V1_START
def _tool_result_status(
    result,
):
    """
    根据 ToolExecutor 当前统一错误返回格式，
    将工具结果分成 success / error。
    """

    value = str(result)

    if value.startswith(
        "工具参数不是有效的JSON："
    ):
        return "error"

    if value.startswith(
        "工具参数格式错误："
    ):
        return "error"

    if value.startswith(
        "不存在工具："
    ):
        return "error"

    if (
        value.startswith("工具 ")
        and " 执行失败：" in value
    ):
        return "error"

    return "success"


def _measure_tool_execute(func):
    """
    统计每次真实工具执行的总耗时。
    """

    def wrapper(
        self,
        tool_call,
        *args,
        **kwargs
    ):

        function = getattr(
            tool_call,
            "function",
            None,
        )

        tool_name = str(
            getattr(
                function,
                "name",
                "unknown",
            )
        )

        start_time = (
            time.perf_counter()
        )

        status = "error"

        try:

            result = func(
                self,
                tool_call,
                *args,
                **kwargs
            )

            status = (
                _tool_result_status(
                    result
                )
            )

            return result

        finally:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            record_tool_call(
                tool_name,
                elapsed,
            )

            record_tool_result(
                tool_name,
                status,
            )

            print(
                "[Timing] "
                f"工具 {tool_name}："
                f"{elapsed:.3f} 秒"
            )

    return wrapper
# TOOL_TIMING_V1_END


class ToolExecutor:
    """负责解析、执行并记录模型发起的工具调用。"""
    def __init__(self, available_tools):
        self.available_tools = available_tools
        # 只记录工具名称
        self.called_tools = []
        # 记录工具名称、参数和返回结果
        self.call_records = []

    # ==========================
    # 清空本轮工具记录
    # ==========================

    def reset_called_tools(self):
        self.called_tools = []
        self.call_records = []

    # ==========================
    # 获取工具名称
    # ==========================

    def get_called_tools(self):
        return self.called_tools.copy()

    # ==========================
    # 获取完整工具调用记录
    # ==========================

    def get_call_records(self):
        return self.call_records.copy()

    # ==========================
    # 执行工具
    # ==========================

    @traceable(name="Tool.execute", run_type="tool", tags=["tool"])
    @_measure_tool_execute
    def execute(self, tool_call):
        tool_name = tool_call.function.name
        # LANGSMITH_TOOL_METADATA_V1_START
        # Trace 本身叫 Tool.execute，
        # metadata 中保存真正的工具名。
        # 即使 LangSmith SDK 本身异常，
        # 也绝不影响业务逻辑。
        try:
            current_run = (
                get_current_run_tree()
            )

            if current_run is not None:
                current_run.metadata[
                    "tool_name"
                ] = str(tool_name)

                current_run.tags.append(
                    f"tool:{tool_name}"
                )

        except Exception:
            pass
        # LANGSMITH_TOOL_METADATA_V1_END

        raw_arguments = (
            tool_call.function.arguments
            or "{}"
        )
        # 记录工具名称
        self.called_tools.append(tool_name)
        arguments = {}

        # ==========================
        # 1. 解析参数
        # ==========================

        try:
            arguments = json.loads(
                raw_arguments
            )
        except json.JSONDecodeError as exc:
            result = (
                "工具参数不是有效的JSON："
                f"{exc}"
            )
            self.call_records.append(
                {
                    "tool_name": tool_name,
                    "arguments": raw_arguments,
                    "result": result
                }
            )
            return result

        if not isinstance(arguments, dict):
            result = (
                "工具参数格式错误："
                "参数必须是JSON对象。"
            )
            self.call_records.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result
                }
            )
            return result

        # ==========================
        # 2. 查找工具
        # ==========================

        tool = self.available_tools.get(
            tool_name
        )

        if tool is None:
            result = f"不存在工具：{tool_name}"
            self.call_records.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result
                }
            )
            return result

        # ==========================
        # 3. 打印工具日志
        # ==========================

        print("\n======================")
        print("🤖 Agent调用工具")
        print("工具名称:", tool_name)
        print("参数:", arguments)
        print("======================")

        # ==========================
        # 4. 执行工具
        # ==========================

        try:
            result = tool(
                **arguments
            )
        except Exception as exc:
            result = (
                f"工具 {tool_name} 执行失败："
                f"{exc}"
            )
        result = str(result)
        print(
            "工具返回:",
            result[:300]
        )

        # ==========================
        # 5. 保存完整调用记录
        # ==========================

        self.call_records.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result
            }
        )
        return result