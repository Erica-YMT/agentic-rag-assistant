import json

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

    def execute(self, tool_call):
        tool_name = tool_call.function.name
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