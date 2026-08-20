from app.core.stream_events import event_print as print
import time
import json

from app.agent.tool_policy import ToolPolicy
from app.core.observability import (
    record_tool_call,
    record_tool_governance,
    record_tool_result,
)
from app.db.tool_audit import tool_audit_store
from app.agent.tool_reliability import (
    ToolReliabilityController,
    tool_reliability_controller,
)
from langsmith import get_current_run_tree, traceable


# TOOL_TIMING_V1_START
def _tool_result_status(result):
    """根据统一返回格式，将工具结果分成 success / error。"""

    value = str(result)

    error_prefixes = (
        "工具参数不是有效的JSON：",
        "工具参数格式错误：",
        "不存在工具：",
        "工具调用被策略拒绝：",
    )

    if value.startswith(error_prefixes):
        return "error"

    if value.startswith("工具 ") and any(
        marker in value
        for marker in (
            " 执行失败：",
            " 执行超时：",
            " 执行被限流：",
            " 执行被熔断：",
            " 执行排队失败：",
        )
    ):
        return "error"

    return "success"


def _measure_tool_execute(func):
    """统计每次 ToolExecutor 调用的总耗时。"""

    def wrapper(self, tool_call, *args, **kwargs):
        function = getattr(tool_call, "function", None)
        tool_name = str(getattr(function, "name", "unknown"))
        start_time = time.perf_counter()
        status = "error"

        try:
            result = func(self, tool_call, *args, **kwargs)
            status = _tool_result_status(result)
            return result
        finally:
            elapsed = time.perf_counter() - start_time

            record_tool_call(tool_name, elapsed)
            record_tool_result(tool_name, status)

            print(
                "[Timing] "
                f"工具 {tool_name}："
                f"{elapsed:.3f} 秒"
            )

    return wrapper
# TOOL_TIMING_V1_END


class ToolExecutor:
    """负责治理、解析、执行并审计模型发起的工具调用。"""

    def __init__(
        self,
        available_tools,
        *,
        policy=None,
        audit_store=None,
        reliability=None,
    ):
        self.available_tools = available_tools
        self.policy = policy or ToolPolicy()
        self.audit_store = audit_store or tool_audit_store
        self.reliability = reliability or tool_reliability_controller

        # 当前执行上下文，由 Agent.run() 每轮刷新。
        self.user_id = None
        self.user_role = "user"
        self.session_id = None

        # 只记录真正进入执行阶段的工具名称。
        self.called_tools = []
        # 记录工具名称、参数、治理决定和返回结果。
        self.call_records = []

    def set_context(
        self,
        *,
        user_id=None,
        role="user",
        session_id=None,
    ):
        self.user_id = int(user_id) if user_id is not None else None
        self.user_role = str(role or "user").strip().lower() or "user"
        self.session_id = str(session_id) if session_id else None

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

    def _append_record(
        self,
        *,
        tool_name,
        arguments,
        result,
        decision,
        risk_level,
        reason,
    ):
        self.call_records.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": str(result),
                "decision": decision,
                "risk_level": risk_level,
                "policy_reason": reason,
            }
        )

    def _start_audit(
        self,
        *,
        tool_name,
        arguments,
        risk_level,
        decision,
        reason,
        status="pending",
    ):
        return self.audit_store.start(
            user_id=self.user_id,
            session_id=self.session_id,
            tool_name=tool_name,
            role=self.user_role,
            risk_level=risk_level,
            decision=decision,
            reason=reason,
            arguments=arguments,
            status=status,
        )

    # ==========================
    # 执行工具
    # ==========================

    @traceable(name="Tool.execute", run_type="tool", tags=["tool"])
    @_measure_tool_execute
    def execute(self, tool_call):
        tool_name = str(tool_call.function.name)

        # LANGSMITH_TOOL_METADATA_V1_START
        try:
            current_run = get_current_run_tree()

            if current_run is not None:
                current_run.metadata["tool_name"] = tool_name
                current_run.tags.append(f"tool:{tool_name}")

        except Exception:
            pass
        # LANGSMITH_TOOL_METADATA_V1_END

        raw_arguments = tool_call.function.arguments or "{}"

        # ==========================
        # 1. 解析参数
        # ==========================

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            result = f"工具参数不是有效的JSON：{exc}"
            arguments = {"_raw": str(raw_arguments)[:4000]}

            self._start_audit(
                tool_name=tool_name,
                arguments=arguments,
                risk_level="unknown",
                decision="deny",
                reason="工具参数不是有效 JSON。",
                status="error",
            )
            record_tool_governance("unknown", "deny")
            self._append_record(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                decision="deny",
                risk_level="unknown",
                reason="工具参数不是有效 JSON。",
            )
            return result

        if not isinstance(arguments, dict):
            result = "工具参数格式错误：参数必须是JSON对象。"
            audit_arguments = {"_value": str(arguments)[:4000]}

            self._start_audit(
                tool_name=tool_name,
                arguments=audit_arguments,
                risk_level="unknown",
                decision="deny",
                reason="工具参数必须是 JSON 对象。",
                status="error",
            )
            record_tool_governance("unknown", "deny")
            self._append_record(
                tool_name=tool_name,
                arguments=audit_arguments,
                result=result,
                decision="deny",
                risk_level="unknown",
                reason="工具参数必须是 JSON 对象。",
            )
            return result

        # ==========================
        # 2. Tool Policy
        # ==========================

        decision = self.policy.check(
            tool_name=tool_name,
            arguments=arguments,
            role=self.user_role,
        )

        metric_tool_name = (
            tool_name
            if tool_name in self.available_tools
            else "unknown"
        )
        metric_decision = "allow" if decision.allowed else "deny"
        record_tool_governance(metric_tool_name, metric_decision)

        audit_id = self._start_audit(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=decision.risk_level,
            decision=metric_decision,
            reason=decision.reason,
            status="pending" if decision.allowed else "blocked",
        )

        if not decision.allowed:
            result = f"工具调用被策略拒绝：{decision.reason}"
            self._append_record(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                decision="deny",
                risk_level=decision.risk_level,
                reason=decision.reason,
            )
            print("⛔ Tool Policy 拒绝：", tool_name, decision.reason)
            return result

        # ==========================
        # 3. 查找工具
        # ==========================

        tool = self.available_tools.get(tool_name)

        if tool is None:
            result = f"不存在工具：{tool_name}"
            self.audit_store.finish(
                audit_id,
                status="error",
                result=result,
                elapsed_seconds=0.0,
            )
            self._append_record(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                decision="allow",
                risk_level=decision.risk_level,
                reason=decision.reason,
            )
            return result

        # ==========================
        # 4. 打印工具日志
        # ==========================

        print("\n======================")
        print("🤖 Agent调用工具")
        print("工具名称:", tool_name)
        print("风险等级:", decision.risk_level)
        print("参数:", arguments)
        print("======================")

        # ==========================
        # 5. 真正执行工具
        # ==========================

        self.called_tools.append(tool_name)
        execution_start = time.perf_counter()

        execution_arguments = dict(arguments)

        # search_knowledge 的用户作用域只能来自已认证 Session，
        # 不能让模型自己伪造 user_id。审计仍记录模型原始参数。
        if tool_name == "search_knowledge":
            execution_arguments["_user_id"] = self.user_id

        reliability_result = self.reliability.execute(
            tool_name=tool_name,
            arguments=execution_arguments,
            function=tool,
        )
        elapsed_seconds = time.perf_counter() - execution_start
        result = str(reliability_result.value)
        status = reliability_result.status

        self.audit_store.finish(
            audit_id,
            status=status,
            result=result,
            elapsed_seconds=elapsed_seconds,
        )

        print("工具返回:", result[:300])

        # ==========================
        # 6. 保存完整调用记录
        # ==========================

        self._append_record(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            decision="allow",
            risk_level=decision.risk_level,
            reason=decision.reason,
        )
        return result
