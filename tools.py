import ast
import operator

from knowledge_base import search_knowledge
from web_search import search_web


# =========================
# 安全计算规则
# =========================

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# =========================
# 安全解析数学表达式
# =========================

def evaluate_expression(
    node
):
    # 普通数字
    if isinstance(
        node,
        ast.Constant
    ):
        # bool 是 int 的子类，
        # 因此需要单独拒绝。
        if isinstance(
            node.value,
            bool
        ):
            raise ValueError(
                "只允许使用数字"
            )

        if isinstance(
            node.value,
            (int, float)
        ):
            return node.value

        raise ValueError(
            "只允许使用数字"
        )

    # 二元运算，例如 1 + 2
    if isinstance(
        node,
        ast.BinOp
    ):
        operator_function = (
            BINARY_OPERATORS.get(
                type(node.op)
            )
        )

        if operator_function is None:
            raise ValueError(
                "包含不支持的运算符"
            )

        left_value = evaluate_expression(
            node.left
        )

        right_value = evaluate_expression(
            node.right
        )

        # 防止指数过大
        if isinstance(
            node.op,
            ast.Pow
        ):
            if abs(right_value) > 100:
                raise ValueError(
                    "指数过大"
                )

        return operator_function(
            left_value,
            right_value
        )

    # 一元运算，例如 -5
    if isinstance(
        node,
        ast.UnaryOp
    ):
        operator_function = (
            UNARY_OPERATORS.get(
                type(node.op)
            )
        )

        if operator_function is None:
            raise ValueError(
                "包含不支持的一元运算符"
            )

        operand_value = (
            evaluate_expression(
                node.operand
            )
        )

        return operator_function(
            operand_value
        )

    raise ValueError(
        "表达式包含不允许的内容"
    )


# =========================
# 工具：安全计算器
# =========================

def calculator(
    expression
):
    try:
        if not isinstance(
            expression,
            str
        ):
            raise TypeError(
                "表达式必须是字符串"
            )

        expression = (
            expression.strip()
        )

        if not expression:
            raise ValueError(
                "表达式不能为空"
            )

        if len(expression) > 200:
            raise ValueError(
                "表达式过长"
            )

        expression_tree = ast.parse(
            expression,
            mode="eval"
        )

        result = evaluate_expression(
            expression_tree.body
        )

        return str(
            result
        )

    except ZeroDivisionError:
        return (
            "计算错误：不能除以零"
        )

    except (
        SyntaxError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        return (
            f"计算错误：{exc}"
        )


# =========================
# Agent 可执行工具注册表
# =========================

available_tools = {
    "search_knowledge": (
        search_knowledge
    ),
    "calculator": calculator,
    "search_web": search_web,
}
