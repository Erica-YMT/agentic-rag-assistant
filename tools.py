import ast
import operator

from pathlib import Path

from client import config
from knowledge_base import KnowledgeBase




# =========================
# 项目根目录
# =========================


PROJECT_ROOT = Path(
    __file__
).resolve().parent




# =========================
# 读取知识库配置
# =========================


embedding_config = config.get(
    "embedding",
    {}
)


model_path_value = embedding_config.get(
    "model_path"
)


index_path_value = embedding_config.get(
    "index_path",
    "faiss_index"
)


top_k = int(
    embedding_config.get(
        "top_k",
        3
    )
)




# =========================
# 检查模型路径配置
# =========================


if not model_path_value:

    raise ValueError(
        "config.toml 中缺少 "
        "[embedding].model_path"
    )




# =========================
# 处理模型路径
# =========================


model_path = Path(
    model_path_value
)


# 如果填写的是相对路径，
# 就以项目根目录为基准

if not model_path.is_absolute():

    model_path = (
        PROJECT_ROOT
        /
        model_path
    )




# =========================
# 处理FAISS索引路径
# =========================


index_path = Path(
    index_path_value
)


# 如果填写的是相对路径，
# 就以项目根目录为基准

if not index_path.is_absolute():

    index_path = (
        PROJECT_ROOT
        /
        index_path
    )




# =========================
# 检查路径是否存在
# =========================


if not model_path.exists():

    raise FileNotFoundError(
        "没有找到Embedding模型："
        f"{model_path}"
    )


if not index_path.exists():

    raise FileNotFoundError(
        "没有找到FAISS索引："
        f"{index_path}"
    )




# =========================
# 初始化知识库
# =========================


kb = KnowledgeBase(

    str(model_path),

    str(index_path)

)




# =========================
# 工具1：知识库搜索
# =========================


def search_knowledge(query):


    result = kb.search(

        query,

        k=top_k

    )


    return result




# =========================
# 安全计算规则
# =========================


BINARY_OPERATORS = {

    ast.Add:
    operator.add,

    ast.Sub:
    operator.sub,

    ast.Mult:
    operator.mul,

    ast.Div:
    operator.truediv,

    ast.FloorDiv:
    operator.floordiv,

    ast.Mod:
    operator.mod,

    ast.Pow:
    operator.pow

}


UNARY_OPERATORS = {

    ast.UAdd:
    operator.pos,

    ast.USub:
    operator.neg

}




# =========================
# 安全解析数学表达式
# =========================


def evaluate_expression(node):


    # 普通数字

    if isinstance(
        node,
        ast.Constant
    ):

        # bool 是 int 的子类，
        # 需要单独拒绝 True 和 False

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


        operand_value = evaluate_expression(
            node.operand
        )


        return operator_function(
            operand_value
        )


    raise ValueError(
        "表达式包含不允许的内容"
    )




# =========================
# 工具2：安全计算器
# =========================


def calculator(expression):


    try:

        if not isinstance(
            expression,
            str
        ):

            raise TypeError(
                "表达式必须是字符串"
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

        return "计算错误:不能除以零"


    except (
        SyntaxError,
        TypeError,
        ValueError,
        OverflowError

    ) as e:

        return f"计算错误:{e}"




# =========================
# 给Agent调用的工具映射
# =========================


available_tools = {


    "search_knowledge":
    search_knowledge,


    "calculator":
    calculator


}