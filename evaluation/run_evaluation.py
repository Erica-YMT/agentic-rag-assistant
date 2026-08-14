import json
import sys # 操控pyhton运行环境
import time # 处理时间，延迟
from collections import Counter # 快速统计
from datetime import datetime # 处理带年月日时分秒的标准时间
from pathlib import Path # 文件路径工具

# ==========================
# 将项目根目录加入导入路径
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from app.agent.agent import Agent


def load_test_cases():
    """读取测试用例。"""

    test_file = (
        Path(__file__).resolve().parent
        / "test_cases.json"
    )

    with open(
        test_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def tools_match(
    expected_tools,
    actual_tools
):
    """
    比较预期工具和实际工具。

    不要求调用顺序一致，
    但要求工具名称和调用次数一致。
    """

    return (
        Counter(expected_tools)
        ==
        Counter(actual_tools)
    )


def keywords_match(
    expected_keywords,
    text
):
    """检查文本是否包含全部预期关键词。"""

    normalized_text = (
        text or ""
    ).lower()

    missing_keywords = []

    for keyword in expected_keywords:

        normalized_keyword = str(
            keyword
        ).lower()

        if normalized_keyword not in normalized_text:

            missing_keywords.append(
                keyword
            )

    passed = (
        len(missing_keywords) == 0
    )

    return passed, missing_keywords


def retrieval_keywords_match(
    expected_keywords,
    call_records
):
    """
    检查 search_knowledge 工具返回的原始内容。

    只读取 search_knowledge 的 result，
    不检查其他工具的结果。
    """

    retrieval_results = []

    for record in call_records:

        if (
            record.get("tool_name")
            ==
            "search_knowledge"
        ):

            retrieval_results.append(
                str(
                    record.get(
                        "result",
                        ""
                    )
                )
            )

    retrieval_text = "\n\n".join(
        retrieval_results
    )

    (
        passed,
        missing_keywords

    ) = keywords_match(

        expected_keywords,
        retrieval_text

    )

    return (
        passed,
        missing_keywords,
        retrieval_text
    )


def run_case_with_retry(
    question,
    session_id,
    max_attempts=1,
    retry_delay=5
):
    """
    执行一个测试案例。

    Agent 内部已经负责模型请求重试，
    因此这里默认不重复执行整个案例。
    """

    last_error = None
    last_actual_tools = []
    last_call_records = []

    for attempt in range(
        1,
        max_attempts + 1
    ):

        test_agent = Agent()

        attempt_session_id = (
            f"{session_id}_attempt_{attempt}"
        )

        try:

            answer = test_agent.run(
                attempt_session_id,
                question
            )

            actual_tools = (
                test_agent.get_called_tools()
            )

            call_records = (
                test_agent.get_call_records()
            )

            return (
                answer,
                actual_tools,
                call_records,
                None,
                attempt
            )

        except Exception as exc:

            last_error = str(exc)

            # 即使最终回答阶段失败，也尽量保留
            # 已经执行过的工具及其返回结果。
            last_actual_tools = (
                test_agent.get_called_tools()
            )

            last_call_records = (
                test_agent.get_call_records()
            )

            print(
                f"第 {attempt} 次执行失败："
                f"{last_error}"
            )

            if attempt < max_attempts:

                print(
                    f"{retry_delay} 秒后重试..."
                )

                time.sleep(
                    retry_delay
                )

    return (
        "",
        last_actual_tools,
        last_call_records,
        last_error,
        max_attempts
    )


def save_evaluation_report(
    case_results,
    summary
):
    """将本次评估结果保存为 JSON 文件。"""

    results_directory = (
        Path(__file__).resolve().parent
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    result_file = (
        results_directory
        / f"evaluation_{timestamp}.json"
    )

    report = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "summary": summary,
        "cases": case_results
    }

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    return result_file


def status_text(status):
    """将内部状态转换为 Markdown 展示文本。"""

    mapping = {
        "passed": "✅ 通过",
        "failed": "❌ 未通过",
        "error": "⚠️ 系统错误"
    }

    return mapping.get(status, str(status))


def format_tools(tool_names):
    """格式化工具名称列表。"""

    if not tool_names:
        return "无"

    return ", ".join(
        str(tool_name)
        for tool_name in tool_names
    )


def escape_markdown_table(text):
    """转义 Markdown 表格中的特殊字符。"""

    return (
        str(text)
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def check_text(value):
    """将布尔检查结果转换为可读文本。"""

    if value is True:
        return "✅ 通过"

    if value is False:
        return "❌ 未通过"

    return "不适用"


def save_markdown_report(
    case_results,
    summary
):
    """生成并保存最新的 Markdown 评估报告。"""

    results_directory = (
        Path(__file__).resolve().parent
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        results_directory
        / "latest_report.md"
    )

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

    accuracy = (
        float(summary.get("accuracy", 0))
        * 100
    )

    execution_rate = (
        float(summary.get("execution_rate", 0))
        * 100
    )

    lines = [
        "# Agent 自动评估报告",
        "",
        f"> 生成时间：{created_at}",
        "",
        "## 评估汇总",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| 测试数量 | "
            f"{summary.get('total_count', 0)} |"
        ),
        (
            "| 正常完成 | "
            f"{summary.get('completed_count', 0)} |"
        ),
        (
            "| 自动通过 | "
            f"{summary.get('passed_count', 0)} |"
        ),
        (
            "| 自动未通过 | "
            f"{summary.get('failed_count', 0)} |"
        ),
        (
            "| 系统错误 | "
            f"{summary.get('error_count', 0)} |"
        ),
        (
            "| 需要人工复核 | "
            f"{summary.get('manual_review_count', 0)} |"
        ),
        f"| 模型评估通过率 | {accuracy:.1f}% |",
        f"| 测试执行成功率 | {execution_rate:.1f}% |",
        "",
        "## 测试案例",
        "",
        (
            "| 案例 | 状态 | 预期工具 | 实际工具 | "
            "工具检查 | 回答检查 | 检索检查 |"
        ),
        "|---|---|---|---|---|---|---|"
    ]

    for case in case_results:

        lines.append(
            "| "
            f"{escape_markdown_table(case.get('name', ''))} | "
            f"{status_text(case.get('status', ''))} | "
            f"{escape_markdown_table(format_tools(case.get('expected_tools', [])))} | "
            f"{escape_markdown_table(format_tools(case.get('actual_tools', [])))} | "
            f"{check_text(case.get('tool_passed'))} | "
            f"{check_text(case.get('keyword_passed'))} | "
            f"{check_text(case.get('retrieval_passed'))} |"
        )

    lines.extend(
        [
            "",
            "## 案例详情",
            ""
        ]
    )

    for index, case in enumerate(
        case_results,
        start=1
    ):

        lines.extend(
            [
                (
                    f"### {index}. "
                    f"{case.get('name', f'案例 {index}')}"
                ),
                "",
                (
                    "- **状态：** "
                    f"{status_text(case.get('status', ''))}"
                ),
                (
                    "- **问题：** "
                    f"{case.get('question', '')}"
                ),
                (
                    "- **预期工具：** "
                    f"{format_tools(case.get('expected_tools', []))}"
                ),
                (
                    "- **实际工具：** "
                    f"{format_tools(case.get('actual_tools', []))}"
                ),
                (
                    "- **工具检查：** "
                    f"{check_text(case.get('tool_passed'))}"
                ),
                (
                    "- **回答关键词检查：** "
                    f"{check_text(case.get('keyword_passed'))}"
                ),
                (
                    "- **检索内容检查：** "
                    f"{check_text(case.get('retrieval_passed'))}"
                ),
                (
                    "- **执行次数：** "
                    f"{case.get('attempts_used', 0)}"
                ),
                (
                    "- **人工复核：** "
                    f"{'需要' if case.get('manual_review') else '不需要'}"
                )
            ]
        )

        error = case.get("error")

        if error:
            lines.append(
                f"- **错误信息：** {error}"
            )

        retrieval_text = case.get(
            "retrieval_text",
            ""
        )

        if retrieval_text:
            lines.extend(
                [
                    "",
                    "**检索内容摘要：**",
                    "",
                    "```text",
                    retrieval_text[:1000],
                    "```"
                ]
            )

        answer = case.get(
            "answer",
            ""
        )

        if answer:
            lines.extend(
                [
                    "",
                    "**模型回答：**",
                    "",
                    answer
                ]
            )

        lines.append("")

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(lines)
        )

    return output_file


def run_evaluation():
    """运行全部 Agent 评估案例。"""

    test_cases = load_test_cases()

    passed_count = 0
    failed_count = 0
    error_count = 0
    manual_review_count = 0

    case_results = []

    print("==========================")
    print("Agent评估开始")
    print("==========================")

    for index, case in enumerate(
        test_cases,
        start=1
    ):

        name = case["name"]
        question = case["question"]

        expected_tools = case.get(
            "expected_tools",
            []
        )

        expected_keywords = case.get(
            "expected_keywords",
            []
        )

        expected_retrieval_keywords = case.get(
            "expected_retrieval_keywords",
            []
        )

        manual_review = case.get(
            "manual_review",
            False
        )

        session_id = (
            f"evaluation_{index}"
        )

        print(f"\n[{index}] {name}")
        print("问题:", question)
        print("预期工具:", expected_tools)
        print("预期关键词:", expected_keywords)

        print(
            "预期检索关键词:",
            expected_retrieval_keywords
        )

        (
            answer,
            actual_tools,
            call_records,
            execution_error,
            attempts_used

        ) = run_case_with_retry(

            question=question,
            session_id=session_id,
            max_attempts=1,
            retry_delay=5

        )

        print(
            "执行次数:",
            attempts_used
        )

        # ==========================
        # 初始化当前案例报告字段
        # ==========================

        case_status = "error"

        tool_passed = None
        keyword_passed = None
        retrieval_passed = None

        missing_keywords = []
        missing_retrieval_keywords = []

        retrieval_text = ""
        passed = False

        # ==========================
        # 系统错误
        # ==========================

        if execution_error is not None:

            case_status = "error"
            error_count += 1

            print(
                "实际工具:",
                actual_tools
            )

            print(
                "工具检查: ⚠️ 未执行完成"
            )

            print(
                "关键词检查: ⚠️ 未执行完成"
            )

            print(
                "检索内容检查: ⚠️ 未执行完成"
            )

            print(
                "模型回答: 未生成"
            )

            print(
                "执行状态: ⚠️ 系统错误"
            )

            print(
                "错误信息:",
                execution_error
            )

        # ==========================
        # 正常执行
        # ==========================

        else:

            tool_passed = tools_match(
                expected_tools,
                actual_tools
            )

            (
                keyword_passed,
                missing_keywords

            ) = keywords_match(

                expected_keywords,
                answer

            )

            (
                retrieval_passed,
                missing_retrieval_keywords,
                retrieval_text

            ) = retrieval_keywords_match(

                expected_retrieval_keywords,
                call_records

            )

            passed = (
                tool_passed
                and keyword_passed
                and retrieval_passed
            )

            print(
                "实际工具:",
                actual_tools
            )

            print(
                "工具检查:",
                "✅ 通过"
                if tool_passed
                else "❌ 未通过"
            )

            print(
                "关键词检查:",
                "✅ 通过"
                if keyword_passed
                else "❌ 未通过"
            )

            if missing_keywords:

                print(
                    "缺少回答关键词:",
                    missing_keywords
                )

            if expected_retrieval_keywords:

                print(
                    "检索内容检查:",
                    "✅ 通过"
                    if retrieval_passed
                    else "❌ 未通过"
                )

                if missing_retrieval_keywords:

                    print(
                        "缺少检索关键词:",
                        missing_retrieval_keywords
                    )

                print(
                    "检索内容摘要:",
                    retrieval_text[:300]
                )

            else:

                print(
                    "检索内容检查: 不适用"
                )

            print(
                "模型回答:",
                answer
            )

            if passed:

                case_status = "passed"
                passed_count += 1

                print(
                    "自动评估: ✅ 通过"
                )

            else:

                case_status = "failed"
                failed_count += 1

                print(
                    "自动评估: ❌ 未通过"
                )

        # ==========================
        # 人工复核
        # ==========================

        if manual_review:

            manual_review_count += 1

            print(
                "人工复核: ⚠️ 需要"
            )

        else:

            print(
                "人工复核: 不需要"
            )

        # ==========================
        # 保存当前案例结果
        # ==========================

        case_results.append(
            {
                "name": name,
                "question": question,
                "status": case_status,
                "attempts_used": attempts_used,

                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "tool_passed": tool_passed,

                "expected_keywords": expected_keywords,
                "keyword_passed": keyword_passed,
                "missing_keywords": missing_keywords,

                "expected_retrieval_keywords": (
                    expected_retrieval_keywords
                ),
                "retrieval_passed": retrieval_passed,
                "missing_retrieval_keywords": (
                    missing_retrieval_keywords
                ),

                "answer": answer,
                "retrieval_text": retrieval_text,
                "call_records": call_records,

                "manual_review": manual_review,
                "error": execution_error
            }
        )

    # ==========================
    # 汇总
    # ==========================

    total_count = len(test_cases)

    completed_count = (
        passed_count
        +
        failed_count
    )

    accuracy = (
        passed_count / completed_count
        if completed_count
        else 0
    )

    execution_rate = (
        completed_count / total_count
        if total_count
        else 0
    )

    print("\n==========================")
    print("评估完成")
    print("==========================")

    print("测试数量:", total_count)
    print("正常完成:", completed_count)
    print("自动通过:", passed_count)
    print("自动未通过:", failed_count)
    print("系统错误:", error_count)

    print(
        "需要人工复核:",
        manual_review_count
    )

    print(
        "模型评估通过率:",
        f"{accuracy:.1%}"
    )

    print(
        "测试执行成功率:",
        f"{execution_rate:.1%}"
    )

    # ==========================
    # 保存评估报告
    # ==========================

    summary = {
        "total_count": total_count,
        "completed_count": completed_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "manual_review_count": manual_review_count,
        "accuracy": round(
            accuracy,
            4
        ),
        "execution_rate": round(
            execution_rate,
            4
        )
    }

    result_file = save_evaluation_report(
        case_results,
        summary
    )

    markdown_file = save_markdown_report(
        case_results,
        summary
    )

    print(
        "JSON评估报告:",
        result_file
    )

    print(
        "Markdown评估报告:",
        markdown_file
    )


if __name__ == "__main__":

    run_evaluation()