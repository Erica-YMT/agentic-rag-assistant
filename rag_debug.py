from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from knowledge_base import (
    get_default_knowledge_base,
)


# =========================
# 延迟加载知识库
# =========================

_knowledge_base = None


def get_knowledge_base():
    """
    第一次进行检索时加载知识库，
    后续检索复用同一个实例。
    """

    global _knowledge_base

    if _knowledge_base is None:
        _knowledge_base = (
            get_default_knowledge_base()
        )

    return _knowledge_base


# =========================
# 执行检索调试
# =========================

def inspect_retrieval(
    query: str,
    top_k: int,
    threshold: float,
) -> tuple[pd.DataFrame, str]:
    query = str(
        query or ""
    ).strip()

    if not query:
        return (
            pd.DataFrame(),
            "请输入需要测试的问题。",
        )

    try:
        top_k = int(
            top_k
        )
    except (
        TypeError,
        ValueError,
    ):
        return (
            pd.DataFrame(),
            "Top-K 必须是整数。",
        )

    if top_k <= 0:
        return (
            pd.DataFrame(),
            "Top-K 必须大于 0。",
        )

    try:
        threshold = float(
            threshold
        )
    except (
        TypeError,
        ValueError,
    ):
        return (
            pd.DataFrame(),
            "距离阈值必须是数字。",
        )

    knowledge_base = (
        get_knowledge_base()
    )

    results = (
        knowledge_base
        .db
        .similarity_search_with_score(
            query,
            k=top_k,
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):
        distance = float(
            distance
        )

        metadata = (
            document.metadata
            or {}
        )

        source = (
            metadata.get("file_name")
            or metadata.get(
                "relative_path"
            )
            or metadata.get("source")
            or "未知来源"
        )

        page = metadata.get(
            "page"
        )

        if isinstance(page, int):
            page_text = str(
                page + 1
            )
        else:
            page_text = ""

        passed = (
            distance
            <= threshold
        )

        content = (
            document
            .page_content
            .strip()
        )

        rows.append(
            {
                "排名": rank,
                "来源": Path(
                    str(source)
                ).name,
                "页码": page_text,
                "FAISS距离": round(
                    distance,
                    4,
                ),
                "是否通过阈值": (
                    "是"
                    if passed
                    else "否"
                ),
                "文本预览": (
                    content[:300]
                ),
            }
        )

    dataframe = (
        pd.DataFrame(
            rows
        )
    )

    passed_count = sum(
        row["是否通过阈值"]
        == "是"
        for row in rows
    )

    summary = (
        f"共召回 {len(rows)} 个文本块，"
        f"其中 {passed_count} 个通过阈值 "
        f"{threshold:.2f}。"
    )

    if rows:
        best_distance = (
            rows[0]["FAISS距离"]
        )

        summary += (
            f" 最佳距离为 "
            f"{best_distance}。"
        )

    if passed_count == 0:
        summary += (
            " 当前问题会被知识库判定为"
            "“没有足够相关资料”。"
        )

    return (
        dataframe,
        summary,
    )


# =========================
# Gradio 调试页面
# =========================

with gr.Blocks(
    title="RAG Retrieval Debugger",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# RAG 检索调试面板

输入一个问题，查看 FAISS 实际召回的文本、来源和距离分数。

- 距离越小，通常越相关
- 距离不超过阈值，才会交给 Agent
- 可用于调整 `top_k` 和 `score_threshold`
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="测试问题",
            placeholder=(
                "例如：这个 Agent 项目的"
                "主要模块有哪些？"
            ),
            lines=2,
            scale=4,
        )

        top_k = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Top-K",
            scale=1,
        )

        threshold = gr.Slider(
            minimum=0.3,
            maximum=1.5,
            value=1.0,
            step=0.05,
            label="距离阈值",
            scale=1,
        )

    inspect_button = gr.Button(
        "开始检索分析",
        variant="primary",
    )

    summary = gr.Markdown()

    result_table = gr.Dataframe(
        headers=[
            "排名",
            "来源",
            "页码",
            "FAISS距离",
            "是否通过阈值",
            "文本预览",
        ],
        interactive=False,
        wrap=True,
    )

    gr.Examples(
        examples=[
            [
                "这个 Agent 项目的主要模块有哪些？",
                5,
                1.0,
            ],
            [
                "build_index.py 的作用是什么？",
                5,
                1.0,
            ],
            [
                "这个项目明年的商业收入目标是多少？",
                5,
                1.0,
            ],
            [
                "唐朝哪位皇帝最喜欢吃西瓜？",
                5,
                1.0,
            ],
        ],
        inputs=[
            query_box,
            top_k,
            threshold,
        ],
    )

    inspect_button.click(
        fn=inspect_retrieval,
        inputs=[
            query_box,
            top_k,
            threshold,
        ],
        outputs=[
            result_table,
            summary,
        ],
    )

    query_box.submit(
        fn=inspect_retrieval,
        inputs=[
            query_box,
            top_k,
            threshold,
        ],
        outputs=[
            result_table,
            summary,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        show_error=True,
    )
