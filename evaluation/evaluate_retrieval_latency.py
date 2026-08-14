from __future__ import annotations

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
import math
import statistics
import time
from pathlib import Path


from rag.knowledge_base import (
    get_default_knowledge_base,
)


# =========================================================
# 基础路径
# =========================================================


CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_cases.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "retrieval_latency_report.json"
)


# 最终返回多少条结果
TOP_K = 3

# 每个问题重复几次
# 10 个问题 × 3 次 = 每种方案 30 个样本
REPEATS = 3


# =========================================================
# 加载评测问题
# =========================================================

def load_cases():
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            "没有找到评测集："
            f"{CASES_PATH}"
        )

    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "retrieval_cases.json "
            "最外层必须是列表"
        )

    cases = []

    for index, item in enumerate(
        data,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        # 兼容 question / query 两种字段名
        question = str(
            item.get("question")
            or item.get("query")
            or ""
        ).strip()

        if not question:
            continue

        cases.append(
            {
                "index": index,
                "question": question,
            }
        )

    if not cases:
        raise ValueError(
            "评测集中没有有效问题"
        )

    return cases


# =========================================================
# P95
# =========================================================

def percentile_95(
    values,
):
    """
    计算 P95。

    例如：
    100 次请求中，
    95% 的请求耗时不会超过这个值。
    """

    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    position = math.ceil(
        0.95 * len(ordered)
    )

    index = max(
        0,
        position - 1,
    )

    return ordered[index]


# =========================================================
# 单次计时
# =========================================================

def measure(
    func,
):
    start = time.perf_counter()

    func()

    elapsed = (
        time.perf_counter()
        - start
    )

    # 转成毫秒
    return (
        elapsed * 1000.0
    )


# =========================================================
# 主程序
# =========================================================

def main():

    cases = load_cases()

    print(
        f"✅ 加载评测问题："
        f"{len(cases)} 条"
    )

    print(
        f"✅ 每个问题重复："
        f"{REPEATS} 次"
    )

    print(
        f"✅ 每种方案样本数："
        f"{len(cases) * REPEATS}"
    )

    print()
    print(
        "正在加载知识库……"
    )


    # 模型、FAISS、BM25、Reranker
    # 只加载一次。
    kb = (
        get_default_knowledge_base()
    )


    # =====================================================
    # 检查当前对象
    # =====================================================

    if not hasattr(
        kb,
        "db",
    ):
        raise RuntimeError(
            "KnowledgeBase 没有 db，"
            "无法测试 FAISS"
        )

    if not hasattr(
        kb,
        "retriever",
    ):
        raise RuntimeError(
            "KnowledgeBase 没有 retriever，"
            "无法测试 Hybrid"
        )

    if not hasattr(
        kb,
        "reranker",
    ):
        raise RuntimeError(
            "KnowledgeBase 没有 reranker，"
            "无法测试 Reranker"
        )


    print(
        "✅ FAISS 可用"
    )

    print(
        "✅ Hybrid Retriever 可用"
    )

    print(
        "✅ Reranker 可用"
    )


    # =====================================================
    # 定义三种方案
    # =====================================================

    def faiss_only(
        question,
    ):
        """
        只运行：
        Embedding + FAISS
        """

        return (
            kb.db
            .similarity_search_with_score(
                question,
                k=TOP_K,
            )
        )


    def hybrid_only(
        question,
    ):
        """
        运行：
        BM25 + FAISS + RRF
        不经过 Reranker。
        """

        return (
            kb.retriever.search(
                question,
                k=TOP_K,
            )
        )


    def hybrid_reranker(
        question,
    ):
        """
        使用当前 KnowledgeBase.search()。

        当前正式流程：
        Hybrid Retrieval
        ↓
        Reranker
        ↓
        Top-K
        """

        return kb.search(
            question,
            k=TOP_K,
        )


    methods = {
        "FAISS Only": (
            faiss_only
        ),
        "Hybrid": (
            hybrid_only
        ),
        "Hybrid + Reranker": (
            hybrid_reranker
        ),
    }


    # =====================================================
    # Warm-up
    #
    # 第一次运行可能包含：
    # tokenizer / 模型缓存 / jieba 等初始化。
    #
    # 不把这些启动成本计算到正式延迟里。
    # =====================================================

    warmup_question = (
        cases[0]["question"]
    )

    print()
    print(
        "🔥 开始 Warm-up……"
    )


    for name, method in (
        methods.items()
    ):

        print(
            f"Warm-up：{name}"
        )

        method(
            warmup_question
        )


    print(
        "✅ Warm-up 完成"
    )


    # =====================================================
    # 正式测试
    # =====================================================

    samples = {
        name: []
        for name in methods
    }


    print()
    print(
        "=" * 70
    )

    print(
        "⏱️ 开始正式延迟评测"
    )

    print(
        "=" * 70
    )


    for case_number, case in enumerate(
        cases,
        start=1,
    ):

        question = (
            case["question"]
        )

        print()
        print(
            f"问题 {case_number}/"
            f"{len(cases)}："
            f"{question}"
        )


        for repeat in range(
            1,
            REPEATS + 1,
        ):

            for (
                name,
                method,
            ) in methods.items():

                elapsed_ms = measure(
                    lambda m=method: m(
                        question
                    )
                )

                samples[
                    name
                ].append(
                    elapsed_ms
                )

                print(
                    f"  第 {repeat} 次 "
                    f"{name:<20}"
                    f"{elapsed_ms:8.2f} ms"
                )


    # =====================================================
    # 汇总
    # =====================================================

    summary = {}


    for name, values in (
        samples.items()
    ):

        average_ms = (
            statistics.mean(
                values
            )
        )

        median_ms = (
            statistics.median(
                values
            )
        )

        p95_ms = (
            percentile_95(
                values
            )
        )

        summary[name] = {
            "samples": len(
                values
            ),
            "average_ms": round(
                average_ms,
                3,
            ),
            "median_ms": round(
                median_ms,
                3,
            ),
            "p95_ms": round(
                p95_ms,
                3,
            ),
            "min_ms": round(
                min(values),
                3,
            ),
            "max_ms": round(
                max(values),
                3,
            ),
        }


    # =====================================================
    # 打印最终表
    # =====================================================

    print()
    print(
        "=" * 78
    )

    print(
        "📊 RAG 延迟评测最终结果"
    )

    print(
        "=" * 78
    )

    print(
        f"{'方案':<25}"
        f"{'样本':>8}"
        f"{'平均耗时':>15}"
        f"{'P95':>15}"
    )

    print(
        "-" * 78
    )


    for name in methods:

        item = summary[name]

        print(
            f"{name:<25}"
            f"{item['samples']:>8}"
            f"{item['average_ms']:>12.2f} ms"
            f"{item['p95_ms']:>12.2f} ms"
        )


    # =====================================================
    # 保存 JSON
    # =====================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    report = {
        "question_count": len(
            cases
        ),
        "repeats": REPEATS,
        "top_k": TOP_K,
        "total_samples_per_method": (
            len(cases)
            * REPEATS
        ),
        "summary": summary,
        "samples_ms": {
            name: [
                round(
                    value,
                    3,
                )
                for value in values
            ]
            for (
                name,
                values,
            ) in samples.items()
        },
    }


    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


    print()
    print(
        "✅ 延迟评测报告已保存："
    )

    print(
        REPORT_PATH
    )


if __name__ == "__main__":
    main()
