from __future__ import annotations

import json
from pathlib import Path

from knowledge_base import (
    get_default_knowledge_base,
)


PROJECT_ROOT = Path(__file__).resolve().parent

CASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_cases.json"
)


# =========================================================
# 加载测试集
# =========================================================

def load_cases():

    with CASE_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# 判断某个 Chunk 是否属于正确结果
# =========================================================

def is_relevant(
    text,
    keywords
):

    text = str(
        text
    ).lower()

    matched = 0

    for keyword in keywords:

        if str(keyword).lower() in text:
            matched += 1

    # 至少命中一半关键词
    required = max(
        1,
        (len(keywords) + 1) // 2
    )

    return matched >= required


# =========================================================
# 计算单个方法指标
# =========================================================

def calculate_metrics(
    all_results,
    cases
):

    hit1 = 0
    hit3 = 0

    reciprocal_ranks = []


    for case, results in zip(
        cases,
        all_results
    ):

        keywords = (
            case["keywords"]
        )

        relevant_ranks = []

        for rank, document in enumerate(
            results,
            start=1
        ):

            text = (
                document.page_content
            )

            if is_relevant(
                text,
                keywords
            ):
                relevant_ranks.append(
                    rank
                )


        if relevant_ranks:

            first_rank = min(
                relevant_ranks
            )

            if first_rank <= 1:
                hit1 += 1

            if first_rank <= 3:
                hit3 += 1

            reciprocal_ranks.append(
                1 / first_rank
            )

        else:

            reciprocal_ranks.append(
                0
            )


    total = len(
        cases
    )

    return {
        "Hit@1":
            hit1 / total,

        "Hit@3":
            hit3 / total,

        "MRR":
            sum(
                reciprocal_ranks
            ) / total,
    }


# =========================================================
# A：FAISS Only
# =========================================================

def search_faiss_only(
    kb,
    question,
    k=3
):

    results = (
        kb.db
        .similarity_search_with_score(
            question,
            k=k
        )
    )

    documents = []

    for document, distance in results:

        if (
            float(distance)
            <= kb.score_threshold
        ):
            documents.append(
                document
            )

    return documents[:k]


# =========================================================
# B：Hybrid
# BM25 + FAISS + RRF
# =========================================================

def search_hybrid(
    kb,
    question,
    k=3
):

    results = (
        kb.retriever.search(
            question,
            k=k
        )
    )

    return [
        item.document
        for item in results
    ]


# =========================================================
# C：Hybrid + Reranker
# =========================================================

def search_reranker(
    kb,
    question,
    k=3
):

    # Reranker 前多召回一些
    candidate_k = max(
        10,
        k
    )

    candidates = (
        kb.retriever.search(
            question,
            k=candidate_k
        )
    )

    ranked = (
        kb.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=k
        )
    )

    return [
        item.document
        for item, score
        in ranked
    ]


# =========================================================
# 显示单题结果
# =========================================================

def show_case(
    number,
    case,
    faiss,
    hybrid,
    reranked
):

    print()
    print(
        "=" * 70
    )

    print(
        f"问题 {number}："
        f"{case['question']}"
    )

    print(
        "目标关键词：",
        case["keywords"]
    )


    methods = [
        (
            "FAISS",
            faiss
        ),
        (
            "Hybrid",
            hybrid
        ),
        (
            "Hybrid + Reranker",
            reranked
        ),
    ]


    for name, results in methods:

        print()
        print(
            f"[{name}]"
        )

        for rank, document in enumerate(
            results,
            start=1
        ):

            relevant = is_relevant(
                document.page_content,
                case["keywords"]
            )

            mark = (
                "✅"
                if relevant
                else "❌"
            )

            preview = (
                document
                .page_content
                .replace(
                    "\n",
                    " "
                )
                .strip()
            )

            print(
                f"{rank}. {mark} "
                f"{preview[:100]}"
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
        "正在加载知识库……"
    )

    kb = (
        get_default_knowledge_base()
    )

    if kb.reranker is None:
        raise RuntimeError(
            "Reranker 当前未启用"
        )


    faiss_results = []
    hybrid_results = []
    reranker_results = []


    for number, case in enumerate(
        cases,
        start=1
    ):

        question = (
            case["question"]
        )

        faiss = search_faiss_only(
            kb,
            question
        )

        hybrid = search_hybrid(
            kb,
            question
        )

        reranked = search_reranker(
            kb,
            question
        )


        faiss_results.append(
            faiss
        )

        hybrid_results.append(
            hybrid
        )

        reranker_results.append(
            reranked
        )


        show_case(
            number,
            case,
            faiss,
            hybrid,
            reranked
        )


    # =====================================================
    # 汇总指标
    # =====================================================

    metrics = {
        "FAISS Only":
            calculate_metrics(
                faiss_results,
                cases
            ),

        "Hybrid":
            calculate_metrics(
                hybrid_results,
                cases
            ),

        "Hybrid + Reranker":
            calculate_metrics(
                reranker_results,
                cases
            ),
    }


    print()
    print()
    print(
        "=" * 70
    )
    print(
        "📊 检索评测最终结果"
    )
    print(
        "=" * 70
    )

    print(
        f"{'方案':<24}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'MRR':>10}"
    )

    print(
        "-" * 54
    )


    for name, values in metrics.items():

        print(
            f"{name:<24}"
            f"{values['Hit@1']:>10.3f}"
            f"{values['Hit@3']:>10.3f}"
            f"{values['MRR']:>10.3f}"
        )


    # =====================================================
    # 保存 JSON
    # =====================================================

    report_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "retrieval_compare_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )


    print()
    print(
        "✅ 评测报告已保存："
    )
    print(
        report_path
    )


if __name__ == "__main__":
    main()
