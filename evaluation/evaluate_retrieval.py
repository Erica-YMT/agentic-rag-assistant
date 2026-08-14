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
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from rag.knowledge_base import KnowledgeBase


CONFIG_PATH = PROJECT_ROOT / "config.toml"
CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_cases.json"
)
REPORT_DIR = PROJECT_ROOT / "evaluation" / "results"


def resolve_project_path(value: str) -> Path:
    path = Path(str(value)).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_knowledge_base(
    config: dict[str, Any],
) -> tuple[KnowledgeBase, int, float]:
    embedding = config["embedding"]

    top_k = int(
        embedding.get("top_k", 5)
    )

    score_threshold = float(
        embedding.get(
            "score_threshold",
            1.0,
        )
    )

    knowledge_base = KnowledgeBase(
        model_dir=str(
            resolve_project_path(
                embedding["model_path"]
            )
        ),
        index_path=str(
            resolve_project_path(
                embedding.get(
                    "index_path",
                    "faiss_index",
                )
            )
        ),
        score_threshold=score_threshold,
    )

    return (
        knowledge_base,
        top_k,
        score_threshold,
    )


def normalize(value: str) -> str:
    return str(value).strip().lower()


def get_source(metadata: dict[str, Any]) -> str:
    source = (
        metadata.get("file_name")
        or metadata.get("relative_path")
        or metadata.get("source")
        or "未知来源"
    )

    return Path(str(source)).name


def keyword_matches(
    text: str,
    keywords: list[str],
) -> list[str]:
    normalized_text = normalize(text)

    return [
        keyword
        for keyword in keywords
        if normalize(keyword) in normalized_text
    ]


def evaluate_case(
    case: dict[str, Any],
    knowledge_base: KnowledgeBase,
    top_k: int,
    score_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = str(case["id"])
    query = str(case["query"])

    should_retrieve = bool(
        case.get("should_retrieve", True)
    )

    expected_sources = {
        Path(str(source)).name
        for source in case.get(
            "expected_sources",
            [],
        )
    }

    expected_keywords = [
        str(keyword)
        for keyword in case.get(
            "expected_keywords",
            [],
        )
    ]

    # 一个文本块至少覆盖多少比例的关键词，
    # 才算真正能够支持回答。
    chunk_match_threshold = float(
        case.get(
            "chunk_match_threshold",
            0.75,
        )
    )

    results = (
        knowledge_base.db
        .similarity_search_with_score(
            query,
            k=top_k,
        )
    )

    detail_rows: list[dict[str, Any]] = []

    first_source_rank: int | None = None
    first_answer_chunk_rank: int | None = None
    passed_texts: list[str] = []

    for rank, (
        document,
        distance,
    ) in enumerate(results, start=1):
        distance = float(distance)
        metadata = document.metadata or {}
        source = get_source(metadata)
        content = document.page_content.strip()

        passed_threshold = (
            distance <= score_threshold
        )

        matches = keyword_matches(
            content,
            expected_keywords,
        )

        if expected_keywords:
            chunk_keyword_coverage = (
                len(matches)
                / len(expected_keywords)
            )
        else:
            chunk_keyword_coverage = 0.0

        is_expected_source = (
            source in expected_sources
        )

        is_answer_chunk = (
            should_retrieve
            and passed_threshold
            and is_expected_source
            and chunk_keyword_coverage
            >= chunk_match_threshold
        )

        if (
            first_source_rank is None
            and is_expected_source
        ):
            first_source_rank = rank

        if (
            first_answer_chunk_rank is None
            and is_answer_chunk
        ):
            first_answer_chunk_rank = rank

        if passed_threshold:
            passed_texts.append(content)

        page = metadata.get("page")

        detail_rows.append(
            {
                "case_id": case_id,
                "query": query,
                "rank": rank,
                "source": source,
                "page": (
                    page + 1
                    if isinstance(page, int)
                    else ""
                ),
                "distance": round(
                    distance,
                    6,
                ),
                "passed_threshold": (
                    passed_threshold
                ),
                "is_expected_source": (
                    is_expected_source
                ),
                "matched_keywords": "、".join(
                    matches
                ),
                "chunk_keyword_coverage": round(
                    chunk_keyword_coverage,
                    4,
                ),
                "is_answer_chunk": (
                    is_answer_chunk
                ),
                "text_preview": (
                    content
                    .replace("\n", " ")[:300]
                ),
            }
        )

    predicted_relevant = any(
        row["passed_threshold"]
        for row in detail_rows
    )

    threshold_correct = (
        predicted_relevant == should_retrieve
    )

    combined_passed_text = "\n".join(
        passed_texts
    )

    all_matches = keyword_matches(
        combined_passed_text,
        expected_keywords,
    )

    overall_keyword_coverage = (
        len(all_matches)
        / len(expected_keywords)
        if expected_keywords
        else None
    )

    source_hit_at_k = (
        first_source_rank is not None
        if expected_sources
        else None
    )

    answer_chunk_hit_at_k = (
        first_answer_chunk_rank is not None
        if should_retrieve
        and expected_keywords
        else None
    )

    source_mrr = (
        1.0 / first_source_rank
        if first_source_rank
        else 0.0
    )

    answer_chunk_mrr = (
        1.0 / first_answer_chunk_rank
        if first_answer_chunk_rank
        else 0.0
    )

    best_distance = (
        float(results[0][1])
        if results
        else None
    )

    summary_row = {
        "case_id": case_id,
        "query": query,
        "should_retrieve": should_retrieve,
        "predicted_relevant": predicted_relevant,
        "threshold_correct": threshold_correct,
        "best_distance": (
            round(best_distance, 6)
            if best_distance is not None
            else None
        ),
        "first_source_rank": (
            first_source_rank or ""
        ),
        "source_hit_at_k": source_hit_at_k,
        "source_mrr": round(
            source_mrr,
            4,
        ),
        "first_answer_chunk_rank": (
            first_answer_chunk_rank or ""
        ),
        "answer_chunk_hit_at_k": (
            answer_chunk_hit_at_k
        ),
        "answer_chunk_mrr": round(
            answer_chunk_mrr,
            4,
        ),
        "overall_keyword_coverage": (
            round(
                overall_keyword_coverage,
                4,
            )
            if overall_keyword_coverage
            is not None
            else None
        ),
    }

    return summary_row, detail_rows


def mean_boolean(
    series: pd.Series,
) -> float:
    valid = series.dropna()

    if valid.empty:
        return 0.0

    return valid.astype(bool).astype(float).mean()


def mean_number(
    series: pd.Series,
) -> float:
    valid = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if valid.empty:
        return 0.0

    return float(valid.mean())


def main() -> None:
    config = load_config()
    cases = load_cases()

    (
        knowledge_base,
        top_k,
        score_threshold,
    ) = load_knowledge_base(config)

    summary_rows = []
    detail_rows = []

    for case in cases:
        summary, details = evaluate_case(
            case=case,
            knowledge_base=knowledge_base,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        summary_rows.append(summary)
        detail_rows.extend(details)

    summary_df = pd.DataFrame(
        summary_rows
    )

    details_df = pd.DataFrame(
        detail_rows
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        REPORT_DIR
        / "rag_eval_v2_summary.csv"
    )

    details_path = (
        REPORT_DIR
        / "rag_eval_v2_details.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    details_df.to_csv(
        details_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_accuracy = mean_boolean(
        summary_df["threshold_correct"]
    )

    source_hit = mean_boolean(
        summary_df["source_hit_at_k"]
    )

    answer_chunk_hit = mean_boolean(
        summary_df[
            "answer_chunk_hit_at_k"
        ]
    )

    source_mrr = mean_number(
        summary_df["source_mrr"]
    )

    answer_chunk_mrr = mean_number(
        summary_df["answer_chunk_mrr"]
    )

    keyword_coverage = mean_number(
        summary_df[
            "overall_keyword_coverage"
        ]
    )

    print("\n" + "=" * 72)
    print("RAG 文本块级自动评测")
    print("=" * 72)
    print(f"评测题数：{len(summary_df)}")
    print(f"Top-K：{top_k}")
    print(
        f"距离阈值：{score_threshold:.2f}"
    )
    print(
        f"阈值判断准确率："
        f"{threshold_accuracy:.2%}"
    )
    print(
        f"来源 Hit@{top_k}："
        f"{source_hit:.2%}"
    )
    print(
        f"来源 MRR：{source_mrr:.4f}"
    )
    print(
        f"答案文本块 Hit@{top_k}："
        f"{answer_chunk_hit:.2%}"
    )
    print(
        "答案文本块 MRR："
        f"{answer_chunk_mrr:.4f}"
    )
    print(
        "整体关键词覆盖率："
        f"{keyword_coverage:.2%}"
    )

    print("\n逐题结果：")

    columns = [
        "case_id",
        "threshold_correct",
        "best_distance",
        "first_source_rank",
        "first_answer_chunk_rank",
        "answer_chunk_hit_at_k",
        "answer_chunk_mrr",
        "overall_keyword_coverage",
    ]

    print(
        summary_df[columns].to_string(
            index=False
        )
    )

    print("\n报告已保存：")
    print("-", summary_path)
    print("-", details_path)


if __name__ == "__main__":
    main()
