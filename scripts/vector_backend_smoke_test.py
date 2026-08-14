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

from pathlib import Path

from config import config
from rag.knowledge_base import (
    DEFAULT_SCORE_THRESHOLD,
    KnowledgeBase,
)




def resolve_path(
    value: str,
) -> Path:

    path = Path(
        str(value)
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def first_source(
    results,
) -> str:

    if not results:
        return ""

    metadata = (
        results[0]
        .document
        .metadata
        or {}
    )

    return str(
        metadata.get(
            "file_name"
        )
        or metadata.get(
            "relative_path"
        )
        or metadata.get(
            "source"
        )
        or ""
    )


def main() -> None:

    embedding_config = (
        config.get(
            "embedding",
            {},
        )
    )

    vector_store_config = (
        config.get(
            "vector_store",
            {},
        )
    )

    assert (
        str(
            vector_store_config.get(
                "backend",
                "",
            )
        ).lower()
        == "faiss"
    ), (
        "正式配置必须继续保持 backend=faiss，"
        "本轮只做双后端预演。"
    )

    model_path = resolve_path(
        embedding_config[
            "model_path"
        ]
    )

    index_path = resolve_path(
        embedding_config.get(
            "index_path",
            "faiss_index",
        )
    )

    score_threshold = float(
        embedding_config.get(
            "score_threshold",
            DEFAULT_SCORE_THRESHOLD,
        )
    )

    milvus_config = (
        config.get(
            "milvus",
            {},
        )
    )

    print(
        "✅ 正式配置仍为 FAISS"
    )

    faiss_kb = KnowledgeBase(
        model_dir=str(
            model_path
        ),
        index_path=str(
            index_path
        ),
        score_threshold=(
            score_threshold
        ),
        vector_backend_name="faiss",
        milvus_settings=(
            milvus_config
        ),
    )

    milvus_kb = KnowledgeBase(
        model_dir=str(
            model_path
        ),
        index_path=str(
            index_path
        ),
        score_threshold=(
            score_threshold
        ),
        vector_backend_name="milvus",
        milvus_settings=(
            milvus_config
        ),
    )

    faiss_count = len(
        faiss_kb.retriever.documents
    )

    milvus_count = len(
        milvus_kb.retriever.documents
    )

    print(
        "FAISS BM25 Corpus：",
        faiss_count,
    )
    print(
        "Milvus BM25 Corpus：",
        milvus_count,
    )

    assert faiss_count > 0
    assert faiss_count == milvus_count

    expected_min_similarity = (
        1.0
        - score_threshold / 2.0
    )

    actual_min_similarity = float(
        milvus_kb
        .vector_backend
        .min_similarity
    )

    assert abs(
        expected_min_similarity
        - actual_min_similarity
    ) < 1e-9

    print(
        "✅ FAISS L2 阈值 -> "
        "Milvus COSINE 阈值：",
        f"{score_threshold:.4f}"
        " -> "
        f"{actual_min_similarity:.4f}",
    )

    query = "青春红音是什么"

    faiss_results = (
        faiss_kb
        .retriever
        .search(
            query=query,
            k=3,
        )
    )

    milvus_results = (
        milvus_kb
        .retriever
        .search(
            query=query,
            k=3,
        )
    )

    assert faiss_results
    assert milvus_results

    faiss_source = first_source(
        faiss_results
    )

    milvus_source = first_source(
        milvus_results
    )

    print(
        "FAISS Top1：",
        faiss_source,
    )
    print(
        "Milvus Top1：",
        milvus_source,
    )

    expected_file = (
        "青春红音2026实践资料.txt"
    )

    assert (
        Path(
            faiss_source
        ).name
        == expected_file
    )

    assert (
        Path(
            milvus_source
        ).name
        == expected_file
    )

    # 验证 Milvus 检索结果仍然能进入
    # 原有 Parent-Child Auto-Merging。
    merged = (
        milvus_kb
        .auto_merger
        .merge(
            milvus_results
        )
    )

    assert merged

    print(
        "✅ Milvus RetrievalResult "
        "可继续进入 Auto-Merging"
    )

    print()
    print(
        "===================================="
    )
    print(
        "✅ ⑦-1B 双后端代码预演全部通过"
    )
    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
