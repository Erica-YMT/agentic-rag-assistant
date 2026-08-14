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

import argparse
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from rag.milvus_store import MilvusStore




def _resolve_project_path(
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


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "⑦-1A Milvus Dense Search Smoke Test"
        )
    )

    parser.add_argument(
        "query",
        help="测试查询",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    embedding_config = (
        config.get(
            "embedding",
            {},
        )
    )

    milvus_config = (
        config.get(
            "milvus",
            {},
        )
    )

    model_path = (
        _resolve_project_path(
            embedding_config[
                "model_path"
            ]
        )
    )

    embedding_model = (
        HuggingFaceEmbeddings(
            model_name=str(
                model_path
            ),
            encode_kwargs={
                "normalize_embeddings":
                    True,
            },
        )
    )

    store = MilvusStore(
        uri=str(
            milvus_config.get(
                "uri",
                "http://milvus-standalone:19530",
            )
        ),
        collection_name=str(
            milvus_config.get(
                "collection_name",
                "agentic_rag_chunks",
            )
        ),
        metric_type=str(
            milvus_config.get(
                "metric_type",
                "COSINE",
            )
        ),
        timeout=float(
            milvus_config.get(
                "timeout",
                30.0,
            )
        ),
    )

    query_vector = (
        embedding_model
        .embed_query(
            args.query
        )
    )

    results = store.search(
        query_vector=query_vector,
        k=args.top_k,
    )

    print()
    print(
        "========== Milvus Dense Search =========="
    )
    print(
        "Query：",
        args.query,
    )

    if not results:
        print(
            "没有检索结果"
        )
        return

    for index, item in enumerate(
        results,
        start=1,
    ):

        metadata = (
            item.document.metadata
            or {}
        )

        source = (
            metadata.get(
                "file_name"
            )
            or metadata.get(
                "relative_path"
            )
            or metadata.get(
                "source"
            )
            or "未知来源"
        )

        page = metadata.get(
            "page"
        )

        if isinstance(
            page,
            int,
        ):
            page_text = (
                f"第 {page + 1} 页"
            )
        else:
            page_text = ""

        content = (
            item.document
            .page_content
            .strip()
            .replace(
                "\n",
                " ",
            )
        )

        print()
        print(
            f"[{index}] "
            f"COSINE={item.score:.4f}"
        )
        print(
            "来源：",
            source,
            page_text,
        )
        print(
            "内容：",
            content[:300],
        )


if __name__ == "__main__":
    main()
