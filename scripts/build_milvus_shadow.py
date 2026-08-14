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

from langchain_community.vectorstores import FAISS
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


def _load_faiss_child_documents(
    *,
    index_path: Path,
    embedding_model,
):
    """
    直接读取正式 FAISS docstore 中已经生成好的 Child Chunk。

    不重新切块、不重新生成 parent_id，
    因此 Milvus 与当前正式 FAISS 使用同一批 Child Document。
    """

    vector_store = (
        FAISS.load_local(
            str(index_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    )

    mapping = getattr(
        vector_store,
        "index_to_docstore_id",
        {},
    )

    docstore = getattr(
        vector_store,
        "docstore",
        None,
    )

    if docstore is None:
        raise RuntimeError(
            "FAISS docstore 不存在"
        )

    documents = []

    for vector_index in sorted(
        mapping
    ):

        document_id = (
            mapping[
                vector_index
            ]
        )

        document = (
            docstore.search(
                document_id
            )
        )

        if hasattr(
            document,
            "page_content",
        ):
            documents.append(
                document
            )

    if not documents:
        raise RuntimeError(
            "没有从 FAISS 中读取到 Child Document"
        )

    return documents


def build_milvus_shadow() -> None:

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

    model_path_value = (
        embedding_config.get(
            "model_path"
        )
    )

    if not model_path_value:
        raise ValueError(
            "[embedding].model_path 缺失"
        )

    model_path = (
        _resolve_project_path(
            model_path_value
        )
    )

    index_path = (
        _resolve_project_path(
            embedding_config.get(
                "index_path",
                "faiss_index",
            )
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Embedding 模型不存在：{model_path}"
        )

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS 索引不存在：{index_path}"
        )

    uri = str(
        milvus_config.get(
            "uri",
            "http://milvus-standalone:19530",
        )
    )

    collection_name = str(
        milvus_config.get(
            "collection_name",
            "agentic_rag_chunks",
        )
    )

    metric_type = str(
        milvus_config.get(
            "metric_type",
            "COSINE",
        )
    )

    timeout = float(
        milvus_config.get(
            "timeout",
            30.0,
        )
    )

    batch_size = max(
        1,
        int(
            milvus_config.get(
                "batch_size",
                128,
            )
        ),
    )

    print()
    print(
        "========== Milvus Shadow Build =========="
    )
    print(
        f"FAISS Source：{index_path}"
    )
    print(
        f"Milvus URI：{uri}"
    )
    print(
        f"Collection：{collection_name}"
    )
    print(
        f"Metric：{metric_type}"
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

    documents = (
        _load_faiss_child_documents(
            index_path=index_path,
            embedding_model=(
                embedding_model
            ),
        )
    )

    print(
        "✅ 从正式 FAISS 读取 Child Chunk：",
        len(documents),
    )

    probe_vector = (
        embedding_model
        .embed_query(
            "Milvus dimension probe"
        )
    )

    dimension = len(
        probe_vector
    )

    if dimension <= 1:
        raise RuntimeError(
            "无法确定 Embedding dimension"
        )

    print(
        "✅ Embedding dimension：",
        dimension,
    )

    store = MilvusStore(
        uri=uri,
        collection_name=(
            collection_name
        ),
        metric_type=(
            metric_type
        ),
        timeout=timeout,
    )

    store.recreate_collection(
        dimension=dimension
    )

    print(
        "✅ Milvus Collection 已创建"
    )

    inserted = 0

    for start in range(
        0,
        len(documents),
        batch_size,
    ):

        batch = documents[
            start:
            start + batch_size
        ]

        vectors = (
            embedding_model
            .embed_documents(
                [
                    str(
                        document.page_content
                    )
                    for document
                    in batch
                ]
            )
        )

        inserted += (
            store.insert_documents(
                documents=batch,
                vectors=vectors,
            )
        )

        print(
            "Milvus 写入进度："
            f"{inserted}/{len(documents)}"
        )

    store.flush()

    stats = store.stats()

    print()
    print(
        "========== Milvus Shadow Build 完成 =========="
    )
    print(
        "写入 Child Chunk：",
        inserted,
    )
    print(
        "Collection Stats：",
        stats,
    )
    print()
    print(
        "✅ FAISS 没有被修改"
    )
    print(
        "✅ 正式 RAG 尚未切换到 Milvus"
    )


if __name__ == "__main__":
    build_milvus_shadow()
