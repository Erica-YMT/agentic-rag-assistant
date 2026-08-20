from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from build_index import load_config
from rag.incremental_index import _embedding_model
from rag.milvus_store import MilvusStore, scoped_collection_name


def _load_faiss_child_documents(*, index_path: Path, embedding_model) -> list:
    vector_store = FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    mapping = getattr(vector_store, "index_to_docstore_id", {}) or {}
    docstore = getattr(vector_store, "docstore", None)
    if docstore is None:
        raise RuntimeError("FAISS docstore 不存在")

    documents = []
    for vector_index in sorted(mapping):
        document = docstore.search(mapping[vector_index])
        if hasattr(document, "page_content"):
            documents.append(document)
    return documents


def _load_parent_store(index_path: Path) -> dict:
    path = index_path / "parent_store.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("parents", {}) or {})


def get_scoped_milvus_store(
    *,
    settings: dict | None = None,
    user_id: int | None = None,
) -> MilvusStore:
    settings = settings or load_config()
    milvus = settings.get("milvus", {})
    collection_name = scoped_collection_name(
        str(milvus.get("collection_name", "agentic_rag_chunks")),
        user_id=user_id,
    )
    return MilvusStore(
        uri=str(milvus.get("uri", "http://milvus-standalone:19530")),
        collection_name=collection_name,
        metric_type=str(milvus.get("metric_type", "COSINE")),
        timeout=float(milvus.get("timeout", 30.0)),
    )


def sync_milvus_from_faiss(
    *,
    index_path: str | Path,
    user_id: int | None = None,
) -> dict:
    """
    Rebuild one scoped Milvus collection from the canonical local FAISS index.

    This is the safe migration/manual-rebuild path. Normal uploads/deletes use
    incremental_update_faiss() and never recreate the collection.
    """

    settings = load_config()
    index_path = Path(index_path).resolve()

    if not (index_path / "index.faiss").is_file() or not (index_path / "index.pkl").is_file():
        raise FileNotFoundError(f"FAISS 索引不存在：{index_path}")

    embedding_model = _embedding_model(settings)
    documents = _load_faiss_child_documents(
        index_path=index_path,
        embedding_model=embedding_model,
    )
    if not documents:
        raise RuntimeError("FAISS 中没有可同步到 Milvus 的 Child Chunk")

    probe_vector = embedding_model.embed_query("Milvus dimension probe")
    dimension = len(probe_vector)
    if dimension <= 1:
        raise RuntimeError("无法确定 Embedding dimension")

    store = get_scoped_milvus_store(settings=settings, user_id=user_id)
    store.recreate_collection(dimension=dimension)

    milvus = settings.get("milvus", {})
    batch_size = max(1, int(milvus.get("batch_size", 128)))
    inserted_children = 0

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        vectors = embedding_model.embed_documents(
            [str(document.page_content) for document in batch]
        )
        inserted_children += store.insert_documents(
            documents=batch,
            vectors=vectors,
        )

    parent_store = _load_parent_store(index_path)
    inserted_parents = store.upsert_parent_records(
        parent_store=parent_store,
        dimension=dimension,
    )
    store.flush()

    return {
        "collection_name": store.collection_name,
        "child_count": inserted_children,
        "parent_count": inserted_parents,
        "dimension": dimension,
    }


def drop_milvus_scope(*, user_id: int | None = None) -> bool:
    settings = load_config()
    store = get_scoped_milvus_store(settings=settings, user_id=user_id)
    return store.drop_collection()
