from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from build_index import (
    PROJECT_ROOT,
    build_index_manifest,
    load_config,
    load_document_file,
    split_documents,
)
from rag.hierarchical_chunks import (
    build_hierarchical_chunks,
    save_parent_store,
)


class IncrementalIndexUnavailable(RuntimeError):
    """当前索引不满足安全增量更新条件，应回退一次全量重建。"""


class _DeleteOnlyEmbeddings(Embeddings):
    """删除 Chunk 时 FAISS 只需要一个 Embeddings 接口，不需要加载真实模型。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("delete-only 更新不应触发 embedding")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("delete-only 更新不应触发 embedding")


@dataclass
class IncrementalIndexResult:
    added_chunks: int
    removed_chunks: int
    added_documents: list
    removed_documents: list
    relative_paths: list[str]
    added_parent_records: dict = field(default_factory=dict)
    removed_parent_ids: list[str] = field(default_factory=list)


def _normalize_relative_path(value: str | Path) -> str:
    return Path(str(value)).as_posix().lstrip("./")


def _relative_path_for_file(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _load_manifest(index_path: Path) -> dict:
    path = index_path / "index_manifest.json"
    if not path.is_file():
        raise IncrementalIndexUnavailable(
            "旧索引缺少 index_manifest.json，需要先全量重建一次"
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IncrementalIndexUnavailable(
            f"index_manifest.json 无法读取：{exc}"
        ) from exc


def _assert_manifest_compatible(index_path: Path, settings: dict) -> None:
    current = _load_manifest(index_path)
    expected = build_index_manifest(settings)
    if current != expected:
        raise IncrementalIndexUnavailable(
            "Chunk/Embedding 配置已变化，需要先全量重建索引"
        )


def _embedding_model(settings: dict):
    embedding = settings.get("embedding", {})
    model_path_value = embedding.get("model_path")
    if not model_path_value:
        raise ValueError("config.toml 中缺少 [embedding].model_path")

    model_path = Path(str(model_path_value)).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"没有找到 Embedding 模型：{model_path}")

    device = str(embedding.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("[embedding].device 只支持 auto / cpu / cuda")

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[embedding].device=cuda，但当前运行环境无法使用 CUDA")
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"

    batch_size = max(1, int(embedding.get("batch_size", 32)))

    return HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": resolved_device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
    )


def _documents_for_paths(vector_store, relative_paths: set[str]):
    ids: list[str] = []
    documents: list = []

    mapping = getattr(vector_store, "index_to_docstore_id", {}) or {}
    docstore = getattr(vector_store, "docstore", None)
    if docstore is None:
        return ids, documents

    for vector_index in sorted(mapping):
        document_id = mapping[vector_index]
        document = docstore.search(document_id)
        if not hasattr(document, "metadata"):
            continue

        source = _normalize_relative_path(
            (document.metadata or {}).get("relative_path", "")
        )
        if source in relative_paths:
            ids.append(str(document_id))
            documents.append(document)

    return ids, documents


def _load_parent_store(index_path: Path) -> dict:
    path = index_path / "parent_store.json"
    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload.get("parents", {}) or {})
    except Exception as exc:
        raise IncrementalIndexUnavailable(
            f"parent_store.json 无法读取：{exc}"
        ) from exc


def _parent_ids_for_paths(parent_store: dict, relative_paths: set[str]) -> list[str]:
    values: list[str] = []
    for parent_id, record in (parent_store or {}).items():
        metadata = dict((record or {}).get("metadata", {}) or {})
        source = _normalize_relative_path(metadata.get("relative_path", ""))
        if source in relative_paths:
            values.append(str(parent_id))
    return values


def _remove_parents_for_paths(parent_store: dict, relative_paths: set[str]) -> dict:
    removed = set(_parent_ids_for_paths(parent_store, relative_paths))
    return {
        parent_id: record
        for parent_id, record in (parent_store or {}).items()
        if str(parent_id) not in removed
    }


def _build_new_chunks(files: list[Path], settings: dict):
    documents = []
    for file_path in files:
        documents.extend(load_document_file(file_path))

    if not documents:
        return [], [], {}

    embedding = settings.get("embedding", {})
    hierarchical = settings.get("hierarchical_chunking", {})
    hierarchical_enabled = bool(hierarchical.get("enabled", True))

    if hierarchical_enabled:
        parents, chunks, parent_store = build_hierarchical_chunks(
            documents=documents,
            parent_chunk_size=int(hierarchical.get("parent_chunk_size", 1200)),
            parent_chunk_overlap=int(hierarchical.get("parent_chunk_overlap", 120)),
            child_chunk_size=int(hierarchical.get("child_chunk_size", 400)),
            child_chunk_overlap=int(hierarchical.get("child_chunk_overlap", 80)),
        )
        return chunks, parents, parent_store

    chunks = split_documents(
        documents=documents,
        chunk_size=int(embedding.get("chunk_size", 600)),
        chunk_overlap=int(embedding.get("chunk_overlap", 100)),
    )
    return chunks, [], {}


def _sync_milvus_incremental(
    *,
    result: IncrementalIndexResult,
    embedding_model,
    settings: dict,
    user_id: int | None,
) -> None:
    """Synchronize the exact changed Child/Parent records to scoped Milvus."""

    from rag.milvus_store import MilvusStore, scoped_collection_name

    milvus = settings.get("milvus", {})
    collection_name = scoped_collection_name(
        str(milvus.get("collection_name", "agentic_rag_chunks")),
        user_id=user_id,
    )
    store = MilvusStore(
        uri=str(milvus.get("uri", "http://milvus-standalone:19530")),
        collection_name=collection_name,
        metric_type=str(milvus.get("metric_type", "COSINE")),
        timeout=float(milvus.get("timeout", 30.0)),
    )

    if not store.collection_exists():
        raise IncrementalIndexUnavailable(
            f"Milvus Collection 不存在：{collection_name}，需要先执行一次完整同步"
        )
    if not store.schema_compatible():
        raise IncrementalIndexUnavailable(
            f"Milvus Collection 使用旧 Schema：{collection_name}，需要先完整同步迁移"
        )

    removed_ids = [
        store._chunk_id(document)
        for document in result.removed_documents
    ]
    if removed_ids:
        store.delete_chunk_ids(removed_ids)
    if result.removed_parent_ids:
        store.delete_parent_ids(result.removed_parent_ids)

    vector_dimension: int | None = None
    if result.added_documents:
        batch_size = max(1, int(milvus.get("batch_size", 128)))
        for start in range(0, len(result.added_documents), batch_size):
            batch = result.added_documents[start:start + batch_size]
            vectors = embedding_model.embed_documents(
                [str(document.page_content) for document in batch]
            )
            if vectors and vector_dimension is None:
                vector_dimension = len(vectors[0])
            store.upsert_documents(documents=batch, vectors=vectors)

    if result.added_parent_records:
        if vector_dimension is None:
            # Parent additions always originate from newly built Child chunks.
            # This branch is only a defensive guard against inconsistent state.
            raise RuntimeError("新增 Parent 时无法确定 Milvus vector dimension")
        store.upsert_parent_records(
            parent_store=result.added_parent_records,
            dimension=vector_dimension,
        )

    store.flush()


def incremental_update_faiss(
    *,
    index_path: str | Path,
    upsert_files: list[str | Path] | None = None,
    delete_relative_paths: list[str | Path] | None = None,
    sync_milvus: bool = False,
    milvus_user_id: int | None = None,
) -> IncrementalIndexResult:
    """
    对现有 FAISS 索引按“文件”增量更新，并可同步同 scope Milvus。

    - 覆盖/上传：仅删除该文件旧 Chunk，再只 embedding 新 Chunk；
    - 删除：仅删除对应 Chunk，不加载真实 Embedding；
    - Parent Store 按 relative_path 替换/删除；
    - Milvus 同步 Child + Parent；private 使用独立 user Collection；
    - manifest/schema 不兼容时拒绝增量，交由上层安全全量重建。
    """

    settings = load_config()
    index_path = Path(index_path).resolve()

    if not (index_path / "index.faiss").is_file() or not (index_path / "index.pkl").is_file():
        raise IncrementalIndexUnavailable("FAISS 索引不存在，需要先全量构建")

    _assert_manifest_compatible(index_path, settings)

    files = [Path(value).resolve() for value in (upsert_files or [])]
    for file_path in files:
        if not file_path.is_file():
            raise FileNotFoundError(f"增量索引文件不存在：{file_path}")

    upsert_relative_paths = {_relative_path_for_file(path) for path in files}
    delete_paths = {
        _normalize_relative_path(value)
        for value in (delete_relative_paths or [])
        if str(value).strip()
    }
    affected_paths = upsert_relative_paths | delete_paths

    if not affected_paths:
        return IncrementalIndexResult(0, 0, [], [], [])

    new_chunks, _parents, new_parent_store = _build_new_chunks(files, settings)
    embedding_model = _embedding_model(settings) if new_chunks else _DeleteOnlyEmbeddings()

    vector_store = FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    old_ids, old_documents = _documents_for_paths(vector_store, affected_paths)
    if old_ids:
        vector_store.delete(ids=old_ids)

    if new_chunks:
        vector_store.add_documents(new_chunks)

    vector_store.save_local(str(index_path))

    hierarchical_enabled = bool(
        settings.get("hierarchical_chunking", {}).get("enabled", True)
    )
    removed_parent_ids: list[str] = []
    if hierarchical_enabled:
        parent_store = _load_parent_store(index_path)
        removed_parent_ids = _parent_ids_for_paths(parent_store, affected_paths)
        parent_store = _remove_parents_for_paths(parent_store, affected_paths)
        parent_store.update(new_parent_store)
        save_parent_store(index_path / "parent_store.json", parent_store)

    result = IncrementalIndexResult(
        added_chunks=len(new_chunks),
        removed_chunks=len(old_documents),
        added_documents=list(new_chunks),
        removed_documents=list(old_documents),
        relative_paths=sorted(affected_paths),
        added_parent_records=dict(new_parent_store),
        removed_parent_ids=removed_parent_ids,
    )

    if sync_milvus:
        _sync_milvus_incremental(
            result=result,
            embedding_model=embedding_model,
            settings=settings,
            user_id=(None if milvus_user_id is None else int(milvus_user_id)),
        )

    return result
