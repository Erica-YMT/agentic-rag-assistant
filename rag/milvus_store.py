from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from pymilvus import DataType, MilvusClient


@dataclass
class MilvusSearchResult:
    document: Document
    score: float
    chunk_id: str


def scoped_collection_name(
    base_name: str,
    *,
    user_id: int | None = None,
) -> str:
    """Return the public or hard-isolated private Milvus collection name."""

    base = re.sub(r"[^A-Za-z0-9_]", "_", str(base_name or "").strip())
    if not base:
        base = "agentic_rag_chunks"
    if base[0].isdigit():
        base = f"c_{base}"

    if user_id is None:
        return base[:255]

    suffix = f"_user_{int(user_id)}"
    return f"{base[: max(1, 255 - len(suffix))]}{suffix}"


class MilvusStore:
    """
    Milvus vector store used by the same RAG lifecycle as FAISS.

    One scoped collection contains two record types:
    - child: searchable Child Chunk with a real embedding;
    - parent: Auto-Merging Parent record with a zero vector.

    Vector search always filters to child records. Parent records are queried only
    as scalar data, so Milvus mode no longer depends on local parent_store.json.
    Public and private data are placed in different collections.
    """

    CONTENT_MAX_BYTES = 16384
    PRIMARY_MAX_LENGTH = 96
    REQUIRED_FIELDS = {
        "chunk_id",
        "record_type",
        "content",
        "metadata",
        "vector",
    }

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        metric_type: str = "COSINE",
        timeout: float = 30.0,
    ) -> None:
        self.uri = str(uri).strip()
        self.collection_name = str(collection_name).strip()
        self.metric_type = str(metric_type).strip().upper()
        self.timeout = float(timeout)

        if not self.uri:
            raise ValueError("Milvus uri 不能为空")
        if not self.collection_name:
            raise ValueError("Milvus collection_name 不能为空")
        if self.metric_type not in {"COSINE", "IP", "L2"}:
            raise ValueError("Milvus metric_type 只支持 COSINE / IP / L2")

        try:
            self.client = MilvusClient(uri=self.uri)
        except Exception as exc:
            raise RuntimeError(
                "无法连接 Milvus："
                f"uri={self.uri}; error={exc}"
            ) from exc

    @staticmethod
    def _json_safe_metadata(
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                metadata or {},
                ensure_ascii=False,
                default=str,
            )
        )

    @classmethod
    def _chunk_id(cls, document: Document) -> str:
        """Generate a stable Child Chunk primary key."""

        metadata = cls._json_safe_metadata(document.metadata)
        child_id = str(metadata.get("child_id") or "").strip()
        relative_path = str(
            metadata.get("relative_path")
            or metadata.get("source")
            or metadata.get("file_name")
            or ""
        ).strip()

        if child_id:
            identity = f"child|{relative_path}|{child_id}"
        else:
            identity = json.dumps(
                {
                    "page_content": str(document.page_content),
                    "metadata": metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @classmethod
    def _parent_record_id(cls, parent_id: str) -> str:
        return f"parent:{str(parent_id).strip()}"

    def collection_exists(self) -> bool:
        return self.collection_name in set(self.client.list_collections())

    def describe_collection(self) -> dict:
        if not self.collection_exists():
            return {}
        value = self.client.describe_collection(
            collection_name=self.collection_name,
            timeout=self.timeout,
        )
        return dict(value or {})

    def schema_compatible(self) -> bool:
        """Detect old Shadow collections that do not contain record_type."""

        if not self.collection_exists():
            return False

        try:
            description = self.describe_collection()
            raw_fields = description.get("fields")
            if raw_fields is None:
                raw_fields = (description.get("schema") or {}).get("fields", [])

            names = {
                str(field.get("name") or field.get("field_name") or "")
                for field in (raw_fields or [])
                if isinstance(field, dict)
            }
            return self.REQUIRED_FIELDS.issubset(names)
        except Exception:
            return False

    def ensure_collection(
        self,
        *,
        dimension: int,
    ) -> bool:
        """Create a new compatible collection, never silently drop an old one."""

        if self.collection_exists():
            if not self.schema_compatible():
                raise RuntimeError(
                    "Milvus Collection 使用旧 Schema，需要执行一次完整同步迁移"
                )
            return False

        self.recreate_collection(dimension=dimension)
        return True

    def recreate_collection(
        self,
        *,
        dimension: int,
    ) -> None:
        dimension = int(dimension)
        if dimension <= 1:
            raise ValueError("Embedding dimension 必须大于 1")

        if self.collection_exists():
            self.client.drop_collection(
                collection_name=self.collection_name,
                timeout=self.timeout,
            )

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.PRIMARY_MAX_LENGTH,
        )
        schema.add_field(
            field_name="record_type",
            datatype=DataType.VARCHAR,
            max_length=16,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=self.CONTENT_MAX_BYTES,
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            timeout=self.timeout,
        )

    def drop_collection(self) -> bool:
        if not self.collection_exists():
            return False
        self.client.drop_collection(
            collection_name=self.collection_name,
            timeout=self.timeout,
        )
        return True

    def _validate_content(self, content: str) -> None:
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > self.CONTENT_MAX_BYTES:
            raise ValueError(
                "Chunk 文本超过 Milvus VARCHAR 上限："
                f"{content_bytes} bytes"
            )

    def _build_child_rows(
        self,
        *,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> list[dict]:
        if len(documents) != len(vectors):
            raise ValueError("documents 与 vectors 数量不一致")

        rows: list[dict] = []
        for document, vector in zip(documents, vectors):
            content = str(document.page_content)
            self._validate_content(content)
            rows.append(
                {
                    "chunk_id": self._chunk_id(document),
                    "record_type": "child",
                    "content": content,
                    "metadata": self._json_safe_metadata(document.metadata),
                    "vector": [float(value) for value in vector],
                }
            )
        return rows

    def _build_parent_rows(
        self,
        *,
        parent_store: dict,
        dimension: int,
    ) -> list[dict]:
        dimension = int(dimension)
        if dimension <= 1:
            raise ValueError("Embedding dimension 必须大于 1")

        # COSINE 对零向量没有意义；Parent 永远被 record_type filter 排除，
        # 因此使用固定单位占位向量，只为满足 Collection 的向量字段要求。
        placeholder_vector = [0.0] * dimension
        placeholder_vector[0] = 1.0
        rows: list[dict] = []

        for parent_id, record in (parent_store or {}).items():
            record = dict(record or {})
            content = str(record.get("page_content") or "")
            self._validate_content(content)
            metadata = self._json_safe_metadata(record.get("metadata") or {})
            metadata.setdefault("parent_id", str(parent_id))
            metadata.setdefault("chunk_level", "parent")

            rows.append(
                {
                    "chunk_id": self._parent_record_id(str(parent_id)),
                    "record_type": "parent",
                    "content": content,
                    "metadata": metadata,
                    "vector": list(placeholder_vector),
                }
            )

        return rows

    def insert_documents(
        self,
        *,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> int:
        rows = self._build_child_rows(documents=documents, vectors=vectors)
        if not rows:
            return 0
        self.client.insert(
            collection_name=self.collection_name,
            data=rows,
            timeout=self.timeout,
        )
        return len(rows)

    def upsert_documents(
        self,
        *,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> int:
        rows = self._build_child_rows(documents=documents, vectors=vectors)
        if not rows:
            return 0
        self.client.upsert(
            collection_name=self.collection_name,
            data=rows,
            timeout=self.timeout,
        )
        return len(rows)

    def upsert_parent_records(
        self,
        *,
        parent_store: dict,
        dimension: int,
    ) -> int:
        rows = self._build_parent_rows(
            parent_store=parent_store,
            dimension=dimension,
        )
        if not rows:
            return 0
        self.client.upsert(
            collection_name=self.collection_name,
            data=rows,
            timeout=self.timeout,
        )
        return len(rows)

    def delete_chunk_ids(self, chunk_ids: list[str]) -> int:
        ids = list(dict.fromkeys(str(value) for value in chunk_ids if value))
        if not ids or not self.collection_exists():
            return 0

        result = self.client.delete(
            collection_name=self.collection_name,
            ids=ids,
            timeout=self.timeout,
        )
        if isinstance(result, dict):
            return int(result.get("delete_count", result.get("delete_cnt", len(ids))))
        return len(ids)

    def delete_parent_ids(self, parent_ids: list[str]) -> int:
        return self.delete_chunk_ids(
            [self._parent_record_id(value) for value in parent_ids]
        )

    def flush(self) -> None:
        if self.collection_exists():
            # Collection-scoped flush works in both Milvus Standalone and
            # Milvus Lite. ``flush_all`` is not implemented by Milvus Lite.
            self.client.flush(
                collection_name=self.collection_name,
                timeout=self.timeout,
            )

    def stats(self) -> dict:
        if not self.collection_exists():
            return {"row_count": 0}
        return dict(
            self.client.get_collection_stats(
                collection_name=self.collection_name,
                timeout=self.timeout,
            )
            or {}
        )

    def _query_records(
        self,
        *,
        record_type: str,
    ) -> list[dict]:
        stats = self.stats()
        row_count = int(stats.get("row_count", 0))
        if row_count <= 0:
            return []
        if row_count >= 16384:
            raise RuntimeError(
                "Milvus Collection 已达到 16384 条以上，"
                "当前本地 BM25/Parent 全量读取方案停止使用；"
                "后续应切换 query_iterator 或 Milvus 原生稀疏检索。"
            )

        return list(
            self.client.query(
                collection_name=self.collection_name,
                filter=f'record_type == "{record_type}"',
                output_fields=[
                    "chunk_id",
                    "record_type",
                    "content",
                    "metadata",
                ],
                limit=row_count,
                timeout=self.timeout,
            )
            or []
        )

    def list_documents(self) -> list[Document]:
        """Read all searchable Child Documents for the local BM25 side."""

        rows = sorted(
            self._query_records(record_type="child"),
            key=lambda item: str(item.get("chunk_id", "")),
        )
        return [
            Document(
                page_content=str(row.get("content", "")),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows
        ]

    def list_parent_records(self) -> dict[str, dict]:
        """Read Auto-Merging Parent records from Milvus itself."""

        parents: dict[str, dict] = {}
        for row in self._query_records(record_type="parent"):
            metadata = dict(row.get("metadata") or {})
            parent_id = str(
                metadata.get("parent_id")
                or str(row.get("chunk_id", "")).removeprefix("parent:")
            ).strip()
            if not parent_id:
                continue
            parents[parent_id] = {
                "page_content": str(row.get("content", "")),
                "metadata": metadata,
            }
        return parents

    def search(
        self,
        *,
        query_vector: list[float],
        k: int = 5,
    ) -> list[MilvusSearchResult]:
        if not self.collection_exists():
            return []

        k = max(1, int(k))
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[[float(value) for value in query_vector]],
            anns_field="vector",
            filter='record_type == "child"',
            limit=k,
            output_fields=["content", "metadata", "record_type"],
            search_params={
                "metric_type": self.metric_type,
                "params": {},
            },
            timeout=self.timeout,
        )

        if not raw:
            return []

        results: list[MilvusSearchResult] = []
        for hit in raw[0]:
            entity = hit.get("entity") or {}
            metadata = entity.get("metadata") or {}
            results.append(
                MilvusSearchResult(
                    document=Document(
                        page_content=str(entity.get("content", "")),
                        metadata=dict(metadata),
                    ),
                    score=float(hit.get("distance", 0.0)),
                    chunk_id=str(hit.get("id", "")),
                )
            )
        return results
