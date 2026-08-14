from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from pymilvus import DataType, MilvusClient

# VECTOR_BACKEND_PHASE1B_V1


@dataclass
class MilvusSearchResult:
    document: Document
    score: float
    chunk_id: str


class MilvusStore:
    """
    ⑦-1A Milvus 实验后端。

    当前只负责：
    - 创建 / 重建实验 Collection
    - 写入现有 FAISS 中的 Child Document
    - Dense Vector Search

    正式 RAG 尚未切换到这里。
    """

    CONTENT_MAX_BYTES = 16384

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        metric_type: str = "COSINE",
        timeout: float = 30.0,
    ) -> None:

        self.uri = str(uri).strip()
        self.collection_name = str(
            collection_name
        ).strip()

        self.metric_type = (
            str(metric_type)
            .strip()
            .upper()
        )

        self.timeout = float(timeout)

        if not self.uri:
            raise ValueError(
                "Milvus uri 不能为空"
            )

        if not self.collection_name:
            raise ValueError(
                "Milvus collection_name 不能为空"
            )

        if self.metric_type not in {
            "COSINE",
            "IP",
            "L2",
        }:
            raise ValueError(
                "Milvus metric_type 只支持 "
                "COSINE / IP / L2"
            )

        self.client = MilvusClient(
            uri=self.uri
        )

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
    def _chunk_id(
        cls,
        document: Document,
    ) -> str:

        payload = json.dumps(
            {
                "page_content":
                    str(
                        document.page_content
                    ),

                "metadata":
                    cls._json_safe_metadata(
                        document.metadata
                    ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    def recreate_collection(
        self,
        *,
        dimension: int,
    ) -> None:

        dimension = int(dimension)

        if dimension <= 1:
            raise ValueError(
                "Embedding dimension 必须大于 1"
            )

        collections = set(
            self.client.list_collections()
        )

        if (
            self.collection_name
            in collections
        ):
            self.client.drop_collection(
                collection_name=(
                    self.collection_name
                ),
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
            max_length=64,
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

        index_params = (
            self.client
            .prepare_index_params()
        )

        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

        self.client.create_collection(
            collection_name=(
                self.collection_name
            ),
            schema=schema,
            index_params=index_params,
            timeout=self.timeout,
        )

    def insert_documents(
        self,
        *,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> int:

        if len(documents) != len(vectors):
            raise ValueError(
                "documents 与 vectors 数量不一致"
            )

        rows = []

        for document, vector in zip(
            documents,
            vectors,
        ):
            content = str(
                document.page_content
            )

            content_bytes = len(
                content.encode("utf-8")
            )

            if (
                content_bytes
                > self.CONTENT_MAX_BYTES
            ):
                raise ValueError(
                    "Chunk 文本超过 Milvus VARCHAR "
                    f"上限：{content_bytes} bytes"
                )

            rows.append(
                {
                    "chunk_id":
                        self._chunk_id(
                            document
                        ),

                    "content":
                        content,

                    "metadata":
                        self._json_safe_metadata(
                            document.metadata
                        ),

                    "vector":
                        [
                            float(value)
                            for value
                            in vector
                        ],
                }
            )

        if not rows:
            return 0

        self.client.insert(
            collection_name=(
                self.collection_name
            ),
            data=rows,
            timeout=self.timeout,
        )

        return len(rows)

    def flush(self) -> None:
        self.client.flush_all(
            timeout=self.timeout
        )

    def stats(self) -> dict:
        return (
            self.client
            .get_collection_stats(
                collection_name=(
                    self.collection_name
                ),
                timeout=self.timeout,
            )
        )

    def list_documents(
        self,
    ) -> list[Document]:
        """
        读取当前 Collection 的全部 Child Document，
        供本地 BM25 建索引。

        ⑦-1B 只用于当前小型知识库。
        当行数达到 16384 以上时，
        后续应改成 query_iterator / Milvus 原生 BM25，
        不在这里偷偷截断数据。
        """

        stats = self.stats()

        row_count = int(
            stats.get(
                "row_count",
                0,
            )
        )

        if row_count <= 0:
            return []

        if row_count >= 16384:
            raise RuntimeError(
                "Milvus Collection 已达到 16384 条以上，"
                "⑦-1B 的本地 BM25 全量读取方案停止使用。"
            )

        rows = self.client.query(
            collection_name=(
                self.collection_name
            ),
            filter="",
            output_fields=[
                "chunk_id",
                "content",
                "metadata",
            ],
            limit=row_count,
            timeout=self.timeout,
        )

        rows = sorted(
            rows,
            key=lambda item:
                str(
                    item.get(
                        "chunk_id",
                        "",
                    )
                ),
        )

        documents = []

        for row in rows:

            metadata = (
                row.get(
                    "metadata"
                )
                or {}
            )

            documents.append(
                Document(
                    page_content=str(
                        row.get(
                            "content",
                            "",
                        )
                    ),
                    metadata=dict(
                        metadata
                    ),
                )
            )

        return documents


    def search(
        self,
        *,
        query_vector: list[float],
        k: int = 5,
    ) -> list[MilvusSearchResult]:

        k = max(
            1,
            int(k),
        )

        raw = self.client.search(
            collection_name=(
                self.collection_name
            ),
            data=[
                [
                    float(value)
                    for value
                    in query_vector
                ]
            ],
            anns_field="vector",
            limit=k,
            output_fields=[
                "content",
                "metadata",
            ],
            search_params={
                "metric_type":
                    self.metric_type,
                "params": {},
            },
            timeout=self.timeout,
        )

        if not raw:
            return []

        results = []

        for hit in raw[0]:

            entity = (
                hit.get("entity")
                or {}
            )

            metadata = (
                entity.get("metadata")
                or {}
            )

            document = Document(
                page_content=str(
                    entity.get(
                        "content",
                        "",
                    )
                ),
                metadata=dict(
                    metadata
                ),
            )

            results.append(
                MilvusSearchResult(
                    document=document,
                    score=float(
                        hit.get(
                            "distance",
                            0.0,
                        )
                    ),
                    chunk_id=str(
                        hit.get(
                            "id",
                            "",
                        )
                    ),
                )
            )

        return results
