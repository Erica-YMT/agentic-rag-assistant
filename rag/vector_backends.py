from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

# VECTOR_BACKEND_PHASE1B_V1


@dataclass
class VectorSearchHit:
    """
    统一向量检索结果。

    score:
        统一使用“越大越相关”的相似度语义。

    legacy_distance:
        仅 FAISS 兼容使用，保留原来的 L2 distance，
        避免已有调试/评测代码失去原始距离信息。
    """

    document: object
    score: float
    legacy_distance: float | None = None


class FAISSBackend:
    """FAISS 向量后端适配器。"""

    backend_name = "faiss"

    def __init__(
        self,
        *,
        db,
        score_threshold: float,
    ) -> None:

        self.db = db
        self.score_threshold = float(
            score_threshold
        )

    @classmethod
    def load(
        cls,
        *,
        index_path: str,
        embedding_model,
        score_threshold: float,
    ):
        db = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )

        return cls(
            db=db,
            score_threshold=score_threshold,
        )

    @classmethod
    def from_existing_store(
        cls,
        *,
        vector_store,
        score_threshold: float,
    ):
        """
        兼容旧测试/旧调用：
        HybridRetriever(vector_store=..., score_threshold=...)
        """

        return cls(
            db=vector_store,
            score_threshold=score_threshold,
        )

    @staticmethod
    def _distance_to_similarity(
        distance: float,
    ) -> float:
        """
        当前项目的 Embedding 在建库与查询端都做 L2 normalize，
        而原 FAISS 默认使用 squared L2 distance。

        对单位向量：
            squared_l2 = 2 - 2 * cosine

        因此：
            cosine = 1 - squared_l2 / 2
        """

        return (
            1.0
            - float(distance) / 2.0
        )

    def list_documents(
        self,
    ) -> list:

        documents = []

        index_mapping = getattr(
            self.db,
            "index_to_docstore_id",
            {},
        )

        docstore = getattr(
            self.db,
            "docstore",
            None,
        )

        if docstore is None:
            return documents

        for vector_index in sorted(
            index_mapping
        ):
            document_id = (
                index_mapping[
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

        return documents

    def search(
        self,
        *,
        query: str,
        k: int,
    ) -> list[VectorSearchHit]:

        raw_results = (
            self.db
            .similarity_search_with_score(
                query,
                k=max(1, int(k)),
            )
        )

        results = []

        for document, distance in raw_results:

            distance = float(
                distance
            )

            # 完整保留旧 FAISS 阈值语义：
            # distance 越小越相关。
            if (
                distance
                > self.score_threshold
            ):
                continue

            results.append(
                VectorSearchHit(
                    document=document,
                    score=(
                        self._distance_to_similarity(
                            distance
                        )
                    ),
                    legacy_distance=distance,
                )
            )

        return results


class MilvusBackend:
    """Milvus Dense Vector 后端适配器。"""

    backend_name = "milvus"

    def __init__(
        self,
        *,
        embedding_model,
        score_threshold: float,
        settings: dict | None,
    ) -> None:

        settings = dict(
            settings or {}
        )

        metric_type = str(
            settings.get(
                "metric_type",
                "COSINE",
            )
        ).strip().upper()

        # ⑦-1B 先只支持 COSINE，
        # 避免把 L2/IP 的阈值方向混在一起。
        if metric_type != "COSINE":
            raise ValueError(
                "⑦-1B MilvusBackend 目前只支持 "
                "metric_type = COSINE"
            )

        self.embedding_model = (
            embedding_model
        )

        self.score_threshold = float(
            score_threshold
        )

        if not (
            0.0
            <= self.score_threshold
            <= 4.0
        ):
            raise ValueError(
                "当前 FAISS score_threshold "
                "必须在 0 到 4 之间，"
                "才能映射到单位向量的 COSINE 阈值。"
            )

        # 当前项目的向量都 normalize_embeddings=True。
        # FAISS squared L2 threshold:
        # d = 2 - 2*cos
        # => cos = 1 - d/2
        self.min_similarity = (
            1.0
            - self.score_threshold / 2.0
        )

        # 重要：
        # pymilvus 仍然是 lazy import。
        # 这样正式 API 目前即使还没成功 rebuild 镜像，
        # 只要 backend=faiss 就不会因为缺少 pymilvus 启动失败。
        from .milvus_store import MilvusStore

        self.store = MilvusStore(
            uri=str(
                settings.get(
                    "uri",
                    "http://milvus-standalone:19530",
                )
            ),
            collection_name=str(
                settings.get(
                    "collection_name",
                    "agentic_rag_chunks",
                )
            ),
            metric_type=metric_type,
            timeout=float(
                settings.get(
                    "timeout",
                    30.0,
                )
            ),
        )

        # An empty or legacy collection must never look like a healthy backend:
        # otherwise retrieval silently returns no evidence and bypasses the
        # configured FAISS fallback. Full synchronization is the explicit
        # operation that creates/migrates the collection.
        if not self.store.collection_exists():
            raise RuntimeError(
                "Milvus Collection 不存在，请先执行完整同步："
                f"{self.store.collection_name}"
            )
        if not self.store.schema_compatible():
            raise RuntimeError(
                "Milvus Collection Schema 不兼容，请先执行完整同步迁移："
                f"{self.store.collection_name}"
            )

    def list_documents(
        self,
    ) -> list:
        return (
            self.store
            .list_documents()
        )

    def load_parent_records(
        self,
    ) -> dict:
        """Milvus 模式下直接从同一 Collection 读取 Parent records。"""
        return (
            self.store
            .list_parent_records()
        )

    def search(
        self,
        *,
        query: str,
        k: int,
    ) -> list[VectorSearchHit]:

        query_vector = (
            self.embedding_model
            .embed_query(
                str(query)
            )
        )

        raw_results = (
            self.store.search(
                query_vector=query_vector,
                k=max(1, int(k)),
            )
        )

        results = []

        for item in raw_results:

            similarity = float(
                item.score
            )

            # COSINE 越大越相关。
            if (
                similarity
                < self.min_similarity
            ):
                continue

            results.append(
                VectorSearchHit(
                    document=(
                        item.document
                    ),
                    score=similarity,
                    legacy_distance=None,
                )
            )

        return results
