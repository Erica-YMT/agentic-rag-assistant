from __future__ import annotations

from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi
import time
from observability import record_rag_stage, record_rag_result
from langsmith import traceable


@dataclass
class RetrievalResult:
    document: object
    fusion_score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_distance: float | None = None
    bm25_score: float | None = None

    @property
    def retrieval_type(self) -> str:
        if (
            self.vector_rank is not None
            and self.bm25_rank is not None
        ):
            return "Hybrid"

        if self.vector_rank is not None:
            return "Embedding"

        return "BM25"




# HYBRID_RETRIEVAL_TIMING_V1
def _measure_rag_stage(label):
    """统计 RAG 子阶段耗时。"""

    def decorator(func):

        def wrapper(*args, **kwargs):

            start_time = (
                time.perf_counter()
            )

            status = "success"

            try:
                return func(
                    *args,
                    **kwargs
                )

            except Exception:
                status = "error"
                raise

            finally:
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                record_rag_stage(
                    label,
                    elapsed,
                )

                record_rag_result(
                    label,
                    status,
                )

                print(
                    "[Timing] "
                    f"{label}："
                    f"{elapsed:.3f} 秒"
                )

        return wrapper

    return decorator
# HYBRID_RETRIEVAL_TIMING_V1_END


class HybridRetriever:
    """
    BM25 + FAISS 混合检索器。

    流程：
    1. FAISS 做语义检索；
    2. BM25 做关键词检索；
    3. 使用 RRF 根据排名融合；
    4. 返回融合后的 Top-K。
    """

    def __init__(
        self,
        vector_store,
        score_threshold: float,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.score_threshold = float(
            score_threshold
        )

        self.candidate_k = max(
            1,
            int(candidate_k)
        )

        self.rrf_k = max(
            1,
            int(rrf_k)
        )

        self.documents = (
            self._load_documents()
        )

        if not self.documents:
            raise RuntimeError(
                "无法从 FAISS 索引中读取文档，"
                "不能创建 BM25 索引。"
            )

        tokenized_corpus = [
            self._tokenize(
                document.page_content
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        print(
            "✅ Hybrid Retriever 已加载："
            f"{len(self.documents)} 个 Chunk"
        )


    # =====================================================
    # 中文分词
    # =====================================================

    @staticmethod
    def _tokenize(
        text: str
    ) -> list[str]:

        text = str(
            text or ""
        ).strip().lower()

        if not text:
            return []

        return [
            token.strip()
            for token
            in jieba.cut_for_search(
                text
            )
            if token.strip()
        ]


    # =====================================================
    # 从现有 FAISS docstore 读取全部 Chunk
    # 不重新加载原始 PDF
    # =====================================================

    def _load_documents(
        self
    ) -> list:

        documents = []

        index_mapping = getattr(
            self.vector_store,
            "index_to_docstore_id",
            {}
        )

        docstore = getattr(
            self.vector_store,
            "docstore",
            None
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
                "page_content"
            ):
                documents.append(
                    document
                )

        return documents


    # =====================================================
    # 文档唯一标识
    # =====================================================

    @staticmethod
    def _document_key(
        document
    ) -> tuple:

        metadata = (
            document.metadata
            or {}
        )

        source = (
            metadata.get(
                "relative_path"
            )
            or metadata.get(
                "source"
            )
            or metadata.get(
                "file_name"
            )
            or ""
        )

        page = metadata.get(
            "page"
        )

        return (
            str(source),
            str(page),
            document.page_content
        )


    # =====================================================
    # RRF 分数
    # =====================================================

    def _rrf_score(
        self,
        rank: int
    ) -> float:

        return (
            1.0
            /
            (
                self.rrf_k
                + rank
            )
        )


    # =====================================================
    # 混合检索
    # =====================================================

    @traceable(name="Hybrid Retrieval", run_type="retriever", tags=["rag", "hybrid"])
    @_measure_rag_stage("Hybrid Retrieval")
    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[RetrievalResult]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        k = max(
            1,
            int(k)
        )

        candidate_k = max(
            self.candidate_k,
            k * 3
        )


        # =================================================
        # 1. FAISS Embedding 检索
        # =================================================

        vector_results = (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=candidate_k
            )
        )


        # =================================================
        # 2. BM25 关键词检索
        # =================================================

        query_tokens = (
            self._tokenize(
                query
            )
        )

        bm25_scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )

        bm25_indexes = sorted(
            range(
                len(bm25_scores)
            ),
            key=lambda index:
                float(
                    bm25_scores[index]
                ),
            reverse=True
        )[:candidate_k]


        # =================================================
        # 3. RRF 融合
        # =================================================

        merged = {}


        # ---------- Embedding ----------
        vector_rank = 0

        for document, distance in vector_results:

            distance = float(
                distance
            )

            # 沿用原项目的相关性阈值
            if (
                distance
                > self.score_threshold
            ):
                continue

            vector_rank += 1

            key = (
                self._document_key(
                    document
                )
            )

            if key not in merged:
                merged[key] = RetrievalResult(
                    document=document,
                    fusion_score=0.0
                )

            item = merged[key]

            item.vector_rank = (
                vector_rank
            )

            item.vector_distance = (
                distance
            )

            item.fusion_score += (
                self._rrf_score(
                    vector_rank
                )
            )


        # ---------- BM25 ----------
        bm25_rank = 0

        for document_index in bm25_indexes:

            score = float(
                bm25_scores[
                    document_index
                ]
            )

            # 完全没有关键词命中时不加入
            if score <= 0:
                continue

            bm25_rank += 1

            document = (
                self.documents[
                    document_index
                ]
            )

            key = (
                self._document_key(
                    document
                )
            )

            if key not in merged:
                merged[key] = RetrievalResult(
                    document=document,
                    fusion_score=0.0
                )

            item = merged[key]

            item.bm25_rank = (
                bm25_rank
            )

            item.bm25_score = (
                score
            )

            item.fusion_score += (
                self._rrf_score(
                    bm25_rank
                )
            )


        # =================================================
        # 4. 按融合分数排序
        # =================================================

        results = sorted(
            merged.values(),
            key=lambda item:
                item.fusion_score,
            reverse=True
        )

        return results[:k]
