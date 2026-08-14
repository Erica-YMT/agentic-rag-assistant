from __future__ import annotations

import torch
from modelscope import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import time
from app.core.observability import record_rag_stage, record_rag_result
from langsmith import traceable




# RERANKER_TIMING_V1
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
# RERANKER_TIMING_V1_END


class CrossEncoderReranker:
    """
    使用 Cross-Encoder 对第一阶段检索结果重新排序。

    输入：
        query + 多个候选文档

    输出：
        按相关性重新排序后的结果
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
    ):
        self.model_name_or_path = (
            model_name_or_path
        )

        self.max_length = int(
            max_length
        )

        print(
            "正在加载 Reranker：",
            self.model_name_or_path
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name_or_path
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_name_or_path
            )
        )

        self.model.eval()

        print(
            "✅ Reranker 加载完成"
        )


    @traceable(name="Reranker", run_type="chain", tags=["rag", "reranker"])
    @_measure_rag_stage("Reranker")
    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 3,
    ):
        """
        candidates:
            HybridRetriever 返回的
            RetrievalResult 列表。

        返回：
            [
                (RetrievalResult, reranker_score),
                ...
            ]
        """

        if not candidates:
            return []

        pairs = [
            [
                query,
                item.document.page_content
            ]
            for item in candidates
        ]

        inputs = (
            self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
        )

        with torch.no_grad():

            scores = (
                self.model(
                    **inputs,
                    return_dict=True
                )
                .logits
                .view(-1)
                .float()
                .cpu()
                .tolist()
            )

        ranked_results = sorted(
            zip(
                candidates,
                scores
            ),
            key=lambda item:
                item[1],
            reverse=True
        )

        return ranked_results[
            :max(
                1,
                int(top_k)
            )
        ]
