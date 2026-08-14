from app.core.stream_events import event_print as print
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from .retriever import HybridRetriever
from .vector_backends import (
    FAISSBackend,
    MilvusBackend,
)
from .reranker import CrossEncoderReranker
from .auto_merger import AutoMerger
from .corrective_rag import CorrectiveRAGController
from .rag_graph import ComplexRAGController


# =========================
# 基础配置
# =========================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# FAISS 距离越小，通常表示越相关。
# 当前知识库的有效问题距离约为 0.79，
# 因此先使用 1.0 作为初始阈值。
DEFAULT_SCORE_THRESHOLD = 1.0

# VECTOR_BACKEND_PHASE1B_V1


class KnowledgeBase:
    """
    统一知识库入口。

    Vector Backend 可选：
    - faiss
    - milvus

    上层 Auto-Merging / Reranker / Corrective RAG
    不感知底层向量数据库。
    """

    def __init__(
        self,
        model_dir: str,
        index_path: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        vector_backend_name: str = "faiss",
        milvus_settings: dict | None = None,
    ):
        self.score_threshold = float(
            score_threshold
        )

        self.vector_backend_name = (
            str(
                vector_backend_name
            )
            .strip()
            .lower()
        )

        # 查询端必须和建库端使用相同的
        # Embedding 模型与归一化配置。
        self.embedding_model = (
            HuggingFaceEmbeddings(
                model_name=model_dir,
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )
        )

        if (
            self.vector_backend_name
            == "faiss"
        ):

            self.vector_backend = (
                FAISSBackend.load(
                    index_path=index_path,
                    embedding_model=(
                        self.embedding_model
                    ),
                    score_threshold=(
                        self.score_threshold
                    ),
                )
            )

            # 保留 self.db，兼容旧调试/测试代码。
            self.db = (
                self.vector_backend.db
            )

        elif (
            self.vector_backend_name
            == "milvus"
        ):

            self.vector_backend = (
                MilvusBackend(
                    embedding_model=(
                        self.embedding_model
                    ),
                    score_threshold=(
                        self.score_threshold
                    ),
                    settings=(
                        milvus_settings
                        or {}
                    ),
                )
            )

            self.db = (
                self.vector_backend.store
            )

        else:
            raise ValueError(
                "[vector_store].backend "
                "只支持 faiss / milvus"
            )

        # BM25 + Embedding 混合检索器
        self.retriever = HybridRetriever(
            vector_backend=(
                self.vector_backend
            ),
        )

        # =========================
        # Reranker 配置
        # =========================

        # =========================
        # Parent-Child Auto-Merging
        # =========================

        auto_merge_config = (
            config.get(
                "auto_merge",
                {},
            )
        )

        self.auto_merger = (
            AutoMerger(
                index_path=index_path,
                settings=(
                    auto_merge_config
                ),
            )
        )

        reranker_config = config.get(
            "reranker",
            {}
        )

        self.reranker_enabled = bool(
            reranker_config.get(
                "enabled",
                True
            )
        )

        self.reranker_candidate_k = int(
            reranker_config.get(
                "candidate_k",
                10
            )
        )

        self.reranker_top_k = int(
            reranker_config.get(
                "top_k",
                3
            )
        )

        if self.reranker_enabled:

            self.reranker = (
                CrossEncoderReranker(
                    model_name_or_path=str(
                        reranker_config.get(
                            "model",
                            "BAAI/bge-reranker-base"
                        )
                    ),
                    max_length=int(
                        reranker_config.get(
                            "max_length",
                            512
                        )
                    ),
                )
            )

        else:
            self.reranker = None

    @staticmethod
    def _get_source_info(
        metadata: dict
    ) -> tuple[str, str]:
        """从 metadata 中读取文件名和 PDF 页码。"""

        metadata = metadata or {}

        file_name = metadata.get(
            "file_name"
        )

        if not file_name:
            source = (
                metadata.get("relative_path")
                or metadata.get("source")
                or "未知来源"
            )

            file_name = Path(
                str(source)
            ).name

        page = metadata.get(
            "page"
        )

        if isinstance(page, int):
            page_text = f"第 {page + 1} 页"
        else:
            page_text = ""

        return str(file_name), page_text

    def search(
        self,
        query: str,
        k: int = 5
    ) -> str:
        """
        检索知识库。

        只有 FAISS 距离不超过相关性阈值的文档，
        才会返回给 Agent。
        """

        if not isinstance(query, str):
            return (
                "知识库检索失败："
                "query 必须是字符串。"
            )

        query = query.strip()

        if not query:
            return (
                "知识库检索失败："
                "查询内容不能为空。"
            )

        try:
            k = int(k)
        except (
            TypeError,
            ValueError
        ):
            return (
                "知识库检索失败："
                "k 必须是整数。"
            )

        if k <= 0:
            return (
                "知识库检索失败："
                "k 必须大于 0。"
            )

        # 第一阶段：
        # BM25 + Embedding 先召回更多候选
        candidate_k = max(
            int(k),
            self.reranker_candidate_k
        )

        retrieval_results = (
            self.retriever.search(
                query=query,
                k=candidate_k,
            )
        )

        if not retrieval_results:
            return (
                "当前知识库没有检索到足够相关的资料，"
                "无法根据知识库回答该问题。"
            )

        # ==========================================
        # Parent-Child Auto-Merging
        # ==========================================

        retrieval_results = (
            self.auto_merger.merge(
                retrieval_results
            )
        )

        # 第二阶段：
        # Cross-Encoder Reranker 精排
        if self.reranker is not None:

            final_results = (
                self.reranker.rerank(
                    query=query,
                    candidates=retrieval_results,
                    top_k=min(
                        int(k),
                        self.reranker_top_k
                    ),
                )
            )

        else:

            final_results = [
                (
                    item,
                    None
                )
                for item
                in retrieval_results[:k]
            ]


        results = [
            "以下是知识库混合检索 + Reranker 结果："
        ]

        for index, (
            item,
            reranker_score
        ) in enumerate(
            final_results,
            start=1
        ):
            document = (
                item.document
            )

            metadata = (
                document.metadata
                or {}
            )

            file_name, page_text = (
                self._get_source_info(
                    metadata
                )
            )

            if page_text:
                source_text = (
                    f"{file_name}，"
                    f"{page_text}"
                )
            else:
                source_text = (
                    file_name
                )

            content = (
                document
                .page_content
                .strip()
            )

            if reranker_score is None:
                score_text = ""
            else:
                score_text = (
                    f"Reranker 分数："
                    f"{reranker_score:.4f}\n"
                )

            result = (
                f"[资料 {index}]\n"
                f"来源：{source_text}\n"
                f"第一阶段检索："
                f"{item.retrieval_type}\n"
                f"{score_text}"
                f"内容：\n"
                f"{content}"
            )

            results.append(
                result
            )

        return "\n\n".join(
            results
        )


# =========================
# 默认知识库实例
# =========================

_default_knowledge_base = None
_default_top_k = 3


def _resolve_project_path(
    path_value,
    config_name
):
    """
    将配置中的路径转换为绝对路径。

    相对路径以项目根目录为基准。
    """

    if not path_value:
        raise ValueError(
            "config.toml 中缺少 "
            f"{config_name}"
        )

    path = Path(
        str(path_value)
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


def get_default_knowledge_base():
    """
    根据 config.toml 创建默认知识库。

    第一次调用时加载模型和索引，
    后续调用复用已经创建的对象。
    """

    global _default_knowledge_base
    global _default_top_k

    if _default_knowledge_base is not None:
        return _default_knowledge_base

    embedding_config = config.get(
        "embedding",
        {}
    )

    vector_store_config = config.get(
        "vector_store",
        {}
    )

    vector_backend_name = str(
        vector_store_config.get(
            "backend",
            "faiss",
        )
    ).strip().lower()

    if vector_backend_name not in {
        "faiss",
        "milvus",
    }:
        raise ValueError(
            "[vector_store].backend "
            "只支持 faiss / milvus"
        )

    milvus_config = config.get(
        "milvus",
        {}
    )

    model_path = _resolve_project_path(
        embedding_config.get(
            "model_path"
        ),
        "[embedding].model_path"
    )

    index_path = _resolve_project_path(
        embedding_config.get(
            "index_path",
            "faiss_index"
        ),
        "[embedding].index_path"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "没有找到 Embedding 模型："
            f"{model_path}"
        )

    if not index_path.exists():
        raise FileNotFoundError(
            "没有找到本地索引目录（Auto-Merging 仍需要 parent_store.json）："
            f"{index_path}"
        )

    try:
        top_k = int(
            embedding_config.get(
                "top_k",
                3
            )
        )
    except (
        TypeError,
        ValueError
    ) as exc:
        raise ValueError(
            "[embedding].top_k 必须是整数"
        ) from exc

    if not 1 <= top_k <= 20:
        raise ValueError(
            "[embedding].top_k "
            "必须在 1 到 20 之间"
        )

    try:
        score_threshold = float(
            embedding_config.get(
                "score_threshold",
                DEFAULT_SCORE_THRESHOLD
            )
        )
    except (
        TypeError,
        ValueError
    ) as exc:
        raise ValueError(
            "[embedding].score_threshold "
            "必须是数字"
        ) from exc

    _default_top_k = top_k

    _default_knowledge_base = (
        KnowledgeBase(
            model_dir=str(
                model_path
            ),
            index_path=str(
                index_path
            ),
            score_threshold=(
                score_threshold
            ),
            vector_backend_name=(
                vector_backend_name
            ),
            milvus_settings=(
                milvus_config
            ),
        )
    )

    return _default_knowledge_base


def reload_default_knowledge_base():
    """
    重新加载默认知识库。

    用于知识库索引重建完成后，
    丢弃旧的 KnowledgeBase 实例并按配置重新加载。
    """

    global _default_knowledge_base

    _default_knowledge_base = None

    return get_default_knowledge_base()


def _search_knowledge_single(
    query
):
    """
    Agent 使用的知识库搜索工具。

    Corrective RAG 完整流程：

    第一次检索
    -> Grade #1
    -> 必要时 Query Rewrite
    -> 第二次检索
    -> Grade #2
    -> 返回证据或明确提示证据不足。
    """

    knowledge_base = (
        get_default_knowledge_base()
    )

    # ==========================================
    # 第一次 Hybrid Retrieval
    # ==========================================

    first_result = (
        knowledge_base.search(
            query,
            k=_default_top_k
        )
    )

    controller = getattr(
        _search_knowledge_single,
        "_corrective_controller",
        None,
    )

    if controller is None:
        controller = (
            CorrectiveRAGController()
        )

        _search_knowledge_single._corrective_controller = (
            controller
        )

    # 配置关闭时，保持原始 RAG 行为
    if not controller.enabled:
        return first_result


    # ==========================================
    # Grade #1
    # ==========================================

    first_grade = (
        controller.grade(
            question=query,
            retrieval_result=first_result,
        )
    )

    print(
        "[CorrectiveRAG] Grade #1："
        f"sufficient={first_grade.sufficient} | "
        f"confidence={first_grade.confidence:.2f} | "
        f"reason={first_grade.reason}"
    )

    if first_grade.sufficient:
        print(
            "[CorrectiveRAG] "
            "首次检索证据足够，直接返回。"
        )

        return first_result


    # ==========================================
    # Query Rewrite
    # ==========================================

    rewritten_query = (
        controller.rewrite(
            question=query,
            retrieval_result=first_result,
            grade_reason=(
                first_grade.reason
            ),
        )
    )

    print(
        "[CorrectiveRAG] Query Rewrite："
        f"{query} -> {rewritten_query}"
    )


    # 如果重写失败或没有变化，
    # 不做完全相同的重复检索。
    if (
        not rewritten_query
        or rewritten_query.strip()
        == str(query).strip()
    ):

        print(
            "[CorrectiveRAG] "
            "Query Rewrite 没有产生有效新查询。"
        )

        return (
            "【知识库证据不足】\n"
            f"原始问题：{query}\n"
            f"原因：{first_grade.reason}\n\n"
            "首次检索证据不足，"
            "并且查询重写没有产生有效的新查询。"
            "当前知识库无法可靠支持完整回答。"
            "请明确告知用户知识库证据不足，"
            "不要猜测或编造答案。"
        )


    # ==========================================
    # 第二次 Hybrid Retrieval
    # ==========================================

    second_result = (
        knowledge_base.search(
            rewritten_query,
            k=_default_top_k,
        )
    )

    print(
        "[CorrectiveRAG] "
        "已完成第二次检索。"
    )


    # ==========================================
    # Grade #2
    # ==========================================

    second_grade = (
        controller.grade(
            question=query,
            retrieval_result=second_result,
        )
    )

    print(
        "[CorrectiveRAG] Grade #2："
        f"sufficient={second_grade.sufficient} | "
        f"confidence={second_grade.confidence:.2f} | "
        f"reason={second_grade.reason}"
    )


    # 第二次证据足够
    if second_grade.sufficient:

        print(
            "[CorrectiveRAG] "
            "第二次检索证据足够，"
            "返回第二次检索结果。"
        )

        return second_result


    # ==========================================
    # 第二次仍然不足：停止
    # ==========================================

    print(
        "[CorrectiveRAG] "
        "第二次检索后证据仍不足，"
        "停止继续重写。"
    )

    return (
        "【知识库证据不足】\n"
        "系统已经执行："
        "首次检索 → 证据评判 → "
        "查询重写 → 第二次检索 → "
        "再次证据评判。\n\n"
        f"原始问题：{query}\n"
        f"重写查询：{rewritten_query}\n"
        f"证据不足原因：{second_grade.reason}\n\n"
        "当前知识库没有足够可靠的资料"
        "支持完整回答。"
        "请明确告知用户知识库证据不足，"
        "不要根据不完整资料猜测或编造答案。"
    )


# ==========================================================
# Complex RAG
# ==========================================================

_complex_rag_controller = None


def _raw_search_knowledge(
    query
):
    """
    只执行原始 Hybrid Retrieval。

    供 LangGraph 子问题 Worker 使用。
    不进行额外 LLM Grade，避免多个 Worker
    同时产生过多模型请求。
    """

    knowledge_base = (
        get_default_knowledge_base()
    )

    return knowledge_base.search(
        query,
        k=_default_top_k,
    )


def search_knowledge(
    query
):
    """
    Agent 统一知识库入口。

    简单问题：
        原 Corrective RAG

    复杂问题：
        Complexity
        -> Question Decomposition
        -> LangGraph Send 并行检索
        -> Merge
        -> Coverage Grade
    """

    global _complex_rag_controller

    if (
        _complex_rag_controller
        is None
    ):
        _complex_rag_controller = (
            ComplexRAGController(
                raw_retrieve=(
                    _raw_search_knowledge
                ),
                simple_retrieve=(
                    _search_knowledge_single
                ),
            )
        )

    return (
        _complex_rag_controller.run(
            query
        )
    )
