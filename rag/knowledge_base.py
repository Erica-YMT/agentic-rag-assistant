from app.core.stream_events import event_print as print
from pathlib import Path

import torch
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
from .dynamic_modes import classify_mode


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
        shared_components=None,
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

        # 查询端必须和建库端使用相同的 Embedding 模型与归一化配置。
        # 私有知识库复用公共 KnowledgeBase 的模型对象，避免每个用户再加载一份。
        embedding_config = config.get(
            "embedding",
            {}
        )

        if shared_components is not None:
            self.embedding_model = shared_components.embedding_model
        else:
            embedding_device = str(
                embedding_config.get(
                    "device",
                    "auto",
                )
            ).strip().lower()

            if embedding_device not in {
                "auto",
                "cpu",
                "cuda",
            }:
                raise ValueError(
                    "[embedding].device "
                    "只支持 auto / cpu / cuda"
                )

            if embedding_device == "auto":
                resolved_embedding_device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )
            elif embedding_device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "[embedding].device=cuda，"
                        "但当前运行环境无法使用 CUDA"
                    )
                resolved_embedding_device = "cuda"
            else:
                resolved_embedding_device = "cpu"

            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_dir,
                model_kwargs={
                    "device": resolved_embedding_device,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
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

        milvus_parent_records = None
        if self.vector_backend_name == "milvus":
            milvus_parent_records = (
                self.vector_backend
                .load_parent_records()
            )

        self.auto_merger = (
            AutoMerger(
                index_path=index_path,
                settings=(
                    auto_merge_config
                ),
                parent_records=milvus_parent_records,
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

        if shared_components is not None:
            self.reranker = getattr(
                shared_components,
                "reranker",
                None,
            )
            self.reranker_enabled = bool(
                getattr(
                    shared_components,
                    "reranker_enabled",
                    False,
                )
            )

        elif self.reranker_enabled:

            try:
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

            except Exception as error:
                print(
                    "[Reranker] 加载失败，"
                    "已自动降级为 Hybrid Retrieval："
                    f"{error}"
                )
                self.reranker = None
                self.reranker_enabled = False

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

            try:
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

            except Exception as error:
                print(
                    "[Reranker] 推理失败，"
                    "本次请求自动降级为 Hybrid Retrieval："
                    f"{error}"
                )

                final_results = [
                    (
                        item,
                        None
                    )
                    for item
                    in retrieval_results[:k]
                ]

        else:

            final_results = [
                (
                    item,
                    None
                )
                for item
                in retrieval_results[:k]
            ]


        if self.reranker is not None:
            result_title = (
                "以下是知识库混合检索 + Reranker 结果："
            )
        else:
            result_title = (
                "以下是知识库混合检索结果："
            )

        results = [
            result_title
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
# 公共 / 私有知识库实例
# =========================

_default_knowledge_base = None
_private_knowledge_bases: dict[int, KnowledgeBase] = {}
_private_kb_cache_limit = 16
_default_top_k = 3


def _resolve_project_path(
    path_value,
    config_name
):
    """将配置中的路径转换为绝对路径；相对路径以项目根目录为基准。"""

    if not path_value:
        raise ValueError(
            "config.toml 中缺少 "
            f"{config_name}"
        )

    path = Path(str(path_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _vector_backend_name() -> str:
    backend = str(
        config.get("vector_store", {}).get("backend", "faiss")
    ).strip().lower()
    if backend not in {"faiss", "milvus"}:
        raise ValueError("[vector_store].backend 只支持 faiss / milvus")
    return backend


def _score_threshold() -> float:
    try:
        return float(
            config.get("embedding", {}).get(
                "score_threshold",
                DEFAULT_SCORE_THRESHOLD,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("[embedding].score_threshold 必须是数字") from exc


def _scoped_milvus_settings(user_id: int | None = None) -> dict:
    settings = dict(config.get("milvus", {}) or {})
    if user_id is None:
        return settings

    from .milvus_store import scoped_collection_name

    settings["collection_name"] = scoped_collection_name(
        str(settings.get("collection_name", "agentic_rag_chunks")),
        user_id=int(user_id),
    )
    return settings


def get_default_knowledge_base():
    """Create/cache the public KnowledgeBase."""

    global _default_knowledge_base
    global _default_top_k

    if _default_knowledge_base is not None:
        return _default_knowledge_base

    embedding_config = config.get("embedding", {})
    vector_backend_name = _vector_backend_name()

    model_path = _resolve_project_path(
        embedding_config.get("model_path"),
        "[embedding].model_path",
    )
    index_path = _resolve_project_path(
        embedding_config.get("index_path", "faiss_index"),
        "[embedding].index_path",
    )

    if not model_path.exists():
        raise FileNotFoundError(f"没有找到 Embedding 模型：{model_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"没有找到本地索引目录：{index_path}")

    try:
        top_k = int(embedding_config.get("top_k", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError("[embedding].top_k 必须是整数") from exc
    if not 1 <= top_k <= 20:
        raise ValueError("[embedding].top_k 必须在 1 到 20 之间")

    _default_top_k = top_k

    try:
        _default_knowledge_base = KnowledgeBase(
            model_dir=str(model_path),
            index_path=str(index_path),
            score_threshold=_score_threshold(),
            vector_backend_name=vector_backend_name,
            milvus_settings=_scoped_milvus_settings(),
        )
    except Exception as exc:
        fallback_enabled = bool(
            config.get("vector_store", {}).get("fallback_to_faiss", True)
        )
        if vector_backend_name != "milvus" or not fallback_enabled:
            raise

        print(
            "⚠️ Milvus 公共知识库加载失败，自动回退 FAISS："
            f"{exc}"
        )
        _default_knowledge_base = KnowledgeBase(
            model_dir=str(model_path),
            index_path=str(index_path),
            score_threshold=_score_threshold(),
            vector_backend_name="faiss",
            milvus_settings={},
        )

    return _default_knowledge_base


def get_user_knowledge_base(user_id: int | str):
    """
    Return the authenticated user's private KnowledgeBase when it exists.

    The private KB reuses the public KB's embedding/reranker objects, so it does
    not load another large model per user. The cache only keeps a small number
    of lightweight backend/BM25 objects.
    """

    user_id = int(user_id)
    if user_id <= 0:
        return None

    cached = _private_knowledge_bases.get(user_id)
    if cached is not None:
        return cached

    embedding_config = config.get("embedding", {})
    model_path = _resolve_project_path(
        embedding_config.get("model_path"),
        "[embedding].model_path",
    )
    base_index_path = _resolve_project_path(
        embedding_config.get("index_path", "faiss_index"),
        "[embedding].index_path",
    )
    private_index_path = base_index_path / "users" / str(user_id)

    if not (
        (private_index_path / "index.faiss").is_file()
        and (private_index_path / "index.pkl").is_file()
    ):
        return None

    public_kb = get_default_knowledge_base()
    requested_backend = _vector_backend_name()

    try:
        private_kb = KnowledgeBase(
            model_dir=str(model_path),
            index_path=str(private_index_path),
            score_threshold=_score_threshold(),
            vector_backend_name=requested_backend,
            milvus_settings=_scoped_milvus_settings(user_id),
            shared_components=public_kb,
        )
    except Exception as exc:
        fallback_enabled = bool(
            config.get("vector_store", {}).get("fallback_to_faiss", True)
        )
        if requested_backend != "milvus" or not fallback_enabled:
            raise

        print(
            f"⚠️ 用户 {user_id} 私有 Milvus 加载失败，自动回退 FAISS：{exc}"
        )
        private_kb = KnowledgeBase(
            model_dir=str(model_path),
            index_path=str(private_index_path),
            score_threshold=_score_threshold(),
            vector_backend_name="faiss",
            milvus_settings={},
            shared_components=public_kb,
        )

    if len(_private_knowledge_bases) >= _private_kb_cache_limit:
        oldest_user_id = next(iter(_private_knowledge_bases))
        _private_knowledge_bases.pop(oldest_user_id, None)

    _private_knowledge_bases[user_id] = private_kb
    return private_kb


def reload_default_knowledge_base():
    """Invalidate public and private KB caches after a public rebuild/update."""

    global _default_knowledge_base
    _default_knowledge_base = None
    _private_knowledge_bases.clear()
    return get_default_knowledge_base()


def reload_user_knowledge_base(user_id: int | str) -> None:
    """Invalidate one private KB without eagerly loading models/indexes."""

    _private_knowledge_bases.pop(int(user_id), None)


def _search_scoped_knowledge(
    query: str,
    *,
    user_id: int | None,
    k: int,
) -> str:
    """Search public + authenticated user's private KB with hard scope isolation."""

    parts = [
        "【公共知识库】\n"
        + get_default_knowledge_base().search(query, k=k)
    ]

    if user_id is not None:
        private_kb = get_user_knowledge_base(int(user_id))
        if private_kb is not None:
            parts.append(
                "【我的私有知识库】\n"
                + private_kb.search(query, k=k)
            )

    return "\n\n".join(parts)


def _search_knowledge_single(
    query,
    _user_id=None,
    _k=None,
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

    # ==========================================
    # 第一次 Public + Private Scoped Retrieval
    # ==========================================

    user_id = None if _user_id is None else int(_user_id)

    first_result = _search_scoped_knowledge(
        str(query),
        user_id=user_id,
        k=int(_k or _default_top_k),
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

    second_result = _search_scoped_knowledge(
        rewritten_query,
        user_id=user_id,
        k=int(_k or _default_top_k),
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

_complex_rag_controllers: dict[tuple[int, str], ComplexRAGController] = {}


def _raw_search_knowledge(
    query,
    _user_id=None,
    _k=None,
):
    """Raw scoped Hybrid Retrieval for Complex-RAG workers."""

    user_id = None if _user_id is None else int(_user_id)
    return _search_scoped_knowledge(
        str(query),
        user_id=user_id,
        k=int(_k or _default_top_k),
    )


def search_knowledge(
    query,
    _user_id=None,
):
    """
    Agent unified RAG entry.

    `_user_id` is injected by ToolExecutor from the authenticated session and is
    never exposed in the LLM tool schema. It controls access to the caller's
    hard-isolated private FAISS/Milvus scope.
    """

    user_id = None if _user_id is None else int(_user_id)
    mode = classify_mode(query)
    cache_key = (int(user_id or 0), mode.name)

    controller = _complex_rag_controllers.get(cache_key)
    if controller is None:
        if len(_complex_rag_controllers) >= _private_kb_cache_limit:
            oldest_key = next(iter(_complex_rag_controllers))
            _complex_rag_controllers.pop(oldest_key, None)

        controller = ComplexRAGController(
            raw_retrieve=(
                lambda subquery, uid=user_id:
                _raw_search_knowledge(
                    subquery,
                    _user_id=uid,
                    _k=mode.top_k,
                )
            ),
            simple_retrieve=(
                lambda simple_query, uid=user_id:
                _search_knowledge_single(
                    simple_query,
                    _user_id=uid,
                    _k=mode.top_k,
                )
            ),
        )
        _complex_rag_controllers[cache_key] = controller

    return f"【动态 RAG 模式：{mode.name}】\n{controller.run(query)}"
