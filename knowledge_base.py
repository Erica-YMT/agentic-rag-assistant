from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from retriever import HybridRetriever
from reranker import CrossEncoderReranker


# =========================
# 基础配置
# =========================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


# FAISS 距离越小，通常表示越相关。
# 当前知识库的有效问题距离约为 0.79，
# 因此先使用 1.0 作为初始阈值。
DEFAULT_SCORE_THRESHOLD = 1.0


class KnowledgeBase:
    """负责加载和检索本地 FAISS 知识库。"""

    def __init__(
        self,
        model_dir: str,
        index_path: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.score_threshold = float(
            score_threshold
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

        # 只能加载自己生成或可信来源的索引文件。
        self.db = FAISS.load_local(
            index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

        # BM25 + Embedding 混合检索器
        self.retriever = HybridRetriever(
            vector_store=self.db,
            score_threshold=self.score_threshold,
        )

        # =========================
        # Reranker 配置
        # =========================

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
            "没有找到 FAISS 索引："
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
        )
    )

    return _default_knowledge_base


def reload_default_knowledge_base():
    """
    重新加载默认知识库。

    用于知识库索引重建完成后，
    丢弃旧的 KnowledgeBase 实例并加载最新 FAISS 索引。
    """

    global _default_knowledge_base

    _default_knowledge_base = None

    return get_default_knowledge_base()


def search_knowledge(
    query
):
    """
    提供给 Agent 使用的知识库搜索工具。

    这个函数是工具层与 KnowledgeBase 类
    之间的简单适配入口。
    """

    knowledge_base = (
        get_default_knowledge_base()
    )

    return knowledge_base.search(
        query,
        k=_default_top_k
    )
