from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from agent import Agent
from schemas import (
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    HealthResponse,
    RebuildKnowledgeResponse,
    SearchRequest,
    SearchResponse,
    ToolCallRecord,
)

from pathlib import Path

import tools as tools_module

from build_index import build_index
from config import config
from knowledge_base import KnowledgeBase

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期。

    服务启动时加载共享知识库；
    服务关闭时记录停止日志。
    """
    start_time = time.perf_counter()

    logger.info("正在预加载知识库……")

    try:
        if getattr(
            tools_module,
            "kb",
            None,
        ) is None:
            tools_module.kb = (
                create_knowledge_base()
            )

    except Exception:
        logger.exception(
            "知识库预加载失败"
        )
        raise

    elapsed_seconds = round(
        time.perf_counter() - start_time,
        2,
    )

    logger.info(
        "知识库预加载完成，耗时 %.2f 秒",
        elapsed_seconds,
    )

    yield

    logger.info(
        "Agentic RAG Assistant API 已停止"
    )


# 创建 FastAPI 应用
app = FastAPI(
    title="Agentic RAG Assistant API",
    description="基于 Agent、工具调用和 RAG 检索的问答接口",
    version="1.0.0",
    lifespan=lifespan,
)

# === Local web CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 每个 session_id 保存一个 Agent
_agents: dict[str, Agent] = {}

# 防止同一个会话同时执行多个请求，导致消息记录混乱
_session_locks = {}

# 保护上面两个字典
_registry_lock = Lock()

# 防止多个请求同时重建知识库
_knowledge_lock = Lock()


# 当前项目根目录，也就是 api.py 所在目录
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_value: str) -> Path:
    """
    把 config.toml 中的路径转换成绝对路径。

    绝对路径：直接使用。
    相对路径：以项目根目录为基准。
    """
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def create_knowledge_base() -> KnowledgeBase:
    """
    根据 config.toml 中的 [embedding] 配置，
    创建并返回一个新的 KnowledgeBase 对象。
    """
    embedding_config = config.get(
        "embedding",
        {},
    )

    model_path_value = embedding_config.get(
        "model_path"
    )

    if not model_path_value:
        raise ValueError(
            "config.toml 中缺少 "
            "[embedding].model_path"
        )

    model_path = resolve_project_path(
        str(model_path_value)
    )

    index_path = resolve_project_path(
        str(
            embedding_config.get(
                "index_path",
                "faiss_index",
            )
        )
    )

    score_threshold = float(
        embedding_config.get(
            "score_threshold",
            1.0,
        )
    )

    return KnowledgeBase(
        model_dir=str(model_path),
        index_path=str(index_path),
        score_threshold=score_threshold,
    )


def get_session_agent(session_id: str):
    """
    根据 session_id 获取 Agent。

    如果这个 session_id 第一次出现，就创建新的 Agent。
    """
    with _registry_lock:
        if session_id not in _agents:
            logger.info("创建新会话 Agent：%s", session_id)

            _agents[session_id] = Agent()
            _session_locks[session_id] = Lock()

        return _agents[session_id], _session_locks[session_id]


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="检查服务状态",
)
def health():
    """
    检查 FastAPI 服务是否已经正常启动。
    """
    return HealthResponse(
        status="ok",
        service="Agentic RAG Assistant API",
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="与 Agent 对话",
)
def chat(request: ChatRequest):
    """
    接收用户问题，调用现有 Agent，并返回最终回答。
    """
    start_time = time.perf_counter()

    agent, session_lock = get_session_agent(
        request.session_id
    )

    try:
        # 同一个会话一次只处理一个问题，
        # 防止聊天记录和工具调用记录混乱。
        with session_lock:
            answer = agent.run(
                session_id=request.session_id,
                question=request.question,
            )

            # 获取本轮调用过的工具名称
            called_tools = list(
                agent.get_called_tools() or []
            )

            # 获取本轮完整工具调用记录
            raw_call_records = list(
                agent.get_call_records() or []
            )

        # 把普通字典转换为 Pydantic 数据模型
        tool_calls = []

        for record in raw_call_records:
            tool_calls.append(
                ToolCallRecord(
                    tool_name=str(
                        record.get(
                            "tool_name",
                            "unknown",
                        )
                    ),
                    arguments=record.get(
                        "arguments",
                        {},
                    ) or {},
                    result=str(
                        record.get(
                            "result",
                            "",
                        )
                    ),
                )
            )

        # 计算整个 /chat 请求的总耗时
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.info(
            "会话完成：session_id=%s，耗时=%.2f 秒，工具=%s",
            request.session_id,
            elapsed_seconds,
            called_tools,
        )

        return ChatResponse(
            answer=str(answer),
            session_id=request.session_id,
            called_tools=called_tools,
            tool_calls=tool_calls,
            elapsed_seconds=elapsed_seconds,
        )

    except APITimeoutError as error:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.exception(
            "模型接口响应超时，session_id=%s，耗时=%.2f 秒",
            request.session_id,
            elapsed_seconds,
        )

        raise HTTPException(
            status_code=504,
            detail="模型响应超时，请稍后重试",
        ) from error

    except APIConnectionError as error:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.exception(
            "模型接口连接失败，session_id=%s，耗时=%.2f 秒",
            request.session_id,
            elapsed_seconds,
        )

        raise HTTPException(
            status_code=502,
            detail="暂时无法连接模型服务，请稍后重试",
        ) from error

    except RateLimitError as error:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.exception(
            "模型接口请求过多，session_id=%s，耗时=%.2f 秒",
            request.session_id,
            elapsed_seconds,
        )

        raise HTTPException(
            status_code=429,
            detail="模型服务当前请求较多，请稍后重试",
        ) from error

    except InternalServerError as error:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.exception(
            "上游模型服务异常，session_id=%s，耗时=%.2f 秒",
            request.session_id,
            elapsed_seconds,
        )

        raise HTTPException(
            status_code=503,
            detail="上游模型服务暂时异常，请稍后重试",
        ) from error

    except Exception as error:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.exception(
            "Agent 请求执行失败，session_id=%s，耗时=%.2f 秒",
            request.session_id,
            elapsed_seconds,
        )

        raise HTTPException(
            status_code=500,
            detail="Agent 运行失败，请查看服务端终端日志",
        ) from error

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="直接检索知识库",
)
def search(request: SearchRequest):
    """
    不调用大模型，直接执行 Embedding 和 FAISS 检索。
    """
    try:
        kb = getattr(
            tools_module,
            "kb",
            None,
        )

        if kb is None:
            raise RuntimeError(
                "知识库尚未加载"
            )

        result = kb.search(
            query=request.query,
            k=request.top_k,
        )

        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            result=str(result),
        )

    except Exception as error:
        logger.exception(
            "知识库检索失败，query=%s",
            request.query,
        )

        raise HTTPException(
            status_code=500,
            detail="知识库检索失败，请查看服务端日志",
        ) from error


@app.post(
    "/knowledge/rebuild",
    response_model=RebuildKnowledgeResponse,
    summary="重建并重新加载知识库",
)
def rebuild_knowledge():
    """
    根据 data/knowledge 中的文档重建 FAISS，
    并替换 FastAPI 当前使用的知识库对象。
    """
    start_time = time.perf_counter()

    try:
        with _knowledge_lock:
            logger.info(
                "开始重建知识库……"
            )

            # 第一步：在磁盘上重新生成 FAISS 索引
            build_index()

            # 第二步：重新加载新的索引
            new_kb = create_knowledge_base()

            # 第三步：替换 tools.py 当前使用的知识库
            tools_module.kb = new_kb

        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.info(
            "知识库重建完成，耗时 %.2f 秒",
            elapsed_seconds,
        )

        return RebuildKnowledgeResponse(
            ok=True,
            message=(
                "知识库已重建并重新加载，"
                "无需重启 FastAPI。"
            ),
            elapsed_seconds=elapsed_seconds,
        )

    except Exception as error:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            2,
        )

        logger.exception(
            "知识库重建失败，耗时 %.2f 秒",
            elapsed_seconds,
        )

        raise HTTPException(
            status_code=500,
            detail="知识库重建失败，请查看服务端日志",
        ) from error


@app.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    summary="清空指定会话",
)
def delete_session(session_id: str):
    """
    删除指定 session_id 对应的 Agent 和聊天记录。
    """

    with _registry_lock:
        existed = session_id in _agents

        _agents.pop(session_id, None)
        _session_locks.pop(session_id, None)

    if existed:
        logger.info("已清空会话：%s", session_id)

        return DeleteSessionResponse(
            session_id=session_id,
            deleted=True,
            message="会话已清空",
        )

    return DeleteSessionResponse(
        session_id=session_id,
        deleted=False,
        message="没有找到该会话",
    )


# === Chat history routes ===
from history_routes import router as history_router

app.include_router(history_router)


# === Explicit user memory ===
from explicit_memory import ExplicitMemoryMiddleware
from user_memory_routes import router as user_memory_router

app.add_middleware(ExplicitMemoryMiddleware)
app.include_router(user_memory_router)

