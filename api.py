from app.core.stream_events import stream_event_session
from fastapi.responses import StreamingResponse
import re
import threading
import queue
import json
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.memory.chat_memory import Memory
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    HealthResponse,
    RebuildKnowledgeResponse,
    SearchRequest,
    SearchResponse,
    ToolCallRecord,
)

from app.core.observability import install_observability
from app.services.agent_session import agent_session_service
from app.services.knowledge_service import knowledge_service
from app.auth.router import get_current_user, require_admin

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
        knowledge_service.prepare_document_storage()
        knowledge_service.preload()

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

# OBSERVABILITY_V1_START
install_observability(app)
# OBSERVABILITY_V1_END


# === Local web CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session_agent(
    session_id: str,
):
    """
    兼容入口。

    真正 Session Registry 实现在：
    app.services.agent_session
    """

    return agent_session_service.get_session_agent(
        session_id
    )


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
def chat(
    request: ChatRequest,
    current_user = Depends(get_current_user),
):
    """
    接收用户问题，调用现有 Agent，并返回最终回答。
    """
    start_time = time.perf_counter()

    # MULTI_USER_ISOLATION_V1
    user_id = int(current_user["id"])
    owner_memory = Memory()

    try:
        owner_memory.ensure_session_owner(
            session_id=request.session_id,
            user_id=user_id,
        )
    except PermissionError as error:
        raise HTTPException(
            status_code=404,
            detail="没有找到该会话",
        ) from error

    agent, session_lock = get_session_agent(
        request.session_id
    )

    bind_user = getattr(
        agent,
        "bind_user",
        None,
    )

    if callable(bind_user):
        bind_user(user_id)
    else:
        setattr(agent, "user_id", user_id)

    try:
        # 同一个会话一次只处理一个问题，
        # 防止聊天记录和工具调用记录混乱。
        with session_lock:

            # Session Lock 仍然保持在最外层。
            #
            # Global Semaphore 由
            # AgentSessionService 统一管理。
            with agent_session_service.global_run_slot(
                request.session_id
            ):

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
    dependencies=[Depends(require_admin)],
)
def search(request: SearchRequest):
    """
    不调用大模型，直接执行知识库检索。
    """

    try:
        result = knowledge_service.search(
            query=request.query,
            top_k=request.top_k,
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
    dependencies=[Depends(require_admin)],
)
def rebuild_knowledge():
    """
    重建知识库并热重载默认知识库。
    """

    start_time = time.perf_counter()

    try:

        elapsed_seconds = (
            knowledge_service.rebuild()
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
            time.perf_counter()
            - start_time,
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
def delete_session(
    session_id: str,
    current_user = Depends(get_current_user),
):
    """只允许当前用户删除自己的会话。"""

    user_id = int(current_user["id"])
    memory = Memory()

    deleted_from_database = memory.delete_session(
        session_id=session_id,
        user_id=user_id,
    )

    existed_in_memory = False

    if deleted_from_database:
        existed_in_memory = (
            agent_session_service.remove_session(
                session_id
            )
        )

    deleted = (
        deleted_from_database
        or existed_in_memory
    )

    if deleted:
        logger.info(
            "已清空当前用户会话及聊天记录：%s",
            session_id,
        )

        return DeleteSessionResponse(
            session_id=session_id,
            deleted=True,
            message="会话及聊天记录已清空",
        )

    return DeleteSessionResponse(
        session_id=session_id,
        deleted=False,
        message="没有找到该会话",
    )


# MULTI_USER_ISOLATION_V1


# AUTH_JWT_V1_START
# === Authentication routes ===
from app.auth.router import router as auth_router

app.include_router(auth_router)
# AUTH_JWT_V1_END


# === Chat history routes ===
from app.routes.history import router as history_router

app.include_router(history_router)


# === Scoped knowledge documents ===
from app.routes.documents import router as documents_router

app.include_router(documents_router)


# === Explicit user memory ===
from app.memory.explicit_memory import ExplicitMemoryMiddleware
from app.routes.user_memory import router as user_memory_router

app.add_middleware(ExplicitMemoryMiddleware)
app.include_router(user_memory_router)


# CHAT_STREAMING_V1_START

def _stream_json_line(
    payload: dict,
) -> bytes:
    """
    NDJSON:
    一行就是一个完整 JSON Event。
    """

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def _extract_stream_sources(
    data: dict,
) -> list[str]:

    sources = []

    for record in (
        data.get(
            "tool_calls",
            []
        )
        or []
    ):

        if not isinstance(
            record,
            dict,
        ):
            continue

        if (
            str(
                record.get(
                    "tool_name",
                    ""
                )
            )
            != "search_knowledge"
        ):
            continue

        result = str(
            record.get(
                "result",
                "",
            )
            or ""
        )

        for source in re.findall(
            r"^来源：(.+)$",
            result,
            flags=re.MULTILINE,
        ):
            source = (
                source.strip()
            )

            if (
                source
                and source
                not in sources
            ):
                sources.append(
                    source
                )

    return sources


@app.post(
    "/chat/stream",
    summary="与 Agent 流式对话",
)
def chat_stream(
    request: ChatRequest,
    current_user = Depends(get_current_user),
):
    """
    Streaming V1。

    过程：
        Agent/RAG 实时 Trace
        -> 最终答案分片
        -> done metadata

    原 /chat 完全保留，
    因此 CLI、测试和旧网页接口都不受影响。
    """

    event_queue = (
        queue.Queue()
    )

    finished = object()


    def push_event(
        event: dict,
    ):
        event_queue.put(
            event
        )


    def worker():

        try:

            with stream_event_session(
                push_event
            ):

                # 直接复用正式 /chat 函数，
                # 因此原有：
                #
                # Session Lock
                # Global Concurrency
                # Agent.run
                # Tool Calling
                # Error handling
                #
                # 都继续有效。
                response = chat(
                    request,
                    current_user,
                )


            if hasattr(
                response,
                "model_dump",
            ):
                data = (
                    response.model_dump()
                )

            elif hasattr(
                response,
                "dict",
            ):
                data = (
                    response.dict()
                )

            elif isinstance(
                response,
                dict,
            ):
                data = dict(
                    response
                )

            else:
                data = {
                    "answer":
                        str(
                            response
                        ),

                    "session_id":
                        request.session_id,

                    "called_tools":
                        [],

                    "tool_calls":
                        [],

                    "elapsed_seconds":
                        0.0,
                }


            answer = str(
                data.get(
                    "answer",
                    "",
                )
                or ""
            )


            # =================================================
            # Final answer transport streaming
            #
            # 注意：
            # 这是服务器分片传输，
            # 不是上游模型 token-level stream。
            # =================================================

            chunk_size = 24

            for start in range(
                0,
                len(answer),
                chunk_size,
            ):

                push_event(
                    {
                        "type":
                            "answer_delta",

                        "text":
                            answer[
                                start:
                                start
                                + chunk_size
                            ],
                    }
                )


            sources = (
                _extract_stream_sources(
                    data
                )
            )


            push_event(
                {
                    "type":
                        "done",

                    "session_id":
                        str(
                            data.get(
                                "session_id",
                                request.session_id,
                            )
                        ),

                    "called_tools":
                        list(
                            data.get(
                                "called_tools",
                                [],
                            )
                            or []
                        ),

                    "sources":
                        sources,

                    "elapsed_seconds":
                        data.get(
                            "elapsed_seconds",
                            0.0,
                        ),
                }
            )


        except HTTPException as exc:

            push_event(
                {
                    "type":
                        "error",

                    "message":
                        str(
                            exc.detail
                        ),

                    "status_code":
                        exc.status_code,
                }
            )


        except Exception as exc:

            logger.exception(
                "Streaming Agent 请求失败："
                "session_id=%s",
                request.session_id,
            )

            push_event(
                {
                    "type":
                        "error",

                    "message":
                        (
                            "Streaming Agent "
                            "运行失败："
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                }
            )


        finally:

            event_queue.put(
                finished
            )


    thread = threading.Thread(
        target=worker,
        daemon=True,
        name=(
            "agent-stream-"
            + str(
                request.session_id
            )[:20]
        ),
    )

    thread.start()


    def generate():

        # 先发一个事件，
        # 防止代理/浏览器等待首包。
        yield _stream_json_line(
            {
                "type": "connected",
                "message":
                    "Streaming connection ready",
            }
        )

        while True:

            event = (
                event_queue.get()
            )

            if event is finished:
                break

            yield _stream_json_line(
                event
            )


    return StreamingResponse(
        generate(),
        media_type=(
            "application/x-ndjson"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )

# CHAT_STREAMING_V1_END


# RBAC_AUTH_V1
