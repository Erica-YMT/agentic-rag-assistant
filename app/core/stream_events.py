from __future__ import annotations

import builtins
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable


_current_emitter: ContextVar[
    Callable[[dict], None] | None
] = ContextVar(
    "stream_event_emitter",
    default=None,
)


_active_emitters = set()
_active_lock = threading.Lock()


def _sanitize(
    text: str,
) -> str:
    """
    避免把超长日志或疑似密钥直接推到前端。
    """

    text = str(
        text
        or ""
    ).strip()

    if not text:
        return ""

    # 简单隐藏常见密钥形式。
    text = re.sub(
        r"sk-[A-Za-z0-9_-]{8,}",
        "sk-***",
        text,
    )

    text = re.sub(
        r"Bearer\s+\S+",
        "Bearer ***",
        text,
        flags=re.IGNORECASE,
    )

    # 防止工具结果过长撑爆 UI。
    if len(text) > 600:
        text = (
            text[:600]
            + "..."
        )

    return text


def _classify(
    text: str,
) -> str:

    lower = text.lower()


    # ==========================================
    # Complex RAG
    # ==========================================

    if (
        "complexrag" in lower
        or "planner" in lower
        or "worker " in lower
        or "coverage grade" in lower
    ):
        return "complex"


    # ==========================================
    # Corrective RAG
    # ==========================================

    if (
        "correctiverag" in lower
        or "query rewrite" in lower
        or "grade #" in lower
    ):
        return "corrective"


    # ==========================================
    # Auto-Merging
    # ==========================================

    if (
        "automerge" in lower
        or "parent 提升" in text
    ):
        return "auto_merge"


    # ==========================================
    # LLM / Model
    # ==========================================

    if (
        "模型调用" in text
        or "模型请求" in text
        or "模型服务" in text
        or "llm" in lower
    ):
        return "model"


    # ==========================================
    # Hybrid Retrieval / Reranker
    # ==========================================

    if (
        "hybrid retrieval" in lower
        or "hybrid retriever" in lower
        or "reranker" in lower
        or "faiss" in lower
        or "bm25" in lower
    ):
        return "retrieval"


    # ==========================================
    # Tool
    # ==========================================

    if (
        "执行工具" in text
        or "调用工具" in text
        or "工具已执行" in text
        or "tool" in lower
    ):
        return "tool"


    # ==========================================
    # Memory
    # ==========================================

    if (
        "longtermmemory" in lower
        or "长期记忆" in text
        or "memory" in lower
    ):
        return "memory"


    # ==========================================
    # Error
    # ==========================================

    if (
        "error" in lower
        or "失败" in text
        or "异常" in text
    ):
        return "error"


    # ==========================================
    # Agent timing / default
    # ==========================================

    if (
        "agent 总耗时" in lower
        or "agent" in lower
    ):
        return "agent"


    return "agent"


def _get_emitter():

    emitter = (
        _current_emitter.get()
    )

    if emitter is not None:
        return emitter

    # LangGraph 并行 Worker 可能在其他线程。
    #
    # 如果当前整个进程只有一个 streaming 请求，
    # 可以安全地把 Worker 事件发送给这个请求。
    #
    # 如果同时有多个 streaming 请求，
    # 宁可不发送子线程事件，也绝不串台。
    with _active_lock:

        if len(_active_emitters) == 1:
            return next(
                iter(
                    _active_emitters
                )
            )

    return None


def emit_event(
    event_type: str,
    **payload,
):

    emitter = _get_emitter()

    if emitter is None:
        return

    event = {
        "type": event_type,
        **payload,
    }

    try:
        emitter(event)

    except Exception:
        # UI 流事件绝不能影响正式 Agent。
        pass


def event_print(
    *args,
    **kwargs,
):
    """
    正常 print + 同时尝试推送 Streaming Trace。
    """

    builtins.print(
        *args,
        **kwargs,
    )

    try:
        text = " ".join(
            str(item)
            for item in args
        )

        text = _sanitize(
            text
        )

        if not text:
            return

        # 分隔线没有 UI 价值。
        if (
            text
            and set(text)
            <= {
                "=",
                "-",
                " ",
            }
        ):
            return

        emit_event(
            "trace",
            stage=_classify(
                text
            ),
            message=text,
        )

    except Exception:
        pass


@contextmanager
def stream_event_session(
    emitter: Callable[[dict], None],
):
    """
    给当前 Agent 请求绑定事件出口。
    """

    token = (
        _current_emitter.set(
            emitter
        )
    )

    with _active_lock:
        _active_emitters.add(
            emitter
        )

    try:

        emit_event(
            "trace",
            stage="agent",
            message="Agent 开始处理问题",
        )

        yield

    finally:

        with _active_lock:
            _active_emitters.discard(
                emitter
            )

        _current_emitter.reset(
            token
        )
