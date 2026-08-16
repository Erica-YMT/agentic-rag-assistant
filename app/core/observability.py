from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)


# ============================================================
# 应用信息
# ============================================================

APP_INFO = Info(
    "agent_app",
    "Agentic RAG Assistant application information",
)

APP_INFO.info(
    {
        "service": "agentic-rag-api",
        "version": "1.0.0",
    }
)


# ============================================================
# HTTP 指标
# ============================================================

HTTP_REQUESTS_TOTAL = Counter(
    "agent_http_requests_total",
    "Total number of HTTP requests",
    [
        "method",
        "route",
        "status",
    ],
)


HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "agent_http_request_duration_seconds",
    "HTTP request duration in seconds",
    [
        "method",
        "route",
    ],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)


HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "agent_http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    [
        "method",
        "route",
    ],
)



# AGENT_INTERNAL_METRICS_V1_START

# ============================================================
# Agent 内部指标
# ============================================================

AGENT_LLM_CALLS_TOTAL = Counter(
    "agent_llm_calls_total",
    "Total number of logical LLM calls",
)


AGENT_LLM_DURATION_SECONDS = Histogram(
    "agent_llm_duration_seconds",
    "Logical LLM call duration in seconds",
    buckets=(
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
    ),
)


AGENT_TOOL_CALLS_TOTAL = Counter(
    "agent_tool_calls_total",
    "Total number of tool executions",
    [
        "tool",
    ],
)


AGENT_TOOL_DURATION_SECONDS = Histogram(
    "agent_tool_duration_seconds",
    "Tool execution duration in seconds",
    [
        "tool",
    ],
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)


AGENT_RAG_STAGE_CALLS_TOTAL = Counter(
    "agent_rag_stage_calls_total",
    "Total number of RAG stage executions",
    [
        "stage",
    ],
)


AGENT_RAG_STAGE_DURATION_SECONDS = Histogram(
    "agent_rag_stage_duration_seconds",
    "RAG stage duration in seconds",
    [
        "stage",
    ],
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    ),
)


def record_llm_call(
    elapsed_seconds: float,
) -> None:

    AGENT_LLM_CALLS_TOTAL.inc()

    AGENT_LLM_DURATION_SECONDS.observe(
        float(elapsed_seconds)
    )


def record_tool_call(
    tool_name: str,
    elapsed_seconds: float,
) -> None:

    tool_name = (
        str(tool_name).strip()
        or "unknown"
    )

    AGENT_TOOL_CALLS_TOTAL.labels(
        tool=tool_name
    ).inc()

    AGENT_TOOL_DURATION_SECONDS.labels(
        tool=tool_name
    ).observe(
        float(elapsed_seconds)
    )


def record_rag_stage(
    stage: str,
    elapsed_seconds: float,
) -> None:

    stage = (
        str(stage).strip()
        or "unknown"
    )

    AGENT_RAG_STAGE_CALLS_TOTAL.labels(
        stage=stage
    ).inc()

    AGENT_RAG_STAGE_DURATION_SECONDS.labels(
        stage=stage
    ).observe(
        float(elapsed_seconds)
    )


# AGENT_INTERNAL_METRICS_V1_END

# AGENT_RESULT_METRICS_V1_START

AGENT_LLM_RESULTS_TOTAL = Counter(
    "agent_llm_results_total",
    "Logical LLM call results",
    [
        "status",
    ],
)


AGENT_TOOL_RESULTS_TOTAL = Counter(
    "agent_tool_results_total",
    "Tool execution results",
    [
        "tool",
        "status",
    ],
)


AGENT_RAG_STAGE_RESULTS_TOTAL = Counter(
    "agent_rag_stage_results_total",
    "RAG stage execution results",
    [
        "stage",
        "status",
    ],
)


def record_llm_result(
    status: str,
) -> None:

    AGENT_LLM_RESULTS_TOTAL.labels(
        status=status
    ).inc()


def record_tool_result(
    tool_name: str,
    status: str,
) -> None:

    tool_name = (
        str(tool_name).strip()
        or "unknown"
    )

    AGENT_TOOL_RESULTS_TOTAL.labels(
        tool=tool_name,
        status=status,
    ).inc()


def record_rag_result(
    stage: str,
    status: str,
) -> None:

    stage = (
        str(stage).strip()
        or "unknown"
    )

    AGENT_RAG_STAGE_RESULTS_TOTAL.labels(
        stage=stage,
        status=status,
    ).inc()


# 预初始化，避免 Grafana 在 0 次错误时显示 No data。
for _status in (
    "success",
    "error",
):
    AGENT_LLM_RESULTS_TOTAL.labels(
        status=_status
    )


for _tool in (
    "search_knowledge",
    "search_web",
    "calculator",
):
    for _status in (
        "success",
        "error",
    ):
        AGENT_TOOL_RESULTS_TOTAL.labels(
            tool=_tool,
            status=_status,
        )


for _stage in (
    "Hybrid Retrieval",
    "Reranker",
):
    for _status in (
        "success",
        "error",
    ):
        AGENT_RAG_STAGE_RESULTS_TOTAL.labels(
            stage=_stage,
            status=_status,
        )


# AGENT_RESULT_METRICS_V1_END




# ============================================================
# Streaming Chat metrics
#
# 普通 HTTP middleware 在 StreamingResponse 返回时就结束计时，
# 因此单独记录从 /chat/stream 开始到流真正结束的生命周期。
# ============================================================

CHAT_STREAM_DURATION_SECONDS = Histogram(
    "agent_chat_stream_duration_seconds",
    "End-to-end duration of streaming chat responses in seconds",
    buckets=(
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        30.0,
        45.0,
        60.0,
        90.0,
        120.0,
        180.0,
        300.0,
    ),
)

CHAT_STREAM_IN_PROGRESS = Gauge(
    "agent_chat_stream_in_progress",
    "Number of streaming chat responses currently in progress",
)

CHAT_STREAM_TOTAL = Counter(
    "agent_chat_stream_total",
    "Total number of completed streaming chat responses",
    ["status"],
)


def _fallback_route(
    path: str,
) -> str:
    """
    避免把 session_id / memory_id
    直接作为 Prometheus label。

    否则每个 ID 都可能生成一组新指标，
    导致 label cardinality 过高。
    """

    if path.startswith(
        "/sessions/"
    ):
        return "/sessions/{session_id}"

    if path.startswith(
        "/memory/"
    ):
        return "/memory/{memory_id}"

    if path.startswith(
        "/history/"
    ):
        return "/history/{session_id}"

    return path


def _resolved_route(
    request: Request,
    fallback: str,
) -> str:
    """
    FastAPI 完成路由匹配以后，
    优先使用路由模板。

    例如：

    /sessions/abc123

    会记录成：

    /sessions/{session_id}
    """

    route = request.scope.get(
        "route"
    )

    route_path = getattr(
        route,
        "path",
        None,
    )

    if route_path:
        return str(route_path)

    return fallback


def install_observability(
    app: FastAPI,
) -> None:
    """
    给 FastAPI 安装 Prometheus 指标采集。

    可重复调用保护：
    防止开发热加载时重复注册。
    """

    if getattr(
        app.state,
        "prometheus_installed",
        False,
    ):
        return

    app.state.prometheus_installed = True


    @app.middleware("http")
    async def prometheus_middleware(
        request: Request,
        call_next,
    ):
        # Prometheus 自己抓 /metrics 时
        # 不统计它自己，避免监控流量污染业务数据。
        if request.url.path == "/metrics":
            return await call_next(
                request
            )

        method = request.method

        fallback_route = (
            _fallback_route(
                request.url.path
            )
        )

        HTTP_REQUESTS_IN_PROGRESS.labels(
            method=method,
            route=fallback_route,
        ).inc()

        start = time.perf_counter()

        status_code = 500

        try:
            response = await call_next(
                request
            )

            status_code = (
                response.status_code
            )

            return response

        finally:
            elapsed = (
                time.perf_counter()
                - start
            )

            route = _resolved_route(
                request,
                fallback_route,
            )

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                route=route,
                status=str(status_code),
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                route=route,
            ).observe(
                elapsed
            )

            HTTP_REQUESTS_IN_PROGRESS.labels(
                method=method,
                route=fallback_route,
            ).dec()


    @app.get(
        "/metrics",
        include_in_schema=False,
    )
    def metrics():
        """
        Prometheus 抓取入口。
        """

        return Response(
            content=generate_latest(),
            headers={
                "Content-Type":
                    CONTENT_TYPE_LATEST
            },
        )
