"""拦截用户明确提出的长期记忆指令。"""

from __future__ import annotations

import json
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.memory.chat_memory import Memory
from app.auth.router import authenticate_access_token
from app.memory.user_memory import (
    UserMemoryStore,
    contains_sensitive_content,
    extract_explicit_memory,
)


USER_MEMORY = UserMemoryStore()
CHAT_MEMORY = Memory()


class ExplicitMemoryMiddleware(BaseHTTPMiddleware):
    """处理“请记住……”指令，不调用大模型。"""

    async def dispatch(self, request: Request, call_next):
        if (
            request.method != "POST"
            or request.url.path != "/chat"
        ):
            return await call_next(request)

        try:
            body = await request.body()
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await call_next(request)

        if not isinstance(payload, dict):
            return await call_next(request)

        question = str(payload.get("question", "")).strip()
        memory_content = extract_explicit_memory(question)

        if memory_content is None:
            return await call_next(request)

        authorization = str(
            request.headers.get("Authorization", "")
        ).strip()

        scheme, _, token = authorization.partition(" ")

        if (
            scheme.lower() != "bearer"
            or not token.strip()
        ):
            return JSONResponse(
                {"detail": "登录状态无效或已经过期"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            current_user = authenticate_access_token(
                token.strip()
            )
        except HTTPException as exc:
            return JSONResponse(
                {"detail": exc.detail},
                status_code=exc.status_code,
                headers=exc.headers or {},
            )

        # MULTI_USER_ISOLATION_V1
        user_id = int(current_user["id"])
        memory_user_id = str(user_id)

        session_id = str(
            payload.get("session_id")
            or f"memory-{uuid.uuid4().hex}"
        )

        try:
            CHAT_MEMORY.ensure_session_owner(
                session_id=session_id,
                user_id=user_id,
            )
        except PermissionError:
            return JSONResponse(
                {"detail": "没有找到该会话"},
                status_code=404,
            )

        if contains_sensitive_content(memory_content):
            answer = (
                "这段内容可能包含密码、密钥、Token 或其他敏感信息，"
                "为了安全，我没有把它保存到长期记忆。"
            )
        else:
            USER_MEMORY.save(
                content=memory_content,
                source_session_id=session_id,
                user_id=memory_user_id,
            )
            answer = f"已记住：{memory_content}"

        CHAT_MEMORY.add_message(
            session_id=session_id,
            role="user",
            content=question,
            user_id=user_id,
        )

        CHAT_MEMORY.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            user_id=user_id,
        )

        return JSONResponse(
            {
                "answer": answer,
                "session_id": session_id,
                "called_tools": [],
                "tool_calls": [],
                "elapsed_seconds": 0.0,
            }
        )


# RBAC_AUTH_V1
# MULTI_USER_ISOLATION_V1
