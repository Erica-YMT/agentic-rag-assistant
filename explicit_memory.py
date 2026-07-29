"""拦截用户明确提出的长期记忆指令。"""

from __future__ import annotations

import json
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from memory import Memory
from user_memory import (
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

        session_id = str(
            payload.get("session_id")
            or f"memory-{uuid.uuid4().hex}"
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
            )

            answer = f"已记住：{memory_content}"

        # 记忆指令本身也保存在普通聊天历史中。
        CHAT_MEMORY.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        CHAT_MEMORY.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
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
