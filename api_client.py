import uuid

import httpx

from config import config


# =========================
# 读取 FastAPI 配置
# =========================

api_config = config.get(
    "api_client",
    {}
)

base_url = api_config.get(
    "base_url",
    "http://127.0.0.1:8000"
).rstrip("/")

timeout = float(
    api_config.get(
        "timeout",
        120.0
    )
)


def _auth_headers(
    access_token: str | None,
) -> dict[str, str]:
    token = str(access_token or "").strip()
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}"
    }


# RBAC_AUTH_V1

class AgentAPIClient:
    """
    Agentic RAG Assistant 的 HTTP 客户端。

    负责调用：
    - GET /health
    - POST /chat
    - POST /search
    - DELETE /sessions/{session_id}
    """

    def __init__(
        self,
        api_base_url: str = base_url,
        request_timeout: float = timeout,
    ):
        self.base_url = api_base_url.rstrip("/")
        self.timeout = request_timeout
        self.session_id = f"client-{uuid.uuid4().hex}"

        self._client = httpx.Client(
            timeout=self.timeout
        )

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict:
        """
        统一发送 FastAPI 请求。

        AgentAPIClient 实例复用同一个 httpx.Client，
        避免每个方法重复创建连接池。
        """
        response = self._client.request(
            method,
            f"{self.base_url}{path}",
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """
        检查 FastAPI 服务状态。
        """
        return self._request(
            "GET",
            "/health",
        )

    def chat(
        self,
        question: str,
        session_id: str | None = None,
        access_token: str | None = None,
    ) -> dict:
        """
        调用 Agent 完整问答接口。
        """
        if not question.strip():
            raise ValueError("question 不能为空")

        if session_id is None:
            effective_session_id = self.session_id
        else:
            effective_session_id = str(session_id).strip()
            if not effective_session_id:
                raise ValueError("session_id 不能为空")

        return self._request(
            "POST",
            "/chat",
            json={
                "question": question,
                "session_id": effective_session_id,
            },
            headers=_auth_headers(access_token),
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        access_token: str | None = None,
    ) -> dict:
        """
        直接调用知识库检索接口。
        """
        if not query.strip():
            raise ValueError("query 不能为空")

        return self._request(
            "POST",
            "/search",
            json={
                "query": query,
                "top_k": top_k,
            },
            headers=_auth_headers(access_token),
        )

    def delete_session(
        self,
        session_id: str,
        access_token: str | None = None,
    ) -> dict:
        """
        清空指定会话。
        """
        if not session_id.strip():
            raise ValueError("session_id 不能为空")

        return self._request(
            "DELETE",
            f"/sessions/{session_id}",
            headers=_auth_headers(access_token),
        )

    def current_user(
        self,
        access_token: str,
    ) -> dict:
        """验证 Bearer Token，并返回当前用户。"""

        return self._request(
            "GET",
            "/auth/me",
            headers=_auth_headers(access_token),
        )


    def rebuild_knowledge(
        self,
        access_token: str | None = None,
    ) -> dict:
        """
        请求 FastAPI 重建并重新加载知识库。
        """
        return self._request(
            "POST",
            "/knowledge/rebuild",
            json={},
            headers=_auth_headers(access_token),
        )


    def list_documents(
        self,
        access_token: str,
    ) -> dict:
        return self._request(
            "GET",
            "/documents",
            headers=_auth_headers(access_token),
        )

    def upload_documents(
        self,
        *,
        body: bytes,
        content_type: str,
        rebuild: bool,
        access_token: str,
    ) -> dict:
        headers = _auth_headers(access_token)
        headers["Content-Type"] = str(content_type)

        return self._request(
            "POST",
            "/documents/upload",
            params={
                "rebuild": str(bool(rebuild)).lower()
            },
            content=body,
            headers=headers,
        )

    def delete_documents(
        self,
        *,
        document_ids: list[int],
        access_token: str,
    ) -> dict:
        return self._request(
            "POST",
            "/documents/delete",
            json={
                "document_ids": [
                    int(value)
                    for value in document_ids
                ]
            },
            headers=_auth_headers(access_token),
        )

    def rebuild_documents(
        self,
        *,
        access_token: str,
    ) -> dict:
        return self._request(
            "POST",
            "/documents/rebuild",
            json={},
            headers=_auth_headers(access_token),
        )



# 提供一个默认客户端对象
api_client = AgentAPIClient()
