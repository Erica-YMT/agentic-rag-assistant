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

    def health(self) -> dict:
        """
        检查 FastAPI 服务状态。
        """
        with httpx.Client(
            timeout=self.timeout
        ) as client:
            response = client.get(
                f"{self.base_url}/health"
            )

            response.raise_for_status()
            return response.json()

    def chat(
        self,
        question: str,
        session_id: str = "web-default",
    ) -> dict:
        """
        调用 Agent 完整问答接口。
        """
        if not question.strip():
            raise ValueError("question 不能为空")

        with httpx.Client(
            timeout=self.timeout
        ) as client:
            response = client.post(
                f"{self.base_url}/chat",
                json={
                    "question": question,
                    "session_id": session_id,
                },
            )

            response.raise_for_status()
            return response.json()

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> dict:
        """
        直接调用知识库检索接口。
        """
        if not query.strip():
            raise ValueError("query 不能为空")

        with httpx.Client(
            timeout=self.timeout
        ) as client:
            response = client.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                },
            )

            response.raise_for_status()
            return response.json()

    def delete_session(
        self,
        session_id: str,
    ) -> dict:
        """
        清空指定会话。
        """
        if not session_id.strip():
            raise ValueError("session_id 不能为空")

        with httpx.Client(
            timeout=self.timeout
        ) as client:
            response = client.delete(
                f"{self.base_url}/sessions/{session_id}"
            )

            response.raise_for_status()
            return response.json()

    def rebuild_knowledge(self) -> dict:
        """
        请求 FastAPI 重建并重新加载知识库。
        """
        with httpx.Client(
            timeout=self.timeout
        ) as client:
            response = client.post(
                f"{self.base_url}/knowledge/rebuild",
                json={},
            )

            response.raise_for_status()

            return response.json()



# 提供一个默认客户端对象
api_client = AgentAPIClient()
