import uuid

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 接口接收的数据。"""

    question: str = Field(
        min_length=1,
        max_length=5000,
        description="用户提出的问题",
    )

    session_id: str = Field(
        default_factory=lambda: f"api-{uuid.uuid4().hex}",
        min_length=1,
        max_length=100,
        description="会话 ID，用于区分不同用户或不同对话",
    )

class ToolCallRecord(BaseModel):
    """Agent 的一次工具调用记录。"""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str

class ChatResponse(BaseModel):
    """POST /chat 接口返回的数据。"""

    answer: str
    session_id: str

    called_tools: list[str] = Field(
        default_factory=list,
        description="本轮调用过的工具名称",
    )

    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="本轮完整工具调用记录",
    )

    elapsed_seconds: float = Field(
        ge=0,
        description="本次请求总耗时，单位为秒",
    )


class HealthResponse(BaseModel):
    """GET /health 接口返回的数据。"""

    status: str
    service: str
class SearchRequest(BaseModel):
    """POST /search 接收的数据。"""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="知识库检索关键词",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="返回的候选文档数量",
    )


class SearchResponse(BaseModel):
    """POST /search 返回的数据。"""

    query: str
    top_k: int
    result: str

class DeleteSessionResponse(BaseModel):
    """DELETE /sessions/{session_id} 返回的数据。"""

    session_id: str
    deleted: bool
    message: str

class RebuildKnowledgeResponse(BaseModel):
    """POST /knowledge/rebuild 返回的数据。"""

    ok: bool
    message: str

    elapsed_seconds: float = Field(
        ge=0,
        description="重建和重新加载知识库的总耗时",
    )

class KnowledgeDocumentItem(BaseModel):
    id: int
    filename: str
    scope: str
    owner_user_id: int | None = None
    size_bytes: int = 0
    index_status: str = "pending"
    created_at: str
    updated_at: str
    can_delete: bool = False


class KnowledgeDocumentListResponse(BaseModel):
    documents: list[KnowledgeDocumentItem] = Field(default_factory=list)
    public_count: int = 0
    private_count: int = 0
    public_index_ready: bool = False
    private_index_ready: bool = False


class KnowledgeDocumentDeleteRequest(BaseModel):
    document_ids: list[int] = Field(min_length=1, max_length=100)


class KnowledgeDocumentMutationResponse(KnowledgeDocumentListResponse):
    ok: bool = True
    message: str
    index_rebuilt: bool = False
    elapsed_seconds: float = 0.0
