"""聊天历史查询接口。"""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.memory.chat_memory import Memory
from app.auth.router import get_current_user


router = APIRouter(
    prefix="/history",
    tags=["聊天历史"],
    dependencies=[Depends(get_current_user)],
)

@lru_cache(maxsize=1)
def get_memory() -> Memory:
    return Memory()


@router.get("/sessions")
def list_sessions(
    limit: int = Query(default=100, ge=1, le=500),
    current_user = Depends(get_current_user),
):
    return {
        "items": get_memory().list_sessions(
            limit=limit,
            user_id=int(current_user["id"]),
        ),
    }


@router.get("/search")
def search_history(
    keyword: str = Query(default="", max_length=200),
    session_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    current_user = Depends(get_current_user),
):
    return {
        "keyword": keyword,
        "items": get_memory().search_messages(
            keyword=keyword,
            session_id=session_id,
            limit=limit,
            user_id=int(current_user["id"]),
        ),
    }


@router.get("/session/{session_id}")
def get_session(
    session_id: str,
    current_user = Depends(get_current_user),
):
    messages = get_memory().get_session_messages(
        session_id=session_id,
        user_id=int(current_user["id"]),
    )

    if not messages:
        raise HTTPException(
            status_code=404,
            detail="没有找到该会话",
        )

    return {
        "session_id": session_id,
        "items": messages,
    }


@router.delete("/session/{session_id}")
def delete_session(
    session_id: str,
    current_user = Depends(get_current_user),
):
    deleted = get_memory().delete_session(
        session_id=session_id,
        user_id=int(current_user["id"]),
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="没有找到该会话",
        )

    return {
        "ok": True,
        "session_id": session_id,
    }


# MULTI_USER_ISOLATION_V1


@router.get("/ui", response_class=HTMLResponse)
def history_ui():
    """简单的聊天历史搜索页面。"""
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>聊天历史</title>

    <style>
        body {
            max-width: 1000px;
            margin: 0 auto;
            padding: 30px;
            font-family: Arial, "Microsoft YaHei", sans-serif;
            background: #f5f6f8;
        }

        h1 {
            margin-bottom: 20px;
        }

        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
        }

        input {
            flex: 1;
            padding: 12px;
            border: 1px solid #cccccc;
            border-radius: 8px;
        }

        button {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }

        .message {
            margin-bottom: 12px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            border-left: 4px solid #999999;
        }

        .message.user {
            border-left-color: #2563eb;
        }

        .message.assistant {
            border-left-color: #16a34a;
        }

        .meta {
            margin-bottom: 8px;
            color: #666666;
            font-size: 13px;
        }

        .content {
            white-space: pre-wrap;
            line-height: 1.7;
        }

        .empty {
            padding: 30px;
            text-align: center;
            color: #777777;
        }
    </style>
</head>

<body>
    <h1>历史聊天记录</h1>

    <div class="search-box">
        <input
            id="keyword"
            placeholder="输入关键词，例如 Docker、FAISS"
        >
        <button onclick="searchHistory()">搜索</button>
    </div>

    <div id="result">
        <div class="empty">输入关键词后搜索聊天记录。</div>
    </div>

    <script>
        function escapeHtml(value) {
            return String(value ?? "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;");
        }

        async function searchHistory() {
            const keyword = document
                .getElementById("keyword")
                .value
                .trim();

            const result = document.getElementById("result");

            try {
                const response = await fetch(
                    `/history/search?keyword=${encodeURIComponent(keyword)}`
                );

                const data = await response.json();
                const items = data.items || [];

                if (!items.length) {
                    result.innerHTML =
                        '<div class="empty">没有找到相关聊天记录。</div>';
                    return;
                }

                result.innerHTML = items.map(item => `
                    <div class="message ${escapeHtml(item.role)}">
                        <div class="meta">
                            ${escapeHtml(item.role)}
                            ·
                            ${escapeHtml(item.session_id)}
                            ·
                            ${escapeHtml(item.created_at)}
                        </div>

                        <div class="content">
                            ${escapeHtml(item.content)}
                        </div>
                    </div>
                `).join("");
            } catch (error) {
                result.innerHTML = `
                    <div class="empty">
                        搜索失败：${escapeHtml(error.message)}
                    </div>
                `;
            }
        }

        document
            .getElementById("keyword")
            .addEventListener("keydown", event => {
                if (event.key === "Enter") {
                    searchHistory();
                }
            });
    </script>
</body>
</html>
        """
    )

# RBAC_AUTH_V1
