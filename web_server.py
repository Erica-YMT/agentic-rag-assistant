"""Serve the standalone web.html UI with the project's real Agent and RAG APIs."""

from __future__ import annotations

import argparse
import errno
import json
import re
import tomllib
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from api_client import api_client


PROJECT_ROOT = Path(__file__).resolve().parent
PAGE_PATH = PROJECT_ROOT / "web.html"
CONFIG_PATH = PROJECT_ROOT / "config.toml"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def extract_sources(call_records: list[dict[str, Any]]) -> list[str]:
    """Extract unique source labels from the real RAG tool response."""

    sources: list[str] = []
    for record in call_records:
        if record.get("tool_name") != "search_knowledge":
            continue
        for source in re.findall(r"^来源：(.+)$", str(record.get("result", "")), re.MULTILINE):
            source = source.strip()
            if source and source not in sources:
                sources.append(source)
    return sources


class Runtime:
    """网页层运行时适配器：将浏览器请求转发给 FastAPI。"""

    def chat(
        self,
        session_id: str,
        question: str,
        access_token: str,
    ) -> dict[str, Any]:
        """
        通过 api_client.py 调用 FastAPI 的 POST /chat。
        """

        result = api_client.chat(
            question=question,
            session_id=session_id,
            access_token=access_token,
        )

        call_records = result.get(
            "tool_calls",
            [],
        ) or []

        return {
            "answer": str(
                result.get("answer")
                or "未获得有效回答。"
            ),
            "session_id": str(
                result.get("session_id")
                or session_id
            ),
            # 保持 web.html 原来使用的字段名 tools
            "tools": list(
                result.get("called_tools")
                or []
            ),
            "sources": extract_sources(
                call_records
            ),
            "elapsed_seconds": float(
                result.get("elapsed_seconds")
                or 0.0
            ),
        }

    def clear_session(
        self,
        session_id: str,
        access_token: str,
    ) -> str:
        """
        删除 FastAPI 中保存的会话，
        然后生成一个新的网页会话 ID。
        """

        if session_id:
            api_client.delete_session(
                session_id,
                access_token=access_token,
            )

        return f"web-{uuid.uuid4().hex}"

    def list_documents(
        self,
        access_token: str,
    ) -> dict[str, Any]:
        return api_client.list_documents(
            access_token=access_token
        )

    def upload_documents(
        self,
        *,
        body: bytes,
        content_type: str,
        rebuild: bool,
        access_token: str,
    ) -> dict[str, Any]:
        return api_client.upload_documents(
            body=body,
            content_type=content_type,
            rebuild=rebuild,
            access_token=access_token,
        )

    def delete_documents(
        self,
        *,
        document_ids: list[int],
        access_token: str,
    ) -> dict[str, Any]:
        return api_client.delete_documents(
            document_ids=document_ids,
            access_token=access_token,
        )

    def rebuild_index(
        self,
        access_token: str,
    ) -> dict[str, Any]:
        return api_client.rebuild_documents(
            access_token=access_token
        )



def resolve_project_path(path_value: str) -> Path:
    path = Path(str(path_value)).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


RUNTIME = Runtime()


class AgentHTTPServer(ThreadingHTTPServer):
    """Do not keep shutdown hostage to an unresponsive upstream model request."""

    daemon_threads = True
    block_on_close = False


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "AgenticRAG/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"ok": False, "error": message}, status)


    def send_upstream_error(self, exc: httpx.HTTPStatusError) -> None:
        try:
            payload = exc.response.json()
        except Exception:
            payload = {}

        message = (
            payload.get("detail")
            or payload.get("error")
            or f"后端请求失败：{exc.response.status_code}"
        )

        try:
            status_code = HTTPStatus(exc.response.status_code)
        except ValueError:
            status_code = HTTPStatus.BAD_GATEWAY

        self.send_error_json(str(message), status_code)

    # RBAC_AUTH_V1
    def require_user(
        self,
    ) -> tuple[str, dict[str, Any]] | None:
        authorization = str(
            self.headers.get("Authorization", "")
        ).strip()

        scheme, _, token = authorization.partition(" ")

        if scheme.lower() != "bearer" or not token.strip():
            self.send_error_json(
                "请先登录。",
                HTTPStatus.UNAUTHORIZED,
            )
            return None

        token = token.strip()

        try:
            user = api_client.current_user(token)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 403:
                self.send_error_json(
                    "当前账号已被停用。",
                    HTTPStatus.FORBIDDEN,
                )
            else:
                self.send_error_json(
                    "登录状态无效或已经过期。",
                    HTTPStatus.UNAUTHORIZED,
                )
            return None
        except httpx.HTTPError:
            self.send_error_json(
                "暂时无法验证登录状态。",
                HTTPStatus.BAD_GATEWAY,
            )
            return None

        return token, user

    def require_admin(
        self,
    ) -> tuple[str, dict[str, Any]] | None:
        authenticated = self.require_user()
        if authenticated is None:
            return None

        token, user = authenticated

        if str(user.get("role", "")).lower() != "admin":
            self.send_error_json(
                "该操作仅管理员可执行。",
                HTTPStatus.FORBIDDEN,
            )
            return None

        return token, user

    def read_body(self, limit: int = MAX_UPLOAD_BYTES) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error_json("无效的请求长度。")
            return None
        if length <= 0:
            self.send_error_json("请求内容不能为空。")
            return None
        if length > limit:
            self.send_error_json("文件总大小不能超过 20 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return None
        return self.rfile.read(length)

    def read_json(self) -> dict[str, Any] | None:
        body = self.read_body(1024 * 1024)
        if body is None:
            return None
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error_json("请求必须是有效的 JSON。")
            return None
        if not isinstance(value, dict):
            self.send_error_json("请求必须是 JSON 对象。")
            return None
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/web.html"}:
            body = PAGE_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            authenticated = self.require_user()
            if authenticated is None:
                return
            access_token, _ = authenticated
            try:
                documents = RUNTIME.list_documents(access_token)
            except httpx.HTTPStatusError as exc:
                self.send_upstream_error(exc)
                return
            self.send_json(
                {
                    "ok": True,
                    "configured": CONFIG_PATH.exists(),
                    "index_ready": bool(documents.get("public_index_ready")),
                    **documents,
                }
            )
            return
        if path == "/api/documents":
            authenticated = self.require_user()
            if authenticated is None:
                return
            access_token, _ = authenticated
            try:
                documents = RUNTIME.list_documents(access_token)
            except httpx.HTTPStatusError as exc:
                self.send_upstream_error(exc)
                return
            self.send_json({"ok": True, **documents})
            return
        self.send_error_json("接口不存在。", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/chat":
                self.handle_chat()
            elif path == "/api/session/clear":
                self.handle_clear_session()
            elif path == "/api/documents/rebuild":
                self.handle_rebuild()
            elif path == "/api/documents/delete":
                self.handle_delete()
            elif path == "/api/documents/upload-only":
                self.handle_upload_only()
            elif path == "/api/documents/upload":
                self.handle_upload()
            else:
                self.send_error_json("接口不存在。", HTTPStatus.NOT_FOUND)
        except httpx.HTTPStatusError as exc:
            self.send_upstream_error(exc)
        except httpx.HTTPError as exc:
            print(f"Upstream request failed: {type(exc).__name__}: {exc}")
            self.send_error_json("暂时无法连接 FastAPI 后端。", HTTPStatus.BAD_GATEWAY)
        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}")
            self.send_error_json(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_chat(self) -> None:
        authenticated = self.require_user()
        if authenticated is None:
            return
        access_token, _ = authenticated

        payload = self.read_json()
        if payload is None:
            return
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_error_json("请输入问题。")
            return
        if len(question) > 12_000:
            self.send_error_json("问题不能超过 12000 个字符。")
            return
        session_id = str(payload.get("session_id") or f"web-{uuid.uuid4().hex}")
        result = RUNTIME.chat(
            session_id,
            question,
            access_token,
        )
        self.send_json({"ok": True, **result})

    def handle_clear_session(self) -> None:
        authenticated = self.require_user()
        if authenticated is None:
            return
        access_token, _ = authenticated

        payload = self.read_json()
        if payload is None:
            return
        session_id = str(payload.get("session_id", ""))
        self.send_json({
            "ok": True,
            "session_id": RUNTIME.clear_session(
                session_id,
                access_token,
            ),
        })

    def handle_rebuild(self) -> None:
        authenticated = self.require_user()
        if authenticated is None:
            return
        access_token, _ = authenticated

        if self.read_body(1024) is None:
            return

        result = RUNTIME.rebuild_index(access_token)
        self.send_json({"ok": True, **result})

    def handle_delete(self) -> None:
        authenticated = self.require_user()
        if authenticated is None:
            return
        access_token, _ = authenticated

        payload = self.read_json()
        if payload is None:
            return

        selected = payload.get("document_ids")
        if not isinstance(selected, list) or not selected:
            self.send_error_json("请至少选择一个可删除文档。")
            return

        result = RUNTIME.delete_documents(
            document_ids=[int(value) for value in selected],
            access_token=access_token,
        )
        self.send_json({"ok": True, **result})

    def _forward_upload(self, *, rebuild: bool) -> None:
        authenticated = self.require_user()
        if authenticated is None:
            return
        access_token, _ = authenticated

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error_json("上传请求必须使用 multipart/form-data。")
            return

        body = self.read_body()
        if body is None:
            return

        result = RUNTIME.upload_documents(
            body=body,
            content_type=content_type,
            rebuild=rebuild,
            access_token=access_token,
        )
        self.send_json({"ok": True, **result})

    def handle_upload_only(self) -> None:
        self._forward_upload(rebuild=False)

    def handle_upload(self) -> None:
        self._forward_upload(rebuild=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic RAG web interface.")
    parser.add_argument("--host", default="0.0.0.0", help="Local bind address.")
    parser.add_argument("--port", type=int, default=8001, help="Preferred local port.")
    args = parser.parse_args()

    server = None
    for port in range(args.port, args.port + 10):
        try:
            server = AgentHTTPServer((args.host, port), ApiHandler)
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    if server is None:
        raise RuntimeError(f"端口 {args.port}-{args.port + 9} 均不可用。")

    print(f"Agentic RAG Assistant is running at http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
