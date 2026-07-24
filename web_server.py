"""Serve the standalone web.html UI with the project's real Agent and RAG APIs."""

from __future__ import annotations

import argparse
import errno
import json
import re
import threading
import tomllib
import uuid
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
PAGE_PATH = PROJECT_ROOT / "web.html"
CONFIG_PATH = PROJECT_ROOT / "config.toml"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
BUILD_LOCK = threading.RLock()


def list_documents() -> list[str]:
    """Return the supported files in the knowledge directory."""

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (
            path.relative_to(KNOWLEDGE_DIR).as_posix()
            for path in KNOWLEDGE_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=str.lower,
    )


def safe_document_path(relative_name: str) -> Path | None:
    """Resolve a user-supplied document name without allowing path traversal."""

    candidate = (KNOWLEDGE_DIR / str(relative_name)).resolve()
    try:
        candidate.relative_to(KNOWLEDGE_DIR.resolve())
    except ValueError:
        return None

    if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    return candidate


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
    """Lazily load costly ML resources and serialize mutations to the index."""

    def __init__(self) -> None:
        self.agent_class: Any | None = None
        self.tools_module: Any | None = None
        self.lock = threading.RLock()
        self.agents: dict[str, Any] = {}
        self.session_locks: dict[str, threading.RLock] = {}

    def _load_agent(self) -> None:
        if self.agent_class is not None:
            return
        # These imports validate config.toml and load the existing FAISS index.
        from agent import Agent
        import tools as tools_module

        self.agent_class = Agent
        self.tools_module = tools_module

    def chat(self, session_id: str, question: str) -> dict[str, Any]:
        with self.lock:
            self._load_agent()
            assert self.agent_class is not None
            agent = self.agents.setdefault(session_id, self.agent_class())
            session_lock = self.session_locks.setdefault(session_id, threading.RLock())

        # A slow upstream model request affects only its own conversation.
        with session_lock:
            answer = str(agent.run(session_id=session_id, question=question) or "未获得有效回答。")
            called_tools = agent.get_called_tools()
            sources = extract_sources(agent.get_call_records())

        return {
            "answer": answer,
            "session_id": session_id,
            "tools": called_tools,
            "sources": sources,
        }

    def clear_session(self, session_id: str) -> str:
        with self.lock:
            agent = self.agents.pop(session_id, None)
            self.session_locks.pop(session_id, None)
            if agent is not None:
                sessions = getattr(getattr(agent, "memory", None), "sessions", None)
                if isinstance(sessions, dict):
                    sessions.pop(session_id, None)
        return f"web-{uuid.uuid4().hex}"

    def rebuild_index(self) -> None:
        """Build the index and replace the in-process RAG store, if loaded."""

        from build_index import build_index

        with BUILD_LOCK, self.lock:
            build_index()
            if self.tools_module is None:
                return

            from knowledge_base import KnowledgeBase

            with CONFIG_PATH.open("rb") as file:
                config = tomllib.load(file)
            embedding = config.get("embedding", {})
            model_path_value = embedding.get("model_path")
            if not model_path_value:
                raise ValueError("config.toml 中缺少 [embedding].model_path")
            model_path = resolve_project_path(str(model_path_value))
            index_path = resolve_project_path(embedding.get("index_path", "faiss_index"))
            self.tools_module.kb = KnowledgeBase(
                model_dir=str(model_path),
                index_path=str(index_path),
                score_threshold=float(embedding.get("score_threshold", 1.0)),
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
            self.send_json(
                {
                    "ok": True,
                    "configured": CONFIG_PATH.exists(),
                    "documents": list_documents(),
                    "index_ready": (PROJECT_ROOT / "faiss_index").is_dir(),
                }
            )
            return
        if path == "/api/documents":
            self.send_json({"ok": True, "documents": list_documents()})
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
            elif path == "/api/documents/upload":
                self.handle_upload()
            else:
                self.send_error_json("接口不存在。", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}")
            self.send_error_json(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_chat(self) -> None:
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
        result = RUNTIME.chat(session_id, question)
        self.send_json({"ok": True, **result})

    def handle_clear_session(self) -> None:
        payload = self.read_json()
        if payload is None:
            return
        session_id = str(payload.get("session_id", ""))
        self.send_json({"ok": True, "session_id": RUNTIME.clear_session(session_id)})

    def handle_rebuild(self) -> None:
        if self.read_body(1024) is None:
            return
        RUNTIME.rebuild_index()
        self.send_json({"ok": True, "message": "知识库已重建并在当前服务中生效。", "documents": list_documents()})

    def handle_delete(self) -> None:
        payload = self.read_json()
        if payload is None:
            return
        selected = payload.get("documents")
        if not isinstance(selected, list) or not selected:
            self.send_error_json("请至少选择一个文档。")
            return
        paths = list(dict.fromkeys(filter(None, (safe_document_path(str(name)) for name in selected))))
        current_documents = list_documents()
        if not paths:
            self.send_error_json("没有可删除的有效文档。")
            return
        if len(paths) >= len(current_documents):
            self.send_error_json("不能删除全部知识文档，请至少保留一个。")
            return
        with BUILD_LOCK:
            deleted = []
            for document in paths:
                deleted.append(document.relative_to(KNOWLEDGE_DIR).as_posix())
                document.unlink()
            RUNTIME.rebuild_index()
        self.send_json({"ok": True, "message": f"已删除并重建知识库：{'、'.join(deleted)}", "documents": list_documents()})

    def handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error_json("上传请求必须使用 multipart/form-data。")
            return
        body = self.read_body()
        if body is None:
            return
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
        )
        KNOWN = []
        rejected = []
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        for part in message.iter_parts():
            name = part.get_filename()
            if not name:
                continue
            safe_name = Path(name).name
            if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
                rejected.append(safe_name)
                continue
            content = part.get_payload(decode=True) or b""
            if not content:
                rejected.append(safe_name)
                continue
            (KNOWLEDGE_DIR / safe_name).write_bytes(content)
            KNOWN.append(safe_name)
        if not KNOWN:
            self.send_error_json("没有导入有效文件，仅支持非空 PDF、Markdown 和 TXT。")
            return
        RUNTIME.rebuild_index()
        detail = f"已导入并重建 {len(KNOWN)} 个文件：{'、'.join(KNOWN)}。"
        if rejected:
            detail += f" 未导入：{'、'.join(rejected)}。"
        self.send_json({"ok": True, "message": detail, "documents": list_documents()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic RAG web interface.")
    parser.add_argument("--host", default="0.0.0.0", help="Local bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Preferred local port.")
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
