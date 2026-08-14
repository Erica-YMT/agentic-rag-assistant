"""Knowledge service: public/private document ownership and index lifecycle."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from threading import Lock
from typing import Any

from build_index import build_index
from config import config
from app.db.knowledge_documents import (
    KnowledgeDocumentStore,
    sha256_bytes,
)
from rag.knowledge_base import (
    get_default_knowledge_base,
    reload_default_knowledge_base,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = PROJECT_ROOT / "data" / "knowledge"
PUBLIC_KNOWLEDGE_DIR = KNOWLEDGE_ROOT / "public"
USER_KNOWLEDGE_ROOT = KNOWLEDGE_ROOT / "users"
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class KnowledgeService:
    """Owns knowledge document storage, ownership checks and rebuild locks."""

    def __init__(self):
        self._rebuild_lock = Lock()
        self.document_store = KnowledgeDocumentStore()

    def prepare_document_storage(self) -> None:
        """Create scoped directories and register existing public files."""
        PUBLIC_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        USER_KNOWLEDGE_ROOT.mkdir(parents=True, exist_ok=True)
        self.document_store.sync_public_directory(
            public_dir=PUBLIC_KNOWLEDGE_DIR,
            project_root=PROJECT_ROOT,
        )

    def preload(self):
        return get_default_knowledge_base()

    def search(self, query: str, top_k: int):
        """Admin diagnostic search against the current public knowledge base."""
        kb = get_default_knowledge_base()
        return kb.search(query=query, k=top_k)

    def _default_index_path(self) -> Path:
        embedding_config = config.get("embedding", {})
        return _resolve_project_path(
            embedding_config.get("index_path", "faiss_index")
        )

    def private_document_dir(self, user_id: int) -> Path:
        return (USER_KNOWLEDGE_ROOT / str(int(user_id))).resolve()

    def private_index_dir(self, user_id: int) -> Path:
        return (self._default_index_path() / "users" / str(int(user_id))).resolve()

    def public_index_ready(self) -> bool:
        index_path = self._default_index_path()
        return (
            (index_path / "index.faiss").is_file()
            and (index_path / "index.pkl").is_file()
        )

    def private_index_ready(self, user_id: int) -> bool:
        index_path = self.private_index_dir(user_id)
        return (
            (index_path / "index.faiss").is_file()
            and (index_path / "index.pkl").is_file()
        )

    @staticmethod
    def _is_admin(current_user: dict[str, Any]) -> bool:
        return str(current_user.get("role", "")).lower() == "admin"

    def list_documents(self, current_user: dict[str, Any]) -> dict[str, Any]:
        self.prepare_document_storage()

        user_id = int(current_user["id"])
        is_admin = self._is_admin(current_user)
        rows = self.document_store.list_visible(user_id=user_id)

        documents = []
        public_count = 0
        private_count = 0

        for row in rows:
            scope = str(row["scope"])
            owner_user_id = row.get("owner_user_id")

            if scope == "public":
                public_count += 1
                can_delete = is_admin
            else:
                private_count += 1
                can_delete = int(owner_user_id) == user_id

            documents.append({
                "id": int(row["id"]),
                "filename": str(row["filename"]),
                "scope": scope,
                "owner_user_id": (
                    None if owner_user_id is None else int(owner_user_id)
                ),
                "size_bytes": int(row.get("size_bytes") or 0),
                "index_status": str(row.get("index_status") or "pending"),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "can_delete": bool(can_delete),
            })

        return {
            "documents": documents,
            "public_count": public_count,
            "private_count": private_count,
            "public_index_ready": self.public_index_ready(),
            "private_index_ready": self.private_index_ready(user_id),
        }

    def save_documents(
        self,
        *,
        current_user: dict[str, Any],
        uploads: list[tuple[str, bytes]],
    ) -> list[str]:
        if not uploads:
            raise ValueError("没有可上传的文档")

        self.prepare_document_storage()

        user_id = int(current_user["id"])
        is_admin = self._is_admin(current_user)

        if is_admin:
            scope = "public"
            owner_user_id = None
            destination = PUBLIC_KNOWLEDGE_DIR
        else:
            scope = "private"
            owner_user_id = user_id
            destination = self.private_document_dir(user_id)

        destination.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []

        for original_name, content in uploads:
            safe_name = Path(str(original_name)).name
            suffix = Path(safe_name).suffix.lower()

            if not safe_name or suffix not in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"不支持的文件：{safe_name or original_name}；"
                    "仅支持 PDF、Markdown、TXT"
                )

            if not content:
                raise ValueError(f"文件为空：{safe_name}")

            path = (destination / safe_name).resolve()
            path.relative_to(destination.resolve())

            temp_path = path.with_name(path.name + ".uploading")
            temp_path.write_bytes(content)
            temp_path.replace(path)

            self.document_store.upsert_document(
                filename=safe_name,
                storage_path=path.relative_to(PROJECT_ROOT).as_posix(),
                scope=scope,
                owner_user_id=owner_user_id,
                uploaded_by_user_id=user_id,
                sha256=sha256_bytes(content),
                size_bytes=len(content),
                index_status="pending",
            )

            saved.append(safe_name)

        return saved

    def _rebuild_public_locked(self) -> None:
        if not any(
            path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
            for path in PUBLIC_KNOWLEDGE_DIR.iterdir()
        ):
            raise ValueError("公共知识库没有可用于建库的文档")

        build_index(data_path_override=PUBLIC_KNOWLEDGE_DIR)

        vector_backend = str(
            config.get("vector_store", {}).get("backend", "faiss")
        ).strip().lower()

        if vector_backend not in {"faiss", "milvus"}:
            raise ValueError("[vector_store].backend 只支持 faiss / milvus")

        if vector_backend == "milvus":
            logger.info("开始同步公共 Milvus Collection……")
            from scripts.build_milvus_shadow import build_milvus_shadow
            build_milvus_shadow()
            logger.info("公共 Milvus Collection 同步完成")

        reload_default_knowledge_base()
        self.document_store.mark_scope_status(
            scope="public",
            owner_user_id=None,
            index_status="indexed",
        )

    def _rebuild_private_locked(self, user_id: int) -> None:
        document_dir = self.private_document_dir(user_id)
        index_dir = self.private_index_dir(user_id)

        document_dir.mkdir(parents=True, exist_ok=True)

        has_documents = any(
            path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
            for path in document_dir.iterdir()
        )

        if not has_documents:
            if index_dir.exists():
                shutil.rmtree(index_dir)
            return

        # Phase 1 builds a hard-isolated private FAISS index.
        # Agent + Milvus private retrieval is connected in the next phases.
        build_index(
            data_path_override=document_dir,
            index_path_override=index_dir,
        )

        self.document_store.mark_scope_status(
            scope="private",
            owner_user_id=int(user_id),
            index_status="indexed",
        )

    def rebuild_for_user(self, current_user: dict[str, Any]) -> float:
        start_time = time.perf_counter()
        user_id = int(current_user["id"])

        with self._rebuild_lock:
            if self._is_admin(current_user):
                logger.info("开始重建公共知识库……")
                self._rebuild_public_locked()
            else:
                logger.info("开始重建用户 %s 的私有知识库……", user_id)
                self._rebuild_private_locked(user_id)

        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info("知识库重建完成，耗时 %.2f 秒", elapsed)
        return elapsed

    def rebuild(self) -> float:
        """Compatibility: legacy admin rebuild means rebuild public knowledge base."""
        start_time = time.perf_counter()
        with self._rebuild_lock:
            self._rebuild_public_locked()
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info("公共知识库重建完成，耗时 %.2f 秒", elapsed)
        return elapsed

    def delete_documents(
        self,
        *,
        current_user: dict[str, Any],
        document_ids: list[int],
        rebuild: bool = True,
    ) -> tuple[list[str], bool, float]:
        user_id = int(current_user["id"])
        is_admin = self._is_admin(current_user)

        unique_ids = list(dict.fromkeys(int(value) for value in document_ids))
        if not unique_ids:
            raise ValueError("请至少选择一个文档")

        rows = []
        for document_id in unique_ids:
            row = self.document_store.get_by_id(document_id)
            if row is None:
                raise FileNotFoundError(f"文档不存在：{document_id}")

            scope = str(row["scope"])
            owner_user_id = row.get("owner_user_id")

            allowed = (
                (scope == "public" and is_admin)
                or (
                    scope == "private"
                    and owner_user_id is not None
                    and int(owner_user_id) == user_id
                )
            )

            if not allowed:
                raise PermissionError("无权删除该知识文档")

            rows.append(row)

        selected_public_count = sum(
            1
            for row in rows
            if str(row["scope"]) == "public"
        )

        if selected_public_count:
            current_public_count = sum(
                1
                for path in PUBLIC_KNOWLEDGE_DIR.iterdir()
                if path.is_file()
                and path.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if selected_public_count >= current_public_count:
                raise ValueError(
                    "不能删除全部公共知识文档，请至少保留一个"
                )

        deleted_names = []
        deleted_public = False
        deleted_private = False

        for row in rows:
            path = _resolve_project_path(str(row["storage_path"]))
            path.relative_to(KNOWLEDGE_ROOT.resolve())

            if path.exists() and path.is_file():
                path.unlink()

            self.document_store.delete_by_id(int(row["id"]))
            deleted_names.append(str(row["filename"]))

            if str(row["scope"]) == "public":
                deleted_public = True
            else:
                deleted_private = True

        index_rebuilt = False
        elapsed = 0.0

        if rebuild:
            start = time.perf_counter()
            with self._rebuild_lock:
                if deleted_public:
                    remaining_public = any(
                        p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
                        for p in PUBLIC_KNOWLEDGE_DIR.iterdir()
                    )
                    if not remaining_public:
                        raise ValueError("公共知识库不能删除到 0 个文档")
                    self._rebuild_public_locked()
                    index_rebuilt = True

                if deleted_private:
                    self._rebuild_private_locked(user_id)
                    index_rebuilt = True

            elapsed = round(time.perf_counter() - start, 2)

        return deleted_names, index_rebuilt, elapsed


knowledge_service = KnowledgeService()
