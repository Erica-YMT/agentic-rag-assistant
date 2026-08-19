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
from rag.incremental_index import (
    IncrementalIndexUnavailable,
    incremental_update_faiss,
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
    ) -> list[dict[str, Any]]:
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

        saved: list[dict[str, Any]] = []

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

            row = self.document_store.upsert_document(
                filename=safe_name,
                storage_path=path.relative_to(PROJECT_ROOT).as_posix(),
                scope=scope,
                owner_user_id=owner_user_id,
                uploaded_by_user_id=user_id,
                sha256=sha256_bytes(content),
                size_bytes=len(content),
                index_status="pending",
            )

            saved.append(row)

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

    def _vector_backend_name(self) -> str:
        backend = str(
            config.get("vector_store", {}).get("backend", "faiss")
        ).strip().lower()
        if backend not in {"faiss", "milvus"}:
            raise ValueError("[vector_store].backend 只支持 faiss / milvus")
        return backend

    def _incremental_public_locked(self, rows: list[dict[str, Any]]) -> None:
        document_ids = [int(row["id"]) for row in rows]

        if not self.public_index_ready():
            logger.info("公共索引不存在，自动回退全量构建")
            self._rebuild_public_locked()
            return

        try:
            result = incremental_update_faiss(
                index_path=self._default_index_path(),
                upsert_files=[
                    _resolve_project_path(str(row["storage_path"]))
                    for row in rows
                ],
                sync_milvus=(self._vector_backend_name() == "milvus"),
            )
            logger.info(
                "公共知识库增量完成：removed=%s added=%s",
                result.removed_chunks,
                result.added_chunks,
            )
        except IncrementalIndexUnavailable as exc:
            logger.info("公共索引不满足增量条件，回退全量重建：%s", exc)
            self._rebuild_public_locked()
            return
        except Exception:
            self.document_store.mark_documents_status(
                document_ids=document_ids,
                index_status="error",
            )
            raise

        reload_default_knowledge_base()
        self.document_store.mark_documents_status(
            document_ids=document_ids,
            index_status="indexed",
        )

    def _incremental_private_locked(
        self,
        user_id: int,
        rows: list[dict[str, Any]],
    ) -> None:
        document_ids = [int(row["id"]) for row in rows]
        index_dir = self.private_index_dir(user_id)

        if not self.private_index_ready(user_id):
            logger.info("用户 %s 私有索引不存在，自动回退全量构建", user_id)
            self._rebuild_private_locked(user_id)
            return

        try:
            result = incremental_update_faiss(
                index_path=index_dir,
                upsert_files=[
                    _resolve_project_path(str(row["storage_path"]))
                    for row in rows
                ],
            )
            logger.info(
                "用户 %s 私有知识库增量完成：removed=%s added=%s",
                user_id,
                result.removed_chunks,
                result.added_chunks,
            )
        except IncrementalIndexUnavailable as exc:
            logger.info(
                "用户 %s 私有索引不满足增量条件，回退全量重建：%s",
                user_id,
                exc,
            )
            self._rebuild_private_locked(user_id)
            return
        except Exception:
            self.document_store.mark_documents_status(
                document_ids=document_ids,
                index_status="error",
            )
            raise

        self.document_store.mark_documents_status(
            document_ids=document_ids,
            index_status="indexed",
        )

    def index_documents(
        self,
        *,
        current_user: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> float:
        """上传后的默认路径：只更新本次变化文档；必要时自动全量兜底。"""

        if not rows:
            return 0.0

        start_time = time.perf_counter()
        user_id = int(current_user["id"])

        with self._rebuild_lock:
            if self._is_admin(current_user):
                self._incremental_public_locked(rows)
            else:
                self._incremental_private_locked(user_id, rows)

        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info("知识库索引更新完成，耗时 %.2f 秒", elapsed)
        return elapsed

    def _delete_public_index_locked(self, rows: list[dict[str, Any]]) -> None:
        if not self.public_index_ready():
            self._rebuild_public_locked()
            return

        try:
            result = incremental_update_faiss(
                index_path=self._default_index_path(),
                delete_relative_paths=[str(row["storage_path"]) for row in rows],
                sync_milvus=(self._vector_backend_name() == "milvus"),
            )
            logger.info(
                "公共知识库增量删除完成：removed=%s",
                result.removed_chunks,
            )
        except IncrementalIndexUnavailable as exc:
            logger.info("公共增量删除不可用，回退全量重建：%s", exc)
            self._rebuild_public_locked()
            return

        reload_default_knowledge_base()

    def _delete_private_index_locked(
        self,
        user_id: int,
        rows: list[dict[str, Any]],
    ) -> None:
        index_dir = self.private_index_dir(user_id)

        if self.document_store.count_private(owner_user_id=user_id) <= 0:
            if index_dir.exists():
                shutil.rmtree(index_dir)
            return

        if not self.private_index_ready(user_id):
            self._rebuild_private_locked(user_id)
            return

        try:
            result = incremental_update_faiss(
                index_path=index_dir,
                delete_relative_paths=[str(row["storage_path"]) for row in rows],
            )
            logger.info(
                "用户 %s 私有知识库增量删除完成：removed=%s",
                user_id,
                result.removed_chunks,
            )
        except IncrementalIndexUnavailable as exc:
            logger.info(
                "用户 %s 私有增量删除不可用，回退全量重建：%s",
                user_id,
                exc,
            )
            self._rebuild_private_locked(user_id)

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
                    public_rows = [
                        row for row in rows if str(row["scope"]) == "public"
                    ]
                    self._delete_public_index_locked(public_rows)
                    index_rebuilt = True

                if deleted_private:
                    private_rows = [
                        row for row in rows if str(row["scope"]) == "private"
                    ]
                    self._delete_private_index_locked(user_id, private_rows)
                    index_rebuilt = True

            elapsed = round(time.perf_counter() - start, 2)

        return deleted_names, index_rebuilt, elapsed


knowledge_service = KnowledgeService()
