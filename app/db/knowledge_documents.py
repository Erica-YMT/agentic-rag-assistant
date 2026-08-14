"""PostgreSQL metadata store for public/private knowledge documents."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import hashlib

from app.auth.user_store import UserStore
from app.db.postgres import postgres_connection


_schema_lock = Lock()
_schema_ready = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_schema() -> None:
    global _schema_ready

    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return

        # knowledge_documents references users.id; make sure users exists first.
        UserStore()

        with postgres_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL UNIQUE,
                    scope TEXT NOT NULL,
                    owner_user_id BIGINT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE,
                    uploaded_by_user_id BIGINT NULL
                        REFERENCES users(id)
                        ON DELETE SET NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL DEFAULT 0,
                    index_status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    CONSTRAINT ck_knowledge_documents_scope
                    CHECK (scope IN ('public', 'private')),

                    CONSTRAINT ck_knowledge_documents_owner
                    CHECK (
                        (scope = 'public' AND owner_user_id IS NULL)
                        OR
                        (scope = 'private' AND owner_user_id IS NOT NULL)
                    ),

                    CONSTRAINT ck_knowledge_documents_index_status
                    CHECK (index_status IN ('pending', 'indexed', 'error')),

                    CONSTRAINT ck_knowledge_documents_size
                    CHECK (size_bytes >= 0)
                )
                """
            )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_knowledge_documents_public_filename
                ON knowledge_documents (LOWER(filename))
                WHERE scope = 'public'
                """
            )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_knowledge_documents_private_owner_filename
                ON knowledge_documents (
                    owner_user_id,
                    LOWER(filename)
                )
                WHERE scope = 'private'
                """
            )

        _schema_ready = True


class KnowledgeDocumentStore:
    """Knowledge document metadata + ownership access layer."""

    def __init__(self) -> None:
        _ensure_schema()

    def upsert_document(
        self,
        *,
        filename: str,
        storage_path: str,
        scope: str,
        owner_user_id: int | None,
        uploaded_by_user_id: int | None,
        sha256: str,
        size_bytes: int,
        index_status: str = "pending",
    ) -> dict[str, Any]:
        scope = str(scope).strip().lower()
        filename = Path(str(filename)).name
        now = utc_now()

        if scope not in {"public", "private"}:
            raise ValueError(f"不支持的知识文档 scope：{scope}")

        if scope == "public":
            owner_user_id = None
        elif owner_user_id is None:
            raise ValueError("private 文档必须包含 owner_user_id")

        with postgres_connection() as connection:
            if scope == "public":
                existing = connection.execute(
                    """
                    SELECT id
                    FROM knowledge_documents
                    WHERE scope = 'public'
                      AND LOWER(filename) = LOWER(%s)
                    LIMIT 1
                    """,
                    (filename,),
                ).fetchone()
            else:
                existing = connection.execute(
                    """
                    SELECT id
                    FROM knowledge_documents
                    WHERE scope = 'private'
                      AND owner_user_id = %s
                      AND LOWER(filename) = LOWER(%s)
                    LIMIT 1
                    """,
                    (int(owner_user_id), filename),
                ).fetchone()

            if existing is None:
                row = connection.execute(
                    """
                    INSERT INTO knowledge_documents (
                        filename,
                        storage_path,
                        scope,
                        owner_user_id,
                        uploaded_by_user_id,
                        sha256,
                        size_bytes,
                        index_status,
                        created_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        filename,
                        storage_path,
                        scope,
                        owner_user_id,
                        uploaded_by_user_id,
                        str(sha256),
                        int(size_bytes),
                        str(index_status),
                        now,
                        now,
                    ),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET storage_path = %s,
                        uploaded_by_user_id = %s,
                        sha256 = %s,
                        size_bytes = %s,
                        index_status = %s,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        storage_path,
                        uploaded_by_user_id,
                        str(sha256),
                        int(size_bytes),
                        str(index_status),
                        now,
                        int(existing["id"]),
                    ),
                ).fetchone()

        if row is None:
            raise RuntimeError("保存知识文档元数据失败")
        return dict(row)

    def sync_public_directory(
        self,
        *,
        public_dir: Path,
        project_root: Path,
    ) -> None:
        public_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(public_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".pdf", ".md", ".txt"}:
                continue

            self.upsert_document(
                filename=path.name,
                storage_path=path.relative_to(project_root).as_posix(),
                scope="public",
                owner_user_id=None,
                uploaded_by_user_id=None,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                index_status="indexed",
            )

    def list_visible(
        self,
        *,
        user_id: int,
    ) -> list[dict[str, Any]]:
        with postgres_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM knowledge_documents
                WHERE scope = 'public'
                   OR (
                       scope = 'private'
                       AND owner_user_id = %s
                   )
                ORDER BY
                    CASE scope
                        WHEN 'public' THEN 0
                        ELSE 1
                    END,
                    LOWER(filename),
                    id
                """,
                (int(user_id),),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_by_id(
        self,
        document_id: int,
    ) -> dict[str, Any] | None:
        with postgres_connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM knowledge_documents
                WHERE id = %s
                LIMIT 1
                """,
                (int(document_id),),
            ).fetchone()

        return None if row is None else dict(row)

    def delete_by_id(
        self,
        document_id: int,
    ) -> bool:
        with postgres_connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM knowledge_documents
                WHERE id = %s
                """,
                (int(document_id),),
            )

        return cursor.rowcount > 0

    def mark_scope_status(
        self,
        *,
        scope: str,
        owner_user_id: int | None,
        index_status: str,
    ) -> None:
        now = utc_now()
        scope = str(scope).strip().lower()

        with postgres_connection() as connection:
            if scope == "public":
                connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET index_status = %s,
                        updated_at = %s
                    WHERE scope = 'public'
                    """,
                    (str(index_status), now),
                )
            else:
                connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET index_status = %s,
                        updated_at = %s
                    WHERE scope = 'private'
                      AND owner_user_id = %s
                    """,
                    (str(index_status), now, int(owner_user_id)),
                )

    def count_private(
        self,
        *,
        owner_user_id: int,
    ) -> int:
        with postgres_connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM knowledge_documents
                WHERE scope = 'private'
                  AND owner_user_id = %s
                """,
                (int(owner_user_id),),
            ).fetchone()
        return int(row["count"] if row else 0)
