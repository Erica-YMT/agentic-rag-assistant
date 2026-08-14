from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
]


def _create_splitter(
    chunk_size: int,
    chunk_overlap: int,
):
    chunk_size = int(
        chunk_size
    )

    chunk_overlap = int(
        chunk_overlap
    )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size 必须大于 0"
        )

    if (
        chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_overlap 必须 >= 0 "
            "且小于 chunk_size"
        )

    return (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function=len,
        )
    )


def build_hierarchical_chunks(
    documents: list,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
):
    """
    构建两层 Parent-Child Chunk。

    Parent：
        不进入 FAISS，
        保存到 parent_store.json。

    Child：
        进入 FAISS，
        metadata 中保存 parent_id。
    """

    if (
        int(parent_chunk_size)
        <= int(child_chunk_size)
    ):
        raise ValueError(
            "parent_chunk_size "
            "必须大于 child_chunk_size"
        )

    parent_splitter = (
        _create_splitter(
            parent_chunk_size,
            parent_chunk_overlap,
        )
    )

    child_splitter = (
        _create_splitter(
            child_chunk_size,
            child_chunk_overlap,
        )
    )

    parent_documents = []
    child_documents = []

    parent_store = {}

    for (
        source_document_index,
        source_document,
    ) in enumerate(
        documents
    ):

        parent_candidates = (
            parent_splitter
            .split_documents(
                [source_document]
            )
        )

        for (
            parent_index,
            parent_document,
        ) in enumerate(
            parent_candidates
        ):

            metadata = dict(
                parent_document.metadata
                or {}
            )

            source = (
                metadata.get(
                    "relative_path"
                )
                or metadata.get(
                    "source"
                )
                or metadata.get(
                    "file_name"
                )
                or "unknown"
            )

            page = metadata.get(
                "page"
            )

            identity = (
                f"{source}|"
                f"{page}|"
                f"{source_document_index}|"
                f"{parent_index}|"
                f"{parent_document.page_content}"
            )

            parent_id = (
                hashlib.sha1(
                    identity.encode(
                        "utf-8"
                    )
                )
                .hexdigest()[:20]
            )

            parent_metadata = {
                **metadata,
                "chunk_level": "parent",
                "parent_id": parent_id,
                "parent_index": (
                    parent_index
                ),
            }

            parent_document.metadata = (
                parent_metadata
            )

            children = (
                child_splitter
                .split_documents(
                    [parent_document]
                )
            )

            if not children:
                continue

            child_count = len(
                children
            )

            parent_document.metadata[
                "parent_child_count"
            ] = child_count

            for (
                child_index,
                child_document,
            ) in enumerate(
                children
            ):

                child_metadata = dict(
                    child_document.metadata
                    or {}
                )

                child_metadata.update(
                    {
                        "chunk_level":
                            "child",

                        "parent_id":
                            parent_id,

                        "child_id":
                            (
                                f"{parent_id}:"
                                f"c{child_index:03d}"
                            ),

                        "child_index":
                            child_index,

                        "parent_child_count":
                            child_count,
                    }
                )

                child_document.metadata = (
                    child_metadata
                )

                child_documents.append(
                    child_document
                )

            parent_documents.append(
                parent_document
            )

            parent_store[
                parent_id
            ] = {
                "page_content":
                    parent_document
                    .page_content,

                "metadata":
                    dict(
                        parent_document
                        .metadata
                    ),
            }

    if not child_documents:
        raise RuntimeError(
            "Parent-Child 切块后"
            "没有生成任何 Child"
        )

    return (
        parent_documents,
        child_documents,
        parent_store,
    )


def save_parent_store(
    path: str | Path,
    parent_store: dict,
):
    """
    保存 Parent Chunk。

    使用 JSON，
    避免额外引入数据库。
    """

    path = Path(
        path
    )

    payload = {
        "version": 1,
        "parent_count": (
            len(parent_store)
        ),
        "parents": parent_store,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
