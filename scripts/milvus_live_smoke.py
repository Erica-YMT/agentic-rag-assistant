"""Live Milvus acceptance checks for public and private RAG scopes.

This script is intentionally non-destructive. It only reads collections that
were created by ``build_milvus_shadow.py`` or the document lifecycle service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from rag.incremental_index import _embedding_model
from rag.milvus_sync import get_scoped_milvus_store


def _source(document) -> str:
    metadata = document.metadata or {}
    return str(
        metadata.get("relative_path")
        or metadata.get("file_name")
        or metadata.get("source")
        or ""
    )


def _check_scope(*, user_id: int | None, query_vector: list[float], query: str) -> dict:
    store = get_scoped_milvus_store(settings=config, user_id=user_id)
    if not store.collection_exists():
        raise RuntimeError(
            f"Milvus collection 不存在: {store.collection_name}"
        )
    if not store.schema_compatible():
        raise RuntimeError(
            f"Milvus schema 不兼容: {store.collection_name}"
        )

    documents = store.list_documents()
    parents = store.list_parent_records()
    results = store.search(query_vector=query_vector, k=3)

    if not documents:
        raise RuntimeError(f"Milvus collection 没有 Child records: {store.collection_name}")
    if not parents:
        raise RuntimeError(f"Milvus collection 没有 Parent records: {store.collection_name}")
    if not results:
        raise RuntimeError(f"Milvus 检索没有结果: {store.collection_name}, query={query!r}")

    return {
        "collection": store.collection_name,
        "user_id": user_id,
        "child_count": len(documents),
        "parent_count": len(parents),
        "top_source": _source(results[0].document),
        "top_score": results[0].score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus public/private live smoke test")
    parser.add_argument("--query", default="青春红音是什么")
    parser.add_argument("--private-user-id", type=int)
    parser.add_argument("--other-private-user-id", type=int)
    args = parser.parse_args()

    embedding_model = _embedding_model(config)
    query_vector = embedding_model.embed_query(args.query)

    checks = [_check_scope(user_id=None, query_vector=query_vector, query=args.query)]

    if args.private_user_id is not None:
        private = _check_scope(
            user_id=args.private_user_id,
            query_vector=query_vector,
            query=args.query,
        )
        if private["collection"] == checks[0]["collection"]:
            raise RuntimeError("公共与私有 Milvus collection 不得相同")

        expected_marker = f"users/{args.private_user_id}/"
        if private["top_source"] and expected_marker not in private["top_source"].replace("\\", "/"):
            raise RuntimeError(
                "私有检索结果来源不属于当前用户: "
                f"{private['top_source']}"
            )
        checks.append(private)

    if args.other_private_user_id is not None:
        if args.private_user_id is None:
            raise ValueError("--other-private-user-id 需要同时提供 --private-user-id")
        other = _check_scope(
            user_id=args.other_private_user_id,
            query_vector=query_vector,
            query=args.query,
        )
        private_collection = checks[1]["collection"]
        if other["collection"] in {checks[0]["collection"], private_collection}:
            raise RuntimeError("不同用户必须使用不同的私有 Milvus collection")
        checks.append(other)

    print(json.dumps({"ok": True, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
