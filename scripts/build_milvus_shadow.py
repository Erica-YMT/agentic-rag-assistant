from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from rag.milvus_sync import sync_milvus_from_faiss


def _resolve_project_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_milvus_shadow() -> None:
    """
    Backward-compatible command name.

    The old "shadow" implementation has been retired. This now performs the
    canonical public FAISS -> Milvus full synchronization, including Parent
    records required by Auto-Merging.
    """

    embedding = config.get("embedding", {})
    index_path = _resolve_project_path(
        embedding.get("index_path", "faiss_index")
    )

    print("\n========== Milvus Full Sync ==========")
    print(f"FAISS Source：{index_path}")

    summary = sync_milvus_from_faiss(
        index_path=index_path,
        user_id=None,
    )

    print(f"✅ Collection：{summary['collection_name']}")
    print(f"✅ Child Chunk：{summary['child_count']}")
    print(f"✅ Parent Chunk：{summary['parent_count']}")
    print(f"✅ Embedding dimension：{summary['dimension']}")


if __name__ == "__main__":
    build_milvus_shadow()
