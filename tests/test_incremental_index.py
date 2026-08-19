import json

from langchain_core.documents import Document

import rag.incremental_index as incremental
from build_index import build_index_manifest
from rag.hierarchical_chunks import build_hierarchical_chunks


class FakeDocstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, document_id):
        return self.documents[document_id]


class FakeVectorStore:
    def __init__(self, documents):
        self.index_to_docstore_id = {
            index: document_id
            for index, document_id in enumerate(documents)
        }
        self.docstore = FakeDocstore(documents)
        self.deleted_ids = []
        self.added_documents = []
        self.saved_path = None

    def delete(self, ids):
        self.deleted_ids.extend(ids)
        return True

    def add_documents(self, documents):
        self.added_documents.extend(documents)
        return [f"new-{i}" for i in range(len(documents))]

    def save_local(self, path):
        self.saved_path = str(path)


def _settings():
    return {
        "embedding": {
            "model_path": "models/bge",
            "chunk_size": 600,
            "chunk_overlap": 100,
        },
        "hierarchical_chunking": {
            "enabled": True,
            "parent_chunk_size": 1200,
            "parent_chunk_overlap": 120,
            "child_chunk_size": 400,
            "child_chunk_overlap": 80,
        },
    }


def test_parent_id_is_independent_of_document_order():
    target = Document(
        page_content="目标文档内容。" * 80,
        metadata={"relative_path": "data/knowledge/public/target.txt"},
    )
    other = Document(
        page_content="其他文档内容。" * 80,
        metadata={"relative_path": "data/knowledge/public/other.txt"},
    )

    _, target_children_alone, _ = build_hierarchical_chunks(
        documents=[target],
        parent_chunk_size=300,
        parent_chunk_overlap=30,
        child_chunk_size=120,
        child_chunk_overlap=20,
    )
    _, children_with_other, _ = build_hierarchical_chunks(
        documents=[other, target],
        parent_chunk_size=300,
        parent_chunk_overlap=30,
        child_chunk_size=120,
        child_chunk_overlap=20,
    )

    alone_ids = {
        doc.metadata["parent_id"]
        for doc in target_children_alone
    }
    reordered_target_ids = {
        doc.metadata["parent_id"]
        for doc in children_with_other
        if doc.metadata.get("relative_path") == "data/knowledge/public/target.txt"
    }

    assert alone_ids == reordered_target_ids


def test_manifest_changes_when_chunking_changes():
    settings = _settings()
    before = build_index_manifest(settings)

    settings["hierarchical_chunking"]["child_chunk_size"] = 500
    after = build_index_manifest(settings)

    assert before != after


def test_incremental_update_replaces_only_target_file(monkeypatch, tmp_path):
    monkeypatch.setattr(incremental, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(incremental, "load_config", _settings)
    monkeypatch.setattr(incremental, "_assert_manifest_compatible", lambda *_: None)
    monkeypatch.setattr(incremental, "_embedding_model", lambda *_: object())

    target_file = tmp_path / "data" / "knowledge" / "public" / "target.txt"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("new target", encoding="utf-8")

    old_target = Document(
        page_content="old target",
        metadata={"relative_path": "data/knowledge/public/target.txt"},
    )
    untouched = Document(
        page_content="keep me",
        metadata={"relative_path": "data/knowledge/public/other.txt"},
    )
    fake_store = FakeVectorStore({"old-target": old_target, "other": untouched})
    monkeypatch.setattr(incremental.FAISS, "load_local", lambda *args, **kwargs: fake_store)

    new_child = Document(
        page_content="new target child",
        metadata={"relative_path": "data/knowledge/public/target.txt"},
    )
    new_parent = {
        "new-parent": {
            "page_content": "new target parent",
            "metadata": {"relative_path": "data/knowledge/public/target.txt"},
        }
    }
    monkeypatch.setattr(
        incremental,
        "_build_new_chunks",
        lambda *_: ([new_child], [], new_parent),
    )

    index_path = tmp_path / "faiss_index"
    index_path.mkdir()
    (index_path / "index.faiss").write_bytes(b"fake")
    (index_path / "index.pkl").write_bytes(b"fake")
    (index_path / "parent_store.json").write_text(
        json.dumps(
            {
                "version": 1,
                "parents": {
                    "old-parent": {
                        "page_content": "old",
                        "metadata": {"relative_path": "data/knowledge/public/target.txt"},
                    },
                    "other-parent": {
                        "page_content": "other",
                        "metadata": {"relative_path": "data/knowledge/public/other.txt"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = incremental.incremental_update_faiss(
        index_path=index_path,
        upsert_files=[target_file],
    )

    assert fake_store.deleted_ids == ["old-target"]
    assert fake_store.added_documents == [new_child]
    assert result.removed_chunks == 1
    assert result.added_chunks == 1

    parents = json.loads(
        (index_path / "parent_store.json").read_text(encoding="utf-8")
    )["parents"]
    assert "old-parent" not in parents
    assert "other-parent" in parents
    assert "new-parent" in parents


def test_old_index_without_manifest_requires_full_rebuild(tmp_path):
    index_path = tmp_path / "faiss_index"
    index_path.mkdir()
    (index_path / "index.faiss").write_bytes(b"fake")
    (index_path / "index.pkl").write_bytes(b"fake")

    try:
        incremental._load_manifest(index_path)
    except incremental.IncrementalIndexUnavailable as exc:
        assert "全量重建" in str(exc)
    else:
        raise AssertionError("旧索引缺少 manifest 时必须拒绝直接增量")
