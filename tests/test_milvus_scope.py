import json
import sys
import types
from types import SimpleNamespace


class _AuditStore:
    def start(self, **kwargs):
        return 1

    def finish(self, *args, **kwargs):
        return None


def _tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
    )


def test_search_knowledge_user_scope_is_injected_by_executor():
    from app.agent.tool_executor import ToolExecutor

    captured = {}

    def fake_search_knowledge(query, _user_id=None):
        captured["query"] = query
        captured["user_id"] = _user_id
        return "ok"

    executor = ToolExecutor(
        {"search_knowledge": fake_search_knowledge},
        audit_store=_AuditStore(),
    )
    executor.set_context(user_id=42, role="user", session_id="s-1")

    result = executor.execute(
        _tool_call("search_knowledge", {"query": "private note"})
    )

    assert result == "ok"
    assert captured == {"query": "private note", "user_id": 42}
    assert executor.get_call_records()[0]["arguments"] == {"query": "private note"}


def test_private_milvus_collection_name_is_hard_isolated(monkeypatch):
    # The helper itself does not need a real Milvus server. Stub only the SDK
    # import so this unit test stays offline.
    fake_pymilvus = types.ModuleType("pymilvus")
    fake_pymilvus.DataType = SimpleNamespace(
        VARCHAR="VARCHAR",
        JSON="JSON",
        FLOAT_VECTOR="FLOAT_VECTOR",
    )
    fake_pymilvus.MilvusClient = object
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    sys.modules.pop("rag.milvus_store", None)
    from rag.milvus_store import scoped_collection_name

    public_name = scoped_collection_name("agentic_rag_chunks")
    user_8 = scoped_collection_name("agentic_rag_chunks", user_id=8)
    user_9 = scoped_collection_name("agentic_rag_chunks", user_id=9)

    assert public_name == "agentic_rag_chunks"
    assert user_8 == "agentic_rag_chunks_user_8"
    assert user_9 == "agentic_rag_chunks_user_9"
    assert user_8 != user_9

    sys.modules.pop("rag.milvus_store", None)


def test_auto_merger_can_use_milvus_parent_records_without_local_file(tmp_path):
    from rag.auto_merger import AutoMerger

    parents = {
        "p-1": {
            "page_content": "parent content",
            "metadata": {
                "parent_id": "p-1",
                "parent_child_count": 2,
                "relative_path": "data/knowledge/users/42/private.txt",
            },
        }
    }

    merger = AutoMerger(
        index_path=str(tmp_path),
        settings={
            "enabled": True,
            "min_child_hits": 2,
            "merge_ratio": 0.5,
            "max_parent_chars": 2400,
        },
        parent_records=parents,
    )

    assert merger.enabled is True
    assert merger.parents == parents
    assert not (tmp_path / "parent_store.json").exists()


def test_milvus_backend_rejects_missing_collection(monkeypatch):
    fake_module = types.ModuleType("rag.milvus_store")

    class MissingCollectionStore:
        collection_name = "agentic_rag_chunks"

        def __init__(self, **kwargs):
            pass

        def collection_exists(self):
            return False

        def schema_compatible(self):
            return False

    fake_module.MilvusStore = MissingCollectionStore
    monkeypatch.setitem(sys.modules, "rag.milvus_store", fake_module)

    from rag.vector_backends import MilvusBackend

    try:
        MilvusBackend(
            embedding_model=object(),
            score_threshold=1.0,
            settings={"metric_type": "COSINE"},
        )
    except RuntimeError as exc:
        assert "Collection 不存在" in str(exc)
    else:
        raise AssertionError("缺失 Milvus collection 必须拒绝启动")
