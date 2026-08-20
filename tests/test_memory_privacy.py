from app.memory.chat_memory import Memory
from app.privacy.pii import contains_pii, sanitize_text


def test_pii_sanitizer_masks_common_identifiers():
    value = sanitize_text("联系 a@example.com 或 13800138000，token=sk-test_12345678")
    assert "a@example.com" not in value
    assert "13800138000" not in value
    assert "sk-test_12345678" not in value
    assert contains_pii(value) is False


def test_memory_compaction_keeps_recent_and_adds_summary(tmp_path):
    memory = Memory(db_path=tmp_path / "memory.db")
    session_id = "s-compact"
    for index in range(6):
        memory.add_message(session_id, "user", f"question {index}")
        memory.add_message(session_id, "assistant", f"answer {index}")

    result = memory.compact_session(session_id, keep_recent=4)
    assert result["compacted"] is True
    messages = memory.get_messages(session_id)
    assert any("【会话摘要】" in item["content"] for item in messages)
    assert len(messages) <= 5
