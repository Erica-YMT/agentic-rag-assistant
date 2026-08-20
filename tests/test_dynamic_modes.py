from rag.dynamic_modes import CHAT, CONSULT, RISK, classify_mode


def test_dynamic_rag_modes():
    assert classify_mode("你好").name == CHAT.name
    assert classify_mode("请分析这个方案").name == CONSULT.name
    assert classify_mode("这个合同有什么法律风险").name == RISK.name
