import asyncio
import threading
import time

import httpx

import api


# =========================================================
# Fake Agent
#
# 不调用真实 LLM。
# 每次 run 固定睡 2 秒，
# 用来模拟一次比较慢的模型请求。
# =========================================================

class FakeAgent:

    def run(
        self,
        session_id,
        question,
    ):
        print(
            f"[FakeAgent] 开始：{session_id}"
        )

        time.sleep(2)

        print(
            f"[FakeAgent] 完成：{session_id}"
        )

        return (
            f"并发测试完成：{session_id}"
        )

    def get_called_tools(self):
        return []

    def get_call_records(self):
        return []


# =========================================================
# Fake Session Registry
# =========================================================

fake_agents = {}
fake_locks = {}

registry_lock = threading.Lock()


def fake_get_session_agent(
    session_id,
):
    with registry_lock:

        if session_id not in fake_agents:

            fake_agents[session_id] = (
                FakeAgent()
            )

            fake_locks[session_id] = (
                threading.Lock()
            )

        return (
            fake_agents[session_id],
            fake_locks[session_id],
        )


# 替换 api.py 原来的 get_session_agent
api.get_session_agent = (
    fake_get_session_agent
)


# =========================================================
# 单个请求
# =========================================================

async def send_request(
    client,
    session_id,
):
    start = time.perf_counter()

    response = await client.post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "并发测试",
        },
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"[请求完成] "
        f"{session_id}："
        f"{elapsed:.2f} 秒，"
        f"HTTP {response.status_code}"
    )

    return elapsed


# =========================================================
# 测试一：不同 session
# =========================================================

async def test_different_sessions(
    client,
):
    print()
    print("=" * 60)
    print(
        "测试 1：不同 session_id"
    )
    print("=" * 60)

    start = time.perf_counter()

    await asyncio.gather(
        send_request(
            client,
            "user-a",
        ),
        send_request(
            client,
            "user-b",
        ),
    )

    total = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"不同会话总耗时："
        f"{total:.2f} 秒"
    )

    return total


# =========================================================
# 测试二：相同 session
# =========================================================

async def test_same_session(
    client,
):
    print()
    print("=" * 60)
    print(
        "测试 2：相同 session_id"
    )
    print("=" * 60)

    start = time.perf_counter()

    await asyncio.gather(
        send_request(
            client,
            "same-user",
        ),
        send_request(
            client,
            "same-user",
        ),
    )

    total = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"相同会话总耗时："
        f"{total:.2f} 秒"
    )

    return total


# =========================================================
# 主程序
# =========================================================

async def main():

    transport = httpx.ASGITransport(
        app=api.app
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=20,
    ) as client:

        different_total = (
            await test_different_sessions(
                client
            )
        )

        same_total = (
            await test_same_session(
                client
            )
        )


    print()
    print("=" * 60)
    print("📊 并发测试结论")
    print("=" * 60)

    print(
        f"不同 session："
        f"{different_total:.2f} 秒"
    )

    print(
        f"相同 session："
        f"{same_total:.2f} 秒"
    )


    if (
        different_total < 3
        and same_total > 3.5
    ):
        print()
        print(
            "✅ FastAPI 不同会话可以并发"
        )

        print(
            "✅ 同一会话被 session lock "
            "正确串行化"
        )

    else:
        print()
        print(
            "⚠️ 结果与预期不同，"
            "需要继续检查并发结构"
        )


if __name__ == "__main__":
    asyncio.run(main())
