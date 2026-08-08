import asyncio
import threading
import time

import httpx

import api


class FakeAgent:

    def run(
        self,
        session_id,
        question,
    ):
        print(
            f"▶️ 开始：{session_id}"
        )

        time.sleep(2)

        print(
            f"✅ 完成：{session_id}"
        )

        return "ok"

    def get_called_tools(self):
        return []

    def get_call_records(self):
        return []


agents = {}
locks = {}

registry_lock = threading.Lock()


def fake_get_session_agent(
    session_id,
):
    with registry_lock:

        if session_id not in agents:
            agents[session_id] = FakeAgent()
            locks[session_id] = (
                threading.Lock()
            )

        return (
            agents[session_id],
            locks[session_id],
        )


api.get_session_agent = (
    fake_get_session_agent
)


async def request(
    client,
    session_id,
):

    start = time.perf_counter()

    response = await client.post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "并发限流测试",
        },
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"{session_id} 请求耗时："
        f"{elapsed:.2f} 秒"
    )

    return response


async def main():

    transport = httpx.ASGITransport(
        app=api.app
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        start = time.perf_counter()

        await asyncio.gather(
            request(
                client,
                "user-1",
            ),
            request(
                client,
                "user-2",
            ),
            request(
                client,
                "user-3",
            ),
        )

        total = (
            time.perf_counter()
            - start
        )


    print()
    print(
        "===== 最终结果 ====="
    )

    print(
        f"3 个不同用户总耗时："
        f"{total:.2f} 秒"
    )


    if 3.5 <= total <= 5.0:
        print(
            "✅ 最大并发数 2 生效"
        )
    else:
        print(
            "⚠️ 并发结果与预期不同"
        )


if __name__ == "__main__":
    asyncio.run(main())
