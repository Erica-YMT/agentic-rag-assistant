"""
Agent Session Service.

负责：
1. session_id -> Agent
2. Session Lock
3. Global Agent Semaphore
4. 内存 Session 删除

不负责 HTTP、SQLite、RAG、Streaming。
"""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import time
from threading import BoundedSemaphore, Lock

from config import config
from app.agent.agent import Agent


logger = logging.getLogger(__name__)


class AgentSessionService:

    def __init__(self):

        self._agents: dict[str, Agent] = {}
        self._session_locks = {}
        self._registry_lock = Lock()

        concurrency_config = config.get(
            "concurrency",
            {},
        )

        try:
            max_concurrent_chats = int(
                concurrency_config.get(
                    "max_concurrent_chats",
                    2,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            max_concurrent_chats = 2

        if not 1 <= max_concurrent_chats <= 32:
            raise ValueError(
                "[concurrency].max_concurrent_chats "
                "必须在 1 到 32 之间"
            )

        self._max_concurrent_chats = (
            max_concurrent_chats
        )

        self._agent_semaphore = (
            BoundedSemaphore(
                max_concurrent_chats
            )
        )

        logger.info(
            "Agent 最大并发数：%s",
            max_concurrent_chats,
        )

    @property
    def max_concurrent_chats(self) -> int:
        return self._max_concurrent_chats

    def get_session_agent(
        self,
        session_id: str,
    ):

        with self._registry_lock:

            if session_id not in self._agents:

                logger.info(
                    "创建新会话 Agent：%s",
                    session_id,
                )

                self._agents[
                    session_id
                ] = Agent()

                self._session_locks[
                    session_id
                ] = Lock()

            return (
                self._agents[
                    session_id
                ],
                self._session_locks[
                    session_id
                ],
            )

    @contextmanager
    def global_run_slot(
        self,
        session_id: str,
    ) -> Iterator[None]:

        wait_start = time.perf_counter()

        self._agent_semaphore.acquire()

        wait_seconds = (
            time.perf_counter()
            - wait_start
        )

        try:

            if wait_seconds >= 0.01:

                logger.info(
                    "Agent 并发排队："
                    "session_id=%s，"
                    "等待=%.3f 秒",
                    session_id,
                    wait_seconds,
                )

            yield

        finally:

            self._agent_semaphore.release()

    def remove_session(
        self,
        session_id: str,
    ) -> bool:

        with self._registry_lock:

            existed = (
                session_id
                in self._agents
            )

            self._agents.pop(
                session_id,
                None,
            )

            self._session_locks.pop(
                session_id,
                None,
            )

        return existed


agent_session_service = AgentSessionService()
