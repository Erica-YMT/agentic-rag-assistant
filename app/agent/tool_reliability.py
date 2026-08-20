"""Reliability controls for synchronous Agent tool calls.

The controller deliberately sits outside Tool Policy: policy decides whether a
call is allowed, while this module decides whether an allowed call can be
scheduled and how failures are handled.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

from config import config
from app.db.tool_dlq import tool_dead_letter_store


@dataclass(frozen=True)
class ToolReliabilitySettings:
    max_queue_size: int = 32
    enqueue_timeout_seconds: float = 0.05
    execution_timeout_seconds: float = 60.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.25
    rate_limit_per_second: float = 0.0
    rate_limit_burst: int = 1
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0
    dlq_max_size: int = 1000

    @classmethod
    def from_config(cls) -> "ToolReliabilitySettings":
        raw = dict(config.get("tool_reliability", {}) or {})

        def positive_float(name: str, default: float) -> float:
            try:
                value = float(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if value >= 0 else default

        def positive_int(name: str, default: int, minimum: int = 0) -> int:
            try:
                value = int(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if value >= minimum else default

        return cls(
            max_queue_size=max(1, positive_int("max_queue_size", 32, 1)),
            enqueue_timeout_seconds=positive_float("enqueue_timeout_seconds", 0.05),
            execution_timeout_seconds=max(
                0.001,
                positive_float("execution_timeout_seconds", 60.0),
            ),
            max_attempts=max(1, positive_int("max_attempts", 2, 1)),
            retry_backoff_seconds=positive_float("retry_backoff_seconds", 0.25),
            rate_limit_per_second=positive_float("rate_limit_per_second", 0.0),
            rate_limit_burst=max(1, positive_int("rate_limit_burst", 1, 1)),
            circuit_failure_threshold=max(
                1,
                positive_int("circuit_failure_threshold", 5, 1),
            ),
            circuit_recovery_seconds=positive_float("circuit_recovery_seconds", 30.0),
            dlq_max_size=max(1, positive_int("dlq_max_size", 1000, 1)),
        )


@dataclass(frozen=True)
class DeadLetter:
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    attempts: int
    created_at: float


@dataclass(frozen=True)
class ToolExecutionResult:
    value: str
    status: str
    attempts: int
    elapsed_seconds: float


@dataclass
class _CircuitState:
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


class _TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = float(rate)
        self.capacity = float(max(1, burst))
        self.tokens = self.capacity
        self.updated_at = time.monotonic()

    def take(self) -> bool:
        if self.rate <= 0:
            return True
        now = time.monotonic()
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class ToolReliabilityController:
    """Queue, timeout, retry, rate-limit, circuit-break and DLQ controls."""

    def __init__(
        self,
        settings: ToolReliabilitySettings | None = None,
        *,
        dlq_store=None,
    ) -> None:
        self.settings = settings or ToolReliabilitySettings.from_config()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.max_queue_size,
            thread_name_prefix="agent-tool",
        )
        self._queue_slots = threading.BoundedSemaphore(self.settings.max_queue_size)
        self._lock = threading.RLock()
        self._buckets: dict[str, _TokenBucket] = {}
        self._circuits: dict[str, _CircuitState] = {}
        self._dlq: deque[DeadLetter] = deque(maxlen=self.settings.dlq_max_size)
        self._dlq_store = dlq_store or tool_dead_letter_store

    def _bucket(self, tool_name: str) -> _TokenBucket:
        with self._lock:
            return self._buckets.setdefault(
                tool_name,
                _TokenBucket(
                    self.settings.rate_limit_per_second,
                    self.settings.rate_limit_burst,
                ),
            )

    def _circuit(self, tool_name: str) -> _CircuitState:
        with self._lock:
            return self._circuits.setdefault(tool_name, _CircuitState())

    def _reject_circuit(self, tool_name: str) -> str | None:
        state = self._circuit(tool_name)
        now = time.monotonic()
        with self._lock:
            if state.opened_at is None:
                return None
            if now - state.opened_at < self.settings.circuit_recovery_seconds:
                return "circuit_open"
            if state.probe_in_flight:
                return "circuit_open"
            state.probe_in_flight = True
            return None

    def _record_success(self, tool_name: str) -> None:
        state = self._circuit(tool_name)
        with self._lock:
            state.failures = 0
            state.opened_at = None
            state.probe_in_flight = False

    def _record_failure(self, tool_name: str) -> None:
        state = self._circuit(tool_name)
        with self._lock:
            state.failures += 1
            state.probe_in_flight = False
            if state.failures >= self.settings.circuit_failure_threshold:
                state.opened_at = time.monotonic()

    def _dead_letter(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        attempts: int,
    ) -> None:
        item = DeadLetter(
            tool_name=tool_name,
            arguments=dict(arguments),
            reason=reason,
            attempts=int(attempts),
            created_at=time.time(),
        )
        with self._lock:
            self._dlq.append(item)
        try:
            self._dlq_store.enqueue(
                tool_name=item.tool_name,
                arguments=item.arguments,
                reason=item.reason,
                attempts=item.attempts,
            )
        except Exception:
            pass

    def dlq_snapshot(self) -> list[DeadLetter]:
        with self._lock:
            return list(self._dlq)

    def drain_dlq(self, limit: int | None = None) -> list[DeadLetter]:
        with self._lock:
            count = len(self._dlq) if limit is None else max(0, int(limit))
            values = []
            for _ in range(min(count, len(self._dlq))):
                values.append(self._dlq.popleft())
            return values

    def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
    ) -> ToolExecutionResult:
        started_at = time.perf_counter()
        name = str(tool_name)

        circuit_reason = self._reject_circuit(name)
        if circuit_reason:
            self._dead_letter(
                tool_name=name,
                arguments=arguments,
                reason=circuit_reason,
                attempts=0,
            )
            return ToolExecutionResult(
                value=f"工具 {name} 执行被熔断：暂时不可用。",
                status=circuit_reason,
                attempts=0,
                elapsed_seconds=time.perf_counter() - started_at,
            )

        if not self._bucket(name).take():
            self._dead_letter(
                tool_name=name,
                arguments=arguments,
                reason="rate_limited",
                attempts=0,
            )
            return ToolExecutionResult(
                value=f"工具 {name} 执行被限流：请稍后重试。",
                status="rate_limited",
                attempts=0,
                elapsed_seconds=time.perf_counter() - started_at,
            )

        def invoke() -> Any:
            return function(**arguments)

        attempts = 0
        last_error: Exception | None = None
        terminal_status: str | None = None
        for attempts in range(1, self.settings.max_attempts + 1):
            if not self._queue_slots.acquire(
                timeout=self.settings.enqueue_timeout_seconds,
            ):
                last_error = RuntimeError("工具队列已满")
                terminal_status = "queue_full"
                break

            future: Future[Any] = self._executor.submit(invoke)
            future.add_done_callback(lambda _: self._queue_slots.release())
            try:
                try:
                    value = future.result(
                        timeout=self.settings.execution_timeout_seconds,
                    )
                    self._record_success(name)
                    return ToolExecutionResult(
                        value=str(value),
                        status="success",
                        attempts=attempts,
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
                except TimeoutError as exc:
                    last_error = exc
                    # Timed-out functions may continue in a worker. Retrying a
                    # potentially side-effecting operation would duplicate it.
                    break
                except Exception as exc:
                    last_error = exc
                    if attempts < self.settings.max_attempts:
                        time.sleep(self.settings.retry_backoff_seconds * attempts)
            finally:
                if future.done() and last_error is not None:
                    # Calling result retrieves the exception and keeps executor
                    # implementations that log unobserved failures quiet.
                    try:
                        future.result()
                    except Exception:
                        pass

        if terminal_status != "queue_full":
            self._record_failure(name)
        if terminal_status == "queue_full":
            status = terminal_status
            message = f"工具 {name} 执行排队失败：队列已满。"
        elif isinstance(last_error, TimeoutError):
            status = "timeout"
            message = f"工具 {name} 执行超时：超过 {self.settings.execution_timeout_seconds:.3g} 秒。"
        else:
            status = "error"
            message = f"工具 {name} 执行失败：{last_error}"

        self._dead_letter(
            tool_name=name,
            arguments=arguments,
            reason=status,
            attempts=attempts,
        )
        return ToolExecutionResult(
            value=message,
            status=status,
            attempts=attempts,
            elapsed_seconds=time.perf_counter() - started_at,
        )


tool_reliability_controller = ToolReliabilityController()
