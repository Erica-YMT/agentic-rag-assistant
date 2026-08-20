from __future__ import annotations

import threading
import time

from app.agent.tool_reliability import (
    ToolReliabilityController,
    ToolReliabilitySettings,
)
from app.db.tool_dlq import _redact


def settings(**overrides):
    values = ToolReliabilitySettings(
        max_queue_size=2,
        enqueue_timeout_seconds=0.02,
        execution_timeout_seconds=0.2,
        max_attempts=2,
        retry_backoff_seconds=0.001,
        rate_limit_per_second=0.0,
        rate_limit_burst=1,
        circuit_failure_threshold=3,
        circuit_recovery_seconds=0.2,
        dlq_max_size=20,
    )
    return ToolReliabilitySettings(
        **{**values.__dict__, **overrides}
    )


class FakeDLQStore:
    def __init__(self):
        self.items = []

    def enqueue(self, **kwargs):
        self.items.append(kwargs)


def test_dlq_redacts_sensitive_arguments():
    assert _redact({"token": "secret", "query": "safe"}) == {
        "token": "***REDACTED***",
        "query": "safe",
    }


def test_retry_reexecutes_transient_failure():
    controller = ToolReliabilityController(settings(max_attempts=2))
    calls = []

    def flaky(value):
        calls.append(value)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return "ok"

    result = controller.execute(
        tool_name="flaky",
        arguments={"value": 7},
        function=flaky,
    )

    assert result.status == "success"
    assert result.attempts == 2
    assert calls == [7, 7]
    assert controller.dlq_snapshot() == []


def test_timeout_goes_to_dlq():
    durable_dlq = FakeDLQStore()
    controller = ToolReliabilityController(
        settings(max_attempts=1),
        dlq_store=durable_dlq,
    )

    result = controller.execute(
        tool_name="slow",
        arguments={},
        function=lambda: time.sleep(0.5),
    )

    assert result.status == "timeout"
    assert controller.dlq_snapshot()[0].reason == "timeout"
    assert durable_dlq.items[0]["reason"] == "timeout"


def test_rate_limit_and_circuit_breaker():
    controller = ToolReliabilityController(
        settings(
            max_attempts=1,
            rate_limit_per_second=1.0,
            rate_limit_burst=1,
            circuit_failure_threshold=2,
        )
    )

    assert controller.execute(
        tool_name="limited",
        arguments={},
        function=lambda: "ok",
    ).status == "success"
    assert controller.execute(
        tool_name="limited",
        arguments={},
        function=lambda: "ok",
    ).status == "rate_limited"

    circuit = ToolReliabilityController(
        settings(
            max_attempts=1,
            circuit_failure_threshold=2,
        )
    )
    failing = lambda: (_ for _ in ()).throw(RuntimeError("down"))
    assert circuit.execute(
        tool_name="broken",
        arguments={},
        function=failing,
    ).status == "error"
    assert circuit.execute(
        tool_name="broken",
        arguments={},
        function=failing,
    ).status == "error"
    assert circuit.execute(
        tool_name="broken",
        arguments={},
        function=failing,
    ).status == "circuit_open"


def test_queue_full_is_dead_lettered_without_opening_circuit():
    controller = ToolReliabilityController(
        settings(
            max_queue_size=1,
            enqueue_timeout_seconds=0.01,
            execution_timeout_seconds=0.5,
            max_attempts=1,
            circuit_failure_threshold=1,
        )
    )
    started = threading.Event()
    release = threading.Event()

    def blocked():
        started.set()
        release.wait(1.0)
        return "ok"

    worker = threading.Thread(
        target=lambda: controller.execute(
            tool_name="busy",
            arguments={},
            function=blocked,
        )
    )
    worker.start()
    assert started.wait(1.0)

    result = controller.execute(
        tool_name="busy",
        arguments={},
        function=lambda: "second",
    )
    assert result.status == "queue_full"
    assert controller.dlq_snapshot()[-1].reason == "queue_full"
    release.set()
    worker.join(timeout=2.0)
