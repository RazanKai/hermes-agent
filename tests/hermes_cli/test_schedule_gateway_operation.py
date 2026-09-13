"""``PluginContext.schedule_gateway_operation`` — bounded work off the hook path.

A hook callback is abandoned when it overruns its wall-clock cap, and the work
keeps running unreachable. That is fine for a lookup and unacceptable for a
plugin operation that mutates external state (a model load, a provisioning call).
This surface lets such work run on the gateway's loop with its OWN timeout, under
the gateway's task ownership, while the turn returns immediately.
"""

import asyncio
import threading

import pytest

from hermes_cli.plugins import PluginContext, PluginManager


class _FakeRunner:
    def __init__(self, loop):
        self._loop = loop
        self._background_tasks = set()


def _ctx(runner, manager=None):
    """A PluginContext wired to a plausible manager/runner pair.

    ``PluginContext`` is normally constructed by the manager, so build the
    minimum it actually reads rather than mocking the surface under test. The
    manifest only needs to be truthy — ``plugin_id`` is used for log labels.
    """
    from hermes_cli.plugins import PluginManifest

    mgr = manager if manager is not None else PluginManager()
    manifest = PluginManifest(name="probe-plugin", version="0.0.0", kind="standalone")
    ctx = PluginContext(manifest=manifest, manager=mgr)
    mgr._gateway_runner = runner
    return ctx


@pytest.mark.linux_only
def test_operation_runs_on_the_gateway_loop():
    loop = asyncio.new_event_loop()
    runner = _FakeRunner(loop)
    ctx = _ctx(runner)
    seen = {}

    async def work():
        seen["thread"] = threading.current_thread().name
        return "done"

    loop_thread_name = {}
    started = threading.Thread(
        target=lambda: (loop_thread_name.setdefault("n", threading.current_thread().name),
                        loop.run_forever()),
        daemon=True, name="gateway-loop")
    started.start()
    try:
        future = ctx.schedule_gateway_operation("probe", work(), timeout=5)
        assert future is not None, "a running loop must accept the operation"
        assert future.result(timeout=5) == "done"
        # The work ran on the gateway loop's thread, NOT the caller's — that is
        # the whole point: it is not the plugin's abandoned hook worker.
        assert seen["thread"] == loop_thread_name["n"] == "gateway-loop"
        assert seen["thread"] != threading.current_thread().name
    finally:
        loop.call_soon_threadsafe(loop.stop)
        started.join(timeout=5)
        loop.close()


@pytest.mark.linux_only
def test_operation_is_owned_by_the_gateway_for_cancellation():
    """The task must be visible to the gateway, or shutdown cannot cancel it."""
    loop = asyncio.new_event_loop()
    runner = _FakeRunner(loop)
    ctx = _ctx(runner)
    gate = threading.Event()

    async def long_work():
        await asyncio.sleep(30)

    started = threading.Thread(target=lambda: loop.run_forever(), daemon=True)
    started.start()
    try:
        assert ctx.schedule_gateway_operation("slow", long_work(), timeout=60) is not None
        deadline = threading.Event()
        for _ in range(50):
            if runner._background_tasks:
                break
            deadline.wait(0.02)
        assert runner._background_tasks, "operation not registered with the gateway"
    finally:
        for task in list(runner._background_tasks):
            loop.call_soon_threadsafe(task.cancel)
        loop.call_soon_threadsafe(loop.stop)
        started.join(timeout=5)
        loop.close()


@pytest.mark.linux_only
def test_operation_timeout_is_its_own_not_the_hook_cap():
    """A bounded operation must time out and NOT propagate the failure."""
    loop = asyncio.new_event_loop()
    runner = _FakeRunner(loop)
    ctx = _ctx(runner)
    observed = []

    async def too_slow():
        await asyncio.sleep(5)
        observed.append("completed")

    started = threading.Thread(target=lambda: loop.run_forever(), daemon=True)
    started.start()
    try:
        future = ctx.schedule_gateway_operation("slow", too_slow(), timeout=0.1)
        assert future is not None
        assert future.result(timeout=5) is None, "a timeout returns None, never raises"
        assert observed == [], "the work must not have finished"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        started.join(timeout=5)
        loop.close()


def test_operation_failure_does_not_raise_into_the_caller():
    loop = asyncio.new_event_loop()
    runner = _FakeRunner(loop)
    ctx = _ctx(runner)

    async def boom():
        raise RuntimeError("provider exploded")

    started = threading.Thread(target=lambda: loop.run_forever(), daemon=True)
    started.start()
    try:
        future = ctx.schedule_gateway_operation("bad", boom(), timeout=5)
        assert future is not None
        assert future.result(timeout=5) is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        started.join(timeout=5)
        loop.close()


def test_no_loop_returns_none_so_the_caller_can_fall_back():
    """CLI/TUI have no gateway loop: the plugin keeps its synchronous path."""
    ctx = _ctx(_FakeRunner(None))

    async def work():
        return "never"

    coro = work()
    assert ctx.schedule_gateway_operation("probe", coro) is None
    coro.close()  # the caller owns it when we decline


def test_closed_loop_returns_none():
    loop = asyncio.new_event_loop()
    loop.close()
    ctx = _ctx(_FakeRunner(loop))

    async def work():
        return "never"

    coro = work()
    assert ctx.schedule_gateway_operation("probe", coro) is None
    coro.close()
