"""Tests for the generic pre-cache ``resolve_turn_route`` hook.

The hook resolves the effective route after ``_resolve_turn_agent_config`` and
before ``_resolve_turn_agent`` compares the cache signature. No consumer must
mean a byte-identical route (standard behavior unchanged).
"""

import pytest

from gateway import run_route_hook
from gateway.config import Platform
from gateway.run_route_hook import ROUTING_API_VERSION, resolve_turn_route
from types import SimpleNamespace
from unittest.mock import MagicMock


def _route(model="base-model", provider="base-provider"):
    return {
        "model": model,
        "runtime": {
            "provider": provider, "api_key": "k", "base_url": "https://x/v1",
            "api_mode": "chat_completions", "requested_provider": None,
            "capabilities": {}, "credential_pool": None, "max_tokens": None,
            "command": None, "args": [],
        },
    }


def test_no_consumer_returns_configured_route_unchanged(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    route = _route()
    assert resolve_turn_route("sess", route, {"platform": "matrix"}) is route


def test_consumer_replacement_wins(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    replacement = _route(model="other-model", provider="other")
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda name, **kw: [None, replacement])
    out = resolve_turn_route("sess", _route(), None)
    assert out is replacement


def test_invalid_results_keep_configured_route(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: ["a string", {"model": "x"}, {"runtime": {}}, 42],
    )
    route = _route()
    assert resolve_turn_route("sess", route) is route


def test_hook_failure_keeps_configured_route(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

    def _boom(name, **kw):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _boom)
    route = _route()
    assert resolve_turn_route("sess", route) is route


def test_hook_receives_session_configured_and_metadata(monkeypatch):
    seen = {}
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

    def _capture(name, **kw):
        seen.update(kw)
        return [None]

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _capture)
    route = _route()
    resolve_turn_route("sess-9", route, {"platform": "matrix", "message_chars": 12})
    assert seen["session_key"] == "sess-9"
    assert seen["configured_route"] is route
    assert seen["turn_metadata"] == {"platform": "matrix", "message_chars": 12}
    assert seen["routing_api_version"] == ROUTING_API_VERSION


def test_route_change_busts_cached_agent_signature():
    """A rewritten model/provider must change the cache key so the stale agent
    is evicted, not reused (FR-13 via the existing signature)."""
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    before = _route()
    after = _route(model="qwen-kraken", provider="custom")
    sig = lambda r: GatewayAgentCacheMixin._agent_config_signature(  # noqa: E731
        r["model"], r["runtime"], ["messaging"], "", cache_keys={})
    assert sig(before) != sig(after)
    assert sig(before) == sig(_route())


def test_mixin_helper_delegates_to_hook(monkeypatch):
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    helper = GatewayAgentCacheMixin._resolve_effective_turn_route
    route = _route()
    fake_self = object.__new__(GatewayAgentCacheMixin)
    assert helper(fake_self, "s", route, platform="matrix", message_chars=3) is route


def test_mixin_helper_forwards_turn_capability_facts(monkeypatch):
    """A multimodal turn must reach lane selection as a fact, not be inferred.

    ``needs_multimodal`` is a hard lane constraint for the router; when the helper
    sent only platform/message length, the constraint was unreachable and a
    text-only lane could serve a turn carrying an image.
    """
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    seen = {}
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

    def _capture(name, **kw):
        seen.update(kw)
        return [None]

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _capture)
    helper = GatewayAgentCacheMixin._resolve_effective_turn_route
    fake_self = object.__new__(GatewayAgentCacheMixin)
    helper(fake_self, "s", _route(), platform="matrix", message_chars=3,
           needs_multimodal=True, max_context_tokens=200_000)
    assert seen["turn_metadata"]["needs_multimodal"] is True
    assert seen["turn_metadata"]["max_context_tokens"] == 200_000


def test_background_task_session_key_call_shape():
    """/bg passed ONE argument to a two-argument method, killing every /bg task.

    Argument binding is the failure, so assert it directly rather than relying on
    a full background-task run.
    """
    import inspect

    from gateway.run import GatewayRunner

    params = list(inspect.signature(
        GatewayRunner._resolve_session_key_or_none).parameters)
    assert params == ["self", "source", "session_key"], params
    runner = object.__new__(GatewayRunner)
    # The fixed call shape must bind.
    runner._resolve_session_key_or_none(None, None)


def test_main_turn_call_site_forwards_multimodal_and_context():
    """The MAIN turn call must state its real requirements, not just the /bg path.

    The review found the main call site at gateway/run_turn_runner.py supplied
    neither modality nor context, so a text-only lane could serve a turn
    carrying an image. Driving the real TurnRunner body is the only way to prove
    the call SITE passes them (a direct helper call works on the unfixed
    revision too).
    """
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    seen = {}

    class _Runner(MagicMock):
        _provider_routing = {}

        def _resolve_session_agent_runtime(self, **_kw):
            return "local-model", {"api_key": "k", "base_url": "http://x/v1"}

        def _resolve_session_reasoning_config(self, **_kw):
            return None

        def _resolve_session_service_tier(self, **_kw):
            return None

        def _resolve_turn_agent_config(self, _msg, model, _rt):
            return {"model": model, "runtime": {"base_url": "http://x/v1"}}

        def _resolve_effective_turn_route(self, session_key, route, **kw):
            seen["session_key"] = session_key
            seen["kwargs"] = kw
            raise RuntimeError("stop here: route resolution captured")

        # Native image buffer: this session has pixels buffered for the turn.
        def _pending_native_image_paths(self, session_key):
            return ["/tmp/pic.png"] if session_key == "sess-img" else []

        def _consume_pending_native_image_paths(self, session_key):
            return []

    runner = _Runner()
    runner.config = SimpleNamespace(streaming=None)
    runner._get_system_prompt_for_channel.return_value = None

    ctx = TurnContext(
        source=SimpleNamespace(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="look at this",
        history=[{"role": "user", "content": "earlier question"},
                 {"role": "assistant", "content": "earlier answer"}],
        session_id="sid", session_key="sess-img", user_config={},
        AIAgent=None, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )
    with pytest.raises(RuntimeError):
        TurnRunner(runner, ctx).run_sync()

    assert seen["session_key"] == "sess-img"
    assert seen["kwargs"]["needs_multimodal"] is True, (
        "the main turn call did not tell the router the turn carries an image")
    # Context requirement is the turn's real size, so a too-small lane is
    # rejected rather than selected on score alone.
    assert isinstance(seen["kwargs"]["max_context_tokens"], int)
    assert seen["kwargs"]["max_context_tokens"] >= 10

    # The peek must NOT consume the buffer: those same paths are attached to the
    # request later by _native_image_run_message.
    consumed = []
    ctx2 = TurnContext(source=ctx.source, message="m", history=[], session_id="s",
                       session_key="sess-img", user_config={}, AIAgent=None,
                       resolve_display_setting=lambda *_a: False,
                       _run_still_current=lambda: True,
                       _hooks_ref=SimpleNamespace(loaded_hooks=False))

    class _Peek(_Runner):
        def _resolve_effective_turn_route(self, session_key, route, **kw):
            return route

        def _pending_native_image_paths(self, session_key):
            return ["/tmp/pic.png"]

        def _consume_pending_native_image_paths(self, session_key):
            consumed.append(session_key)
            return []

    facts = TurnRunner(_Peek(), ctx2)._turn_route_facts("cli", "m")
    assert facts["needs_multimodal"] is True
    assert consumed == [], "the route hook consumed the image buffer"
