"""The prefix triple is established BEFORE the route hook, and reaches it.

Why the ordering is load-bearing:

1. FR-12(b) defines the triple as the session's frozen-prefix state at the
   pre-cache route-resolution point — the moment the route is decided.
2. FR-20 needs the hash in the hook's `turn_metadata` so the plugin's decision
   record carries the prefix the request was actually built with.

Computed after the hook, it satisfies neither: the value is defined too late to
describe the routing decision, and the plugin can only ever see an empty string.
That was the shipped state (`prefix_hash: ""` in every record), and it is invisible
at runtime — a record with a blank field just looks like a record.

Follows the established pattern in `test_resolve_turn_route.py`: drive the real
`TurnRunner.run_sync` body and capture the hook's kwargs by raising inside the
stubbed call site. A direct helper call would pass on the unfixed revision too.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.turn_context import TurnContext


def _run_and_capture(monkeypatch, *, route_model, base_url, session_key):
    """Drive the real turn body; return the kwargs the hook was called with."""
    from gateway.run_turn_runner import TurnRunner

    seen = {}

    class _Runner(MagicMock):
        _provider_routing = {}

        def _resolve_session_agent_runtime(self, **_kw):
            return route_model, {"api_key": "k", "base_url": base_url}

        def _resolve_session_reasoning_config(self, **_kw):
            return None

        def _resolve_session_service_tier(self, **_kw):
            return None

        def _resolve_turn_agent_config(self, _msg, model, _rt):
            return {"model": model, "runtime": {"base_url": base_url}}

        def _resolve_effective_turn_route(self, session_key, route, **kw):
            seen["session_key"] = session_key
            seen["kwargs"] = kw
            seen["route_at_hook"] = route
            raise RuntimeError("stop here: route resolution captured")

        def _pending_native_image_paths(self, session_key):
            return []

        def _consume_pending_native_image_paths(self, session_key):
            return []

    runner = _Runner()
    runner.config = SimpleNamespace(streaming=None)
    runner._get_system_prompt_for_channel.return_value = None

    ctx = TurnContext(
        source=SimpleNamespace(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="hello",
        history=[],
        session_id="sid", session_key=session_key, user_config={},
        AIAgent=None, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )
    with pytest.raises(RuntimeError):
        TurnRunner(runner, ctx).run_sync()
    return seen


def test_hook_receives_prefix_hash_for_a_local_route(monkeypatch):
    """A gate-routed turn must hand the hook the prefix hash it was built with."""
    monkeypatch.setenv("HERMES_MODEL_ROUTER_URL", "http://gate:8090")
    from gateway import prefix_freeze
    prefix_freeze.reset()
    try:
        seen = _run_and_capture(
            monkeypatch,
            route_model="qwen-kraken",
            base_url="http://gate:8090/v1",
            session_key="sess-local",
        )
    finally:
        prefix_freeze.reset()

    kw = seen["kwargs"]
    assert "prefix_hash" in kw, (
        "the hook was called without prefix_hash, so the plugin's decision record "
        "cannot carry it (FR-20) — the triple is being observed after the hook")
    assert kw["prefix_hash"].startswith("ph1:"), (
        f"prefix_hash is not a ph1 fingerprint: {kw['prefix_hash']!r}")
    assert "tool_set_version" in kw


def test_hook_receives_no_prefix_hash_for_a_non_local_route(monkeypatch):
    """§7 per-route scoping: a cloud turn is not prefixed, and must not pretend to be."""
    monkeypatch.setenv("HERMES_MODEL_ROUTER_URL", "http://gate:8090")
    from gateway import prefix_freeze
    prefix_freeze.reset()
    try:
        seen = _run_and_capture(
            monkeypatch,
            route_model="muse",
            base_url="https://opencode.ai/zen/v1",
            session_key="sess-cloud",
        )
    finally:
        prefix_freeze.reset()

    assert seen["kwargs"].get("prefix_hash") == "", (
        "a non-local route reported a prefix hash, which would put a phantom "
        "prefix in the decision record")


def test_the_same_session_reports_a_stable_hash_across_turns(monkeypatch):
    """§8(a): the trailing message must not move the prefix hash."""
    monkeypatch.setenv("HERMES_MODEL_ROUTER_URL", "http://gate:8090")
    from gateway import prefix_freeze
    prefix_freeze.reset()
    try:
        first = _run_and_capture(monkeypatch, route_model="qwen-kraken",
                                 base_url="http://gate:8090/v1",
                                 session_key="sess-stable")["kwargs"]["prefix_hash"]
        second = _run_and_capture(monkeypatch, route_model="qwen-kraken",
                                  base_url="http://gate:8090/v1",
                                  session_key="sess-stable")["kwargs"]["prefix_hash"]
    finally:
        prefix_freeze.reset()

    assert first == second, (
        "the prefix hash changed between two identical turns in one session; the "
        "per-session freeze is not holding")
