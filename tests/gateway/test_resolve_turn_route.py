"""Tests for the generic pre-cache ``resolve_turn_route`` hook.

The hook resolves the effective route after ``_resolve_turn_agent_config`` and
before ``_resolve_turn_agent`` compares the cache signature. No consumer must
mean a byte-identical route (standard behavior unchanged).
"""

import pytest

from gateway import run_route_hook
from gateway.run_route_hook import ROUTING_API_VERSION, resolve_turn_route


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
