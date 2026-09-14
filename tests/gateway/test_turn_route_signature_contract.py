"""The middle layer must ACCEPT every keyword the turn runner passes.

The bug this pins, found on the FIRST real Matrix turn through the peer install
(not by any test):

    TypeError: GatewayAgentCacheMixin._resolve_effective_turn_route()
               got an unexpected keyword argument 'prefix_hash'

The turn runner computes the FR-22 prefix triple and forwards `prefix_hash` /
`tool_set_version` into `_resolve_effective_turn_route`. That method declared its
parameters explicitly (no `**kwargs`), so the new keywords raised. Every turn died
on the way to the hook.

Why the existing tests missed it: they drove `resolve_turn_route` — the hook
function one layer BELOW this method — so the intermediate signature was never
exercised. A test that calls the layer under the caller proves nothing about the
call the caller actually makes.

These tests assert the contract directly: whatever the turn runner passes must be
accepted here.
"""
import inspect

import pytest

from gateway.run_agent_cache import GatewayAgentCacheMixin


def test_signature_accepts_the_prefix_keywords():
    """The exact keywords the turn runner forwards must be declared."""
    sig = inspect.signature(GatewayAgentCacheMixin._resolve_effective_turn_route)
    params = sig.parameters
    assert "prefix_hash" in params, (
        "the turn runner passes prefix_hash, which this signature does not "
        "declare — every turn raises TypeError before reaching the hook")
    assert "tool_set_version" in params, (
        "the turn runner passes tool_set_version, which this signature does not "
        "declare")


def test_prefix_keywords_are_optional():
    """Callers that predate the prefix triple (or a non-local route) must still work."""
    sig = inspect.signature(GatewayAgentCacheMixin._resolve_effective_turn_route)
    for name in ("prefix_hash", "tool_set_version"):
        p = sig.parameters[name]
        assert p.default is not inspect.Parameter.empty, (
            f"{name} must have a default; otherwise every existing caller that "
            "does not pass it breaks")


def test_prefix_keywords_are_keyword_only():
    """Positional order must not become load-bearing between the two layers."""
    sig = inspect.signature(GatewayAgentCacheMixin._resolve_effective_turn_route)
    for name in ("prefix_hash", "tool_set_version"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} should be keyword-only, matching the other turn facts")


def test_metadata_carries_the_prefix_only_when_present(monkeypatch):
    """Empty values are OMITTED, so a non-local route reports no phantom prefix.

    Drives the real method with a stubbed hook, capturing what it forwards.
    """
    from gateway import run_route_hook

    seen = {}

    def fake_resolve(session_key, route, turn_metadata=None):
        seen["metadata"] = dict(turn_metadata or {})
        return route

    monkeypatch.setattr(run_route_hook, "resolve_turn_route", fake_resolve)

    class _Host(GatewayAgentCacheMixin):
        pass

    host = _Host()
    route = {"model": "qwen-kraken", "runtime": {"base_url": "http://gate:8090/v1"}}

    # With a prefix: both fields forwarded.
    host._resolve_effective_turn_route(
        "s1", dict(route), platform="cli", needs_multimodal=False,
        prefix_hash="ph1:abc", tool_set_version="v1")
    assert seen["metadata"].get("prefix_hash") == "ph1:abc"
    assert seen["metadata"].get("tool_set_version") == "v1"

    # Without: absent, not blank.
    host._resolve_effective_turn_route("s1", dict(route), platform="cli")
    assert "prefix_hash" not in seen["metadata"], (
        "an empty prefix_hash was forwarded; a blank field reads as a measured "
        "zero to anyone reading the decision history")
    assert "tool_set_version" not in seen["metadata"]
