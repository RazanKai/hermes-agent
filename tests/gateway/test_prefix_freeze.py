"""FR-22 §8(a): deterministic prompt-prefix construction (no GPU).

Asserts INVARIANTS, never frozen hash values: the contract is that two turns
differing only in the trailing message produce the same prefix hash, and that the
system+tools bytes are byte-identical across those turns.

Also pins the cross-repo agreement: the Hermes-side freeze and the router-side
``prefix_stability`` implement §6's scheme, so a prefix hash computed on either
side must match for the same inputs. If those two drift, the decision records and
the prompt construction would disagree about what "same prefix" means.
"""
import os
import sys

import pytest

from gateway.prefix_freeze import (
    PREFIX_INVALIDATING,
    STABLE_PREFIX,
    TAIL_CARRIED,
    PrefixFreeze,
    PrefixTriple,
    canonical_tools_bytes,
    classify_field,
    diff_triples,
    freeze_for,
    is_local_route,
    reset,
)

ROUTER_REPO = os.environ.get("HERMES_MODEL_ROUTER_REPO", "/home/nazar/src/hermes-model-router")

TOOLS = ["read_file", "terminal", "web_search"]
DEFS = [
    '{"name":"read_file","parameters":{"type":"object"}}',
    '{"name":"terminal","parameters":{"type":"object"}}',
    '{"name":"web_search","parameters":{"type":"object"}}',
]
SYSTEM = "You are Hermes. Stable system block."
TEMPLATE = "chat_template@v1"


@pytest.fixture(autouse=True)
def clean():
    reset()
    yield
    reset()


def test_same_prefix_across_turns_differing_only_in_the_trailing_message():
    """§8(a) core invariant: the trailing message must not move the prefix."""
    freeze = PrefixFreeze(session_key="s")
    first = freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                           tool_names=TOOLS, tool_definitions=DEFS)
    second = freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                            tool_names=TOOLS, tool_definitions=DEFS)
    assert first["prefix_hash"] == second["prefix_hash"]
    assert second["changed"] == [], second
    assert second["prefix_invalidated"] is False


def test_system_and_tool_bytes_are_byte_identical_across_turns():
    a = canonical_tools_bytes(TOOLS, DEFS)
    b = canonical_tools_bytes(list(reversed(TOOLS)), list(reversed(DEFS)))
    assert a == b, "tool serialization must not depend on input order"


def test_tool_order_does_not_move_the_version():
    assert (PrefixTriple.build(template_content=TEMPLATE, stable_system=SYSTEM,
                               tool_names=TOOLS, tool_definitions=DEFS).tool_set_version
            == PrefixTriple.build(template_content=TEMPLATE, stable_system=SYSTEM,
                                  tool_names=list(reversed(TOOLS)),
                                  tool_definitions=list(reversed(DEFS))).tool_set_version)


def test_a_semantic_change_invalidates_the_prefix():
    """Correctness overrides cache stability (§1)."""
    freeze = PrefixFreeze(session_key="s")
    base = freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                          tool_names=TOOLS, tool_definitions=DEFS)
    moved = freeze.observe(template_content=TEMPLATE,
                           stable_system=SYSTEM + " One more instruction.",
                           tool_names=TOOLS, tool_definitions=DEFS)
    assert moved["prefix_hash"] != base["prefix_hash"]
    assert "stable_system_hash" in moved["changed"]
    assert moved["prefix_invalidated"] is True


def test_tool_definition_change_invalidates_and_requests_a_rebuild():
    """§3: legitimate mid-session, but must rebuild at a turn boundary."""
    freeze = PrefixFreeze(session_key="s")
    freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                   tool_names=TOOLS, tool_definitions=DEFS)
    changed = freeze.observe(
        template_content=TEMPLATE, stable_system=SYSTEM, tool_names=TOOLS + ["memory"],
        tool_definitions=DEFS + ['{"name":"memory","parameters":{}}'])
    assert changed["prefix_invalidated"] is True
    assert changed["needs_rebuild"] is True
    assert changed["tool_set_drifted"] is True
    assert freeze.rebuild_requested is True


def test_the_frozen_tool_set_is_what_gets_reused():
    """§3: resolved once, reused verbatim — not re-resolved per turn."""
    freeze = PrefixFreeze(session_key="s")
    freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                   tool_names=TOOLS, tool_definitions=DEFS)
    freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                   tool_names=TOOLS + ["extra"], tool_definitions=DEFS + ["{}"])
    assert freeze.frozen_tools == tuple(sorted(TOOLS)), "the freeze must not move"


def test_freeze_survives_an_agent_rebuild():
    """The freeze lives per session, not per agent — that is the point."""
    a = freeze_for("sess-x")
    a.observe(template_content=TEMPLATE, stable_system=SYSTEM,
              tool_names=TOOLS, tool_definitions=DEFS)
    # A rebuild gets the same freeze back.
    b = freeze_for("sess-x")
    assert b is a
    assert b.observations == 1
    # A different session is independent.
    assert freeze_for("sess-y") is not a


def test_assert_frozen_guards_serialization_before_first_observe():
    with pytest.raises(RuntimeError):
        PrefixFreeze(session_key="s").assert_frozen()


# ── §4: volatile-tail classification ───────────────────────────────────────

@pytest.mark.parametrize("name", ["tool_authorization", "safety_policy",
                                  "instruction_priority", "gateway_provenance"])
def test_authoritative_fields_are_never_tail_carried(name):
    """§4: authority must not ride the trailing user block."""
    assert classify_field(name) == PREFIX_INVALIDATING


@pytest.mark.parametrize("name", ["timestamp", "counters", "routing_hints",
                                  "latest_user_message", "session_identity_line"])
def test_turn_specific_fields_are_tail_carried(name):
    assert classify_field(name) == TAIL_CARRIED


@pytest.mark.parametrize("name", ["system_prompt", "tool_definitions", "template"])
def test_stable_blocks_are_part_of_the_prefix(name):
    assert classify_field(name) == STABLE_PREFIX


def test_unknown_fields_default_to_authoritative():
    """Guessing wrong toward the tail risks silent authorization drift."""
    assert classify_field("brand_new_thing") == PREFIX_INVALIDATING


def test_session_identity_is_suppressed_from_the_prefix():
    """§4: identity travels via turn metadata (FR-15), not prompt text."""
    assert classify_field("session_identity_line") == TAIL_CARRIED
    freeze = PrefixFreeze(session_key="s")
    a = freeze.observe(template_content=TEMPLATE, stable_system=SYSTEM,
                       tool_names=TOOLS, tool_definitions=DEFS)
    reset()
    b = PrefixFreeze(session_key="other").observe(
        template_content=TEMPLATE, stable_system=SYSTEM,
        tool_names=TOOLS, tool_definitions=DEFS)
    assert a["prefix_hash"] == b["prefix_hash"], "session identity leaked into the prefix"


# ── §7 per-route scoping ───────────────────────────────────────────────────

def test_only_the_local_gate_route_is_frozen():
    gate = "http://100.104.186.86:8090"
    assert is_local_route("qwen-kraken", f"{gate}/v1", gate) is True
    assert is_local_route("muse", "https://opencode.ai/zen/v1", gate) is False
    assert is_local_route("muse", "", gate) is False


# ── cross-repo agreement with the router's §6 implementation ───────────────

def test_hash_scheme_agrees_with_the_router_side_contract():
    """Both sides must compute the same prefix hash for the same components.

    The router logs `prefix_hash` in decision records; the fork computes it at
    construction. If the two schemes drift, a record would describe a prefix the
    request did not have.
    """
    if not os.path.isdir(ROUTER_REPO):
        pytest.skip(f"router repo not present at {ROUTER_REPO}")
    sys.path.insert(0, ROUTER_REPO)
    try:
        from prefix_stability import prefix_hash as router_prefix_hash
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"router prefix_stability unavailable: {exc}")

    tbytes = canonical_tools_bytes(TOOLS, DEFS)
    fork = PrefixTriple.build(template_content=TEMPLATE, stable_system=SYSTEM,
                              tool_names=TOOLS, tool_definitions=DEFS)
    router = router_prefix_hash(TEMPLATE, SYSTEM, tbytes, "")
    assert fork.prefix_hash == router, (
        "the fork and the router compute different prefix hashes for identical "
        "components — decision records would not describe the sent prefix")


def test_diff_reports_first_observation_without_claiming_invalidation():
    assert diff_triples(None, {"prefix_hash": "ph1:x"}) == {
        "changed": [], "first_observation": True, "prefix_invalidated": False}
