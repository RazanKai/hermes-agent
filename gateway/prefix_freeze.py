"""Session prefix freeze for the local route (PRD FR-22 §1-§5).

Cost on the local route is prefill-of-delta only while each request begins with
matching rendered tokens backed by resident KV cache. Prompt construction is
already deterministic; what this adds is the *guarantee*, because the failure
mode is silent — a prefix that shifts by one byte still produces a working turn,
just an uncached one, and nothing reports it.

Three jobs, matching §1-§5:

1. **Freeze the tool set per session** (§3). Resolved once, serialized
   deterministically, reused verbatim. Tools are the largest stable block after
   the system prompt and the one most likely to drift (a registry reload, a
   plugin load). A mid-session change is legitimate, so it does not raise: it
   marks the session as needing a rebuild at a turn boundary.
2. **Recompute the triple every turn and report the diff** (§1). Correctness
   overrides cache stability — a semantic change must surface, not be papered
   over.
3. **Classify volatile fields as tail-carried or prefix-invalidating** (§4).
   Turn-specific context rides the trailing user message; anything that affects
   instruction priority, safety policy, tool authorization or provenance must NOT
   be moved to the tail, and changing it invalidates the prefix instead.

Scoped to the local route (§7): callers only engage this for `qwen-kraken`.
Nothing here changes global behaviour, and nothing sets `supports_prompt_cache_key`
(§3 — TabbyAPI does not implement it; reuse is rendered-prefix match only).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger("gateway.run")

HASH_VERSION = "ph1"

# §4: how a per-turn ephemeral field may be treated.
STABLE_PREFIX = "stable-prefix"        # part of the frozen prefix
TAIL_CARRIED = "tail-carried"          # rides the trailing user block
PREFIX_INVALIDATING = "prefix-invalidating"  # changing it must rebuild/invalidate

# §4's inventory, stated once. Moving a field between these classes is a design
# decision, not a refactor: promotion to TAIL_CARRIED is only legal for fields
# that carry no instruction priority, safety policy, authorization or provenance.
FIELD_CLASSIFICATION: dict[str, str] = {
    # Authoritative — never tail-carried, per §4.
    "tool_authorization": PREFIX_INVALIDATING,
    "safety_policy": PREFIX_INVALIDATING,
    "instruction_priority": PREFIX_INVALIDATING,
    "gateway_provenance": PREFIX_INVALIDATING,
    # Frozen with the prefix.
    "system_prompt": STABLE_PREFIX,
    "tool_definitions": STABLE_PREFIX,
    "template": STABLE_PREFIX,
    # Turn-specific; safe at the tail.
    "timestamp": TAIL_CARRIED,
    "counters": TAIL_CARRIED,
    "routing_hints": TAIL_CARRIED,
    "latest_user_message": TAIL_CARRIED,
    "session_identity_line": TAIL_CARRIED,  # §4: suppressed; identity travels via turn metadata
}


def classify_field(name: str) -> str:
    """How this field may be treated. Unknown fields are treated as authoritative.

    Defaulting to PREFIX_INVALIDATING rather than TAIL_CARRIED is deliberate: a
    new field is more likely to carry authority than to be safe at the tail, and
    the cost of guessing wrong in the other direction is silent authorization
    drift.
    """
    return FIELD_CLASSIFICATION.get(name, PREFIX_INVALIDATING)


def _frame(value: bytes) -> bytes:
    """Length-prefix a field (same framing as the router-side contract)."""
    return len(value).to_bytes(8, "big") + value


def canonical_tools_bytes(tool_names: Sequence[str],
                          tool_definitions: Iterable[str]) -> bytes:
    """Canonical tool bytes: sorted names + sorted definitions, each framed.

    Sorting is what makes the version stable across dict iteration order.
    """
    names = "\n".join(sorted(tool_names))
    defs = "\n".join(sorted(tool_definitions))
    return _frame(names.encode()) + _frame(defs.encode())


def tools_version(tool_names: Sequence[str], tool_definitions: Iterable[str]) -> str:
    return hashlib.sha256(canonical_tools_bytes(tool_names, tool_definitions)).hexdigest()[:16]


@dataclass
class PrefixTriple:
    """§1's triple, plus the prefix hash it implies."""

    template_content_hash: str = ""
    stable_system_hash: str = ""
    tool_set_version: str = ""
    tool_names: tuple = ()
    prefix_hash: str = ""

    @classmethod
    def build(cls, *, template_content: str, stable_system: str, tool_names: Sequence[str],
              tool_definitions: Iterable[str], rendering_config: str = "") -> "PrefixTriple":
        """Build the triple and the §6 prefix hash over the same components.

        The prefix hash frames the template CONTENT (not a hash of it) so this
        matches the router-side `prefix_stability.prefix_hash` byte for byte —
        the router logs that hash in decision records, and a record must describe
        the prefix the request actually had. `template_content_hash` is computed
        separately, as the triple's identifying component.
        """
        names = tuple(sorted(tool_names))
        tbytes = canonical_tools_bytes(names, tool_definitions)
        blob = (_frame(template_content.encode())
                + _frame(stable_system.encode())
                + _frame(tbytes)
                + _frame(rendering_config.encode()))
        return cls(
            template_content_hash=hashlib.sha256(template_content.encode()).hexdigest()[:16],
            stable_system_hash=hashlib.sha256(stable_system.encode()).hexdigest()[:16],
            tool_set_version=tools_version(names, tool_definitions),
            tool_names=names,
            prefix_hash=f"{HASH_VERSION}:{hashlib.sha256(blob).hexdigest()}",
        )

    def as_dict(self) -> dict:
        return {
            "template_content_hash": self.template_content_hash,
            "stable_system_hash": self.stable_system_hash,
            "tool_set_version": self.tool_set_version,
            "tool_names": list(self.tool_names),
            "prefix_hash": self.prefix_hash,
        }


def diff_triples(previous: Optional[dict], current: dict) -> dict:
    """Which components moved, and whether the prefix is invalidated (§1)."""
    if not previous:
        return {"changed": [], "first_observation": True, "prefix_invalidated": False}
    keys = ("template_content_hash", "stable_system_hash", "tool_set_version", "prefix_hash")
    changed = [k for k in keys if previous.get(k) != current.get(k)]
    tools_changed = "tool_set_version" in changed
    return {
        "changed": changed,
        "first_observation": False,
        "prefix_invalidated": "prefix_hash" in changed,
        # §3: a tool-set change is legitimate mid-session but must rebuild at a
        # turn boundary. Stated separately so the caller can distinguish it from
        # an unexplained shift.
        "needs_rebuild": tools_changed,
    }


@dataclass
class PrefixFreeze:
    """Per-session prefix state. One instance per session, on the local route.

    Holds the frozen tool set and the last observed triple. Not thread-unsafe in
    a dangerous way: the worst a race can do is log a spurious mismatch.
    """

    session_key: str = ""
    frozen_tools: Optional[tuple] = None
    frozen_tools_version: str = ""
    last_triple: Optional[dict] = None
    rebuild_requested: bool = False
    observations: int = 0
    _definitions: Optional[tuple] = field(default=None, repr=False)

    def observe(self, *, template_content: str, stable_system: str,
                tool_names: Sequence[str], tool_definitions: Iterable[str],
                rendering_config: str = "") -> dict:
        """Recompute the triple, compare with the last, and report.

        Returns ``{"prefix_hash", "changed", "prefix_invalidated", "needs_rebuild"}``.
        A mismatch is LOGGED, never raised: a shifted prefix still produces a
        working turn, so failing it would trade a cost regression for an outage.
        """
        names = tuple(sorted(tool_names))
        defs = tuple(sorted(tool_definitions))
        if self.frozen_tools is None:
            # §3: freeze on first observation and reuse verbatim thereafter.
            self.frozen_tools = names
            self._definitions = defs
            self.frozen_tools_version = tools_version(names, defs)
        elif names != self.frozen_tools:
            # Legitimate (agent rebuild at a turn boundary) but must not pass
            # silently: the caller rebuilds the prefix next turn.
            self.rebuild_requested = True
            logger.warning(
                "FR-22: tool set changed mid-session for %s (%d -> %d tools); "
                "requesting an agent rebuild at the turn boundary",
                self.session_key or "?", len(self.frozen_tools), len(names))

        triple = PrefixTriple.build(
            template_content=template_content, stable_system=stable_system,
            tool_names=names, tool_definitions=defs, rendering_config=rendering_config)
        current = triple.as_dict()
        diff = diff_triples(self.last_triple, current)
        self.observations += 1
        if diff["changed"] and not diff["first_observation"]:
            logger.info("FR-22: prefix shift for session %s: %s",
                        self.session_key or "?", ", ".join(diff["changed"]))
        self.last_triple = current

        # The frozen tools are what must actually be SENT; verify the caller is
        # still using them rather than the freshly resolved set.
        out = dict(current)
        out.update(diff)
        out["frozen_tool_set_version"] = self.frozen_tools_version
        out["tool_set_drifted"] = names != self.frozen_tools
        return out

    def assert_frozen(self) -> None:
        """Guard for callers about to serialize tools: the freeze must hold."""
        if self.frozen_tools is None:
            raise RuntimeError("FR-22: prefix not frozen yet; call observe() first")


# One freeze per session, on the local route only (§7).
_FREEZES: dict = {}


def freeze_for(session_key: str) -> PrefixFreeze:
    """The session's freeze, created on first use.

    Module-level because the gateway builds an agent per turn and the freeze must
    outlive it — the whole point is that the prefix does NOT depend on how many
    times the agent was rebuilt.
    """
    key = session_key or ""
    freeze = _FREEZES.get(key)
    if freeze is None:
        freeze = PrefixFreeze(session_key=key)
        _FREEZES[key] = freeze
    return freeze


def reset(session_key: Optional[str] = None) -> None:
    """Forget one session's freeze, or all of them (tests, session reset)."""
    if session_key is None:
        _FREEZES.clear()
    else:
        _FREEZES.pop(session_key or "", None)


def is_local_route(model: str, base_url: str, gate_base: str) -> bool:
    """§7 per-route scoping: only the local gate route is frozen."""
    base = str(base_url or "")
    return bool(gate_base) and gate_base.rstrip("/") in base
