"""Pre-cache turn-route resolution (model-router branch).

``resolve_turn_route`` runs after :meth:`GatewayTurnMixin._resolve_turn_agent_config`
builds the configured route and before ``TurnRunner._resolve_turn_agent`` looks up
the cached agent, so a route change is visible to the cache-signature comparison
in the same turn. The hook is synchronous and MUST stay fast (policy + advisory
snapshot only): cold model preparation runs as a dedicated gateway operation
with its own timeout, never inside this hook.

Contract (generic; the consumer ships separately):
- Payload: ``session_key``, ``configured_route`` (``{"model", "runtime",
  "request_overrides"?}`` as built by ``_resolve_turn_agent_config``),
  ``turn_metadata`` (small, non-content facts: platform, message length).
- A consumer returns a replacement route dict (``model: str`` + ``runtime:
  dict`` required) or None to keep the configured route. First valid dict wins.
- Invalid results are ignored (configured route kept); a hook exception never
  breaks the turn — ``invoke_hook`` isolates callbacks.

``ROUTING_API_VERSION`` is the consumer compatibility floor: out-of-tree router
plugins declare the minimum version they need and fail clearly below it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.run")

ROUTING_API_VERSION = 1


def resolve_turn_route(
    session_key: Optional[str],
    configured_route: Dict[str, Any],
    turn_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from hermes_cli.lifecycle import has_hook, invoke_hook
    if not has_hook("resolve_turn_route"):
        return configured_route
    try:
        results = invoke_hook(
            "resolve_turn_route",
            session_key=session_key,
            configured_route=configured_route,
            turn_metadata=dict(turn_metadata or {}),
            routing_api_version=ROUTING_API_VERSION,
        )
    except Exception as exc:
        logger.warning("resolve_turn_route hook failed, keeping configured route: %s", exc)
        return configured_route
    for result in results or []:
        if isinstance(result, dict) and isinstance(result.get("model"), str) \
                and isinstance(result.get("runtime"), dict):
            return result
        if result is not None:
            logger.debug("resolve_turn_route ignoring invalid result %r", type(result))
    return configured_route
