"""Tool allowlist policy.

Only tools on an explicit allowlist may run. The check happens where tools are
executed (:class:`openjarvis.tools._stubs.ToolExecutor`), so it covers every
agent, managed agent, skill and channel in the process, not just the tools the
server offers by default. Tools not on the list are refused even if an agent
was configured with them, and a tool added to OpenJarvis later is refused until
someone adds it to the list.

The default allowlist is read-only and local: reasoning, arithmetic, local
calendar/knowledge/memory *reads*, local inference sub-calls, the morning
digest's collector and local text-to-speech. Shell and code execution, file
access, git, web search, HTTP, browsers, messaging, memory/profile writes,
skill authoring and the proactive action queue are all off by default.

Configure in ``config.toml``::

    [security]
    tool_allowlist_enabled = true          # default
    allowed_tools = ["think", "calculator", "calendar_upcoming"]

Setting ``allowed_tools`` replaces the default list. ``["*"]`` allows every
tool (not recommended). ``load_config`` installs the policy for the process.
"""

from __future__ import annotations

import threading
from typing import Any, FrozenSet, Iterable, Optional, Set

__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "configure",
    "is_active",
    "allowed_tools",
    "tool_identities",
    "violation",
    "is_allowed",
]

#: Read-only, local tools allowed unless the config says otherwise.
DEFAULT_ALLOWED_TOOLS: FrozenSet[str] = frozenset(
    {
        # Reasoning and arithmetic
        "think",
        "calculator",
        # Local personal data, read-only
        "calendar_search",
        "calendar_upcoming",
        "digest_collect",
        "knowledge_search",
        "knowledge_sql",  # SELECT-only on the local knowledge store
        "scan_chunks",
        "retrieval",
        "memory_search",
        "memory_retrieve",
        "kg_query",
        "kg_neighbors",
        # Local inference sub-call (subject to the local-only policy)
        "llm",
        # Read-only views of the approval queue
        "get_pending_actions",
        "check_permission",
        # Morning briefing voice (cloud TTS backends are blocked by local-only)
        "text_to_speech",
    }
)

_WILDCARD = "*"

_lock = threading.Lock()
_active = False
_allowed: FrozenSet[str] = DEFAULT_ALLOWED_TOOLS


def configure(enabled: bool, allowed: Optional[Iterable[str]] = None) -> None:
    """Install the policy for this process (called by ``load_config``)."""
    global _active, _allowed
    if allowed is None:
        names: Iterable[str] = DEFAULT_ALLOWED_TOOLS
    elif isinstance(allowed, str):
        names = allowed.split(",")
    else:
        names = allowed
    with _lock:
        _active = bool(enabled)
        _allowed = frozenset(str(n).strip() for n in names if str(n).strip())


def _reset_for_tests() -> None:  # pragma: no cover - test helper
    global _active, _allowed
    with _lock:
        _active = False
        _allowed = DEFAULT_ALLOWED_TOOLS


def is_active() -> bool:
    return _active


def allowed_tools() -> FrozenSet[str]:
    return _allowed


def tool_identities(tool: Any) -> Set[str]:
    """Every name *tool* is known by: advertised name plus the real tool's.

    Wrappers such as the managed-agent ``_SpecOverrideTool`` can advertise a
    different name to the model; unwrapping prevents a disallowed tool from
    running under an allowed name.
    """
    names: Set[str] = set()
    current = tool
    for _ in range(8):  # bounded unwrap
        try:
            name = current.spec.name
        except Exception:
            name = None
        if isinstance(name, str) and name:
            names.add(name)
        tool_id = getattr(type(current), "tool_id", None)
        if isinstance(tool_id, str) and tool_id:
            names.add(tool_id)
        inner = getattr(current, "__dict__", {}).get("_wrapped")
        if inner is None:
            break
        current = inner
    return names


def violation(tool: Any) -> Optional[str]:
    """Return why *tool* may not run, or None when it is allowed."""
    if not _active:
        return None
    allowed = _allowed
    if _WILDCARD in allowed:
        return None
    names = tool_identities(tool)
    if not names:
        return "tool has no identifiable name"
    blocked = sorted(n for n in names if n not in allowed)
    if blocked:
        return (
            f"tool {', '.join(repr(n) for n in blocked)} is not on the allowlist "
            "([security] allowed_tools)"
        )
    return None


def is_allowed(tool: Any) -> bool:
    return violation(tool) is None
