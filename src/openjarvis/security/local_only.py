"""Local-only inference policy.

When enabled, OpenJarvis must never send an inference request (and therefore
never send prompt content such as personal mail or calendar data) to a remote
AI provider. The policy is enforced in code at every place an inference
backend can be created or a cloud provider can be called, so it does not
depend on choosing the right model or on which API keys happen to be set.

The policy is **on by default**. It is active when any of these hold:

* ``[security] local_only_inference = true`` in ``config.toml`` (the default);
* the ``OPENJARVIS_LOCAL_ONLY`` environment variable is truthy;
* :func:`enable` has been called in this process (``load_config`` does this
  when the loaded config has the policy on).

Once enabled in a process it cannot be switched off again in that process:
:func:`enable` is a one-way latch, and the environment variable can only turn
the policy on. Allowing cloud inference requires explicitly setting
``local_only_inference = false`` in the config file and restarting.

What "local" means here:

* the engine key is one of :data:`LOCAL_ENGINE_KEYS` (an allowlist, so a new or
  unknown engine is treated as remote until it is reviewed), and
* the engine does not declare ``is_cloud = True``, and
* if the engine talks to a host over HTTP, that host is this machine
  (``localhost``, a ``127.0.0.0/8`` address, ``::1`` or ``0.0.0.0``).
"""

from __future__ import annotations

import ipaddress
import os
import threading
from typing import Any, Optional
from urllib.parse import urlsplit

__all__ = [
    "LOCAL_ENGINE_KEYS",
    "IN_PROCESS_ENGINE_KEYS",
    "CloudInferenceBlocked",
    "enable",
    "is_enabled",
    "is_loopback_url",
    "check_engine_key",
    "check_engine",
    "engine_violation",
    "ensure_cloud_allowed",
]

#: Engines that run on this machine. Host-based ones are additionally required
#: to point at a loopback address.
LOCAL_ENGINE_KEYS = frozenset(
    {
        "ollama",
        "vllm",
        "llamacpp",
        "sglang",
        "mlx",
        "lmstudio",
        "exo",
        "nexa",
        "uzu",
        "apple_fm",
        "lemonade",
        "gemma_cpp",
        "afm",
    }
)

#: Local engines that run inside this process and have no HTTP host at all.
IN_PROCESS_ENGINE_KEYS = frozenset({"gemma_cpp", "afm"})

_TRUTHY = {"1", "true", "yes", "on"}
_ENV_VAR = "OPENJARVIS_LOCAL_ONLY"

_lock = threading.Lock()
_latched = False


class CloudInferenceBlocked(RuntimeError):
    """Raised when the local-only policy blocks a remote inference backend."""


def enable() -> None:
    """Turn the policy on for the rest of this process (cannot be undone)."""
    global _latched
    with _lock:
        _latched = True


def _reset_for_tests() -> None:  # pragma: no cover - test helper
    global _latched
    with _lock:
        _latched = False


def is_enabled(config: Any = None) -> bool:
    """Return whether the local-only policy is active.

    *config* is an optional ``JarvisConfig``; when given, its
    ``security.local_only_inference`` setting is honoured as well.
    """
    if _latched:
        return True
    if os.environ.get(_ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    if config is not None:
        security = getattr(config, "security", None)
        if security is not None and getattr(security, "local_only_inference", False):
            return True
    return False


def is_loopback_url(url: Optional[str]) -> bool:
    """Return True when *url* (or a bare ``host[:port]``) names this machine."""
    if not url:
        return False
    text = url.strip()
    if "://" not in text:
        text = "http://" + text
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if addr.is_loopback or addr.is_unspecified:
        return True
    mapped = getattr(addr, "ipv4_mapped", None)
    return bool(mapped is not None and (mapped.is_loopback or mapped.is_unspecified))


def check_engine_key(key: str, config: Any = None) -> None:
    """Refuse a registry *key* that is not an allowlisted local engine."""
    if not is_enabled(config):
        return
    if key not in LOCAL_ENGINE_KEYS:
        raise CloudInferenceBlocked(
            f"Engine {key!r} is not an approved local engine and local-only "
            "inference is enabled ([security] local_only_inference)."
        )


def engine_violation(key: Optional[str], engine: Any) -> Optional[str]:
    """Return why *engine* is not local, or None when it is acceptable.

    This ignores whether the policy is enabled; callers decide that.
    """
    if key is not None and key not in LOCAL_ENGINE_KEYS:
        return f"engine {key!r} is not an approved local engine"
    if getattr(engine, "is_cloud", False):
        return f"engine {key or type(engine).__name__!r} is a cloud engine"
    host = getattr(engine, "_host", None)
    if host is None:
        if key in IN_PROCESS_ENGINE_KEYS:
            return None
        if key is None:
            return f"engine {type(engine).__name__!r} has no verifiable local host"
        return f"engine {key!r} has no verifiable local host"
    if not is_loopback_url(str(host)):
        name = key or type(engine).__name__
        return f"engine {name!r} points at non-local host {host!r}"
    return None


def check_engine(key: Optional[str], engine: Any, config: Any = None) -> None:
    """Raise :class:`CloudInferenceBlocked` if *engine* is not local."""
    if not is_enabled(config):
        return
    reason = engine_violation(key, engine)
    if reason is not None:
        raise CloudInferenceBlocked(
            f"Local-only inference is enabled; refusing to use it: {reason}."
        )


def ensure_cloud_allowed(what: str = "cloud inference", config: Any = None) -> None:
    """Raise :class:`CloudInferenceBlocked` if the policy is active.

    Call this at the top of any code path that would contact a remote AI
    provider.
    """
    if is_enabled(config):
        raise CloudInferenceBlocked(
            f"{what} is blocked because local-only inference is enabled "
            "([security] local_only_inference = true)."
        )
