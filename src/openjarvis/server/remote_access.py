"""Remote-access guard for a loopback server reached through Tailscale Serve.

One outermost ASGI middleware (it sees both HTTP and WebSocket traffic):

* **Host check** (DNS-rebinding defence). Only requests addressed to this
  machine (``localhost``, ``127.0.0.1``, ``::1``), to a Tailscale MagicDNS name
  (``*.ts.net``) or to an explicitly configured host are served. A web page
  that rebinds its own domain to 127.0.0.1 is refused before anything else
  runs, even though the API key would also stop it.
* **Tailscale identity** (optional). When ``[server] tailscale_allowed_logins``
  is set, a request that arrived through the local proxy (it carries
  ``X-Forwarded-For``) must also carry a ``Tailscale-User-Login`` from that
  list. Tailscale Serve sets that header itself and strips any copy a client
  sends, so it identifies the tailnet user behind the request.
* **No caching of API responses** (``Cache-Control: no-store``) so mail and
  calendar data do not linger in a phone's browser cache.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Iterable, Optional

Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
TAILSCALE_SUFFIX = ".ts.net"
_PRIVATE_PREFIXES = ("/v1/", "/api/", "/metrics")


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def host_name(host_header: str) -> str:
    """Hostname from a Host header, without port or IPv6 brackets."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host.rstrip(".")


def host_allowed(
    host_header: str,
    *,
    extra_hosts: Iterable[str] = (),
    allow_tailscale: bool = True,
) -> bool:
    host = host_name(host_header)
    if not host:
        return False
    if host in LOOPBACK_HOSTS:
        return True
    if (
        allow_tailscale
        and host.endswith(TAILSCALE_SUFFIX)
        and host != TAILSCALE_SUFFIX[1:]
    ):
        return True
    return host in {h.strip().lower().rstrip(".") for h in extra_hosts if h.strip()}


class RemoteAccessMiddleware:
    def __init__(
        self,
        app: Any,
        *,
        extra_hosts: Iterable[str] = (),
        allow_tailscale_hosts: bool = True,
        tailscale_allowed_logins: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.extra_hosts = tuple(extra_hosts)
        self.allow_tailscale_hosts = allow_tailscale_hosts
        self.allowed_logins = frozenset(
            login.strip().lower() for login in tailscale_allowed_logins if login.strip()
        )

    def _rejection(self, scope: Scope) -> Optional[tuple[int, str]]:
        if not host_allowed(
            _header(scope, b"host"),
            extra_hosts=self.extra_hosts,
            allow_tailscale=self.allow_tailscale_hosts,
        ):
            return 421, "Request host is not allowed for this server."
        if self.allowed_logins and _header(scope, b"x-forwarded-for"):
            login = _header(scope, b"tailscale-user-login").strip().lower()
            if login not in self.allowed_logins:
                return 403, "Tailscale user is not allowed."
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        rejection = self._rejection(scope)
        if rejection is not None:
            status, detail = rejection
            if kind == "websocket":
                # Closing before accept makes the server answer the upgrade
                # with HTTP 403.
                await send({"type": "websocket.close", "code": 1008})
                return
            body = json.dumps({"detail": detail}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        if kind == "http" and str(scope.get("path", "")).startswith(_PRIVATE_PREFIXES):

            async def send_no_store(message: dict) -> None:
                if message.get("type") == "http.response.start":
                    headers = [
                        (k, v)
                        for k, v in message.get("headers") or []
                        if k.lower() not in (b"cache-control", b"pragma")
                    ]
                    headers += [
                        (b"cache-control", b"no-store, private"),
                        (b"pragma", b"no-cache"),
                    ]
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_no_store)
            return

        await self.app(scope, receive, send)
