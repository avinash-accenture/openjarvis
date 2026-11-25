"""Remote-access guard: host allowlist, Tailscale identity, no-store."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from openjarvis.core.config import JarvisConfig  # noqa: E402
from openjarvis.server.app import create_app  # noqa: E402
from openjarvis.server.remote_access import host_allowed, host_name  # noqa: E402

KEY = "oj_sk_" + "k" * 40
AUTH = {"Authorization": f"Bearer {KEY}"}
TS_HOST = "macbook.tail1234.ts.net"


def _app(**server):
    engine = MagicMock()
    engine.engine_id = "ollama"
    engine.list_models.return_value = ["qwen3.5:2b"]
    engine.health.return_value = True
    cfg = JarvisConfig()
    cfg.analytics.enabled = False
    cfg.traces.enabled = False
    cfg.security.enabled = False
    for key, value in server.items():
        setattr(cfg.server, key, value)
    return create_app(
        engine, "qwen3.5:2b", engine_name="ollama", config=cfg, api_key=KEY
    )


def _client(host: str, **server) -> TestClient:
    return TestClient(_app(**server), base_url=f"https://{host}")


# ---------------------------------------------------------------------------
# Host allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "name"),
    [
        ("localhost:8000", "localhost"),
        ("127.0.0.1:8000", "127.0.0.1"),
        ("[::1]:8000", "::1"),
        ("MacBook.Tail1234.TS.NET.", "macbook.tail1234.ts.net"),
        ("macbook.tail1234.ts.net:443", "macbook.tail1234.ts.net"),
    ],
)
def test_host_name_parsing(header: str, name: str) -> None:
    assert host_name(header) == name


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1:8000", "[::1]:8000", TS_HOST, f"{TS_HOST}:443"],
)
def test_allowed_hosts(host: str) -> None:
    assert host_allowed(host)


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",  # DNS rebinding: attacker domain pointed at 127.0.0.1
        "localhost.evil.example",
        "ts.net",
        "evil.ts.net.example",
        "192.168.1.20:8000",
        "",
    ],
)
def test_rejected_hosts(host: str) -> None:
    assert not host_allowed(host)


def test_tailscale_names_can_be_disabled_and_extra_hosts_added() -> None:
    assert not host_allowed(TS_HOST, allow_tailscale=False)
    assert host_allowed("jarvis.home.arpa", extra_hosts=["jarvis.home.arpa"])


@pytest.mark.parametrize("path", ["/health", "/v1/models", "/", "/dashboard"])
def test_rebinding_host_is_refused_before_anything_runs(path: str) -> None:
    client = _client("evil.example")
    resp = client.get(path, headers=AUTH)
    assert resp.status_code == 421
    assert resp.headers["cache-control"] == "no-store"


def test_rebinding_host_refused_for_websocket() -> None:
    client = _client("evil.example")
    with pytest.raises(WebSocketDisconnect) as exc:
        # An absolute URL: TestClient otherwise always uses Host: testserver.
        with client.websocket_connect("ws://evil.example/v1/chat/stream", headers=AUTH):
            pass
    assert exc.value.code == 1008


def test_tailscale_host_websocket_passes_the_host_check() -> None:
    client = _client(TS_HOST)
    # Reaches the endpoint's own auth check (no key -> refused there), which
    # shows the host guard let it through.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"ws://{TS_HOST}/v1/chat/stream"):
            pass
    with client.websocket_connect(f"ws://{TS_HOST}/v1/chat/stream", headers=AUTH) as ws:
        assert ws is not None


def test_tailscale_host_is_served_and_still_needs_the_key() -> None:
    client = _client(TS_HOST)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=AUTH).status_code == 200


# ---------------------------------------------------------------------------
# Tailscale identity
# ---------------------------------------------------------------------------


def test_identity_check_applies_to_forwarded_requests_only() -> None:
    client = _client(TS_HOST, tailscale_allowed_logins=["me@example.com"])
    forwarded = {**AUTH, "X-Forwarded-For": "100.101.102.103"}
    ok = client.get(
        "/v1/models", headers={**forwarded, "Tailscale-User-Login": "Me@Example.com"}
    )
    assert ok.status_code == 200
    stranger = client.get(
        "/v1/models",
        headers={**forwarded, "Tailscale-User-Login": "someone@example.com"},
    )
    assert stranger.status_code == 403
    assert client.get("/v1/models", headers=forwarded).status_code == 403
    # A direct local request (no proxy) is not subject to the identity check.
    assert (
        _client("127.0.0.1:8000", tailscale_allowed_logins=["me@example.com"])
        .get("/v1/models", headers=AUTH)
        .status_code
        == 200
    )


def test_identity_check_off_by_default() -> None:
    client = _client(TS_HOST)
    resp = client.get(
        "/v1/models", headers={**AUTH, "X-Forwarded-For": "100.101.102.103"}
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_api_responses_are_not_cacheable() -> None:
    client = _client(TS_HOST)
    resp = client.get("/v1/models", headers=AUTH)
    assert resp.headers["cache-control"] == "no-store, private"
    assert resp.headers["pragma"] == "no-cache"
    unauth = client.get("/v1/models")
    assert unauth.headers["cache-control"] == "no-store, private"
