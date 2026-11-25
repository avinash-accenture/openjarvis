"""Server access policy: mandatory API key and loopback-only binding."""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.server.auth_middleware import (
    MIN_API_KEY_LENGTH,
    generate_api_key,
    is_loopback_host,
    server_access_problems,
)

STRONG = generate_api_key()


def test_defaults_require_key_and_loopback() -> None:
    cfg = JarvisConfig()
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.require_api_key is True
    assert cfg.server.allow_non_loopback_bind is False


def test_generated_keys_meet_minimum_length() -> None:
    assert len(generate_api_key()) >= MIN_API_KEY_LENGTH


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_hosts(host: str) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "100.101.1.2"])
def test_non_loopback_hosts(host: str) -> None:
    assert not is_loopback_host(host)


def test_loopback_with_strong_key_is_allowed() -> None:
    assert server_access_problems("127.0.0.1", api_key=STRONG) == []


def test_missing_key_is_refused_even_on_loopback() -> None:
    problems = server_access_problems("127.0.0.1", api_key="")
    assert len(problems) == 1
    assert "jarvis auth create-key" in problems[0]


def test_weak_key_is_refused() -> None:
    problems = server_access_problems("127.0.0.1", api_key="password123")
    assert any("too short" in p for p in problems)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "100.101.1.2"])
def test_non_loopback_bind_is_refused_even_with_key(host: str) -> None:
    problems = server_access_problems(host, api_key=STRONG)
    assert any("not loopback" in p for p in problems)


def test_non_loopback_bind_allowed_only_explicitly_and_still_needs_key() -> None:
    assert (
        server_access_problems("0.0.0.0", api_key=STRONG, allow_non_loopback_bind=True)
        == []
    )
    problems = server_access_problems(
        "0.0.0.0", api_key="", allow_non_loopback_bind=True, require_api_key=False
    )
    assert any("No API key" in p for p in problems)


def test_key_requirement_can_be_disabled_for_loopback_only() -> None:
    assert server_access_problems("127.0.0.1", api_key="", require_api_key=False) == []


# ---------------------------------------------------------------------------
# `jarvis serve` start-up
# ---------------------------------------------------------------------------


def _run_serve(monkeypatch, tmp_path, *, config_text: str, args=(), env_key=None):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import importlib

    from click.testing import CliRunner

    from openjarvis.cli import cli

    serve_mod = importlib.import_module("openjarvis.cli.serve")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(config_text)
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(cfg_path))
    if env_key is None:
        monkeypatch.delenv("OPENJARVIS_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENJARVIS_API_KEY", env_key)

    def _no_engine_work(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("engine selection ran before the access check")

    monkeypatch.setattr(serve_mod, "get_engine", _no_engine_work)
    return CliRunner().invoke(cli, ["serve", *args])


def test_serve_refuses_to_start_without_key(monkeypatch, tmp_path) -> None:
    result = _run_serve(monkeypatch, tmp_path, config_text="")
    assert result.exit_code == 1
    assert "Refusing to start" in result.output
    assert "No API key" in result.output


def test_serve_refuses_public_bind(monkeypatch, tmp_path) -> None:
    result = _run_serve(
        monkeypatch,
        tmp_path,
        config_text="",
        args=("--host", "0.0.0.0"),
        env_key=STRONG,
    )
    assert result.exit_code == 1
    assert "not loopback" in result.output


def test_serve_reads_key_from_config_file(monkeypatch, tmp_path) -> None:
    """With a key in [server.auth], the access check passes and start-up goes
    on to engine selection (stubbed here to stop the run)."""
    result = _run_serve(
        monkeypatch,
        tmp_path,
        config_text=f'[server.auth]\napi_key = "{STRONG}"\n',
    )
    assert "Refusing to start" not in result.output
    assert isinstance(result.exception, AssertionError)
    assert "engine selection ran" in str(result.exception)


# ---------------------------------------------------------------------------
# `jarvis auth create-key` / `revoke-key` only touch [server.auth]
# ---------------------------------------------------------------------------


def test_set_server_auth_key_leaves_other_api_keys_alone() -> None:
    from openjarvis.cli.auth_cmd import _set_server_auth_key

    content = (
        "[channel.telegram]\n"
        'api_key = "telegram-secret"\n'
        "\n"
        "[server.auth]\n"
        'api_key = "old"  # comment kept\n'
        "\n"
        "[tools.weather]\n"
        'api_key = "weather-secret"\n'
    )
    updated = _set_server_auth_key(content, "new-key")
    assert 'api_key = "telegram-secret"' in updated
    assert 'api_key = "weather-secret"' in updated
    assert 'api_key = "new-key"  # comment kept' in updated
    assert '"old"' not in updated


def test_set_server_auth_key_appends_section_when_missing() -> None:
    import tomllib

    from openjarvis.cli.auth_cmd import _set_server_auth_key

    updated = _set_server_auth_key('[tools.weather]\napi_key = "w"', "k")
    data = tomllib.loads(updated)
    assert data["server"]["auth"]["api_key"] == "k"
    assert data["tools"]["weather"]["api_key"] == "w"


def test_set_server_auth_key_adds_key_to_existing_empty_section() -> None:
    import tomllib

    from openjarvis.cli.auth_cmd import _set_server_auth_key

    updated = _set_server_auth_key(
        "[server.auth]\n\n[engine]\ndefault = 'ollama'\n", "k"
    )
    data = tomllib.loads(updated)
    assert data["server"]["auth"]["api_key"] == "k"
    assert data["engine"]["default"] == "ollama"


# ---------------------------------------------------------------------------
# `jarvis tunnel` (public Cloudflare URL) is refused by default
# ---------------------------------------------------------------------------


def test_public_tunnel_refused_by_default(monkeypatch, tmp_path) -> None:
    import subprocess

    from click.testing import CliRunner

    from openjarvis.cli.tunnel_cmd import tunnel

    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(cfg))

    def _must_not_run(*a, **k):  # pragma: no cover
        raise AssertionError("cloudflared must not be started")

    monkeypatch.setattr(subprocess, "run", _must_not_run)
    result = CliRunner().invoke(tunnel, [])
    assert result.exit_code == 1
    assert "Refusing to start a public tunnel" in result.output


def test_public_tunnel_runs_only_when_explicitly_allowed(monkeypatch, tmp_path) -> None:
    import subprocess

    from click.testing import CliRunner

    from openjarvis.cli import tunnel_cmd

    cfg = tmp_path / "config.toml"
    cfg.write_text("[server]\nallow_public_tunnel = true\n")
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(cfg))
    monkeypatch.setattr(tunnel_cmd.shutil, "which", lambda _: "/usr/bin/cloudflared")
    calls = []

    def _fake_run(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess(a[0], 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = CliRunner().invoke(tunnel_cmd.tunnel, [])
    assert result.exit_code == 0, result.output
    assert calls and calls[-1][0][0] == "cloudflared"
