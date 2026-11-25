"""`jarvis remote`: Tailscale Serve configuration checks."""

from __future__ import annotations

import json
import os
import stat

import pytest
from click.testing import CliRunner

from openjarvis.cli import remote_cmd
from openjarvis.cli.remote_cmd import evaluate_serve_config

NAME = "macbook.tail1234.ts.net"


def _serve(proxy="http://127.0.0.1:8000", funnel=False, extra=None):
    cfg = {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": proxy}}}},
    }
    if funnel:
        cfg["AllowFunnel"] = {f"{NAME}:443": True}
    cfg.update(extra or {})
    return cfg


def test_good_config_has_no_problems() -> None:
    assert evaluate_serve_config(_serve(), 8000) == []
    assert evaluate_serve_config(_serve("http://localhost:8000"), 8000) == []


def test_funnel_is_a_failure() -> None:
    problems = evaluate_serve_config(_serve(funnel=True), 8000)
    assert any("Funnel is ON" in p for p in problems)


def test_funnel_inside_foreground_session_is_found() -> None:
    cfg = {"Foreground": {"session1": _serve(funnel=True)}}
    assert any("Funnel" in p for p in evaluate_serve_config(cfg, 8000))


@pytest.mark.parametrize(
    ("proxy", "fragment"),
    [
        ("http://192.168.1.5:8000", "non-local target"),
        ("http://127.0.0.1:9000", "port 9000"),
    ],
)
def test_wrong_targets(proxy: str, fragment: str) -> None:
    assert any(fragment in p for p in evaluate_serve_config(_serve(proxy), 8000))


def test_missing_serve_and_plain_http() -> None:
    assert any("not forwarding" in p for p in evaluate_serve_config({}, 8000))
    cfg = _serve(extra={"TCP": {"80": {"HTTP": True}, "443": {"HTTPS": True}}})
    assert any("plain HTTP" in p for p in evaluate_serve_config(cfg, 8000))


# ---------------------------------------------------------------------------
# `jarvis remote check` against a fake tailscale binary
# ---------------------------------------------------------------------------


def _fake_tailscale(tmp_path, *, state="Running", serve=None):
    status = {"BackendState": state, "Self": {"DNSName": NAME + "."}}
    script = tmp_path / "tailscale"
    script.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = status ]; then echo '{json.dumps(status)}'; exit 0; fi\n"
        f"if [ \"$1\" = serve ]; then echo '{json.dumps(serve or {})}'; exit 0; fi\n"
        "exit 1\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[server.auth]\napi_key = "oj_sk_{"k" * 40}"\n')
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(cfg))
    monkeypatch.delenv("OPENJARVIS_API_KEY", raising=False)
    return tmp_path


def _check(monkeypatch, binary):
    monkeypatch.setattr(remote_cmd, "tailscale_binary", lambda: binary)
    return CliRunner().invoke(remote_cmd.remote, ["check", "--skip-http"])


def test_check_passes_for_a_correct_setup(configured, monkeypatch) -> None:
    result = _check(monkeypatch, _fake_tailscale(configured, serve=_serve()))
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_check_fails_when_funnel_is_on(configured, monkeypatch) -> None:
    result = _check(monkeypatch, _fake_tailscale(configured, serve=_serve(funnel=True)))
    assert result.exit_code == 1
    assert "Funnel is ON" in result.output


def test_check_fails_when_tailscale_is_down(configured, monkeypatch) -> None:
    result = _check(
        monkeypatch, _fake_tailscale(configured, state="Stopped", serve=_serve())
    )
    assert result.exit_code == 1
    assert "not connected" in result.output


def test_check_fails_without_tailscale_or_key(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(cfg))
    monkeypatch.delenv("OPENJARVIS_API_KEY", raising=False)
    result = _check(monkeypatch, None)
    assert result.exit_code == 1
    assert "not installed" in result.output
    assert "No API key" in result.output


@pytest.mark.skipif(os.name != "posix", reason="shell script fake")
def test_setup_apply_runs_serve(configured, monkeypatch) -> None:
    log = configured / "calls.txt"
    script = configured / "tailscale"
    script.write_text(f'#!/bin/sh\necho "$@" >> {log}\necho ok\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(remote_cmd, "tailscale_binary", lambda: str(script))
    result = CliRunner().invoke(remote_cmd.remote, ["setup", "--apply"])
    assert result.exit_code == 0, result.output
    assert log.read_text().strip() == "serve --bg --https=443 http://127.0.0.1:8000"


def test_server_launch_agent_plist() -> None:
    import plistlib

    data = plistlib.loads(
        remote_cmd.build_server_plist(
            jarvis_path="/repo/.venv/bin/jarvis", working_dir="/repo", log_dir="/l"
        )
    )
    assert data["Label"] == "com.openjarvis.server"
    assert data["ProgramArguments"] == ["/repo/.venv/bin/jarvis", "serve"]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] == 30
