"""`jarvis briefing`: run, status, and LaunchAgent definitions."""

from __future__ import annotations

import json
import plistlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from openjarvis.cli import briefing_cmd

TZ = ZoneInfo("America/New_York")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(briefing_cmd, "get_config_dir", lambda: tmp_path)
    monkeypatch.setattr(briefing_cmd, "_ollama_ready", lambda host, timeout: True)
    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(cfg))
    return tmp_path


def _ok_metadata(**extra):
    data = {
        "collect_ok": True,
        "tts_ok": True,
        "report_path": "/r.md",
        "audio_path": "",
        "word_count": 250,
        "estimated_seconds": 119,
        "emails_spoken": 9,
        "issues": [],
    }
    data.update(extra)
    return data


def _invoke(*args):
    return CliRunner().invoke(briefing_cmd.briefing, list(args))


def test_successful_run_records_status(home, monkeypatch) -> None:
    monkeypatch.setattr(briefing_cmd, "generate", lambda config: _ok_metadata())
    result = _invoke("run")
    assert result.exit_code == 0, result.output
    assert "Result:   OK" in result.output
    assert "Report:   /r.md" in result.output
    status = json.loads((home / "digests" / "last_run.json").read_text())
    assert status["ok"] is True and status["word_count"] == 250
    assert oct((home / "digests" / "last_run.json").stat().st_mode)[-3:] == "600"


@pytest.mark.parametrize(
    ("metadata", "problem"),
    [
        (_ok_metadata(tts_ok=False), "audio was not generated"),
        (_ok_metadata(collect_ok=False), "data collection failed"),
    ],
)
def test_partial_failure_exits_nonzero(home, monkeypatch, metadata, problem) -> None:
    monkeypatch.setattr(briefing_cmd, "generate", lambda config: metadata)
    result = _invoke("run")
    assert result.exit_code == 1
    assert problem in result.output


def test_exception_is_recorded(home, monkeypatch) -> None:
    def boom(config):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(briefing_cmd, "generate", boom)
    result = _invoke("run")
    assert result.exit_code == 1
    status = json.loads((home / "digests" / "last_run.json").read_text())
    assert status["ok"] is False
    assert "engine exploded" in status["problems"][0]


def test_ollama_down_fails_cleanly(home, monkeypatch) -> None:
    monkeypatch.setattr(briefing_cmd, "_ollama_ready", lambda host, timeout: False)
    monkeypatch.setattr(
        briefing_cmd, "generate", lambda c: pytest.fail("must not generate")
    )
    result = _invoke("run", "--wait-for-ollama", "0")
    assert result.exit_code == 1
    assert "Ollama is not responding" in result.output


class _FixedDatetime(datetime):
    fixed = datetime(2026, 11, 25, 9, 30, tzinfo=TZ)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed.astimezone(tz) if tz else cls.fixed


@pytest.mark.parametrize(
    ("now", "ran_today", "should_run"),
    [
        (datetime(2026, 11, 25, 9, 30, tzinfo=TZ), False, True),
        (datetime(2026, 11, 25, 9, 30, tzinfo=TZ), True, False),
        (datetime(2026, 11, 25, 8, 0, tzinfo=TZ), False, False),  # before 9:10
        (datetime(2026, 11, 25, 13, 0, tzinfo=TZ), False, False),  # after noon
    ],
)
def test_catch_up_rules(home, monkeypatch, now, ran_today, should_run) -> None:
    _FixedDatetime.fixed = now
    monkeypatch.setattr(briefing_cmd, "datetime", _FixedDatetime)
    if ran_today:
        (home / "digests").mkdir(exist_ok=True)
        (home / "digests" / "last_run.json").write_text(
            json.dumps({"ok": True, "finished": now.isoformat()})
        )
    calls = []
    monkeypatch.setattr(
        briefing_cmd, "generate", lambda c: calls.append(1) or _ok_metadata()
    )
    (home / "config.toml").write_text(
        '[digest]\nschedule = "10 9 * * *"\ntimezone = "America/New_York"\n'
    )
    result = _invoke("run", "--catch-up")
    assert result.exit_code == 0, result.output
    assert bool(calls) is should_run


def test_concurrent_run_is_skipped(home, monkeypatch) -> None:
    import fcntl

    (home / "digests").mkdir(exist_ok=True)
    holder = open(home / "digests" / ".briefing.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        monkeypatch.setattr(
            briefing_cmd, "generate", lambda c: pytest.fail("must not run twice")
        )
        result = _invoke("run")
        assert "Another briefing run is in progress." in result.output
    finally:
        holder.close()


def test_launch_agent_plists() -> None:
    main = plistlib.loads(
        briefing_cmd.build_plist(
            jarvis_path="/repo/.venv/bin/jarvis",
            working_dir="/repo",
            hour=9,
            minute=10,
            log_dir="/home/.openjarvis/logs",
        )
    )
    assert main["Label"] == "com.openjarvis.morning-briefing"
    assert main["StartCalendarInterval"] == {"Hour": 9, "Minute": 10}
    assert main["ProgramArguments"] == [
        "/repo/.venv/bin/jarvis",
        "briefing",
        "run",
        "--notify",
        "--play",
    ]
    assert main["WorkingDirectory"] == "/repo"
    assert "RunAtLoad" not in main
    assert main["StandardErrorPath"].endswith("briefing.err.log")

    catch_up = plistlib.loads(
        briefing_cmd.build_catch_up_plist(
            jarvis_path="/repo/.venv/bin/jarvis", working_dir="/repo", log_dir="/l"
        )
    )
    assert catch_up["RunAtLoad"] is True
    assert "--catch-up" in catch_up["ProgramArguments"]


def test_status_without_history(home) -> None:
    result = _invoke("status")
    assert "Schedule: not installed" in result.output
    assert "Last run: none recorded" in result.output


def test_install_requires_macos(home, monkeypatch) -> None:
    monkeypatch.setattr(briefing_cmd.sys, "platform", "linux")
    result = _invoke("install-schedule")
    assert result.exit_code != 0
    assert "only available on macOS" in result.output
