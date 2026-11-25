"""``jarvis briefing`` — run and schedule the local morning briefing.

``run`` generates the briefing once and records the outcome in
``~/.openjarvis/digests/last_run.json``; it exits non-zero on failure.
``install-schedule`` installs a macOS LaunchAgent that runs it daily (9:10 by
default) and catches up at login if the Mac was off at that time.
"""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import click

from openjarvis.core.paths import get_config_dir

LABEL = "com.openjarvis.morning-briefing"
CATCH_UP_UNTIL_HOUR = 12  # a briefing after noon is no longer a morning briefing
_LAUNCHD_PATH = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _digest_dir() -> Path:
    path = get_config_dir() / "digests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_path() -> Path:
    return _digest_dir() / "last_run.json"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def read_status() -> Dict[str, Any]:
    try:
        return json.loads(status_path().read_text())
    except (OSError, ValueError):
        return {}


def _write_status(data: Dict[str, Any]) -> None:
    path = status_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _notify(title: str, message: str) -> None:
    """macOS notification. Messages are generic: no mail content."""
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return
    script = (
        f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _ollama_ready(host: str, timeout: float) -> bool:
    import httpx

    deadline = time.monotonic() + timeout
    while True:
        try:
            if httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(15.0, max(0.0, deadline - time.monotonic())))


class _RunLock:
    """Exclusive, non-blocking lock so overlapping triggers do not double-run."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: Any = None

    def __enter__(self) -> bool:
        import fcntl

        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __exit__(self, *exc: Any) -> None:
        if self._fh is not None:
            self._fh.close()


def _already_ran_today(tz: ZoneInfo) -> bool:
    status = read_status()
    return bool(
        status.get("ok")
        and str(status.get("finished", ""))[:10] == datetime.now(tz).date().isoformat()
    )


def generate(config: Any) -> Dict[str, Any]:
    """Generate one briefing through the SDK; return the agent metadata."""
    from openjarvis.sdk import Jarvis

    with Jarvis() as jarvis:
        result = jarvis.ask_full("Generate my morning digest", agent="morning_digest")
    metadata = dict(result.get("metadata") or {})
    metadata["content"] = result.get("content", "")
    return metadata


@click.group("briefing")
def briefing() -> None:
    """Run and schedule the local morning briefing."""


@briefing.command("run")
@click.option("--play", is_flag=True, help="Play the audio when it is ready.")
@click.option("--notify", is_flag=True, help="Post a macOS notification.")
@click.option(
    "--catch-up",
    is_flag=True,
    help="Only run if today's briefing has not succeeded yet and it is still morning.",
)
@click.option(
    "--wait-for-ollama",
    default=180,
    show_default=True,
    help="Seconds to wait for the local Ollama server before giving up.",
)
def run(play: bool, notify: bool, catch_up: bool, wait_for_ollama: int) -> None:
    """Generate the morning briefing now."""
    from openjarvis.core.config import load_config

    config = load_config()
    tz = ZoneInfo(config.digest.timezone or "America/New_York")
    now = datetime.now(tz)

    if catch_up and (
        _already_ran_today(tz)
        or now.hour >= CATCH_UP_UNTIL_HOUR
        or (now.hour, now.minute) < _scheduled_time(config)
    ):
        click.echo("Catch-up not needed.")
        return

    with _RunLock(_digest_dir() / ".briefing.lock") as acquired:
        if not acquired:
            click.echo("Another briefing run is in progress.")
            return
        status: Dict[str, Any] = {"started": now.isoformat(), "ok": False}
        try:
            host = config.engine.ollama_host or "http://127.0.0.1:11434"
            if not _ollama_ready(host, float(wait_for_ollama)):
                raise RuntimeError(f"Ollama is not responding at {host}.")
            metadata = generate(config)
            problems = []
            if not metadata.get("collect_ok", False):
                problems.append("data collection failed")
            if not metadata.get("tts_ok", False):
                problems.append("audio was not generated")
            status.update(
                {
                    "ok": not problems,
                    "problems": problems,
                    "report_path": metadata.get("report_path", ""),
                    "audio_path": metadata.get("audio_path", ""),
                    "word_count": metadata.get("word_count"),
                    "estimated_seconds": metadata.get("estimated_seconds"),
                    "emails_spoken": metadata.get("emails_spoken"),
                    "issues": metadata.get("issues", []),
                }
            )
        except Exception as exc:  # noqa: BLE001
            status.update({"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]})
            metadata = {}
        status["finished"] = datetime.now(tz).isoformat()
        _write_status(status)

    _print_status(status)
    if status["ok"]:
        if notify:
            minutes = round((status.get("estimated_seconds") or 0) / 60, 1)
            _notify(
                "Morning briefing", f"Your briefing is ready (about {minutes} min)."
            )
        audio = status.get("audio_path")
        if play and audio and Path(audio).exists() and shutil.which("afplay"):
            subprocess.run(["afplay", audio], check=False)
    else:
        if notify:
            _notify(
                "Morning briefing failed", "Run `jarvis briefing status` for details."
            )
        sys.exit(1)


def _scheduled_time(config: Any) -> tuple[int, int]:
    """(hour, minute) from the digest cron schedule; defaults to 9:10."""
    try:
        minute, hour = str(config.digest.schedule).split()[:2]
        return int(hour), int(minute)
    except (ValueError, AttributeError):
        return 9, 10


def _print_status(status: Dict[str, Any]) -> None:
    click.echo(f"Result:   {'OK' if status.get('ok') else 'FAILED'}")
    for problem in status.get("problems") or []:
        click.echo(f"Problem:  {problem}")
    if status.get("word_count") is not None:
        click.echo(
            f"Length:   {status['word_count']} words, "
            f"about {round((status.get('estimated_seconds') or 0) / 60, 1)} minutes"
        )
    if status.get("report_path"):
        click.echo(f"Report:   {status['report_path']}")
    if status.get("audio_path"):
        click.echo(f"Audio:    {status['audio_path']}")
    for issue in status.get("issues") or []:
        click.echo(f"Check:    {issue}")


def build_plist(
    *, jarvis_path: str, working_dir: str, hour: int, minute: int, log_dir: str
) -> bytes:
    """The LaunchAgent definition."""
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [
                jarvis_path,
                "briefing",
                "run",
                "--notify",
                "--play",
            ],
            "WorkingDirectory": working_dir,
            # launchd runs a missed calendar interval once on wake from sleep.
            "StartCalendarInterval": {"Hour": hour, "Minute": minute},
            "EnvironmentVariables": {
                "OPENJARVIS_NO_UPDATE_CHECK": "1",
                "PATH": _LAUNCHD_PATH,
            },
            "StandardOutPath": str(Path(log_dir) / "briefing.log"),
            "StandardErrorPath": str(Path(log_dir) / "briefing.err.log"),
            "ProcessType": "Interactive",
        }
    )


def build_catch_up_plist(*, jarvis_path: str, working_dir: str, log_dir: str) -> bytes:
    """Runs at login: catch up if the Mac was off at the scheduled time."""
    return plistlib.dumps(
        {
            "Label": f"{LABEL}.catch-up",
            "ProgramArguments": [
                jarvis_path,
                "briefing",
                "run",
                "--catch-up",
                "--notify",
            ],
            "WorkingDirectory": working_dir,
            "RunAtLoad": True,
            "EnvironmentVariables": {"OPENJARVIS_NO_UPDATE_CHECK": "1"},
            "StandardOutPath": str(Path(log_dir) / "briefing.log"),
            "StandardErrorPath": str(Path(log_dir) / "briefing.err.log"),
        }
    )


def _catch_up_plist_path() -> Path:
    return plist_path().with_name(f"{LABEL}.catch-up.plist")


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, check=False
    )


@briefing.command("install-schedule")
@click.option(
    "--time", "at", default="", help="HH:MM (default from [digest] schedule)."
)
def install_schedule(at: str) -> None:
    """Install the macOS LaunchAgent for the daily briefing."""
    from openjarvis.core.config import load_config

    if sys.platform != "darwin":
        raise click.ClickException("LaunchAgents are only available on macOS.")
    config = load_config()
    hour, minute = _scheduled_time(config)
    if at:
        try:
            hour, minute = (int(part) for part in at.split(":"))
            assert 0 <= hour < 24 and 0 <= minute < 60
        except (ValueError, AssertionError):
            raise click.ClickException("Use --time HH:MM, e.g. 09:10.")

    jarvis_path = shutil.which("jarvis") or str(
        Path(sys.executable).with_name("jarvis")
    )
    if not Path(jarvis_path).exists():
        raise click.ClickException("Could not find the jarvis executable.")
    working_dir = str(Path.cwd())
    log_dir = get_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_path().parent.mkdir(parents=True, exist_ok=True)
    for path, data in (
        (
            plist_path(),
            build_plist(
                jarvis_path=jarvis_path,
                working_dir=working_dir,
                hour=hour,
                minute=minute,
                log_dir=str(log_dir),
            ),
        ),
        (
            _catch_up_plist_path(),
            build_catch_up_plist(
                jarvis_path=jarvis_path, working_dir=working_dir, log_dir=str(log_dir)
            ),
        ),
    ):
        _launchctl("unload", str(path))
        path.write_bytes(data)
        result = _launchctl("load", "-w", str(path))
        if result.returncode != 0:
            raise click.ClickException(
                f"launchctl load failed: {result.stderr.strip()}"
            )

    tz_note = os.environ.get("TZ") or time.strftime("%Z")
    click.echo(
        f"Installed. The briefing runs daily at {hour:02d}:{minute:02d} "
        f"(this Mac's local time, {tz_note}), and at login if it was missed."
    )
    click.echo(f"Logs: {log_dir}/briefing.log")


@briefing.command("uninstall-schedule")
def uninstall_schedule() -> None:
    """Remove the LaunchAgents."""
    for path in (plist_path(), _catch_up_plist_path()):
        if path.exists():
            _launchctl("unload", "-w", str(path))
            path.unlink()
            click.echo(f"Removed {path}")


@briefing.command("status")
def status() -> None:
    """Show the schedule and the last run."""
    installed = plist_path().exists()
    click.echo(f"Schedule: {'installed' if installed else 'not installed'}")
    if installed and sys.platform == "darwin":
        listed = _launchctl("list", LABEL)
        click.echo(f"Loaded:   {'yes' if listed.returncode == 0 else 'no'}")
    last = read_status()
    if not last:
        click.echo("Last run: none recorded")
        return
    click.echo(f"Last run: {last.get('finished') or last.get('started')}")
    _print_status(last)
