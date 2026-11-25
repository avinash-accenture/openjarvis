"""``jarvis remote`` — private remote access through Tailscale Serve.

The server keeps listening on 127.0.0.1. Tailscale Serve terminates HTTPS on
this Mac's tailnet name (``https://<mac>.<tailnet>.ts.net``) and forwards to
the loopback server, so only devices on your tailnet can reach it and nothing
is exposed to the internet. Tailscale *Funnel* (public exposure) must stay off;
``jarvis remote check`` fails if it is on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import click

_MAC_APP_CLI = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
_LOCAL_TARGETS = ("127.0.0.1", "localhost", "[::1]")


def tailscale_binary() -> Optional[str]:
    found = shutil.which("tailscale")
    if found:
        return found
    if Path(_MAC_APP_CLI).exists():
        return _MAC_APP_CLI
    return None


def _run(binary: str, *args: str) -> Tuple[int, str]:
    try:
        done = subprocess.run(
            [binary, *args], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout or done.stderr or "").strip()


def _walk(node: Any, key: str) -> Iterator[Tuple[Any, Any]]:
    """Yield (parent, value) for every occurrence of *key* in nested JSON."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield node, v
            yield from _walk(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, key)


def evaluate_serve_config(config: Dict[str, Any], port: int) -> List[str]:
    """Problems with a ``tailscale serve status --json`` document."""
    problems: List[str] = []
    for _, allow in _walk(config, "AllowFunnel"):
        if isinstance(allow, dict) and any(allow.values()) or allow is True:
            problems.append(
                "Tailscale Funnel is ON: the server is reachable from the public "
                "internet. Turn it off with `tailscale funnel reset`."
            )
            break

    proxies = [str(v) for _, v in _walk(config, "Proxy")]
    if not proxies:
        problems.append(
            "Tailscale Serve is not forwarding to OpenJarvis. "
            "Run `jarvis remote setup --apply`."
        )
    for proxy in proxies:
        target = proxy.split("://", 1)[-1].split("/", 1)[0]
        host, _, target_port = target.rpartition(":")
        if host not in _LOCAL_TARGETS:
            problems.append(f"Serve forwards to a non-local target: {proxy}")
        elif target_port and target_port != str(port):
            problems.append(
                f"Serve forwards to port {target_port}, but OpenJarvis uses {port}."
            )
    if any(v for _, v in _walk(config, "TCP") if isinstance(v, dict) and "80" in v):
        problems.append("Serve also listens on plain HTTP port 80; use HTTPS only.")
    return problems


def _tailnet_name(binary: str) -> Tuple[Optional[str], List[str]]:
    code, out = _run(binary, "status", "--json")
    if code != 0:
        return None, [f"`tailscale status` failed: {out[:200]}"]
    try:
        status = json.loads(out)
    except ValueError:
        return None, ["Could not parse `tailscale status --json`."]
    problems = []
    if status.get("BackendState") != "Running":
        problems.append(
            f"Tailscale is not connected (state: {status.get('BackendState')})."
        )
    name = str((status.get("Self") or {}).get("DNSName") or "").rstrip(".")
    if not name:
        problems.append(
            "This Mac has no MagicDNS name. Enable MagicDNS for the tailnet."
        )
    return name or None, problems


@click.group("remote")
def remote() -> None:
    """Private remote access through Tailscale (never public)."""


@remote.command("setup")
@click.option("--apply", is_flag=True, help="Run the Tailscale command now.")
def setup(apply: bool) -> None:
    """Show (or apply) the Tailscale Serve command for this server."""
    from openjarvis.core.config import load_config

    port = load_config().server.port
    command = ["serve", "--bg", "--https=443", f"http://127.0.0.1:{port}"]
    binary = tailscale_binary()
    click.echo("OpenJarvis stays on 127.0.0.1. Tailscale Serve forwards your")
    click.echo("tailnet's HTTPS address to it. Funnel (public access) stays off.\n")
    click.echo("  tailscale " + " ".join(command))
    if not apply:
        click.echo("\nRun with --apply to do this now, then `jarvis remote check`.")
        return
    if binary is None:
        raise click.ClickException(
            "Tailscale is not installed. Install it from https://tailscale.com/download"
        )
    code, out = _run(binary, *command)
    click.echo(out)
    if code != 0:
        raise click.ClickException("tailscale serve failed (see output above).")


@remote.command("check")
@click.option(
    "--skip-http", is_flag=True, help="Only check the Tailscale configuration."
)
def check(skip_http: bool) -> None:
    """Verify the remote-access setup end to end."""
    from openjarvis.core.config import load_config
    from openjarvis.server.auth_middleware import server_access_problems

    config = load_config()
    problems: List[str] = []
    notes: List[str] = []

    from openjarvis.cli.serve import _resolve_api_key

    api_key = _resolve_api_key()
    problems += server_access_problems(
        config.server.host,
        api_key=api_key,
        require_api_key=config.server.require_api_key,
        allow_non_loopback_bind=config.server.allow_non_loopback_bind,
    )

    binary = tailscale_binary()
    if binary is None:
        problems.append(
            "Tailscale is not installed. Install it from https://tailscale.com/download"
        )
        name = None
    else:
        name, status_problems = _tailnet_name(binary)
        problems += status_problems
        code, out = _run(binary, "serve", "status", "--json")
        try:
            serve_cfg = json.loads(out) if code == 0 and out else {}
        except ValueError:
            serve_cfg = {}
        problems += evaluate_serve_config(serve_cfg, config.server.port)

    if name and not skip_http and not problems:
        problems += _http_checks(f"https://{name}", api_key)
        notes.append(f"Open https://{name} on your phone (Tailscale connected).")

    for problem in problems:
        click.echo(f"  FAIL  {problem}")
    for note in notes:
        click.echo(f"  NOTE  {note}")
    if problems:
        sys.exit(1)
    click.echo("  PASS  Private remote access is configured correctly.")


def _http_checks(base: str, api_key: str) -> List[str]:
    import httpx

    problems: List[str] = []
    try:
        with httpx.Client(timeout=20) as client:
            if client.get(f"{base}/health").status_code != 200:
                problems.append(f"{base}/health did not answer 200.")
            if client.get(f"{base}/v1/models").status_code != 401:
                problems.append(f"{base}/v1/models answered without an API key.")
            authed = client.get(
                f"{base}/v1/models", headers={"Authorization": f"Bearer {api_key}"}
            )
            if authed.status_code != 200:
                problems.append(
                    f"{base}/v1/models with the API key answered {authed.status_code}."
                )
    except httpx.HTTPError as exc:
        problems.append(f"Could not reach {base}: {exc}")
    return problems


SERVER_LABEL = "com.openjarvis.server"


def server_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVER_LABEL}.plist"


def build_server_plist(*, jarvis_path: str, working_dir: str, log_dir: str) -> bytes:
    """LaunchAgent that keeps ``jarvis serve`` running while you are logged in."""
    import plistlib

    return plistlib.dumps(
        {
            "Label": SERVER_LABEL,
            "ProgramArguments": [jarvis_path, "serve"],
            "WorkingDirectory": working_dir,
            "RunAtLoad": True,
            # Restart after a crash, but not in a tight loop.
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 30,
            "EnvironmentVariables": {"OPENJARVIS_NO_UPDATE_CHECK": "1"},
            "StandardOutPath": str(Path(log_dir) / "server.log"),
            "StandardErrorPath": str(Path(log_dir) / "server.err.log"),
        }
    )


@remote.command("install-server")
def install_server() -> None:
    """Keep `jarvis serve` running in the background (macOS LaunchAgent)."""
    from openjarvis.core.paths import get_config_dir

    if sys.platform != "darwin":
        raise click.ClickException("LaunchAgents are only available on macOS.")
    jarvis_path = shutil.which("jarvis") or str(
        Path(sys.executable).with_name("jarvis")
    )
    if not Path(jarvis_path).exists():
        raise click.ClickException("Could not find the jarvis executable.")
    log_dir = get_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = server_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["launchctl", "unload", str(path)], capture_output=True, check=False)
    path.write_bytes(
        build_server_plist(
            jarvis_path=jarvis_path, working_dir=str(Path.cwd()), log_dir=str(log_dir)
        )
    )
    done = subprocess.run(
        ["launchctl", "load", "-w", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise click.ClickException(f"launchctl load failed: {done.stderr.strip()}")
    click.echo("The OpenJarvis server now starts at login and restarts if it stops.")
    click.echo(f"Logs: {log_dir}/server.log")


@remote.command("uninstall-server")
def uninstall_server() -> None:
    """Stop running `jarvis serve` in the background."""
    path = server_plist_path()
    if path.exists():
        subprocess.run(["launchctl", "unload", "-w", str(path)], check=False)
        path.unlink()
        click.echo(f"Removed {path}")
