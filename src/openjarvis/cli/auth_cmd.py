"""CLI commands for API key management."""

from __future__ import annotations

import os
import re
import stat

import click

from openjarvis.core.config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
)
from openjarvis.server.auth_middleware import generate_api_key

_SECTION_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$", re.M)
_API_KEY_LINE = re.compile(r'^(\s*api_key\s*=\s*)(".*?"|\'.*?\')(.*)$', re.M)


def _server_auth_span(content: str) -> tuple[int, int] | None:
    """Return the (start, end) offsets of the ``[server.auth]`` table body."""
    headers = list(_SECTION_HEADER.finditer(content))
    for i, match in enumerate(headers):
        if match.group(1) == "server.auth":
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            return match.end(), end
    return None


def _set_server_auth_key(content: str, key: str) -> str:
    """Set ``api_key`` inside ``[server.auth]`` only, leaving other tables alone."""
    span = _server_auth_span(content)
    if span is None:
        sep = "" if not content or content.endswith("\n") else "\n"
        return f'{content}{sep}\n[server.auth]\napi_key = "{key}"\n'
    start, end = span
    body = content[start:end]
    if _API_KEY_LINE.search(body):
        body = _API_KEY_LINE.sub(lambda m: f'{m.group(1)}"{key}"{m.group(3)}', body, 1)
    else:
        body = f'\napi_key = "{key}"' + body
    return content[:start] + body + content[end:]


@click.group("auth")
def auth() -> None:
    """Manage API authentication keys."""


@auth.command("create-key")
def create_key() -> None:
    """Generate a new API key and store it in config."""
    key = generate_api_key()
    config_path = DEFAULT_CONFIG_PATH

    # Ensure config directory exists
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing config or start fresh
    if config_path.exists():
        content = config_path.read_text()
    else:
        content = ""

    content = _set_server_auth_key(content, key)

    config_path.write_text(content)
    os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600

    click.echo(f"API key generated: {key}")
    click.echo(f"Stored in: {config_path}")
    click.echo("File permissions set to 0600 (user-only read/write).")


@auth.command("revoke-key")
def revoke_key() -> None:
    """Revoke the current API key."""
    config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        click.echo("No config file found.")
        return

    content = config_path.read_text()
    if _server_auth_span(content) is None:
        click.echo("No [server.auth] API key found in config.")
        return

    config_path.write_text(_set_server_auth_key(content, ""))
    os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    click.echo("API key revoked. The server will not start until a new key exists.")
