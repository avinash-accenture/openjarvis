"""Read-only search over the local Apple Mail database.

Covers the same mailboxes as the morning briefing: each configured account's
Inbox, plus Gmail's All Mail minus messages labelled as not in the Inbox.
Nothing is ever written to Mail's database (it is opened read-only).

Messages are identified by ``m<ROWID>`` (the Envelope Index row id), which is
unique across all accounts and short enough for a small model to cite.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Dict, List, Optional

from openjarvis.connectors import apple_mail as _mail

MAX_LIMIT = 25
BODY_SCAN_LIMIT = 300  # most recent messages read when searching bodies
_ID = re.compile(r"^m(\d+)$")


@dataclass
class MailRef:
    message_id: str
    account: str
    sender: str
    subject: str
    received: datetime
    unread: bool
    flagged: bool
    conversation_id: str
    row_id: int
    body: Optional[str] = None
    body_available: bool = False
    headers: Dict[str, object] = field(default_factory=dict)


def parse_message_id(message_id: str) -> Optional[int]:
    match = _ID.match(str(message_id).strip())
    return int(match.group(1)) if match else None


def _scope() -> tuple[str, list]:
    """SQL fragment limiting rows to the briefing's mailboxes."""
    inbox_urls = [url for name, _, url in _mail._ACCOUNTS if name != "Gmail"]
    clauses = []
    params: list = []
    if inbox_urls:
        clauses.append(f"b.url IN ({','.join('?' for _ in inbox_urls)})")
        params += inbox_urls
    if any(name == "Gmail" for name, _, _ in _mail._ACCOUNTS):
        clauses.append(
            "(m.mailbox = ? AND NOT EXISTS (SELECT 1 FROM server_messages sm "
            "JOIN server_labels sl ON sl.server_message = sm.ROWID "
            "WHERE sm.message = m.ROWID AND sl.label = ?))"
        )
        params += [_mail.GMAIL_ALL_MAIL_MAILBOX, _mail.GMAIL_NOT_INBOX_LABEL]
    if not clauses:
        return "0", []
    return "(" + " OR ".join(clauses) + ")", params


_SELECT = """
SELECT
    m.ROWID AS row_id,
    m.date_received AS date_received,
    m.date_sent AS date_sent,
    m.read AS is_read,
    m.flagged AS is_flagged,
    m.conversation_id AS conversation_id,
    m.mailbox AS mailbox_id,
    s.subject AS subject,
    a.address AS sender,
    b.url AS mailbox_url
FROM messages m
JOIN subjects s ON s.ROWID = m.subject
LEFT JOIN addresses a ON a.ROWID = m.sender
JOIN mailboxes b ON b.ROWID = m.mailbox
"""


def _account_for(row: sqlite3.Row) -> tuple[str, str]:
    """(account name, account id) for a row."""
    if row["mailbox_id"] == _mail.GMAIL_ALL_MAIL_MAILBOX:
        for name, account_id, _ in _mail._ACCOUNTS:
            if name == "Gmail":
                return name, account_id
    for name, account_id, url in _mail._ACCOUNTS:
        if row["mailbox_url"] == url:
            return name, account_id
    return "Unknown", ""


def _row_to_ref(row: sqlite3.Row) -> Optional[MailRef]:
    received = row["date_received"] or row["date_sent"]
    if received is None:
        return None
    name, _ = _account_for(row)
    sender_name, address = parseaddr(str(row["sender"] or ""))
    return MailRef(
        message_id=f"m{int(row['row_id'])}",
        account=name,
        sender=address or sender_name or "Unknown sender",
        subject=str(row["subject"] or "").strip() or "(No subject)",
        received=datetime.fromtimestamp(float(received), timezone.utc).astimezone(
            _mail._LOCAL_TZ
        ),
        unread=not bool(row["is_read"]),
        flagged=bool(row["is_flagged"]),
        conversation_id=str(row["conversation_id"] or ""),
        row_id=int(row["row_id"]),
    )


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _connect() -> sqlite3.Connection:
    conn = _mail._open_mail_db()
    if conn is None:
        raise RuntimeError("Could not open the local Apple Mail database.")
    conn.row_factory = sqlite3.Row
    return conn


def search_messages(
    query: str = "",
    *,
    sender: str = "",
    account: str = "",
    days_back: int = 90,
    limit: int = 10,
    search_bodies: bool = False,
    unread_only: bool = False,
) -> List[MailRef]:
    """Newest-first messages whose subject/sender match every query word."""
    limit = max(1, min(int(limit), MAX_LIMIT))
    since = datetime.now(timezone.utc) - timedelta(days=max(1, int(days_back)))
    scope_sql, scope_params = _scope()
    where = [
        "m.deleted = 0",
        "COALESCE(m.date_received, m.date_sent) >= ?",
        scope_sql,
    ]
    params: list = [since.timestamp(), *scope_params]
    if sender.strip():
        where.append("a.address LIKE ? ESCAPE '\\'")
        params.append(_like(sender.strip()))
    if unread_only:
        where.append("m.read = 0")
    terms = [t for t in re.split(r"\s+", query.strip()) if t]

    def run(term_sql: list, term_params: list, row_limit: int) -> List[MailRef]:
        sql = (
            _SELECT
            + " WHERE "
            + " AND ".join(where + term_sql)
            + " ORDER BY COALESCE(m.date_received, m.date_sent) DESC LIMIT ?"
        )
        conn = _connect()
        try:
            rows = conn.execute(sql, [*params, *term_params, row_limit]).fetchall()
        finally:
            conn.close()
        refs = [_row_to_ref(row) for row in rows]
        return [
            ref
            for ref in refs
            if ref is not None
            and (not account or ref.account.casefold() == account.strip().casefold())
        ]

    term_sql = []
    term_params: list = []
    for term in terms:
        term_sql.append(
            "(s.subject LIKE ? ESCAPE '\\' OR a.address LIKE ? ESCAPE '\\')"
        )
        term_params += [_like(term), _like(term)]
    results = run(term_sql, term_params, limit if not account else MAX_LIMIT * 4)
    results = results[:limit]

    if search_bodies and terms and len(results) < limit:
        seen = {ref.row_id for ref in results}
        for ref in run([], [], BODY_SCAN_LIMIT):
            if len(results) >= limit:
                break
            if ref.row_id in seen:
                continue
            body, _, _ = _load_body(ref)
            text = f"{ref.subject}\n{body}".casefold()
            if all(term.casefold() in text for term in terms):
                results.append(ref)
    return results


def get_message(message_id: str) -> Optional[MailRef]:
    """One message with its body (or None if the id is unknown/out of scope)."""
    row_id = parse_message_id(message_id)
    if row_id is None:
        return None
    scope_sql, scope_params = _scope()
    conn = _connect()
    try:
        row = conn.execute(
            _SELECT + f" WHERE m.ROWID = ? AND m.deleted = 0 AND {scope_sql}",
            [row_id, *scope_params],
        ).fetchone()
    finally:
        conn.close()
    ref = _row_to_ref(row) if row is not None else None
    if ref is None:
        return None
    ref.body, ref.body_available, ref.headers = _load_body(ref)
    return ref


def get_thread(message_id: str, limit: int = 10) -> List[MailRef]:
    """Messages in the same conversation, oldest first."""
    ref = get_message(message_id)
    if ref is None:
        return []
    if not ref.conversation_id or ref.conversation_id == "0":
        return [ref]
    scope_sql, scope_params = _scope()
    conn = _connect()
    try:
        rows = conn.execute(
            _SELECT
            + f" WHERE m.conversation_id = ? AND m.deleted = 0 AND {scope_sql}"
            + " ORDER BY COALESCE(m.date_received, m.date_sent) ASC LIMIT ?",
            [ref.conversation_id, *scope_params, max(1, min(int(limit), MAX_LIMIT))],
        ).fetchall()
    finally:
        conn.close()
    thread = [r for r in (_row_to_ref(row) for row in rows) if r is not None]
    for item in thread:
        item.body, item.body_available, item.headers = _load_body(item)
    return thread or [ref]


# ---------------------------------------------------------------------------
# Locating .emlx files (index cached, rebuilt at most every few minutes)
# ---------------------------------------------------------------------------

_INDEX_TTL = 300.0
_index_lock = threading.Lock()
_index: Dict[str, tuple[float, Dict[str, Path]]] = {}


def _account_root(account_name: str) -> Optional[Path]:
    for name, account_id, _ in _mail._ACCOUNTS:
        if name == account_name:
            if name == "Gmail":
                return _mail._MAIL_ROOT / account_id / "[Gmail].mbox" / "All Mail.mbox"
            return _mail._MAIL_ROOT / account_id
    return None


def _build_index(root: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for path in root.rglob("*.partial.emlx"):
        found.setdefault(path.name.removesuffix(".partial.emlx"), path)
    for path in root.rglob("*.emlx"):
        if not path.name.endswith(".partial.emlx"):
            found[path.name.removesuffix(".emlx")] = path
    return found


def _locate_emlx(account_name: str, row_id: int) -> Optional[Path]:
    root = _account_root(account_name)
    if root is None or not root.is_dir():
        return None
    key = str(root)
    with _index_lock:
        built_at, index = _index.get(key, (0.0, {}))
        path = index.get(str(row_id))
        if path is not None and path.exists():
            return path
        if time.monotonic() - built_at < _INDEX_TTL and index:
            return None
        index = _build_index(root)
        _index[key] = (time.monotonic(), index)
        return index.get(str(row_id))


def _load_body(ref: MailRef) -> tuple[str, bool, dict]:
    path = _locate_emlx(ref.account, ref.row_id)
    if path is None:
        return "", False, {}
    return _mail._unpack_emlx(_mail._read_emlx(path))
