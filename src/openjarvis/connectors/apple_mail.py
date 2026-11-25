"""Read locally synchronized Apple Mail messages from macOS Mail's database."""

from __future__ import annotations

import html
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

from openjarvis.connectors._stubs import BaseConnector, Document, SyncStatus
from openjarvis.core.registry import ConnectorRegistry


_MAIL_DB = Path.home() / "Library/Mail/V9/MailData/Envelope Index"
_MAIL_ROOT = Path.home() / "Library/Mail/V9"

_ACCOUNTS = (
    ("iCloud", "0215DEC8-BEA3-46CA-A1E7-53BDA887E9B2",
     "imap://0215DEC8-BEA3-46CA-A1E7-53BDA887E9B2/INBOX"),
    ("Gmail", "8F4B893C-6CB4-4BB4-A24F-0BD083DDFEA4",
     "imap://8F4B893C-6CB4-4BB4-A24F-0BD083DDFEA4/INBOX"),
    ("Yahoo Avi", "7E7AA7B5-C367-48B2-A2D1-5A6A5ACFA1AF",
     "imap://7E7AA7B5-C367-48B2-A2D1-5A6A5ACFA1AF/INBOX"),
    ("Yahoo Pavi", "B5548858-E285-437C-8BAF-3167AA50E1AF",
     "imap://B5548858-E285-437C-8BAF-3167AA50E1AF/INBOX"),
    ("Outlook", "9F267112-858D-4F70-A69C-4D6969C038AA",
     "ews://9F267112-858D-4F70-A69C-4D6969C038AA/Inbox"),
)
_LOCAL_TZ = ZoneInfo("America/New_York")
# This Mac's Envelope Index: Gmail's "All Mail" mailbox row, and the server
# label that marks messages which are not in the Gmail Inbox.
GMAIL_ALL_MAIL_MAILBOX = 50
GMAIL_NOT_INBOX_LABEL = 72
_MAX_BODY_CHARS = 3000


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)

    def text(self) -> str:
        return " ".join(self.parts)


def _html_to_text(value: str) -> str:
    parser = _HTMLTextParser()
    try:
        parser.feed(value)
        return html.unescape(parser.text())
    except Exception:
        return ""


def _message_body(message) -> str:
    """Extract readable text while excluding attachment payloads."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    parts = message.walk() if message.is_multipart() else [message]

    for part in parts:
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        if part.get_filename():
            continue

        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue

        try:
            content = part.get_content()
        except Exception:
            continue

        if not isinstance(content, str):
            continue

        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(_html_to_text(content))

    result = "\n".join(plain_parts).strip()
    if not result:
        result = "\n".join(html_parts).strip()

    return result[:_MAX_BODY_CHARS]


def _read_gmail_inbox(
    since: datetime,
    max_messages: int,
) -> list[Document]:
    """Read recent incoming Gmail messages from local All Mail and emlx files."""

    total_limit = max(1, int(max_messages))
    documents: list[Document] = []

    gmail_account_id = next(
        account_id
        for account_name, account_id, _ in _ACCOUNTS
        if account_name == "Gmail"
    )

    gmail_root = (
        _MAIL_ROOT
        / gmail_account_id
        / "[Gmail].mbox"
        / "All Mail.mbox"
    )

    if not gmail_root.is_dir():
        raise RuntimeError(
            f"Gmail All Mail directory not found: {gmail_root}"
        )

    data_roots = list(gmail_root.glob("*/Data"))

    if len(data_roots) != 1:
        raise RuntimeError(
            "Expected exactly one Gmail message Data directory; "
            f"found {len(data_roots)}."
        )

    data_root = data_roots[0]

    # Index local message files once. Prefer complete files over partial files.
    emlx_by_id: dict[str, Path] = {}

    for path in data_root.rglob("*.partial.emlx"):
        mail_id = path.name.removesuffix(".partial.emlx")
        emlx_by_id.setdefault(mail_id, path)

    for path in data_root.rglob("*.emlx"):
        if path.name.endswith(".partial.emlx"):
            continue
        mail_id = path.name.removesuffix(".emlx")
        emlx_by_id[mail_id] = path

    conn = _open_mail_db()
    if conn is None:
        raise RuntimeError("Could not open the local Apple Mail database.")

    try:
        rows = conn.execute(
            """
            SELECT
                m.ROWID AS row_id,
                m.message_id AS rfc_message_id,
                m.date_received AS date_received,
                m.read AS read_value,
                m.flagged AS flagged_value,
                s.subject AS subject,
                a.address AS sender
            FROM messages m
            JOIN subjects s ON s.ROWID = m.subject
            LEFT JOIN addresses a ON a.ROWID = m.sender
            WHERE m.mailbox = ?
              AND m.deleted = 0
              AND m.date_received >= ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM server_messages sm
                  JOIN server_labels sl
                    ON sl.server_message = sm.ROWID
                  WHERE sm.message = m.ROWID
                    AND sl.label = ?
              )
            ORDER BY m.date_received DESC
            LIMIT ?
            """,
            (
                GMAIL_ALL_MAIL_MAILBOX,
                since.timestamp(),
                GMAIL_NOT_INBOX_LABEL,
                total_limit,
            ),
        ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Gmail All Mail query failed: {exc}"
        ) from exc
    finally:
        conn.close()

    for row in rows:
        mail_id = str(row["row_id"])

        try:
            timestamp = datetime.fromtimestamp(
                float(row["date_received"]),
                tz=timezone.utc,
            ).astimezone(_LOCAL_TZ)
        except (ValueError, TypeError, OSError):
            continue

        sender_header = row["sender"] or ""
        sender_name, sender_address = parseaddr(sender_header)
        sender = sender_address or sender_name or "Unknown sender"

        subject = (row["subject"] or "").strip() or "(No subject)"

        # Read the body directly from the local Apple Mail message file.
        emlx_path = emlx_by_id.get(mail_id)
        body = ""
        body_available = False
        headers: dict = {}

        if emlx_path is not None:
            body, body_available, headers = _unpack_emlx(_read_emlx(emlx_path))

        rfc_message_id = row["rfc_message_id"] or ""

        documents.append(
            Document(
                doc_id=f"apple_mail:gmail:{mail_id}",
                source="apple_mail",
                doc_type="email",
                title=subject,
                author=sender,
                participants=[sender],
                content=body[:_MAX_BODY_CHARS],
                timestamp=timestamp,
                thread_id=rfc_message_id,
                metadata={
                    "account": "Gmail",
                    "unread": not bool(row["read_value"]),
                    "flagged": bool(row["flagged_value"]),
                    "body_available": body_available,
                    "sender_name": headers.get("from_name", "") or sender_name,
                    "mail_headers": headers,
                    "mailbox": "All Mail",
                    "message_id": rfc_message_id,
                    "mail_id": mail_id,
                },
            )
        )

    return documents





def _header_signals(message) -> dict:
    """Extract the few headers the digest classifier uses; never the body.

    These identify bulk and automated mail far more reliably than wording:
    mailing lists set ``List-Unsubscribe``/``List-Id``, bulk senders set
    ``Precedence: bulk``, and machine-generated mail sets ``Auto-Submitted``.
    """
    def _get(name: str) -> str:
        try:
            value = message.get(name)
        except Exception:  # malformed header
            return ""
        return str(value or "").strip()

    from_name, _ = parseaddr(_get("From"))
    has_calendar = False
    try:
        for part in message.walk():
            if part.get_content_type() == "text/calendar":
                has_calendar = True
                break
    except Exception:
        pass
    return {
        "from_name": from_name.strip().strip('"'),
        "list_unsubscribe": bool(_get("List-Unsubscribe")),
        "list_id": bool(_get("List-Id")),
        "precedence": _get("Precedence").lower(),
        "auto_submitted": _get("Auto-Submitted").lower(),
        "has_calendar": has_calendar,
    }


def _read_emlx(path: Path) -> tuple[str, bool, dict]:
    """Read an Apple Mail .emlx file: (body text, body available, header signals)."""
    try:
        raw = path.read_bytes()
        prefix, separator, payload = raw.partition(b"\n")

        if not separator:
            return "", False, {}

        # Apple Mail prefixes the RFC822 message with its byte length.
        message_length = int(prefix.strip())
        if message_length <= 0 or len(payload) < message_length:
            return "", False, {}

        # Exclude trailing Apple plist metadata.
        message_bytes = payload[:message_length]
        message = BytesParser(policy=policy.default).parsebytes(message_bytes)

        body = _message_body(message)
        return body, bool(body.strip()), _header_signals(message)
    except (OSError, ValueError, TypeError):
        return "", False, {}


def _unpack_emlx(result) -> tuple[str, bool, dict]:
    """Accept (body, available) or (body, available, headers)."""
    body, available, *rest = result
    headers = rest[0] if rest and isinstance(rest[0], dict) else {}
    return body, available, headers


def _stats(documents: list, error: str = "") -> dict:
    """Coverage for one account: how many messages, how many with a body."""
    return {
        "fetched": len(documents),
        "with_body": sum(
            1 for doc in documents if doc.metadata.get("body_available")
        ),
        "error": error,
    }


def _open_mail_db() -> Optional[sqlite3.Connection]:
    try:
        uri = _MAIL_DB.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn
    except (sqlite3.Error, OSError):
        return None


@ConnectorRegistry.register("apple_mail")
class AppleMailConnector(BaseConnector):
    """Read the Outlook Inbox already synchronized by macOS Apple Mail."""

    connector_id = "apple_mail"
    display_name = "Apple Mail (local)"
    auth_type = "local"

    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_per_account: int = 200,
    ) -> None:
        # ``max_messages`` caps the combined result (None = no cap; the digest
        # collector ranks and selects). ``max_per_account`` bounds each query.
        self.max_messages = None if max_messages is None else max(1, int(max_messages))
        self.max_per_account = max(1, int(max_per_account))
        if self.max_messages is not None:
            self.max_per_account = min(self.max_per_account, self.max_messages)
        #: Per-account coverage from the last sync: fetched, with_body, error.
        self.account_stats: dict[str, dict] = {}
        self._items_synced = 0
        self._items_total = 0
        self._last_sync: Optional[datetime] = None
        self._error: Optional[str] = None

    def is_connected(self) -> bool:
        if not _MAIL_DB.is_file():
            return False

        # One missing account must not disable mail entirely; sync() reports
        # missing accounts in ``account_stats``.
        if not any(
            (_MAIL_ROOT / account_id).is_dir()
            for _, account_id, _ in _ACCOUNTS
        ):
            return False

        conn = _open_mail_db()
        if conn is None:
            return False

        try:
            conn.execute("SELECT 1 FROM mailboxes LIMIT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
    def disconnect(self) -> None:
        # The connector holds no credentials or persistent session.
        pass

    def sync_status(self) -> SyncStatus:
        return SyncStatus(
            state="error" if self._error else "idle",
            items_synced=self._items_synced,
            items_total=self._items_total,
            last_sync=self._last_sync,
            error=self._error,
        )


    def sync(
        self,
        *,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
    ) -> Iterator[Document]:
        self._error = None
        self._items_synced = 0
        self._items_total = 0
        self.account_stats = {}

        conn = _open_mail_db()
        if conn is None:
            self._error = "Could not open the local Apple Mail database."
            return

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
        elif since.tzinfo is None:
            since = since.replace(tzinfo=_LOCAL_TZ)

        cutoff = since.timestamp()

        query = """
            SELECT
                m.ROWID AS row_id,
                m.date_received AS date_received,
                m.date_sent AS date_sent,
                m.read AS is_read,
                m.flagged AS is_flagged,
                m.conversation_id AS conversation_id,
                s.subject AS subject,
                a.address AS sender
            FROM messages m
            JOIN subjects s ON s.ROWID = m.subject
            LEFT JOIN addresses a ON a.ROWID = m.sender
            JOIN mailboxes b ON b.ROWID = m.mailbox
            WHERE b.url = ?
              AND m.deleted = 0
              AND COALESCE(m.date_received, m.date_sent) >= ?
            ORDER BY COALESCE(m.date_received, m.date_sent) DESC
            LIMIT ?
        """

        account_documents: list[list[Document]] = []

        try:
            # Fetch a bounded set from each account. The final selection
            # is round-robin so high-volume accounts cannot monopolize it.
            for account_name, account_id, inbox_url in _ACCOUNTS:

                # Gmail Inbox membership is not correctly represented
                # in the local SQLite database, so use AppleScript.
                if account_name == "Gmail":
                    try:
                        gmail_documents = _read_gmail_inbox(
                            datetime.fromtimestamp(
                                float(cutoff),
                                timezone.utc,
                            ),
                            self.max_per_account,
                        )
                    except RuntimeError as exc:
                        self._error = str(exc)
                        self.account_stats[account_name] = _stats([], str(exc))
                        account_documents.append([])
                        continue

                    self.account_stats[account_name] = _stats(gmail_documents)
                    account_documents.append(gmail_documents)
                    continue

                account_root = _MAIL_ROOT / account_id
                if not account_root.is_dir():
                    self.account_stats[account_name] = _stats(
                        [], "Account folder not found in Apple Mail."
                    )
                    account_documents.append([])
                    continue

                # All other accounts use the local Mail database.
                rows = conn.execute(
                    query,
                    (inbox_url, cutoff, self.max_per_account),
                ).fetchall()

                documents: list[Document] = []

                # Index local message files once per account.
                # Prefer complete .emlx files over partial files.
                emlx_by_id: dict[str, Path] = {}

                for path in account_root.rglob("*.partial.emlx"):
                    mail_id = path.name.removesuffix(".partial.emlx")
                    emlx_by_id.setdefault(mail_id, path)

                for path in account_root.rglob("*.emlx"):
                    mail_id = path.name.removesuffix(".emlx")
                    emlx_by_id[mail_id] = path

                for row in rows:
                    row_id = int(row["row_id"])
                    emlx_path = emlx_by_id.get(str(row_id))


                    body = ""
                    body_available = False
                    headers: dict = {}

                    if emlx_path is not None:
                        body, body_available, headers = _unpack_emlx(
                            _read_emlx(emlx_path)
                        )

                    sender_header = str(
                        row["sender"] or ""
                    ).strip()

                    sender_name, sender_address = parseaddr(
                        sender_header
                    )

                    sender = (
                        sender_address
                        or sender_name
                        or "Unknown sender"
                    )

                    subject = (
                        str(row["subject"] or "").strip()
                        or "(No subject)"
                    )

                    received = row["date_received"]
                    if received is None:
                       received = row["date_sent"]

                    if received is None:
                        continue

                    timestamp = datetime.fromtimestamp(
                                     float(received), timezone.utc
                                ).astimezone(_LOCAL_TZ)

                    documents.append(
                        Document(
                            doc_id=(
                                f"apple_mail:{account_id}:{row_id}"
                            ),
                            source="apple_mail",
                            doc_type="email",
                            title=subject,
                            author=sender,
                            participants=[sender],
                            content=body,
                            timestamp=timestamp,
                            thread_id=str(
                                row["conversation_id"] or ""
                            ),
                            metadata={
                                "account": account_name,
                                "unread": not bool(row["is_read"]),
                                "flagged": bool(row["is_flagged"]),
                                "body_available": body_available,
                                "sender_name": (
                                    headers.get("from_name", "") or sender_name
                                ),
                                "mail_headers": headers,
                                "mailbox": "Inbox",
                            },
                        )
                    )

                self.account_stats[account_name] = _stats(documents)
                account_documents.append(documents)
        except sqlite3.Error as exc:
            self._error = (
                f"Apple Mail database query failed: {exc}"
            )
            return
        finally:
            conn.close()

        # Interleave accounts so, when a total cap is set, one high-volume
        # Inbox does not consume the entire limit.
        documents = []
        longest = max(
            (len(items) for items in account_documents),
            default=0,
        )
        cap = self.max_messages

        for index in range(longest):
            for account_items in account_documents:
                if index < len(account_items):
                    documents.append(account_items[index])
                    if cap is not None and len(documents) >= cap:
                        break

            if cap is not None and len(documents) >= cap:
                break

        documents.sort(
            key=lambda doc: doc.timestamp,
            reverse=True,
        )

        self._items_total = len(documents)
        self._items_synced = len(documents)
        self._last_sync = datetime.now(timezone.utc)

        yield from documents

