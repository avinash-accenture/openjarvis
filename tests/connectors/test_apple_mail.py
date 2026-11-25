"""Tests for the read-only local Apple Mail connector."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import openjarvis.connectors.apple_mail as apple_mail


def _make_mail_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE mailboxes (
            ROWID INTEGER PRIMARY KEY,
            url TEXT
        );

        CREATE TABLE messages (
            ROWID INTEGER PRIMARY KEY,
            date_received REAL,
            date_sent REAL,
            read INTEGER,
            flagged INTEGER,
            conversation_id INTEGER,
            subject INTEGER,
            sender INTEGER,
            mailbox INTEGER,
            deleted INTEGER
        );

        CREATE TABLE subjects (
            ROWID INTEGER PRIMARY KEY,
            subject TEXT
        );

        CREATE TABLE addresses (
            ROWID INTEGER PRIMARY KEY,
            address TEXT
        );

        INSERT INTO mailboxes VALUES (
            1, 'imap://test-account/INBOX'
        );

        INSERT INTO subjects VALUES (1, 'Synthetic message');
        INSERT INTO addresses VALUES (1, 'sender@example.test');
        """
    )
    conn.commit()
    return conn


def _insert_message(
    conn,
    *,
    rowid,
    date_received,
    date_sent,
):
    conn.execute(
        """
        INSERT INTO messages VALUES (
            ?, ?, ?, 0, 0, ?, 1, 1, 1, 0
        )
        """,
        (
            rowid,
            date_received,
            date_sent,
            rowid,
        ),
    )


def _configure_connector(monkeypatch, db_path, mail_root):
    account_id = "synthetic-account"

    monkeypatch.setattr(
        apple_mail,
        "_ACCOUNTS",
        (
            (
                "Synthetic",
                account_id,
                "imap://test-account/INBOX",
            ),
        ),
    )
    monkeypatch.setattr(apple_mail, "_MAIL_ROOT", mail_root)
    monkeypatch.setattr(
        apple_mail,
        "_open_mail_db",
        lambda: sqlite3.connect(
            db_path,
            factory=_RowConnection,
        ),
    )
    monkeypatch.setattr(
        apple_mail,
        "_read_emlx",
        lambda path: ("synthetic body", True),
    )


class _RowConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_factory = sqlite3.Row


def test_date_sent_is_used_when_date_received_is_null(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "mail.sqlite"
    conn = _make_mail_db(db_path)

    sent = datetime(2026, 11, 14, 12, tzinfo=timezone.utc).timestamp()
    _insert_message(
        conn,
        rowid=1,
        date_received=None,
        date_sent=sent,
    )
    conn.commit()
    conn.close()

    mail_root = tmp_path / "Mail"
    account_root = mail_root / "synthetic-account"
    account_root.mkdir(parents=True)
    (account_root / "1.emlx").write_text("synthetic")

    _configure_connector(monkeypatch, db_path, mail_root)

    documents = list(
        apple_mail.AppleMailConnector().sync(
            since=datetime(2026, 11, 14, tzinfo=timezone.utc)
        )
    )

    assert len(documents) == 1
    assert documents[0].doc_id.endswith(":1")
    assert documents[0].timestamp.timestamp() == sent


def test_message_with_both_timestamps_null_is_excluded(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "mail.sqlite"
    conn = _make_mail_db(db_path)

    _insert_message(
        conn,
        rowid=1,
        date_received=None,
        date_sent=None,
    )
    conn.commit()
    conn.close()

    mail_root = tmp_path / "Mail"
    account_root = mail_root / "synthetic-account"
    account_root.mkdir(parents=True)
    (account_root / "1.emlx").write_text("synthetic")

    _configure_connector(monkeypatch, db_path, mail_root)

    documents = list(
        apple_mail.AppleMailConnector().sync(
            since=datetime(2026, 11, 14, tzinfo=timezone.utc)
        )
    )

    assert documents == []


def test_messages_are_ordered_by_effective_timestamp_newest_first(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "mail.sqlite"
    conn = _make_mail_db(db_path)

    older = datetime(2026, 11, 14, 10, tzinfo=timezone.utc).timestamp()
    newer = datetime(2026, 11, 14, 14, tzinfo=timezone.utc).timestamp()

    _insert_message(
        conn,
        rowid=1,
        date_received=older,
        date_sent=None,
    )
    _insert_message(
        conn,
        rowid=2,
        date_received=None,
        date_sent=newer,
    )
    conn.commit()
    conn.close()

    mail_root = tmp_path / "Mail"
    account_root = mail_root / "synthetic-account"
    account_root.mkdir(parents=True)
    (account_root / "1.emlx").write_text("synthetic")
    (account_root / "2.emlx").write_text("synthetic")

    _configure_connector(monkeypatch, db_path, mail_root)

    documents = list(
        apple_mail.AppleMailConnector(max_messages=2).sync(
            since=datetime(2026, 11, 14, tzinfo=timezone.utc)
        )
    )

    assert [document.doc_id.rsplit(":", 1)[-1] for document in documents] == [
        "2",
        "1",
    ]


# ---------------------------------------------------------------------------
# Briefing support: headers, per-account coverage, no 15-message total cap
# ---------------------------------------------------------------------------


def _emlx(path, raw: bytes) -> None:
    path.write_bytes(str(len(raw)).encode() + b"\n" + raw + b"<?xml plist?>")


def test_read_emlx_returns_body_and_header_signals(tmp_path) -> None:
    raw = (
        b'From: "Priya Raman" <priya@example.test>\r\n'
        b"Subject: Weekly update\r\n"
        b"List-Unsubscribe: <mailto:unsub@example.test>\r\n"
        b"Precedence: bulk\r\n"
        b"Auto-Submitted: auto-generated\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello there.\r\n"
    )
    path = tmp_path / "1.emlx"
    _emlx(path, raw)
    body, available, headers = apple_mail._read_emlx(path)
    assert body.strip() == "Hello there."
    assert available is True
    assert headers == {
        "from_name": "Priya Raman",
        "list_unsubscribe": True,
        "list_id": False,
        "precedence": "bulk",
        "auto_submitted": "auto-generated",
        "has_calendar": False,
    }


def test_read_emlx_detects_calendar_invites_and_empty_bodies(tmp_path) -> None:
    raw = (
        b"From: organizer@example.test\r\nSubject: Invite\r\n"
        b'Content-Type: multipart/alternative; boundary="b"\r\n\r\n'
        b"--b\r\nContent-Type: text/calendar; method=REQUEST\r\n\r\n"
        b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n--b--\r\n"
    )
    path = tmp_path / "2.emlx"
    _emlx(path, raw)
    body, available, headers = apple_mail._read_emlx(path)
    assert headers["has_calendar"] is True
    assert available is False  # no text body to summarise


def test_unreadable_emlx_is_unavailable(tmp_path) -> None:
    path = tmp_path / "3.emlx"
    path.write_bytes(b"not an emlx file")
    assert apple_mail._read_emlx(path) == ("", False, {})


def _two_account_setup(tmp_path, monkeypatch, *, messages_per_account=10):
    db_path = tmp_path / "mail.sqlite"
    conn = _make_mail_db(db_path)
    conn.execute("INSERT INTO mailboxes VALUES (2, 'imap://second/INBOX')")
    base = datetime(2026, 11, 14, 10, tzinfo=timezone.utc).timestamp()
    rowid = 0
    for mailbox in (1, 2):
        for i in range(messages_per_account):
            rowid += 1
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, NULL, 0, 0, ?, 1, 1, ?, 0)",
                (rowid, base + rowid * 60, rowid, mailbox),
            )
    conn.commit()
    conn.close()

    mail_root = tmp_path / "Mail"
    (mail_root / "acct-one").mkdir(parents=True)
    (mail_root / "acct-two").mkdir(parents=True)
    _configure_connector(monkeypatch, db_path, mail_root)
    monkeypatch.setattr(
        apple_mail,
        "_ACCOUNTS",
        (
            ("One", "acct-one", "imap://test-account/INBOX"),
            ("Two", "acct-two", "imap://second/INBOX"),
            ("Missing", "acct-missing", "imap://missing/INBOX"),
        ),
    )
    return db_path, mail_root


def test_sync_returns_all_messages_in_window_by_default(tmp_path, monkeypatch) -> None:
    _two_account_setup(tmp_path, monkeypatch, messages_per_account=10)
    connector = apple_mail.AppleMailConnector()
    documents = list(
        connector.sync(since=datetime(2026, 11, 14, tzinfo=timezone.utc))
    )
    assert len(documents) == 20  # no 15-message total cap any more
    assert {doc.metadata["account"] for doc in documents} == {"One", "Two"}


def test_sync_reports_per_account_coverage_and_missing_accounts(
    tmp_path, monkeypatch
) -> None:
    _two_account_setup(tmp_path, monkeypatch, messages_per_account=3)
    connector = apple_mail.AppleMailConnector()
    list(connector.sync(since=datetime(2026, 11, 14, tzinfo=timezone.utc)))
    stats = connector.account_stats
    assert stats["One"]["fetched"] == 3
    assert stats["Two"]["fetched"] == 3
    assert stats["Missing"]["fetched"] == 0
    assert "not found" in stats["Missing"]["error"]


def test_is_connected_tolerates_one_missing_account(tmp_path, monkeypatch) -> None:
    db_path, _ = _two_account_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(apple_mail, "_MAIL_DB", db_path)
    assert apple_mail.AppleMailConnector().is_connected() is True


def test_explicit_total_cap_still_round_robins(tmp_path, monkeypatch) -> None:
    _two_account_setup(tmp_path, monkeypatch, messages_per_account=10)
    documents = list(
        apple_mail.AppleMailConnector(max_messages=4).sync(
            since=datetime(2026, 11, 14, tzinfo=timezone.utc)
        )
    )
    accounts = [doc.metadata["account"] for doc in documents]
    assert len(documents) == 4
    assert accounts.count("One") == 2 and accounts.count("Two") == 2
