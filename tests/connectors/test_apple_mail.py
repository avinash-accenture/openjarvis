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
