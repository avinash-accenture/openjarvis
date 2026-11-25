"""Integration tests for the read-only Apple Calendar connector and tools."""

from __future__ import annotations

import sqlite3

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from openjarvis.connectors.apple_calendar import (
    AppleCalendarConnector,
    _to_apple_ts,
)
from openjarvis.core.registry import ToolRegistry


def _make_calendar_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE ZNODE (
            Z_PK INTEGER PRIMARY KEY,
            ZTITLE TEXT,
            ZCOLORSTRING TEXT,
            ZISEVENTCONTAINER INTEGER,
            ZISENABLED INTEGER
        );

        CREATE TABLE ZCALENDARITEM (
            Z_PK INTEGER PRIMARY KEY,
            ZTITLE TEXT,
            ZNOTES TEXT,
            ZSTARTDATE REAL,
            ZENDDATE REAL,
            ZISALLDAY INTEGER,
            ZTIMEZONE TEXT,
            ZLOCALUID TEXT,
            ZUNIVERSALIDENTIFIER TEXT,
            ZSTRUCTUREDLOCATION INTEGER,
            ZHASATTENDEE INTEGER,
            ZLASTMODIFIEDDATE REAL,
            ZCALENDAR INTEGER,
            Z_ENT INTEGER,
            ZRECURRENCERULE TEXT,
            ZRECURRENCEENDDATE REAL,
            ZISDETACHED INTEGER
        );

        CREATE TABLE ZLOCATION (
            Z_PK INTEGER PRIMARY KEY,
            ZDISPLAYNAME TEXT,
            ZTITLE TEXT,
            ZADDRESS TEXT
        );

        CREATE TABLE ZRECURRENCEEXCEPTION (
            ZMASTEREVENT INTEGER,
            ZDETACHEDEVENT INTEGER,
            ZDATE REAL
        );

        INSERT INTO ZNODE
            (Z_PK, ZTITLE, ZCOLORSTRING, ZISEVENTCONTAINER, ZISENABLED)
        VALUES
            (1, 'Work', '#123456', 1, 1);
        """
    )
    return conn


def _insert_event(
    conn,
    *,
    rowid,
    summary,
    start,
    end,
    tz="UTC",
    uuid=None,
    recurrence_rule=None,
):
    event_uid = uuid or f"event-{rowid}"

    conn.execute(
        """
        INSERT INTO ZCALENDARITEM (
            Z_PK,
            ZTITLE,
            ZNOTES,
            ZSTARTDATE,
            ZENDDATE,
            ZISALLDAY,
            ZTIMEZONE,
            ZLOCALUID,
            ZUNIVERSALIDENTIFIER,
            ZSTRUCTUREDLOCATION,
            ZHASATTENDEE,
            ZLASTMODIFIEDDATE,
            ZCALENDAR,
            Z_ENT,
            ZRECURRENCERULE,
            ZRECURRENCEENDDATE,
            ZISDETACHED
        )
        VALUES (?, ?, '', ?, ?, 0, ?, ?, ?, NULL, 0, ?, 1, 5, ?, NULL, 0)
        """,
        (
            rowid,
            summary,
            _to_apple_ts(start),
            _to_apple_ts(end),
            tz,
            event_uid,
            event_uid,
            _to_apple_ts(start),
            recurrence_rule,
        ),
    )


def test_missing_database_is_an_empty_sync(tmp_path):
    connector = AppleCalendarConnector(db_path=str(tmp_path / "missing.sqlite"))
    assert list(connector.sync()) == []


def test_timezone_is_converted_before_display(tmp_path):
    db = tmp_path / "calendar.sqlite"
    conn = _make_calendar_db(db)

    now = datetime.now(timezone.utc)
    start = (now + timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0
    )

    _insert_event(
        conn,
        rowid=1,
        summary="Pacific meeting",
        start=start,
        end=start + timedelta(hours=1),
        tz="America/Los_Angeles",
    )

    conn.commit()
    conn.close()

    document = list(AppleCalendarConnector(db_path=str(db)).sync())[0]
    local_start = start.astimezone(ZoneInfo("America/Los_Angeles"))

    assert local_start.strftime("%-I:%M %p") in document.content
    assert document.metadata["start"] == local_start.isoformat()


def test_recurring_master_outside_window_yields_distinct_occurrences(tmp_path):
    db = tmp_path / "calendar.sqlite"
    conn = _make_calendar_db(db)

    now = datetime.now(timezone.utc)
    first_occurrence = now + timedelta(days=1)
    master = first_occurrence - timedelta(days=7)

    _insert_event(
        conn,
        rowid=1,
        summary="Weekly standup",
        start=master,
        end=master + timedelta(hours=1),
        uuid="weekly",
        recurrence_rule="RRULE:FREQ=WEEKLY;COUNT=3",
    )

    conn.commit()
    conn.close()

    connector = AppleCalendarConnector(
        db_path=str(db),
        days_ahead=10,
    )
    documents = list(connector.sync())

    assert [document.title for document in documents] == [
        "Weekly standup",
        "Weekly standup",
    ]
    assert len({document.doc_id for document in documents}) == 2


def test_recurring_master_is_not_duplicated_by_same_start_occurrence(tmp_path):
    db = tmp_path / "calendar.sqlite"
    conn = _make_calendar_db(db)

    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=1)

    _insert_event(
        conn,
        rowid=1,
        summary="Weekly standup",
        start=start,
        end=start + timedelta(hours=1),
        uuid="weekly",
        recurrence_rule="RRULE:FREQ=WEEKLY;COUNT=3",
    )

    conn.commit()
    conn.close()

    documents = list(AppleCalendarConnector(db_path=str(db)).sync())

    assert [document.title for document in documents] == ["Weekly standup"]
    assert documents[0].doc_id.startswith("apple_calendar:weekly:")


def test_calendar_tools_are_registered_and_executable(tmp_path, monkeypatch):
    import openjarvis.connectors.apple_calendar as calendar_module

    from openjarvis.tools.apple_calendar import (
        CalendarSearchTool,
        CalendarUpcomingTool,
    )

    # Restore registrations if the suite's autouse fixture cleared them.
    if not ToolRegistry.contains("calendar_upcoming"):
        ToolRegistry.register_value("calendar_upcoming", CalendarUpcomingTool)

    if not ToolRegistry.contains("calendar_search"):
        ToolRegistry.register_value("calendar_search", CalendarSearchTool)

    db = tmp_path / "calendar.sqlite"
    conn = _make_calendar_db(db)

    now = datetime.now(timezone.utc)

    _insert_event(
        conn,
        rowid=1,
        summary="Planning session",
        start=now + timedelta(hours=1),
        end=now + timedelta(hours=2),
    )

    conn.commit()
    conn.close()

    monkeypatch.setattr(calendar_module, "_CAL_DB", db)

    upcoming = ToolRegistry.get("calendar_upcoming")().execute(days_ahead=1)
    search = ToolRegistry.get("calendar_search")().execute(query="Planning")

    assert upcoming.success and "Planning session" in upcoming.content
    assert search.success and "Planning session" in search.content
