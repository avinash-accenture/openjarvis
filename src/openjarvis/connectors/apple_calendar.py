"""Apple Calendar connector for macOS Monterey Calendar Cache."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import rrulestr

from openjarvis.connectors._stubs import BaseConnector, Document, SyncStatus
from openjarvis.core.registry import ConnectorRegistry
from openjarvis.tools._stubs import ToolSpec


_CAL_DB = Path.home() / "Library" / "Calendars" / "Calendar Cache"
_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

_DEFAULT_DAYS_AHEAD = 2
_DEFAULT_DAYS_BEHIND = 0
_DEFAULT_TIMEZONE = "America/New_York"


def _apple_ts(seconds: float) -> datetime:
    return _APPLE_EPOCH + timedelta(seconds=float(seconds))


def _to_apple_ts(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) - _APPLE_EPOCH).total_seconds()


def _open_db(path: Path) -> sqlite3.Connection | None:
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except (sqlite3.Error, OSError):
        return None


def _zone(name: str | None, fallback: str = _DEFAULT_TIMEZONE) -> ZoneInfo:
    try:
        return ZoneInfo(name or fallback)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(fallback)


def _display_datetimes(
    start_ts: float | None,
    end_ts: float | None,
    start_tz: str | None,
    all_day: bool,
) -> tuple[datetime, datetime]:
    """Return display datetimes; all-day dates remain UTC date values."""
    if start_ts is None:
        now = datetime.now(timezone.utc)
        return now, now

    start_utc = _apple_ts(start_ts)
    end_utc = _apple_ts(end_ts) if end_ts is not None else start_utc

    if all_day:
        # Apple stores all-day dates at UTC midnight. Local conversion would
        # incorrectly move the event to the previous day in US time zones.
        return start_utc.replace(tzinfo=None), end_utc.replace(tzinfo=None)

    tz = _zone(start_tz)
    return start_utc.astimezone(tz), end_utc.astimezone(tz)


# Preserve the old search-tool result shape and column positions.
_SEARCH_QUERY = """
SELECT
    e.Z_PK AS ROWID,
    e.ZTITLE AS summary,
    e.ZNOTES AS description,
    e.ZSTARTDATE AS start_date,
    e.ZENDDATE AS end_date,
    e.ZISALLDAY AS all_day,
    e.ZTIMEZONE AS start_tz,
    e.ZLOCALUID AS UUID,
    n.ZTITLE AS calendar_title
FROM ZCALENDARITEM e
LEFT JOIN ZNODE n ON e.ZCALENDAR = n.Z_PK
WHERE e.Z_ENT = 5
  AND e.ZTITLE LIKE ?
  AND e.ZSTARTDATE IS NOT NULL
ORDER BY e.ZSTARTDATE DESC
LIMIT ?
"""

_EVENTS_QUERY = """
SELECT
    e.Z_PK AS row_id,
    e.ZTITLE AS title,
    e.ZNOTES AS notes,
    e.ZSTARTDATE AS start_ts,
    e.ZENDDATE AS end_ts,
    e.ZISALLDAY AS all_day,
    e.ZTIMEZONE AS event_tz,
    e.ZSTRUCTUREDLOCATION AS location_id,
    e.ZHASATTENDEE AS has_attendees,
    e.ZLASTMODIFIEDDATE AS last_modified,
    e.ZLOCALUID AS local_uid,
    e.ZUNIVERSALIDENTIFIER AS uuid,
    n.ZTITLE AS calendar_title,
    n.ZCOLORSTRING AS calendar_color,
    e.ZRECURRENCERULE AS recurrence_rule,
    e.ZRECURRENCEENDDATE AS recurrence_end,
    e.ZISDETACHED AS is_detached
FROM ZCALENDARITEM e
JOIN ZNODE n ON e.ZCALENDAR = n.Z_PK
WHERE e.Z_ENT = 5
  AND e.ZSTARTDATE IS NOT NULL
  AND e.ZTITLE IS NOT NULL
  AND n.ZISEVENTCONTAINER = 1
  AND n.ZISENABLED = 1
ORDER BY e.ZSTARTDATE ASC
"""

_EXCEPTIONS_QUERY = """
SELECT ZMASTEREVENT, ZDETACHEDEVENT, ZDATE
FROM ZRECURRENCEEXCEPTION
"""


@ConnectorRegistry.register("apple_calendar")
class AppleCalendarConnector(BaseConnector):
    """Read events from the local macOS Calendar Cache database."""

    connector_id = "apple_calendar"
    display_name = "Apple Calendar"
    auth_type = "local"

    def __init__(
        self,
        db_path: str = "",
        days_ahead: int = _DEFAULT_DAYS_AHEAD,
        days_behind: int = _DEFAULT_DAYS_BEHIND,
        timezone_name: str = _DEFAULT_TIMEZONE,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _CAL_DB
        self._days_ahead = max(0, int(days_ahead))
        self._days_behind = max(0, int(days_behind))
        self._timezone_name = timezone_name
        self._items_synced = 0
        self._items_total = 0
        self._last_sync: Optional[datetime] = None

    def is_connected(self) -> bool:
        return self._db_path.is_file()

    def disconnect(self) -> None:
        # This connector reads a local database and holds no persistent session.
        pass

    def sync_status(self) -> SyncStatus:
        return SyncStatus(
            state="idle",
            items_synced=self._items_synced,
            items_total=self._items_total,
            last_sync=self._last_sync,
        )

    def _display_datetimes(
        self,
        start_ts: float | None,
        end_ts: float | None,
        start_tz: str | None,
        all_day: bool,
    ) -> tuple[datetime, datetime]:
        return _display_datetimes(start_ts, end_ts, start_tz, all_day)

    def _fetch_location(
        self, conn: sqlite3.Connection, location_id: int | None
    ) -> str:
        if not location_id:
            return ""
        row = conn.execute(
            """
            SELECT ZDISPLAYNAME, ZTITLE, ZADDRESS
            FROM ZLOCATION
            WHERE Z_PK = ?
            """,
            (location_id,),
        ).fetchone()
        if not row:
            return ""
        return ", ".join(
            str(value).strip()
            for value in row
            if value and str(value).strip()
        )

    def _event_start(
        self, row: sqlite3.Row, occurrence: datetime | None = None
    ) -> datetime:
        all_day = bool(row["all_day"])
        if occurrence is not None:
            if all_day:
                return occurrence.astimezone(timezone.utc).replace(tzinfo=None)
            return occurrence
        start = _apple_ts(row["start_ts"])
        if all_day:
            return start.replace(tzinfo=None)
        tz = _zone(row["event_tz"])
        return start.astimezone(tz)

    def _event_end(
        self,
        row: sqlite3.Row,
        occurrence_start: datetime | None = None,
    ) -> datetime:
        start_raw = float(row["start_ts"])
        end_raw = (
            float(row["end_ts"])
            if row["end_ts"] is not None
            else start_raw
        )
        duration = timedelta(seconds=max(0, end_raw - start_raw))

        if occurrence_start is not None:
            return occurrence_start + duration

        return self._event_start(row) + duration

    def _occurrence_starts(
        self,
        row: sqlite3.Row,
        window_start: datetime,
        window_end: datetime,
        excluded: set[float],
    ) -> list[datetime]:
        """Return starts for one event row inside the requested time window."""
        start_utc = _apple_ts(row["start_ts"])
        all_day = bool(row["all_day"])
        rule = row["recurrence_rule"]

        if not rule:
            if all_day:
                start = start_utc.replace(tzinfo=None)
                end = self._event_end(row)
                if start.date() < window_end.date() and end.date() > window_start.date():
                    return [start]
            else:
                start = self._event_start(row)
                end = self._event_end(row)
                ws = window_start.astimezone(start.tzinfo)
                we = window_end.astimezone(start.tzinfo)
                if start < we and end >= ws:
                    return [start]
            return []

        # Recurrence is expanded in the event's own timezone. For all-day
        # events UTC is used so date values cannot shift across local midnight.
        if all_day:
            dtstart = start_utc
            query_start = window_start.astimezone(timezone.utc) - timedelta(days=2)
            query_end = window_end.astimezone(timezone.utc) + timedelta(days=2)
        else:
            event_tz = _zone(row["event_tz"])
            dtstart = start_utc.astimezone(event_tz)
            query_start = window_start.astimezone(event_tz) - timedelta(days=2)
            query_end = window_end.astimezone(event_tz) + timedelta(days=2)

        try:
            recurrence = rrulestr(rule, dtstart=dtstart)
            starts = recurrence.between(query_start, query_end, inc=True)
        except (ValueError, TypeError, OverflowError):
            # A malformed recurrence should not prevent non-recurring events
            # from being included in the briefing.
            return []

        duration = timedelta(
            seconds=max(
                0,
                float(row["end_ts"] or row["start_ts"])
                - float(row["start_ts"]),
            )
        )

        result: list[datetime] = []
        for occurrence in starts:
            occurrence_utc = occurrence.astimezone(timezone.utc)
            if any(
                abs(_to_apple_ts(occurrence_utc) - value) < 1
                for value in excluded
            ):
                continue

            if all_day:
                occurrence_date = occurrence_utc.date()
                if (
                    occurrence_date < window_end.date()
                    and (occurrence_date + duration) > window_start.date()
                ):
                    result.append(occurrence_utc.replace(tzinfo=None))
            else:
                end = occurrence + duration
                if occurrence < query_end and end >= query_start:
                    if occurrence < window_end.astimezone(occurrence.tzinfo) and end >= window_start.astimezone(occurrence.tzinfo):
                        result.append(occurrence)

        return result

    def _make_document(
        self,
        row: sqlite3.Row,
        start: datetime,
        location: str,
    ) -> Document:
        all_day = bool(row["all_day"])
        end = self._event_end(row, start)

        if all_day:
            start_text = start.strftime("%A, %B %-d %Y (all day)")
            timestamp = datetime.combine(
                start.date(), time.min, tzinfo=timezone.utc
            )
            start_value = start.date().isoformat()
            end_value = end.date().isoformat()
        else:
            start_text = start.strftime("%A, %B %-d at %-I:%M %p %Z")
            timestamp = start
            start_value = start.isoformat()
            end_value = end.isoformat()

        title = str(row["title"] or "Untitled event").strip()
        calendar = str(row["calendar_title"] or "Calendar")
        parts = [title, start_text, f"Calendar: {calendar}"]

        if location:
            parts.append(f"Location: {location}")
        if row["notes"]:
            parts.append(f"Notes: {str(row['notes']).strip()}")

        uid = row["local_uid"] or row["uuid"] or row["row_id"]
        occurrence_key = start_value.replace(":", "-")
        return Document(
            doc_id=f"apple_calendar:{uid}:{occurrence_key}",
            source="apple_calendar",
            doc_type="calendar_event",
            title=title,
            content="\n".join(parts),
            timestamp=timestamp,
            metadata={
                "start": start_value,
                "end": end_value,
                "calendar": calendar,
                "all_day": all_day,
                "location": location,
                "has_attendees": bool(row["has_attendees"]),
                "event_timezone": row["event_tz"] or "",
            },
        )

    def sync(
        self,
        *,
        since: datetime | None = None,
        cursor: str | None = None,
    ) -> Iterator[Document]:
        # Intentionally do not use `since`: a future event created months ago
        # still belongs in an upcoming-events briefing.
        conn = _open_db(self._db_path)
        if conn is None:
            return

        tz = _zone(self._timezone_name)
        now = datetime.now(tz)
        window_start = datetime.combine(
            (now - timedelta(days=self._days_behind)).date(),
            time.min,
            tzinfo=tz,
        )
        window_end = datetime.combine(
            (now + timedelta(days=self._days_ahead + 1)).date(),
            time.min,
            tzinfo=tz,
        )

        try:
            rows = conn.execute(_EVENTS_QUERY).fetchall()
            exception_rows = conn.execute(_EXCEPTIONS_QUERY).fetchall()

            excluded_by_master: dict[int, set[float]] = {}
            for exception in exception_rows:
                master_id = exception["ZMASTEREVENT"]
                exception_date = exception["ZDATE"]
                if master_id is not None and exception_date is not None:
                    excluded_by_master.setdefault(int(master_id), set()).add(
                        float(exception_date)
                    )

            # Keep detached replacement events as independent rows. The
            # original recurrence instance is suppressed by its exception date.
            documents: list[Document] = []
            for row in rows:
                excluded = excluded_by_master.get(int(row["row_id"]), set())
                starts = self._occurrence_starts(
                    row, window_start, window_end, excluded
                )
                location = self._fetch_location(conn, row["location_id"])

                for start in starts:
                    documents.append(
                        self._make_document(row, start, location)
                    )

            documents.sort(key=lambda doc: doc.timestamp)
            self._items_total = len(documents)
            self._items_synced = len(documents)
            self._last_sync = datetime.now(timezone.utc)

        finally:
            conn.close()

        yield from documents

    def mcp_tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                name="calendar_upcoming",
                description=(
                    "Return upcoming Apple Calendar events from the local "
                    "macOS Calendar database."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "days_ahead": {
                            "type": "integer",
                            "description": "Number of days ahead to look.",
                            "default": 7,
                        }
                    },
                },
                category="knowledge",
            ),
            ToolSpec(
                name="calendar_search",
                description="Search local Apple Calendar event titles.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {
                            "type": "integer",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
                category="knowledge",
            ),
        ]
