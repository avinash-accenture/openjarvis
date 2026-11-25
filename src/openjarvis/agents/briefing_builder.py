# ruff: noqa: E501
"""Deterministic morning briefing: spoken text plus a written report.

Input is the structured ``briefing`` payload produced by ``digest_collect``.
No language model writes any part of the spoken text; every statement comes
from a retrieved field or a fixed template, so the briefing cannot invent
facts. Optional per-email summaries (produced elsewhere and grounding-checked)
are spoken with an explicit "Summary" label; otherwise generic messages get a
labelled "Preview" of their first words.

The spoken text is kept within a listening budget (default five minutes);
when it would run over, previews are shortened, then dropped from
lower-priority items, then the lowest-ranked emails move to the written
report only. Nothing is dropped silently: the briefing says how many.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Spoken order and labels for mail categories.
CATEGORY_LABELS = {
    "ACTION_REQUIRED": "Action required",
    "TIME_SENSITIVE": "Time sensitive",
    "REVIEW": "For review",
    "COMPLETED": "Completed",
    "LOW_PRIORITY": "Lower priority",
}
_CATEGORY_ORDER = list(CATEGORY_LABELS)

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_LONG_NUMBER = re.compile(r"(?<![\d.])\d{5,}(?![\d.])")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NON_SPEECH = re.compile(r"[^\w\s.,:;'’!?&()/@+%$£€₹-]", re.UNICODE)

DEFAULT_WORDS_PER_MINUTE = 126  # macOS `say` at 175 wpm x 0.72 speed
DEFAULT_MAX_MINUTES = 5.0
PREVIEW_WORDS = 25
SHORT_PREVIEW_WORDS = 12


@dataclass
class Briefing:
    """The result of building a briefing."""

    spoken: str
    report_markdown: str
    word_count: int
    estimated_seconds: float
    emails_spoken: int
    emails_report_only: int
    issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def words(text: str) -> int:
    return len(text.split())


def _clean(text: str, limit: int = 160) -> str:
    """Make retrieved text safe and pleasant to speak."""
    text = _URL.sub("", text or "")
    text = _LONG_NUMBER.sub("", text)
    text = _NON_SPEECH.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text.strip(" -–—:")


def _first_words(text: str, count: int) -> str:
    parts = _clean(text, 2000).split()
    if not parts:
        return ""
    clipped = " ".join(parts[:count])
    return clipped + ("…" if len(parts) > count else "")


def _sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def _speak_date(value: date, today: date) -> str:
    if value == today:
        return "today"
    if value == today + timedelta(days=1):
        return "tomorrow"
    if today < value < today + timedelta(days=7):
        return value.strftime("%A")
    return value.strftime("%A, %B ") + str(value.day)


def _humanize_iso_dates(text: str, today: date) -> str:
    def repl(match: re.Match) -> str:
        try:
            value = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return match.group(0)
        spoken = value.strftime("%B ") + str(value.day)
        if value.year != today.year:
            spoken += f", {value.year}"
        return spoken

    return _ISO_DATE.sub(repl, text)


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.combine(date.fromisoformat(value[:10]), datetime.min.time())
        except ValueError:
            return None


def _time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def sender_label(entry: Dict[str, Any]) -> str:
    """Display name when known, else the sender's domain."""
    name = _clean(str(entry.get("sender_name") or ""), 60)
    if name and "@" not in name:
        return name
    sender = str(entry.get("sender") or "")
    if "@" in sender:
        return sender.rsplit("@", 1)[1].strip("> ").lower() or "an unknown sender"
    return _clean(sender, 60) or "an unknown sender"


def _received_phrase(value: str, today: date) -> str:
    dt = _parse_dt(value)
    if dt is None:
        return ""
    day = dt.date()
    if day == today:
        return f"at {_time(dt)}"
    if day == today - timedelta(days=1):
        return f"yesterday at {_time(dt)}"
    return f"on {dt.strftime('%B')} {dt.day} at {_time(dt)}"


# ---------------------------------------------------------------------------
# Mail
# ---------------------------------------------------------------------------


@dataclass
class _MailLine:
    entry: Dict[str, Any]
    rank: int
    detail: str  # "summary", "preview", "note" or ""
    preview_words: int = PREVIEW_WORDS


def _mail_rank(entry: Dict[str, Any]) -> float:
    try:
        rank = float(_CATEGORY_ORDER.index(entry.get("category", "")))
    except ValueError:
        rank = float(len(_CATEGORY_ORDER))
    if entry.get("category") == "REVIEW" and entry.get("status") != "personal":
        rank += 0.5
    return rank


def _email_sentence(
    line: _MailLine,
    today: date,
    summaries: Dict[str, str],
) -> str:
    entry = line.entry
    subject = _clean(str(entry.get("subject") or ""), 140) or "no subject"
    when = _received_phrase(str(entry.get("received") or ""), today)
    head = f"From {sender_label(entry)}"
    if when:
        head += f", {when}"
    parts = [_sentence(f"{head}: {subject}")]

    note = _humanize_iso_dates(_clean(str(entry.get("note") or ""), 300), today)
    summary = summaries.get(str(entry.get("doc_id")), "")
    preview = str(entry.get("preview") or "")

    if line.detail == "summary" and summary:
        parts.append(_sentence(f"Summary: {_clean(summary, 400)}"))
    elif line.detail == "preview" and preview:
        parts.append(_sentence(f"Preview: {_first_words(preview, line.preview_words)}"))
    elif line.detail == "note" and note:
        parts.append(_sentence(note))

    if line.detail in {"preview", "summary"} and not entry.get("body_available", True):
        parts.append("The message body is not available on this Mac yet.")
    return " ".join(parts)


def _default_detail(entry: Dict[str, Any], summaries: Dict[str, str]) -> str:
    if str(entry.get("doc_id")) in summaries:
        return "summary"
    if entry.get("preview"):
        return "preview"
    if entry.get("note"):
        return "note"
    return ""


def _mail_section(
    lines: List[_MailLine],
    stats: Dict[str, Any],
    today: date,
    summaries: Dict[str, str],
    report_only: int,
) -> str:
    in_window = int(stats.get("in_window", len(lines) + report_only))
    spoken = len(lines)
    if in_window == 0:
        return "No new email arrived in the last 24 hours."

    opener = (
        f"In the last 24 hours, {in_window} "
        f"email{'s' if in_window != 1 else ''} arrived across your accounts."
    )
    if spoken:
        opener += (
            f" Here {'are' if spoken != 1 else 'is'} the {spoken} "
            f"that matter{'s' if spoken == 1 else ''} most."
        )
    out = [opener]

    current = None
    low_priority: List[_MailLine] = []
    for line in lines:
        category = line.entry.get("category", "")
        if category == "LOW_PRIORITY":
            low_priority.append(line)
            continue
        if category != current:
            current = category
            out.append(_sentence(CATEGORY_LABELS.get(category, "Other")))
        out.append(_email_sentence(line, today, summaries))
    if low_priority:
        out.append(_low_priority_sentence(low_priority))

    skipped: List[str] = []
    excluded = int(stats.get("excluded_marketing", 0) or 0)
    omitted = sum(int(v) for v in (stats.get("omitted_by_category") or {}).values())
    if report_only:
        skipped.append(f"{report_only} more in the written report")
    if omitted:
        skipped.append(
            f"{omitted} lower-ranked email{'s' if omitted != 1 else ''} not selected"
        )
    if excluded:
        skipped.append(
            f"{excluded} promotional email{'s' if excluded != 1 else ''} skipped"
        )
    if skipped:
        out.append(_sentence("Also: " + "; ".join(skipped)))

    failed = [
        name
        for name, account in (stats.get("accounts") or {}).items()
        if account.get("error")
    ]
    if failed:
        out.append(_sentence("I could not read mail from " + _join(failed)))
    return " ".join(out)


def _low_priority_sentence(lines: List[_MailLine], per_sender: int = 5) -> str:
    """Lower-priority mail, named compactly by sender (each one is in the report)."""
    by_sender: Dict[str, List[str]] = {}
    for line in lines:
        subject = _clean(str(line.entry.get("subject") or ""), 80) or "no subject"
        by_sender.setdefault(sender_label(line.entry), []).append(subject)
    groups = []
    for sender, subjects in by_sender.items():
        shown = subjects[:per_sender]
        text = f"from {sender}: " + _join(shown)
        if len(subjects) > per_sender:
            text += f", and {len(subjects) - per_sender} more"
        groups.append(text)
    count = len(lines)
    return _sentence(
        f"Lower priority, {count} email{'s' if count != 1 else ''}, "
        + "; ".join(groups)
    )


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _event_day(event: Dict[str, Any]) -> Optional[date]:
    dt = _parse_dt(str(event.get("start") or ""))
    return dt.date() if dt else None


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def possible_duplicates(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same title on the same day more than once. Reported, never merged."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for event in events:
        day = _event_day(event)
        key = (day, _norm_title(str(event.get("title") or "")))
        if day and key[1]:
            groups.setdefault(key, []).append(event)
    return [
        {"day": day, "title": items[0].get("title"), "events": items}
        for (day, _), items in groups.items()
        if len(items) > 1
    ]


def _event_phrase(event: Dict[str, Any], today: date) -> str:
    title = _clean(str(event.get("title") or ""), 120) or "Untitled event"
    start = _parse_dt(str(event.get("start") or ""))
    end = _parse_dt(str(event.get("end") or ""))
    location = _clean(str(event.get("location") or ""), 80)

    if event.get("all_day"):
        when = "All day"
        if start and end:
            last = end.date() - timedelta(days=1)  # all-day end is exclusive
            if start.date() < today <= last:
                when = f"Ongoing through {_speak_date(last, today)}"
    elif start:
        when = _time(start)
    else:
        when = "Time not available"
    phrase = f"{when}: {title}"
    if location:
        phrase += f", at {location}"
    return _sentence(phrase)


def _calendar_section(
    events: List[Dict[str, Any]], today: date, horizon_days: int
) -> str:
    if not events:
        return _sentence(
            f"Nothing is on your calendar for today or the next {horizon_days} days"
        )
    by_day: Dict[date, List[Dict[str, Any]]] = {}
    for event in events:
        day = _event_day(event) or today
        if event.get("all_day"):
            end = _parse_dt(str(event.get("end") or ""))
            # Ongoing multi-day events are listed under today.
            if end and day < today < end.date():
                day = today
        by_day.setdefault(day, []).append(event)

    out = ["Your calendar."]
    for day in sorted(by_day):
        items = sorted(
            by_day[day],
            key=lambda e: (not e.get("all_day"), str(e.get("start") or "")),
        )
        label = _speak_date(day, today)
        out.append(_sentence(label[0].upper() + label[1:]))
        out.extend(_event_phrase(event, today) for event in items)

    for dup in possible_duplicates(events):
        times = [
            "all day" if e.get("all_day") else _time(_parse_dt(str(e.get("start"))))
            for e in dup["events"]
            if e.get("all_day") or _parse_dt(str(e.get("start")))
        ]
        out.append(
            _sentence(
                f"Note: {len(dup['events'])} entries are named "
                f"{_clean(str(dup['title']), 80)} on {_speak_date(dup['day'], today)}"
                + (f", at {_join(times)}" if times else "")
                + ". They may be separate events, so both are kept"
            )
        )
    return " ".join(out)


def _special_section(events: List[Dict[str, Any]], today: date) -> str:
    upcoming = sorted(
        (e for e in events if _event_day(e) and _event_day(e) >= today),
        key=lambda e: _event_day(e),
    )
    if not upcoming:
        return ""
    seen = set()
    phrases = []
    for event in upcoming:
        key = (_event_day(event), _norm_title(str(event.get("title") or "")))
        if key in seen:
            continue
        seen.add(key)
        when = _speak_date(_event_day(event), today)
        if when not in {"today", "tomorrow"}:
            when = f"on {when}"
        phrases.append(f"{_clean(str(event.get('title') or ''), 80)} {when}")
    return _sentence("Coming up: " + "; ".join(phrases))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_briefing(
    payload: Dict[str, Any],
    *,
    now: datetime,
    honorific: str = "sir",
    sections: Sequence[str] = ("messages", "calendar"),
    failed_sections: Iterable[str] = (),
    summaries: Optional[Dict[str, str]] = None,
    words_per_minute: float = DEFAULT_WORDS_PER_MINUTE,
    max_minutes: float = DEFAULT_MAX_MINUTES,
    calendar_horizon_days: int = 2,
) -> Briefing:
    """Build the spoken briefing and its written report from *payload*."""
    today = now.date()
    summaries = dict(summaries or {})
    for entry in payload.get("mail") or []:
        if entry.get("summary"):
            summaries.setdefault(str(entry.get("doc_id")), str(entry["summary"]))
    failed = {s.casefold() for s in failed_sections}
    wanted = [s.strip().casefold() for s in sections if str(s).strip()]
    budget = int(words_per_minute * max_minutes)

    # Category rank first; newest first within a category (stable sorts).
    mail = sorted(
        payload.get("mail") or [],
        key=lambda e: str(e.get("received") or ""),
        reverse=True,
    )
    mail.sort(key=_mail_rank)
    stats = payload.get("mail_stats") or {}
    events = payload.get("calendar") or []
    special = payload.get("special_events") or []

    greeting_name = honorific.strip()
    opening = (
        f"Good morning{', ' + greeting_name if greeting_name else ''}. "
        f"It's {now.strftime('%A, %B')} {now.day}."
    )

    lines = [
        _MailLine(entry, rank=i, detail=_default_detail(entry, summaries))
        for i, entry in enumerate(mail)
    ]
    report_only: List[Dict[str, Any]] = []

    def compose() -> str:
        parts = [opening]
        if "messages" in wanted:
            if "messages" in failed and not mail:
                parts.append("I could not retrieve your email this morning.")
            else:
                parts.append(
                    _mail_section(lines, stats, today, summaries, len(report_only))
                )
        if "calendar" in wanted:
            if "calendar" in failed and not events:
                parts.append("I could not retrieve your calendar this morning.")
            else:
                parts.append(_calendar_section(events, today, calendar_horizon_days))
            special_text = _special_section(special, today)
            if special_text:
                parts.append(special_text)
        parts.append("That's your briefing.")
        return " ".join(p for p in parts if p)

    spoken = compose()
    # Fit the listening budget: shorten previews, then drop detail from the
    # lowest-ranked items, then move the lowest-ranked emails to the report.
    if words(spoken) > budget:
        for line in lines:
            line.preview_words = SHORT_PREVIEW_WORDS
        spoken = compose()
    for line in reversed(lines):
        if words(spoken) <= budget:
            break
        if line.detail in {"preview", "summary"} and line.entry.get("category") in {
            "LOW_PRIORITY",
            "COMPLETED",
        }:
            line.detail = "note" if line.entry.get("note") else ""
            spoken = compose()
    while words(spoken) > budget and lines:
        report_only.insert(0, lines.pop().entry)
        spoken = compose()

    word_count = words(spoken)
    seconds = word_count / max(words_per_minute, 1) * 60.0
    issues = validate_spoken(spoken, payload, budget=budget)
    report = _report(
        payload,
        now=now,
        spoken=spoken,
        lines=lines,
        report_only=report_only,
        summaries=summaries,
        word_count=word_count,
        seconds=seconds,
        budget=budget,
        issues=issues,
    )
    return Briefing(
        spoken=spoken,
        report_markdown=report,
        word_count=word_count,
        estimated_seconds=seconds,
        emails_spoken=len(lines),
        emails_report_only=len(report_only),
        issues=issues,
    )


def validate_spoken(text: str, payload: Dict[str, Any], *, budget: int) -> List[str]:
    """Checks that must hold for every briefing. Returns problems found."""
    problems: List[str] = []
    if words(text) > budget:
        problems.append(f"Spoken text is {words(text)} words; budget is {budget}.")
    if _URL.search(text):
        problems.append("Spoken text contains a URL.")
    if _LONG_NUMBER.search(text):
        problems.append("Spoken text contains a long number (possible code or ID).")
    for entry in payload.get("mail") or []:
        if not entry.get("subject") and not entry.get("sender"):
            problems.append(f"Email {entry.get('doc_id')} has no sender or subject.")
    for name, account in (
        (payload.get("mail_stats") or {}).get("accounts") or {}
    ).items():
        if account.get("error"):
            problems.append(f"Mail account {name}: {account['error']}")
        fetched = int(account.get("fetched", 0) or 0)
        with_body = int(account.get("with_body", 0) or 0)
        if fetched and with_body / fetched < 0.5:
            problems.append(
                f"Mail account {name}: only {with_body} of {fetched} messages "
                "have a body downloaded locally."
            )
    return problems


# ---------------------------------------------------------------------------
# Written report
# ---------------------------------------------------------------------------


def _md(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def _report(
    payload: Dict[str, Any],
    *,
    now: datetime,
    spoken: str,
    lines: List[_MailLine],
    report_only: List[Dict[str, Any]],
    summaries: Dict[str, str],
    word_count: int,
    seconds: float,
    budget: int,
    issues: List[str],
) -> str:
    stats = payload.get("mail_stats") or {}
    out: List[str] = [
        f"# Morning briefing — {now.strftime('%A, %B')} {now.day}, {now.year}",
        "",
        f"Generated {now.strftime('%Y-%m-%d %H:%M %Z').strip()}. "
        f"Spoken length: {word_count} words, about {seconds / 60:.1f} minutes "
        f"(budget {budget} words).",
        "",
        "## Validation",
        "",
    ]
    out += [f"- ⚠️ {_md(i)}" for i in issues] or ["- No problems found."]

    out += ["", "## Mail coverage (last 24 hours)", ""]
    accounts = stats.get("accounts") or {}
    if accounts:
        out += [
            "| Account | Messages | With body | Status |",
            "| --- | --- | --- | --- |",
        ]
        for name, account in accounts.items():
            out.append(
                f"| {_md(name)} | {account.get('fetched', 0)} | "
                f"{account.get('with_body', 0)} | {_md(account.get('error') or 'ok')} |"
            )
        out.append("")
    out.append(
        f"In window: {stats.get('in_window', 0)}. Selected: {stats.get('selected', 0)}. "
        f"Promotional skipped: {stats.get('excluded_marketing', 0)}. "
        f"Not selected by rank: {sum((stats.get('omitted_by_category') or {}).values())} "
        f"{dict(stats.get('omitted_by_category') or {}) or ''}. "
        f"Duplicate shipment notices removed: {stats.get('duplicate_shipments_removed', 0)}."
    )

    out += ["", "## Selected email", ""]
    spoken_ids = {str(line.entry.get("doc_id")) for line in lines}
    for i, entry in enumerate(payload.get("mail") or [], 1):
        doc_id = str(entry.get("doc_id"))
        where = "spoken" if doc_id in spoken_ids else "report only"
        out.append(
            f"{i}. **{_md(entry.get('subject')) or '(no subject)'}** — "
            f"{_md(entry.get('sender_name') or entry.get('sender'))} "
            f"<{_md(entry.get('sender'))}>, {_md(entry.get('account'))}, "
            f"received {_md(entry.get('received'))} ({where})"
        )
        out.append(
            f"   - `{entry.get('category')}` / `{entry.get('status')}`; evidence: "
            f"{', '.join(entry.get('evidence') or []) or 'none'}; "
            f"body {'available' if entry.get('body_available') else 'NOT available'}"
            + ("; unread" if entry.get("unread") else "")
            + ("; flagged" if entry.get("flagged") else "")
        )
        if entry.get("note"):
            out.append(f"   - Classification note: {_md(entry.get('note'))}")
        if doc_id in summaries:
            out.append(
                f"   - Summary (local model, grounding-checked): {_md(summaries[doc_id])}"
            )
        if entry.get("preview"):
            out.append(
                f"   - Preview (first words, not a summary): {_md(entry.get('preview'))}"
            )
        for key, label in (
            ("deadline", "Deadline"),
            ("event_date", "Event date"),
            ("effective_date", "Effective date"),
        ):
            if entry.get(key):
                out.append(f"   - {label}: {entry[key]}")
    if not payload.get("mail"):
        out.append("No email selected.")

    out += ["", "## Calendar", ""]
    for event in sorted(
        payload.get("calendar") or [], key=lambda e: str(e.get("start"))
    ):
        out.append(
            f"- {_md(event.get('start'))}"
            + (" (all day)" if event.get("all_day") else "")
            + f" — **{_md(event.get('title'))}** [{_md(event.get('calendar'))}]"
            + (f" — {_md(event.get('location'))}" if event.get("location") else "")
        )
    if not payload.get("calendar"):
        out.append("No events in the window.")
    dups = possible_duplicates(payload.get("calendar") or [])
    if dups:
        out += ["", "### Possible duplicates (not merged)", ""]
        for dup in dups:
            out.append(
                f"- {dup['day']}: {_md(dup['title'])} × {len(dup['events'])} — "
                + "; ".join(
                    f"{_md(e.get('start'))} [{_md(e.get('calendar'))}]"
                    for e in dup["events"]
                )
            )

    out += ["", "## Birthdays and anniversaries", ""]
    special = payload.get("special_events") or []
    for event in sorted(special, key=lambda e: str(e.get("start"))):
        out.append(f"- {_md(event.get('start'))[:10]} — {_md(event.get('title'))}")
    if not special:
        out.append("None in the lookahead window.")

    out += ["", "## Spoken text", "", spoken, ""]
    return "\n".join(out)
