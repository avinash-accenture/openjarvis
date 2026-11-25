# ruff: noqa: E501
"""Digest collection tool — fetches recent data from configured connectors."""

from __future__ import annotations

import html
import json
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from openjarvis.connectors._stubs import Document
from openjarvis.core.registry import ConnectorRegistry, ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.mail_classifier import (
    GENERIC_STATUSES,
    ClassifiedMailItem,
    classify_mail_document,
    classify_mail_document_with_local_llm,
    deduplicate_shipments,
    summarize_mail_with_local_llm,
)

# ---------------------------------------------------------------------------
# Section definitions: ordered list of (section_name, connector_ids)
# ---------------------------------------------------------------------------

_SECTION_ORDER: List[tuple] = [
    ("HEALTH", {"oura", "apple_health", "strava"}),
    (
        "MESSAGES",
        {
            "gmail",
            "gmail_imap",
            "google_tasks",
            "slack",
            "imessage",
            "whatsapp",
            "outlook",
            "apple_mail",
            "notion",
            "github_notifications",
        },
    ),
    ("CALENDAR", {"gcalendar", "apple_calendar"}),
    ("WORLD", {"weather", "hackernews", "news_rss"}),
    ("MUSIC", {"spotify", "apple_music"}),
]

_CONNECTOR_TO_SECTION: Dict[str, str] = {}
for _section_name, _ids in _SECTION_ORDER:
    for _cid in _ids:
        _CONNECTOR_TO_SECTION[_cid] = _section_name


# ---------------------------------------------------------------------------
# Per-connector human-readable formatters
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)

def _format_apple_calendar(doc: Document) -> str:
    """Format a local Apple Calendar event, including ongoing date ranges."""
    start = doc.metadata.get("start", "")
    end = doc.metadata.get("end", "")
    title = doc.title or "(No title)"
    calendar = doc.metadata.get("calendar", "")
    location = doc.metadata.get("location", "")
    all_day = bool(doc.metadata.get("all_day"))

    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end) if end else None

        if all_day:
            today = datetime.now().date()
            if end_dt is not None and start_dt.date() < today < end_dt.date():
                # All-day end dates are exclusive in Apple Calendar.
                last_day = end_dt.date() - timedelta(days=1)
                time_str = (
                    f"Ongoing: {start_dt.strftime('%B %-d')}–"
                    f"{last_day.strftime('%B %-d')}"
                )
            else:
                time_str = f"{start_dt.strftime('%A, %B %-d')} (all day)"
        else:
            now = (
                datetime.now(start_dt.tzinfo)
                if start_dt.tzinfo
                else datetime.now()
            )
            if end_dt is not None and start_dt < now < end_dt:
                last_day = end_dt.date()
                if end_dt.time() == datetime.min.time():
                    last_day -= timedelta(days=1)
                time_str = (
                    f"Ongoing: {start_dt.strftime('%B %-d')}–"
                    f"{last_day.strftime('%B %-d')}"
                )
            else:
                time_str = (
                    f"{start_dt.strftime('%A, %B %-d')} "
                    f"at {_format_time(start_dt)}"
                )
    except (ValueError, TypeError):
        time_str = start or "Time unavailable"

    line = f"[apple_calendar] {time_str} — {title}"
    if calendar:
        line += f" [{calendar}]"
    if location:
        line += f" — {location}"
    return line

def _time_ago(ts: datetime) -> str:
    """Return a human-readable relative time like '2h ago' or '15m ago'."""
    now = datetime.now()
    ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
    delta = now - ts_naive
    total_seconds = max(0, int(delta.total_seconds()))
    if total_seconds < 60:
        return "just now"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m ago"
    if total_seconds < 86400:
        return f"{total_seconds // 3600}h ago"
    days = total_seconds // 86400
    return f"{days}d ago"


def _format_date(ts: datetime) -> str:
    """Format a datetime as 'April 1' style."""
    return ts.strftime("%B %-d") if hasattr(ts, "strftime") else str(ts)


def _format_time(ts: datetime) -> str:
    """Format a datetime as '10:30 AM' style."""
    return ts.strftime("%-I:%M %p") if hasattr(ts, "strftime") else str(ts)


def _parse_content_json(doc: Document) -> Dict[str, Any]:
    """Try to parse the document content as JSON; return {} on failure."""
    try:
        return json.loads(doc.content)
    except (json.JSONDecodeError, TypeError):
        return {}


def _format_oura(doc: Document) -> str:
    """Format an Oura Ring document into a readable line."""
    data = _parse_content_json(doc)
    data_type = doc.metadata.get("data_type", "")
    day = doc.metadata.get("day", "")
    day_str = day or _format_date(doc.timestamp)

    if data_type == "sleep":
        hr = data.get("average_heart_rate", "?")
        hrv = data.get("average_hrv", data.get("average_heart_rate_variability", "?"))
        total = data.get("total_sleep_duration")
        awake = data.get("awake_time")
        score = data.get("score")
        parts = []
        if score is not None:
            parts.append(f"score {score}")
        parts.append(f"avg HR {hr} bpm")
        if hrv != "?":
            parts.append(f"HRV {hrv}")
        if total is not None:
            parts.append(f"{_format_duration(total)} total sleep")
        if awake is not None:
            parts.append(f"awake {_format_duration(awake)}")
        return f"[oura] Sleep — {day_str}: {', '.join(parts)}"

    if data_type == "daily_readiness":
        score = data.get("score", "?")
        temp = data.get(
            "temperature_deviation",
            data.get("temperature_trend_deviation"),
        )
        line = f"[oura] Readiness — {day_str}: score {score}"
        if temp is not None:
            sign = "+" if temp >= 0 else ""
            line += f", temperature deviation {sign}{temp}"
        return line

    if data_type == "daily_activity":
        steps = data.get("steps", "?")
        cal = data.get("total_calories", data.get("active_calories", "?"))
        score = data.get("score")
        parts = []
        if score is not None:
            parts.append(f"score {score}")
        parts.append(f"steps {steps}")
        parts.append(f"calories {cal}")
        return f"[oura] Activity — {day_str}: {', '.join(parts)}"

    # Fallback for unknown Oura doc types
    return f"[oura] {doc.title}"


def _format_apple_health(doc: Document) -> str:
    """Format an Apple Health document."""
    return f"[apple_health] {doc.title}"


def _format_strava(doc: Document) -> str:
    """Format a Strava activity document."""
    return f"[strava] {doc.title}"

_SECURITY_CODE_CONTEXT = re.compile(
    r"\b(?:verification|authentication|security|one[- ]time|otp|"
    r"passcode|password|2[- ]factor|two[- ]factor|mfa|login|sign[- ]in)"
    r"\b",
    re.IGNORECASE,
)

_NUMERIC_CODE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

def _remove_email_css(text: str) -> str:
    """Remove CSS rules, including minified and nested at-rules."""
    selector = re.compile(
          r"@[-\w]+\b"
          r"|(?:\*|body|html|table|td|th|p|div|a|img|head)"
          r"(?:\s|,|\.|#|\[|:|$)"
          r"|[.#][\w-]+(?:\s|,|\{)",
          re.IGNORECASE,
    )
    result = []
    cursor = 0

    while cursor < len(text):
        brace = text.find("{", cursor)

        if brace == -1:
            result.append(text[cursor:])
            break

        # Find a possible CSS selector before this opening brace.
        prefix = text[cursor:brace]
        match = selector.search(prefix)

        if match is None:
            result.append(text[cursor:brace + 1])
            cursor = brace + 1
            continue

        start = cursor + match.start()

        # Preserve any ordinary text before the CSS selector.
        result.append(text[cursor:start])

        depth = 1
        end = brace + 1

        while end < len(text) and depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1

        if depth:
            # Malformed CSS: do not expose the remaining fragment.
            break

        cursor = end

    return "".join(result)

def _sanitize_email_preview(content: str, limit: int) -> str:
    """Clean email markup and remove likely secrets before briefing or TTS."""
    text = html.unescape(content)

    # Remove non-content blocks and CSS that some senders leak into
    # their plain-text email alternatives.
    text = re.sub(
        r"(?is)<(script|style)[^>]*>.*?</(?:script|style)\s*>",
        " ",
        text,
    )
    text = re.sub(r"(?s)<!--.*?-->", " ", text)

    text = _remove_email_css(text)
    # Some email alternatives contain escaped HTML tags.
    text = re.sub(r"\\(?=<)", "", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)

    # Remove an incomplete HTML tag left at the end of a truncated preview.
    text = re.sub(r"<[^>]*$", " ", text)

    # Remove invisible formatting characters, including zero-width and
    # bidirectional control characters.
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cf"
    )

    # Normalize whitespace left by HTML layout and email templates.
    text = " ".join(text.split())

    # Never read authentication links aloud.
    text = _URL_PATTERN.sub("[link omitted]", text)

    # Mask numeric codes when the preview contains security-related context.
    if _SECURITY_CODE_CONTEXT.search(text):
        text = _NUMERIC_CODE.sub("[code redacted]", text)

    return text[:limit]

def _format_gmail(doc: Document) -> str:
    """Format a Gmail email document.

    Includes ``doc_id`` so the proactive agent can reference the
    real Gmail ``messages.get/modify`` id in its action proposals
    instead of hallucinating one.
    """
    sender = doc.author or "Unknown"
    subject = doc.title or "(no subject)"
    ago = _time_ago(doc.timestamp)
    body = _sanitize_email_preview(doc.content, 150) if doc.content else ""
    line = f'[gmail id={doc.doc_id}] From: {sender} — "{subject}" ({ago})'
    if body:
        line += f"\n  Preview: {body}"
    return line


def _format_gmail_imap(doc: Document) -> str:
    """Format a Gmail IMAP email document."""
    sender = doc.author or "Unknown"
    subject = doc.title or "(no subject)"
    ago = _time_ago(doc.timestamp)
    return f'[gmail id={doc.doc_id}] From: {sender} — "{subject}" ({ago})'


def _format_google_tasks(doc: Document) -> str:
    """Format a Google Tasks document."""
    title = doc.title or "Untitled Task"
    status = doc.metadata.get("status", "")
    due = doc.metadata.get("due", "")
    parts = [f"[google_tasks] {title}"]
    extras = []
    if due:
        extras.append(f"due {due}")
    if status == "completed":
        extras.append("completed")
    if extras:
        parts.append(f"({', '.join(extras)})")
    return " ".join(parts)


def _format_slack(doc: Document) -> str:
    """Format a Slack message document."""
    author = doc.author or "Unknown"
    channel = doc.metadata.get("channel", "")
    ago = _time_ago(doc.timestamp)
    snippet = doc.content[:150].replace("\n", " ").strip()
    content_preview = snippet if doc.content else ""
    prefix = f"[slack] #{channel}" if channel else "[slack]"
    line = f"{prefix} {author} ({ago})"
    if content_preview:
        line += f": {content_preview}"
    return line


def _format_imessage(doc: Document) -> str:
    """Format an iMessage document."""
    sender = doc.author or "Unknown"
    ago = _time_ago(doc.timestamp)
    snippet = doc.content[:150].replace("\n", " ").strip()
    content_preview = snippet if doc.content else ""
    line = f"[imessage] {sender} ({ago})"
    if content_preview:
        line += f": {content_preview}"
    return line


def _format_whatsapp(doc: Document) -> str:
    """Format a WhatsApp message document."""
    sender = doc.author or "Unknown"
    content_preview = doc.content[:80].replace("\n", " ").strip() if doc.content else ""
    return f"[whatsapp] {sender}: {content_preview}"


def _format_outlook(doc: Document) -> str:
    """Format an Outlook email document."""
    sender = doc.author or "Unknown"
    subject = doc.title or "(no subject)"
    ago = _time_ago(doc.timestamp)
    return f'[outlook] From: {sender} — "{subject}" ({ago})'

def _format_classified_apple_mail(
    item: Any,
    doc: Document,
) -> str:
    """Format one Apple Mail message with its classification and context."""
    sender = doc.author or "Unknown sender"
    subject = doc.title or "(no subject)"
    ago = _time_ago(doc.timestamp)

    parts = [
        f"[apple_mail][{item.category}][{item.status}]",
    ]

    if item.evidence:
        parts.append(f"[evidence: {', '.join(item.evidence)}]")

    parts.append(f'From: {sender} — "{subject}" ({ago}).')

    if item.safe_summary:
        parts.append(item.safe_summary)

    # Generic classifications describe the *kind* of message, not its
    # content, so they get a short, clearly labelled preview. Specific rule
    # matches (e.g. an overdue payment) keep only their vetted summary.
    needs_preview = item.status in GENERIC_STATUSES

    if needs_preview and doc.content:
        preview = _sanitize_email_preview(doc.content, 240)
        if preview:
            parts.append(f"Preview: {preview}")

    if item.event_date:
        parts.append(f"Event date: {item.event_date}.")
    if item.deadline:
        parts.append(f"Decision deadline: {item.deadline}.")
    if item.effective_date:
        parts.append(f"Effective date: {item.effective_date}.")

    return " ".join(parts)


def _format_classified_apple_mail_section(
    docs: List[Document],
    classified_mail: Dict[str, ClassifiedMailItem],
) -> List[str]:
    """Format every Apple Mail message individually; never group by sender."""
    lines: List[str] = []

    for doc in docs:
        item = classified_mail.get(doc.doc_id)
        if item is not None:
            lines.append(_format_classified_apple_mail(item, doc))

    return lines


def _format_apple_mail(doc: Document) -> str:
    """Format one Apple Mail message with its classification."""
    item = classify_mail_document(doc)
    return _format_classified_apple_mail(item, doc)


def _format_notion(doc: Document) -> str:
    """Format a Notion page document."""
    title = doc.title or "Untitled"
    ago = _time_ago(doc.timestamp)
    return f"[notion] {title} (updated {ago})"


def _format_gcalendar(doc: Document) -> str:
    """Format a Google Calendar event document."""
    title = doc.title or "(No title)"
    time_str = _format_time(doc.timestamp)
    # Try to extract duration from content
    duration_match = (
        re.search(r"When:\s*(.+?)$", doc.content, re.MULTILINE) if doc.content else None
    )
    time_range = ""
    if duration_match:
        when = duration_match.group(1).strip()
        # Extract just the times from the ISO strings
        parts = when.split(" – ")
        if len(parts) == 2:
            try:
                start_dt = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
                diff = end_dt - start_dt
                mins = int(diff.total_seconds() / 60)
                if mins >= 60:
                    hrs = mins // 60
                    remaining = mins % 60
                    duration = f"{hrs} hour" + ("s" if hrs > 1 else "")
                    if remaining:
                        duration += f" {remaining} min"
                else:
                    duration = f"{mins} min"
                time_range = f" ({duration})"
            except (ValueError, TypeError):
                pass
    return f"[gcalendar id={doc.doc_id}] {time_str} — {title}{time_range}"


def _format_spotify(doc: Document) -> str:
    """Format a Spotify recently-played track — returns 'Track — Artist'."""
    # doc.title is already "Track — Artist" from the connector
    return doc.title


def _format_apple_music(doc: Document) -> str:
    """Format an Apple Music track — returns 'Track — Artist'."""
    # doc.title is already "Track — Artist" from the connector
    return doc.title


def _format_weather(doc: Document) -> str:
    """Format a weather document."""
    data = _parse_content_json(doc)
    if doc.doc_type == "current":
        temp = data.get("temp_f", "?")
        cond = data.get("conditions", "?")
        humidity = data.get("humidity", "?")
        return f"[weather] Current: {temp}°F, {cond}, humidity {humidity}%"
    if doc.doc_type == "forecast":
        return f"[weather] Forecast: {doc.content[:200]}"
    return f"[weather] {doc.title}"


def _format_github_notifications(doc: Document) -> str:
    """Format a GitHub notification."""
    reason = doc.metadata.get("reason", "")
    repo = doc.metadata.get("repo", "")
    title = doc.title or "(no title)"
    ago = _time_ago(doc.timestamp)
    reason_str = f" ({reason})" if reason else ""
    repo_str = f" in {repo}" if repo else ""
    return f"[github] {title}{repo_str}{reason_str} ({ago})"


def _format_hackernews(doc: Document) -> str:
    """Format a Hacker News story."""
    score = doc.metadata.get("score", "?")
    comments = doc.metadata.get("descendants", "?")
    return f"[hackernews] {doc.title} (score {score}, {comments} comments)"


def _format_news_rss(doc: Document) -> str:
    """Format an RSS news item."""
    feed_name = doc.metadata.get("feed_name", "")
    prefix = f"[{feed_name}]" if feed_name else "[news]"
    description = doc.content[:150].replace("\n", " ").strip() if doc.content else ""
    line = f"{prefix} {doc.title}"
    if description:
        line += f" — {description}"
    return line


# Map connector IDs to their formatting functions
_FORMATTERS: Dict[str, Any] = {
    "oura": _format_oura,
    "apple_health": _format_apple_health,
    "strava": _format_strava,
    "gmail": _format_gmail,
    "gmail_imap": _format_gmail_imap,
    "google_tasks": _format_google_tasks,
    "slack": _format_slack,
    "imessage": _format_imessage,
    "whatsapp": _format_whatsapp,
    "outlook": _format_outlook,
    "apple_mail": _format_apple_mail,
    "notion": _format_notion,
    "gcalendar": _format_gcalendar,
    "weather": _format_weather,
    "github_notifications": _format_github_notifications,
    "hackernews": _format_hackernews,
    "news_rss": _format_news_rss,
    "spotify": _format_spotify,
    "apple_music": _format_apple_music,
    "apple_calendar": _format_apple_calendar,
}


def _format_doc(source: str, doc: Document) -> str:
    """Format a document using the source-specific formatter, with fallback."""
    formatter = _FORMATTERS.get(source)
    if formatter:
        try:
            return formatter(doc)
        except Exception:  # noqa: BLE001
            pass
    # Fallback: connector name + title
    return f"[{source}] {doc.title}"


def _format_music_section(
    collected_docs: Dict[str, List[Document]],
    music_connectors: set,
) -> List[str]:
    """Format music connectors as grouped lists instead of per-track lines."""
    lines: List[str] = []
    for source in sorted(music_connectors):
        docs = collected_docs.get(source, [])
        if not docs:
            continue
        tracks = []
        for doc in docs:
            tracks.append(doc.title)
        label = "Recently played" if source == "spotify" else "Library"
        lines.append(f"[{source}] {label}: {', '.join(tracks)}")
    return lines


def _filter_unanswered_threads(docs: List[Document]) -> List[Document]:
    """Keep only iMessage threads where the last message is NOT from the user.

    Groups by chat title, finds the most-recent message per chat, and returns
    only that message if ``author != "me"``.  Threads the user has already
    replied to are silently dropped.
    """
    from collections import defaultdict

    by_chat: Dict[str, List[Document]] = defaultdict(list)
    for doc in docs:
        by_chat[doc.title or doc.author or ""].append(doc)

    result: List[Document] = []
    for chat_docs in by_chat.values():
        latest = max(chat_docs, key=lambda d: d.timestamp)
        if latest.author != "me":
            result.append(latest)
    return result


def _filter_pending_invites(docs: List[Document]) -> List[Document]:
    """Keep only calendar events the user has not yet responded to."""
    pending: List[Document] = []
    for doc in docs:
        response_status = doc.metadata.get("response_status", "")
        # Include if status is explicitly needsAction, or if no status recorded
        # (connector may not populate it — safer to include than to drop)
        if response_status in ("needsAction", ""):
            pending.append(doc)
    return pending


# ---------------------------------------------------------------------------
# Selection and structured output for the morning briefing
# ---------------------------------------------------------------------------

_CATEGORY_RANK = {
    "ACTION_REQUIRED": 0,
    "TIME_SENSITIVE": 1,
    "REVIEW": 2,
    "COMPLETED": 4,
    "LOW_PRIORITY": 5,
    "EXCLUDE": 9,
}

_SPECIAL_EVENT = re.compile(r"\b(birthday|bday|b-day|anniversary)\b", re.IGNORECASE)


def mail_priority(item: ClassifiedMailItem) -> float:
    """Lower is more important. Personal mail leads the REVIEW group."""
    rank = float(_CATEGORY_RANK.get(item.category, 6))
    if item.category == "REVIEW" and item.status != "personal":
        rank += 0.5
    return rank


def _count_categories(
    docs: List[Document], classified: Dict[str, ClassifiedMailItem]
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for doc in docs:
        category = classified[doc.doc_id].category
        counts[category] = counts.get(category, 0) + 1
    return counts


def is_special_event(doc: Document) -> bool:
    """Birthday or anniversary, by event title or calendar name."""
    calendar = str(doc.metadata.get("calendar", ""))
    return bool(_SPECIAL_EVENT.search(doc.title or "") or _SPECIAL_EVENT.search(calendar))


def _collect_special_events(
    connector_cls: Any, days_ahead: int, errors: List[str]
) -> List[Document]:
    """Birthdays and anniversaries over a longer horizon than the agenda."""
    try:
        connector = connector_cls(days_ahead=days_ahead)
        return [doc for doc in connector.sync() if is_special_event(doc)]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Birthday/anniversary lookup failed: {exc}")
        return []


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _mail_entry(doc: Document, item: ClassifiedMailItem) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    preview = ""
    if item.status in GENERIC_STATUSES and doc.content:
        preview = _sanitize_email_preview(doc.content, 400)
    return {
        "doc_id": doc.doc_id,
        "account": metadata.get("account", ""),
        "sender": doc.author or "",
        "sender_name": metadata.get("sender_name", "") or "",
        "subject": doc.title or "",
        "received": _iso(doc.timestamp),
        "unread": bool(metadata.get("unread", False)),
        "flagged": bool(metadata.get("flagged", False)),
        "body_available": bool(metadata.get("body_available", bool(doc.content))),
        "category": item.category,
        "status": item.status,
        "evidence": list(item.evidence),
        "note": item.safe_summary,
        "preview": preview,
        "deadline": item.deadline,
        "event_date": item.event_date,
        "effective_date": item.effective_date,
    }


def _event_entry(doc: Document) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "doc_id": doc.doc_id,
        "title": doc.title or "(No title)",
        "calendar": metadata.get("calendar", ""),
        "start": metadata.get("start", "") or _iso(doc.timestamp),
        "end": metadata.get("end", ""),
        "all_day": bool(metadata.get("all_day", False)),
        "location": metadata.get("location", ""),
    }


def _summarize_selected(
    docs: List[Document],
    classified: Dict[str, ClassifiedMailItem],
    *,
    max_summaries: int,
    model: str,
    time_budget: float,
) -> Dict[str, str]:
    """Grounded local summaries for the top generic emails that have a body.

    Specific rule matches keep their vetted wording and are not summarised.
    Stops at *max_summaries* or when *time_budget* seconds are used up.
    """
    import time

    deadline = time.monotonic() + max(0.0, time_budget)
    summaries: Dict[str, str] = {}
    attempts = 0
    for doc in docs:
        if attempts >= max_summaries or time.monotonic() >= deadline:
            break
        item = classified[doc.doc_id]
        if item.status not in GENERIC_STATUSES or item.status == "bulk":
            continue
        if not (doc.content or "").strip():
            continue
        attempts += 1
        remaining = max(5.0, deadline - time.monotonic())
        summary = summarize_mail_with_local_llm(
            doc, model=model, timeout=min(60.0, remaining)
        )
        if summary:
            summaries[doc.doc_id] = summary
    return summaries


def _briefing_payload(
    collected_docs: Dict[str, List[Document]],
    classified_mail: Dict[str, ClassifiedMailItem],
    mail_stats: Dict[str, Any],
    special_events: List[Document],
    summaries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Structured data for the deterministic briefing builder."""
    mail_docs = collected_docs.get("apple_mail", [])
    calendar_docs = collected_docs.get("apple_calendar", [])
    summaries = summaries or {}
    entries = []
    for doc in mail_docs:
        if doc.doc_id not in classified_mail:
            continue
        entry = _mail_entry(doc, classified_mail[doc.doc_id])
        if doc.doc_id in summaries:
            entry["summary"] = summaries[doc.doc_id]
        entries.append(entry)
    return {
        "mail": entries,
        "mail_stats": mail_stats,
        "calendar": [
            _event_entry(doc) for doc in calendar_docs if not is_special_event(doc)
        ],
        "special_events": [_event_entry(doc) for doc in special_events],
    }


@ToolRegistry.register("digest_collect")
class DigestCollectTool(BaseTool):
    """Collect recent data from multiple connectors for digest synthesis."""

    tool_id = "digest_collect"
    is_local = True

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="digest_collect",
            description=(
                "Fetch recent data from configured connectors (email, calendar, "
                "health, tasks, etc.) and return a structured, human-readable "
                "summary grouped by section (Health, Messages, Calendar, Music) "
                "for digest synthesis."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of connector IDs to fetch from "
                            "(e.g., ['gmail', 'oura', 'gcalendar'])."
                        ),
                    },
                    "hours_back": {
                        "type": "number",
                        "description": "How many hours back to look (default: 24).",
                    },
                    "unacted_only": {
                        "type": "boolean",
                        "description": (
                            "When true, only return items the user has not yet acted on: "
                            "unread emails, unanswered iMessage threads, pending calendar invites."
                        ),
                    },
                    "seen_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "doc_ids to exclude (already queued or acted on).",
                    },
                    "mail_limit": {
                        "type": "integer",
                        "description": (
                            "Maximum emails selected across all mail accounts "
                            "(default 15). Marketing is never selected."
                        ),
                    },
                    "max_summaries": {
                        "type": "integer",
                        "description": (
                            "Local-model summaries for up to this many selected "
                            "emails (default 0). Ungrounded summaries are dropped."
                        ),
                    },
                    "summary_model": {
                        "type": "string",
                        "description": "Local model used for summaries.",
                    },
                    "summary_time_budget": {
                        "type": "number",
                        "description": "Seconds allowed for all summaries (default 150).",
                    },
                    "special_days_ahead": {
                        "type": "integer",
                        "description": (
                            "Days ahead to look for calendar birthdays and "
                            "anniversaries (default 14)."
                        ),
                    },
                },
                "required": ["sources"],
            },
            category="data",
            timeout_seconds=120.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        # Ensure connectors are registered
        import openjarvis.connectors  # noqa: F401

        sources: List[str] = params.get("sources", [])
        hours_back: float = params.get("hours_back", 24)
        unacted_only: bool = bool(params.get("unacted_only", False))
        seen_ids: set = set(params.get("seen_ids", []))
        mail_limit: int = max(1, int(params.get("mail_limit", 15) or 15))
        special_days_ahead: int = max(
            0, int(params.get("special_days_ahead", 14) or 0)
        )
        since = datetime.now() - timedelta(hours=hours_back)
        mail_stats: Dict[str, Any] = {}
        special_events: List[Document] = []
        summaries: Dict[str, str] = {}
        max_summaries = max(0, int(params.get("max_summaries", 0) or 0))
        summary_model = str(params.get("summary_model") or "")
        summary_budget = float(params.get("summary_time_budget", 150) or 0)

        # Collect raw documents per source
        collected_docs: Dict[str, List[Document]] = {}
        classified_mail: Dict[str, ClassifiedMailItem] = {}
        errors: List[str] = []
        failed_sources: List[str] = []

        for source in sources:
            if not ConnectorRegistry.contains(source):
                failed_sources.append(source)
                errors.append(f"Connector '{source}' not available")
                continue

            try:
                connector_cls = ConnectorRegistry.get(source)
                connector = connector_cls()

                if not connector.is_connected():
                    failed_sources.append(source)
                    errors.append(
                        f"Connector '{source}' not connected (no credentials)"
                    )
                    continue

                docs: List[Document] = []

                sync_kwargs: Dict[str, Any] = (
                    {} if source == "apple_calendar" else {"since": since}
                )


                for d in connector.sync(**sync_kwargs):

                    if d.doc_id in seen_ids:
                        continue

                    if source == "apple_mail":
                        try:
                            # Deterministic classification only during collection.
                            classified = classify_mail_document(d)
                        except Exception as exc:
                            errors.append(
                                "An Apple Mail message could not be classified; "
                                "it was retained as unclassified."
                            )
                            classified = ClassifiedMailItem(
                                doc_id=d.doc_id,
                                category="REVIEW",
                                status="unclassified",
                                reason=f"Classification failed: {type(exc).__name__}",
                                safe_summary=(
                                    "Classification was unavailable; "
                                    "message retained for review."
                                ),
                                timestamp=d.timestamp,
                                evidence=(
                                    "conservative_fallback",
                                    "classification_error",
                                ),
                            )

                        classified_mail[d.doc_id] = classified

                    docs.append(d)

                if unacted_only and source == "imessage":
                    docs = _filter_unanswered_threads(docs)

                if unacted_only and source == "gcalendar":
                    docs = _filter_pending_invites(docs)

                if source == "apple_mail":
                    before_dedup = len(docs)
                    retained_items = deduplicate_shipments(
                         [classified_mail[d.doc_id] for d in docs]
                         )
                    retained_ids = {item.doc_id for item in retained_items}
                    docs = [d for d in docs if d.doc_id in retained_ids]
                    duplicate_shipments = before_dedup - len(docs)

                    # Rank every message in the window, then select across
                    # all accounts. Marketing (EXCLUDE) is never selected.
                    in_window = len(docs)
                    excluded = [
                        d for d in docs
                        if classified_mail[d.doc_id].category == "EXCLUDE"
                    ]
                    docs = [
                        d for d in docs
                        if classified_mail[d.doc_id].category != "EXCLUDE"
                    ]
                    docs.sort(
                        key=lambda d: (
                            mail_priority(classified_mail[d.doc_id]),
                            -(d.timestamp.timestamp() if d.timestamp else 0),
                        )
                    )
                    omitted = docs[mail_limit:]
                    docs = docs[:mail_limit]
                    mail_stats = {
                        "in_window": in_window,
                        "selected": len(docs),
                        "excluded_marketing": len(excluded),
                        "omitted_by_category": _count_categories(
                            omitted, classified_mail
                        ),
                        "accounts": dict(getattr(connector, "account_stats", {}) or {}),
                        "duplicate_shipments_removed": duplicate_shipments,
                    }

                    # Refine at most one retained REVIEW/unclassified email.
                    # All other emails keep their deterministic classifications.
                    llm_refinements = 0
                    # One local inference attempt per digest keeps collection
                    # within the outer tool timeout.
                    max_llm_refinements = 1

                    for d in docs:
                        item = classified_mail[d.doc_id]

                        if (
                            item.category == "REVIEW"
                            and item.status == "unclassified"
                            and "classification_error" not in item.evidence
                            and llm_refinements < max_llm_refinements
                        ):
                            refined_item = classify_mail_document_with_local_llm(
                                d,
                                deterministic_classifier=classify_mail_document,
                            )
                            classified_mail[d.doc_id] = refined_item
                            llm_refinements += 1
                            if refined_item.status == "unclassified-timeout":
                                break

                    if max_summaries and summary_model:
                        summaries = _summarize_selected(
                            docs,
                            classified_mail,
                            max_summaries=max_summaries,
                            model=summary_model,
                            time_budget=summary_budget,
                        )

                if source == "apple_calendar" and special_days_ahead:
                    special_events = _collect_special_events(
                        connector_cls, special_days_ahead, errors
                    )

                collected_docs[source] = docs
            except Exception as exc:
                failed_sources.append(source)
                errors.append(f"Error fetching from '{source}': {exc}")

        # Group by section and build human-readable output
        summary_parts: List[str] = []
        for section_name, section_connectors in _SECTION_ORDER:
            # Gather all sources that belong to this section and have data
            section_sources = [
                s for s in sources if s in section_connectors and s in collected_docs
            ]
            if not section_sources:
                continue

            section_lines: List[str] = []

            if section_name == "MUSIC":
                # Music gets special grouped formatting
                section_lines = _format_music_section(
                    collected_docs, section_connectors
                )
            else:
                for source in section_sources:
                    if source == "apple_mail":
                        section_lines.extend(
                            _format_classified_apple_mail_section(
                                collected_docs[source], classified_mail
                            )
                        )
                    else:
                        for doc in collected_docs[source]:
                            section_lines.append(_format_doc(source, doc))
            if section_lines:
                summary_parts.append(f"=== {section_name} ===")
                summary_parts.extend(section_lines)
                summary_parts.append("")  # blank line between sections

        # Handle any connectors not in a known section (fallback)
        known_connectors = set()
        for _, cids in _SECTION_ORDER:
            known_connectors |= cids

        uncategorized_sources = [
            s for s in sources if s not in known_connectors and s in collected_docs
        ]
        if uncategorized_sources:
            summary_parts.append("=== OTHER ===")
            for source in uncategorized_sources:
                for doc in collected_docs[source]:
                    summary_parts.append(_format_doc(source, doc))
            summary_parts.append("")

        # Errors at the end, not inline
        if errors:
            summary_parts.append("=== ERRORS ===")
            summary_parts.extend(errors)

        return ToolResult(
            tool_name="digest_collect",
            content="\n".join(summary_parts),
            success=True,
            metadata={
                "sources_queried": sources,
                "sources_ok": list(collected_docs.keys()),
                "sources_failed": failed_sources,
                "errors": errors,
                "total_items": sum(len(v) for v in collected_docs.values()),
                "briefing": _briefing_payload(
                    collected_docs,
                    classified_mail,
                    mail_stats,
                    special_events,
                    summaries,
                ),
            },
        )
