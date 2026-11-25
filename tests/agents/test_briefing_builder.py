"""Deterministic morning briefing builder."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from openjarvis.agents.briefing_builder import (
    build_briefing,
    possible_duplicates,
    sender_label,
    words,
)

NOW = datetime.fromisoformat("2026-11-25T09:10:00-05:00")


def _mail(i, category="REVIEW", status="personal", **extra):
    entry = {
        "doc_id": f"m{i}",
        "account": "Gmail",
        "sender": f"person{i}@example.test",
        "sender_name": f"Person {i}",
        "subject": f"Subject {i}",
        "received": (NOW - timedelta(hours=i)).isoformat(),
        "body_available": True,
        "category": category,
        "status": status,
        "evidence": ["individual_sender"],
        "note": "A message from an individual sender.",
        "preview": f"Body text of message {i} with a few more words in it.",
    }
    entry.update(extra)
    return entry


def _payload(mail=(), calendar=(), special=(), **stats):
    base = {
        "in_window": len(mail),
        "selected": len(mail),
        "excluded_marketing": 0,
        "omitted_by_category": {},
        "accounts": {},
    }
    base.update(stats)
    return {
        "mail": list(mail),
        "mail_stats": base,
        "calendar": list(calendar),
        "special_events": list(special),
    }


def test_each_email_is_spoken_individually_with_sender_subject_time() -> None:
    briefing = build_briefing(_payload([_mail(1), _mail(2)]), now=NOW, honorific="sir")
    text = briefing.spoken
    assert text.startswith("Good morning, sir. It's Wednesday, November 25.")
    assert "From Person 1, at 8:10 AM: Subject 1." in text
    assert "From Person 2, at 7:10 AM: Subject 2." in text
    assert "Here are the 2 that matter most." in text
    assert briefing.emails_spoken == 2


def test_preview_and_summary_are_labelled() -> None:
    summary_mail = _mail(1, summary="Dinner is moved to Saturday.")
    briefing = build_briefing(_payload([summary_mail, _mail(2)]), now=NOW)
    assert "Summary: Dinner is moved to Saturday." in briefing.spoken
    assert "Preview: Body text of message 2" in briefing.spoken
    assert "Summary (local model, grounding-checked)" in briefing.report_markdown
    assert "Preview (first words, not a summary)" in briefing.report_markdown


def test_specific_classification_uses_note_not_preview() -> None:
    overdue = _mail(
        1,
        category="ACTION_REQUIRED",
        status="payment_overdue",
        note="A payment is overdue and the message requests payment.",
        preview="",
    )
    briefing = build_briefing(_payload([overdue]), now=NOW)
    assert "Action required." in briefing.spoken
    assert "A payment is overdue and the message requests payment." in briefing.spoken
    assert "Preview" not in briefing.spoken


def test_categories_are_spoken_in_priority_order() -> None:
    mail = [
        _mail(1, category="LOW_PRIORITY", status="automated"),
        _mail(2, category="ACTION_REQUIRED", status="payment_overdue", preview=""),
        _mail(3, category="REVIEW", status="statement", preview=""),
        _mail(4, category="REVIEW", status="personal"),
    ]
    text = build_briefing(_payload(mail), now=NOW).spoken
    order = [text.index(f"Subject {i}") for i in (2, 4, 3, 1)]
    assert order == sorted(order)


def test_missing_body_is_said_out_loud() -> None:
    entry = _mail(1, body_available=False, preview="")
    entry["status"] = "unclassified"
    entry["note"] = "An email needs review."
    text = build_briefing(_payload([entry]), now=NOW).spoken
    assert "An email needs review." in text

    entry = _mail(2, body_available=False)
    text = build_briefing(_payload([entry]), now=NOW).spoken
    assert "The message body is not available on this Mac yet." in text


def test_iso_dates_in_notes_are_spoken_naturally() -> None:
    entry = _mail(
        1,
        status="statement",
        preview="",
        note=(
            "A statement or bill is available. "
            "The stated payment due date is 2026-12-03."
        ),
    )
    text = build_briefing(_payload([entry]), now=NOW).spoken
    assert "due date is December 3." in text


def test_counts_of_skipped_mail_are_reported() -> None:
    briefing = build_briefing(
        _payload(
            [_mail(1)],
            in_window=40,
            excluded_marketing=22,
            omitted_by_category={"LOW_PRIORITY": 17},
        ),
        now=NOW,
    )
    assert "40 emails arrived" in briefing.spoken
    assert "17 lower-ranked emails not selected" in briefing.spoken
    assert "22 promotional emails skipped" in briefing.spoken


def test_failed_account_is_spoken_and_flagged() -> None:
    accounts = {
        "Gmail": {"fetched": 5, "with_body": 5, "error": ""},
        "Yahoo Pavi": {"fetched": 0, "with_body": 0, "error": "Query failed"},
        "iCloud": {"fetched": 10, "with_body": 2, "error": ""},
    }
    briefing = build_briefing(_payload([_mail(1)], accounts=accounts), now=NOW)
    assert "I could not read mail from Yahoo Pavi." in briefing.spoken
    assert any("Yahoo Pavi" in issue for issue in briefing.issues)
    assert any("only 2 of 10" in issue for issue in briefing.issues)
    assert "| Yahoo Pavi | 0 | 0 | Query failed |" in briefing.report_markdown


def test_no_mail_and_no_events() -> None:
    text = build_briefing(_payload(), now=NOW).spoken
    assert "No new email arrived in the last 24 hours." in text
    assert "Nothing is on your calendar for today or the next 2 days." in text
    assert text.endswith("That's your briefing.")


def test_urls_and_codes_never_spoken() -> None:
    entry = _mail(
        1,
        subject="Order 99812345 confirmed https://shop.test/x",
        preview="Track at https://track.test/abc code 4471182",
    )
    briefing = build_briefing(_payload([entry]), now=NOW)
    assert "http" not in briefing.spoken
    assert not re.search(r"\d{5,}", briefing.spoken)
    assert briefing.issues == []


def test_budget_shortens_then_moves_lowest_ranked_to_report() -> None:
    long_preview = " ".join(["word"] * 60)
    mail = [_mail(i, preview=long_preview) for i in range(1, 16)]
    briefing = build_briefing(_payload(mail), now=NOW, max_minutes=1.0)
    budget = 126
    assert briefing.word_count <= budget
    assert briefing.emails_spoken + briefing.emails_report_only == 15
    assert briefing.emails_report_only > 0
    assert (
        f"{briefing.emails_report_only} more in the written report" in briefing.spoken
    )
    # Highest-ranked (newest) mail is kept; dropped ones are listed in the report.
    assert "Subject 1" in briefing.spoken
    assert "Subject 15" not in briefing.spoken
    assert "Subject 15" in briefing.report_markdown
    assert "(report only)" in briefing.report_markdown


def test_five_minute_budget_holds_for_fifteen_long_emails() -> None:
    long_preview = " ".join(["word"] * 200)
    mail = [_mail(i, preview=long_preview) for i in range(1, 16)]
    briefing = build_briefing(_payload(mail), now=NOW)
    assert briefing.word_count <= 630
    assert briefing.estimated_seconds <= 300


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _event(title, start, *, all_day=False, end="", calendar="Home", location=""):
    return {
        "doc_id": f"e-{title}-{start}",
        "title": title,
        "calendar": calendar,
        "start": start,
        "end": end,
        "all_day": all_day,
        "location": location,
    }


def test_calendar_grouped_by_day_in_order() -> None:
    events = [
        _event("Dentist", "2026-11-26T14:00:00-05:00", location="Main St Dental"),
        _event("Standup", "2026-11-25T09:30:00-05:00"),
        _event("Holiday", "2026-11-26", all_day=True, end="2026-11-27"),
    ]
    text = build_briefing(_payload(calendar=events), now=NOW).spoken
    assert "Your calendar. Today. 9:30 AM: Standup." in text
    assert "Tomorrow. All day: Holiday. 2:00 PM: Dentist, at Main St Dental." in text


def test_ongoing_multi_day_event() -> None:
    events = [_event("Conference", "2026-11-24", all_day=True, end="2026-11-28")]
    text = build_briefing(_payload(calendar=events), now=NOW).spoken
    assert "Ongoing through Friday: Conference." in text


def test_possible_duplicates_are_reported_not_merged() -> None:
    events = [
        _event("Alamo drop-off", "2026-11-27T09:00:00-05:00"),
        _event("Alamo drop off", "2026-11-27T11:00:00-05:00"),
    ]
    assert len(possible_duplicates(events)) == 1
    briefing = build_briefing(_payload(calendar=events), now=NOW)
    assert briefing.spoken.count("Alamo drop") == 3  # both events + the note
    assert "They may be separate events, so both are kept." in briefing.spoken
    assert "Possible duplicates (not merged)" in briefing.report_markdown


def test_birthdays_and_anniversaries() -> None:
    special = [
        _event("Mom's Birthday", "2026-11-26", all_day=True, calendar="Birthdays"),
        _event("Wedding anniversary", "2026-12-03", all_day=True),
        _event("Old birthday", "2026-11-20", all_day=True),
    ]
    text = build_briefing(_payload(special=special), now=NOW).spoken
    assert (
        "Coming up: Mom's Birthday tomorrow; "
        "Wedding anniversary on Thursday, December 3." in text
    )
    assert "Old birthday" not in text


def test_failed_sections() -> None:
    text = build_briefing(
        _payload(), now=NOW, failed_sections={"messages", "calendar"}
    ).spoken
    assert "I could not retrieve your email this morning." in text
    assert "I could not retrieve your calendar this morning." in text


# ---------------------------------------------------------------------------
# No invented content
# ---------------------------------------------------------------------------


def test_every_spoken_sentence_comes_from_data_or_a_template() -> None:
    mail = [_mail(i) for i in range(1, 4)]
    events = [_event("Standup", "2026-11-25T09:30:00-05:00")]
    briefing = build_briefing(_payload(mail, calendar=events), now=NOW)
    allowed = {
        "Good morning, sir.",
        "It's Wednesday, November 25.",
        "In the last 24 hours, 3 emails arrived across your accounts.",
        "Here are the 3 that matter most.",
        "For review.",
        "Your calendar.",
        "Today.",
        "9:30 AM: Standup.",
        "That's your briefing.",
    }
    for i in range(1, 4):
        hour = 9 - i
        allowed.add(f"From Person {i}, at {hour}:10 AM: Subject {i}.")
        allowed.add(f"Preview: Body text of message {i} with a few more words in it.")
    sentences = re.split(r"(?<=[.!?…])\s+", briefing.spoken)
    assert [s for s in sentences if s not in allowed] == []


def test_sender_label() -> None:
    assert sender_label({"sender_name": "Priya", "sender": "p@x.test"}) == "Priya"
    assert sender_label({"sender": "alerts@chase.com"}) == "chase.com"
    assert sender_label({}) == "an unknown sender"
    assert words("a b  c") == 3


def test_lower_priority_mail_is_named_compactly_by_sender() -> None:
    low = [
        _mail(
            i,
            category="LOW_PRIORITY",
            status="automated",
            sender="no-reply@app.test",
            sender_name="",
            subject=f"Alert {i}",
        )
        for i in range(1, 9)
    ]
    briefing = build_briefing(_payload([_mail(20)] + low), now=NOW)
    text = briefing.spoken
    assert (
        "Lower priority, 8 emails, from app.test: "
        "Alert 1, Alert 2, Alert 3, Alert 4 and Alert 5, and 3 more." in text
    )
    assert text.count("Preview:") == 1  # only the personal message
    for i in range(1, 9):  # every one is still listed in the report
        assert f"**Alert {i}**" in briefing.report_markdown
