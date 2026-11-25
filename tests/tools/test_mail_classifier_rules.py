"""Classifier rules added for the morning briefing (v1.1) and grounding."""

from __future__ import annotations

from datetime import datetime

import pytest

from openjarvis.connectors._stubs import Document
from openjarvis.tools.mail_classifier import (
    GENERIC_STATUSES,
    classify_mail_document,
    summary_is_grounded,
)

NOW = datetime(2026, 11, 25, 9, 10).astimezone()


def _doc(subject="", body="", sender="person@example.test", **metadata) -> Document:
    return Document(
        doc_id="d1",
        source="apple_mail",
        doc_type="email",
        title=subject,
        author=sender,
        content=body,
        timestamp=NOW,
        metadata=metadata,
    )


def _classify(**kw):
    return classify_mail_document(_doc(**kw), NOW)


def test_verification_code_is_low_priority() -> None:
    item = _classify(subject="Your sign-in code", body="Your code is 482913.")
    assert (item.category, item.status) == ("LOW_PRIORITY", "verification_code")


def test_security_alert_still_wins_over_code_wording() -> None:
    item = _classify(
        subject="New sign-in",
        body=(
            "New sign-in from a new device. If this was not you, secure your "
            "account. Your verification code is 123456."
        ),
    )
    assert item.status == "security_alert"


def test_calendar_invitation_by_subject_and_by_attachment() -> None:
    assert _classify(subject="Invitation: Team sync @ Fri").status == "invitation"
    item = _classify(subject="Plans", mail_headers={"has_calendar": True})
    assert item.status == "invitation"


def test_rsvp_reply_is_low_priority() -> None:
    item = _classify(subject="Accepted: Team sync @ Fri")
    assert (item.category, item.status) == ("LOW_PRIORITY", "rsvp_reply")


def test_receipt_is_completed_but_cancellation_is_review() -> None:
    assert _classify(subject="Your order #A123 receipt").status == "receipt"
    item = _classify(subject="Your order has been cancelled")
    assert (item.category, item.status) == ("REVIEW", "order_issue")


def test_statement_reports_only_an_explicit_due_date() -> None:
    item = _classify(
        subject="Your statement is ready",
        body="Your statement is ready. Payment due date: December 3, 2026.",
    )
    assert item.status == "statement"
    assert item.deadline == "2026-12-03"
    assert "2026-12-03" in item.safe_summary

    item = _classify(subject="Your statement is ready", body="Log in to view it.")
    assert item.status == "statement" and item.deadline is None
    assert item.category == "REVIEW"  # a bill is not evidence of overdue payment


def test_password_change_is_review() -> None:
    item = _classify(subject="Security", body="Your password was changed today.")
    assert item.status == "account_change"


def test_bulk_headers_mark_low_priority() -> None:
    item = _classify(
        subject="This week at the library",
        mail_headers={"list_unsubscribe": True},
    )
    assert (item.category, item.status) == ("LOW_PRIORITY", "bulk")


def test_bulk_header_does_not_hide_a_specific_rule() -> None:
    item = _classify(
        subject="Shipment update",
        body="Your package is out for delivery.",
        mail_headers={"list_unsubscribe": True},
    )
    assert item.status == "out_for_delivery"


@pytest.mark.parametrize(
    "sender",
    ["no-reply@bank.test", "alerts@bank.test", "notifications@app.test"],
)
def test_role_addresses_are_automated(sender: str) -> None:
    assert _classify(subject="Update", sender=sender).status == "automated"


def test_auto_submitted_header_is_automated() -> None:
    item = _classify(
        subject="Update", mail_headers={"auto_submitted": "auto-generated"}
    )
    assert item.status == "automated"


def test_individual_sender_is_personal() -> None:
    item = _classify(
        subject="Dinner Saturday?",
        sender="priya@example.test",
        sender_name="Priya Raman",
        mail_headers={"from_name": "Priya Raman"},
    )
    assert (item.category, item.status) == ("REVIEW", "personal")


def test_unknown_without_signals_stays_unclassified() -> None:
    item = _classify(subject="Hello", sender="person@example.test")
    assert item.status == "unclassified"


def test_generic_statuses_cover_new_kind_only_rules() -> None:
    for status in ("personal", "automated", "bulk", "invitation", "receipt"):
        assert status in GENERIC_STATUSES
    for status in ("payment_overdue", "security_alert", "statement"):
        assert status not in GENERIC_STATUSES


# ---------------------------------------------------------------------------
# Grounding check for local-model summaries
# ---------------------------------------------------------------------------

SOURCE = (
    "Hi Avi, the plumber from Rapid Rooter will come on Thursday at 2:30 PM. "
    "The quote is $450, payable to Rapid Rooter."
)


@pytest.mark.parametrize(
    "summary",
    [
        "The plumber from Rapid Rooter comes Thursday at 2:30 PM; the quote is $450.",
        "A plumber visit is scheduled for Thursday.",
    ],
)
def test_grounded_summaries_pass(summary: str) -> None:
    assert summary_is_grounded(summary, SOURCE)


@pytest.mark.parametrize(
    "summary",
    [
        "The plumber comes Thursday at 3:30 PM.",  # wrong time
        "The quote is $540.",  # wrong amount
        "Roto-Rooter will come on Thursday.",  # invented company
        "Avi must pay Mike by Friday.",  # invented person
        " ".join(["word"] * 41),  # too long
        "",
    ],
)
def test_ungrounded_summaries_fail(summary: str) -> None:
    assert not summary_is_grounded(summary, SOURCE)
