"""Tests for the digest_collect tool."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from unittest.mock import MagicMock, patch

from openjarvis.connectors._stubs import Document

from openjarvis.core.registry import ConnectorRegistry, ToolRegistry

def test_digest_collect_registered():
    from openjarvis.tools.digest_collect import DigestCollectTool

    if not ToolRegistry.contains("digest_collect"):
        ToolRegistry.register_value("digest_collect", DigestCollectTool)

    assert ToolRegistry.contains("digest_collect")
    assert ToolRegistry.get("digest_collect") is DigestCollectTool

def test_digest_collect_executes():
    from openjarvis.tools.digest_collect import DigestCollectTool

    tool = DigestCollectTool()

    mock_docs = [
        Document(
            doc_id="test-1",
            source="gmail",
            doc_type="email",
            content="Meeting at 3pm",
            title="Team standup",
            author="alice@example.com",
            timestamp=datetime(2026, 4, 1, 10, 0),
        )
    ]

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True
    mock_connector.return_value.sync.return_value = mock_docs

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(ConnectorRegistry, "get", return_value=mock_connector):
            result = tool.execute(sources=["gmail"], hours_back=24)

    assert result.success is True
    assert "=== MESSAGES ===" in result.content
    assert "[gmail id=test-1] From: alice@example.com" in result.content
    assert "Team standup" in result.content
    assert result.metadata["total_items"] == 1
    assert result.metadata["sources_ok"] == ["gmail"]
    assert result.metadata["sources_failed"] == []
    assert result.metadata["errors"] == []

def test_apple_mail_classifies_before_per_source_limit():
    from openjarvis.tools.digest_collect import DigestCollectTool

    mock_docs = [
        Document(
            doc_id=f"promo-{i}",
            source="apple_mail",
            doc_type="email",
            title="Weekly newsletter",
            author="Marketing",
            content="Sale and discount. Unsubscribe.",
            timestamp=datetime(2026, 11, 9, 9, 0),
        )
        for i in range(15)
    ]

    mock_docs.append(
        Document(
            doc_id="important-escrow",
            source="apple_mail",
            doc_type="email",
            title="Escrow shortage notice",
            author="Bank",
            content=(
                "The escrow shortage can be paid before November 30, 2026. "
                "Take No Action and the shortage will be divided evenly "
                "over the next 12 monthly mortgage payments. "
                "New payment starts December 1, 2026."
            ),
            timestamp=datetime(2026, 11, 9, 9, 0),
        )
    )

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True
    mock_connector.return_value.sync.return_value = mock_docs

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(
            ConnectorRegistry, "get", return_value=mock_connector
        ):
            result = DigestCollectTool().execute(
                sources=["apple_mail"],
                hours_back=24,
            )

    assert result.success is True
    assert "Escrow shortage notice"  in result.content
    assert "optional repayment choice" in result.content
    # Marketing is never selected; it is counted, not read out.
    assert result.metadata["total_items"] == 1
    stats = result.metadata["briefing"]["mail_stats"]
    assert stats["in_window"] == 16
    assert stats["excluded_marketing"] == 15

def test_apple_mail_briefing_includes_unclassified_email_context():
    from openjarvis.tools.digest_collect import DigestCollectTool

    mock_docs = [
        Document(
            doc_id="private-mail-1",
            source="apple_mail",
            doc_type="email",
            title="PRIVATE_SUBJECT_TOKEN",
            author="PRIVATE_SENDER_TOKEN",
            content="PRIVATE_BODY_TOKEN",
            timestamp=datetime(2026, 11, 9, 9, 0),
        )
    ]

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True
    mock_connector.return_value.sync.return_value = mock_docs

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(
            ConnectorRegistry, "get", return_value=mock_connector
        ):
            result = DigestCollectTool().execute(
                sources=["apple_mail"],
                hours_back=24,
            )

    assert result.success is True
    assert "PRIVATE_SUBJECT_TOKEN" in result.content
    assert "PRIVATE_SENDER_TOKEN" in result.content
    assert "Preview: PRIVATE_BODY_TOKEN" in result.content
    assert "An email needs review" in result.content

def test_digest_collect_missing_connector():
    from openjarvis.tools.digest_collect import DigestCollectTool

    tool = DigestCollectTool()

    with patch.object(ConnectorRegistry, "contains", return_value=False):
        result = tool.execute(sources=["nonexistent"])

    assert result.success is True  # Partial success
    assert "not available" in result.content
    assert result.metadata["sources_ok"] == []
    assert result.metadata["sources_failed"] == ["nonexistent"]
    assert len(result.metadata["errors"]) == 1
    assert "not available" in result.metadata["errors"][0]

def test_apple_mail_collector_keeps_newest_shipment_status():
    from openjarvis.tools.digest_collect import DigestCollectTool

    older = Document(
        doc_id="shipment-older",
        source="apple_mail",
        doc_type="email",
        title="Costco shipment in transit",
        author="Costco",
        content=(
            "Order 123456. "
            "Tracking number: 1Z999AA10123456784. "
            "Your shipment is in transit."
        ),
        timestamp=datetime(2026, 11, 8, 10, 0),
    )

    newer = Document(
        doc_id="shipment-newer",
        source="apple_mail",
        doc_type="email",
        title="Costco shipment out for delivery",
        author="Costco",
        content=(
            "Order 123456. "
            "Tracking number: 1Z999AA10123456784. "
            "Your shipment is out for delivery."
        ),
        timestamp=datetime(2026, 11, 9, 10, 0),
    )

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True

    # The connector contract is newest-first.
    mock_connector.return_value.sync.return_value = [newer, older]

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(
            ConnectorRegistry, "get", return_value=mock_connector
        ):
            result = DigestCollectTool().execute(
                sources=["apple_mail"],
                hours_back=48,
            )

    assert result.success is True
    assert "out_for_delivery" in result.content
    assert "in_transit" not in result.content
    assert result.metadata["total_items"] == 1

def test_apple_mail_classification_failure_does_not_fail_connector():
    from openjarvis.tools.digest_collect import DigestCollectTool

    mock_docs = [
        Document(
            doc_id="classification-error-1",
            source="apple_mail",
            doc_type="email",
            title="Test message",
            author="sender@example.com",
            content="Test email content",
            timestamp=datetime(2026, 11, 9, 9, 0),
        )
    ]

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True
    mock_connector.return_value.sync.return_value = mock_docs

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(
            ConnectorRegistry, "get", return_value=mock_connector
        ):
            with patch(
                "openjarvis.tools.digest_collect.classify_mail_document",
                side_effect=RuntimeError("Classification failed"),
            ):
                result = DigestCollectTool().execute(
                    sources=["apple_mail"],
                    hours_back=24,
                )

    assert result.success is True
    assert result.metadata["sources_ok"] == ["apple_mail"]
    assert result.metadata["sources_failed"] == []
    assert any(
        "could not be classified" in error
        for error in result.metadata["errors"]
    )
    assert "classification-error-1" not in result.content or "Test message" in result.content
    assert result.metadata["total_items"] == 1

def test_apple_mail_overdue_payment_survives_digest_collection():
    from openjarvis.tools.digest_collect import DigestCollectTool

    mock_docs = [
        Document(
            doc_id="synthetic-overdue-1",
            source="apple_mail",
            doc_type="email",
            title="SYNTHETIC_OVERDUE_SUBJECT_TOKEN",
            author="SYNTHETIC_SENDER_TOKEN",
            content=(
                "SYNTHETIC_OVERDUE_BODY_TOKEN. "
                "Your payment is overdue. Please make the outstanding "
                "payment to bring your account current."
            ),
            timestamp=datetime(2026, 11, 12, 9, 0),
        )
    ]

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True
    mock_connector.return_value.sync.return_value = mock_docs

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(
            ConnectorRegistry, "get", return_value=mock_connector
        ):
            result = DigestCollectTool().execute(
                sources=["apple_mail"],
                hours_back=24,
            )

    assert result.success is True
    assert "[apple_mail][ACTION_REQUIRED][payment_overdue]" in result.content
    assert "A payment is overdue" in result.content

    assert "SYNTHETIC_OVERDUE_SUBJECT_TOKEN" in result.content
    assert "SYNTHETIC_SENDER_TOKEN" in result.content
    assert "SYNTHETIC_OVERDUE_BODY_TOKEN" not in result.content

    assert result.metadata["total_items"] == 1

def test_apple_mail_groups_unclassified_messages_by_sender_domain():
    from openjarvis.tools.digest_collect import DigestCollectTool

    mock_docs = [
        Document(
            doc_id="icici-1",
            source="apple_mail",
            doc_type="email",
            title="PRIVATE_ICICI_SUBJECT_1",
            author="alerts@icici.bank.in",
            content="PRIVATE_ICICI_BODY_1",
            timestamp=datetime(2026, 11, 10, 9, 0),
        ),
        Document(
            doc_id="icici-2",
            source="apple_mail",
            doc_type="email",
            title="PRIVATE_ICICI_SUBJECT_2",
            author="notices@icici.bank.in",
            content="PRIVATE_ICICI_BODY_2",
            timestamp=datetime(2026, 11, 10, 9, 5),
        ),
        Document(
            doc_id="venmo-1",
            source="apple_mail",
            doc_type="email",
            title="PRIVATE_VENMO_SUBJECT",
            author="alerts@venmo.com",
            content="PRIVATE_VENMO_BODY",
            timestamp=datetime(2026, 11, 10, 9, 10),
        ),
    ]

    mock_connector = MagicMock()
    mock_connector.return_value.is_connected.return_value = True
    mock_connector.return_value.sync.return_value = mock_docs

    with patch.object(ConnectorRegistry, "contains", return_value=True):
        with patch.object(
            ConnectorRegistry, "get", return_value=mock_connector
        ):
            result = DigestCollectTool().execute(
                sources=["apple_mail"],
                hours_back=24,
            )

    assert result.success is True
    assert "2 messages from icici.bank.in need review." not in result.content
    # alerts@ / notices@ senders are classified as automated notifications.
    assert "An automated notification." in result.content
    assert "PRIVATE_ICICI_SUBJECT_1" in result.content
    assert "PRIVATE_ICICI_SUBJECT_2" in result.content
    assert "PRIVATE_VENMO_SUBJECT" in result.content
    assert "Preview: PRIVATE_ICICI_BODY_1" in result.content
    assert "Preview: PRIVATE_ICICI_BODY_2" in result.content
    assert "Preview: PRIVATE_VENMO_BODY" in result.content
    assert "alerts@icici.bank.in" in result.content
    assert "alerts@venmo.com" in result.content
    assert result.metadata["total_items"] == 3

def test_email_preview_redacts_verification_codes():
    from openjarvis.tools.digest_collect import _sanitize_email_preview

    content = (
        "Your verification code is 123456. "
        "It expires in 10 minutes. Meeting at 3pm."
    )

    result = _sanitize_email_preview(content, 240)

    assert "123456" not in result
    assert "[code redacted]" in result
    assert "Meeting at 3pm" in result


def test_email_preview_preserves_ordinary_numbers():
    from openjarvis.tools.digest_collect import _sanitize_email_preview

    content = "Your appointment is at 3pm on November 25, 2026."

    result = _sanitize_email_preview(content, 240)

    assert result == content


def test_email_preview_removes_authentication_urls():
    from openjarvis.tools.digest_collect import _sanitize_email_preview

    content = "Please verify your account at https://example.com/token/abc123"

    result = _sanitize_email_preview(content, 240)

    assert "https://" not in result
    assert "[link omitted]" in result


def test_apple_calendar_formats_ongoing_all_day_event():
    from openjarvis.tools.digest_collect import _format_apple_calendar

    today = datetime.now().date()
    start = today - timedelta(days=4)
    end = today + timedelta(days=3)  # Exclusive all-day end date.
    doc = Document(
        doc_id="calendar-ongoing",
        source="apple_calendar",
        doc_type="calendar_event",
        title="Vacation stay",
        content="",
        timestamp=datetime.combine(start, datetime.min.time()),
        metadata={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "all_day": True,
        },
    )

    result = _format_apple_calendar(doc)
    expected_end = end - timedelta(days=1)
    assert "Ongoing:" in result
    assert start.strftime("%B %-d") in result
    assert expected_end.strftime("%B %-d") in result
    assert "Vacation stay" in result


def test_apple_calendar_keeps_future_event_date():
    from openjarvis.tools.digest_collect import _format_apple_calendar

    tomorrow = datetime.now().date() + timedelta(days=1)
    doc = Document(
        doc_id="calendar-future",
        source="apple_calendar",
        doc_type="calendar_event",
        title="Future appointment",
        content="",
        timestamp=datetime.combine(tomorrow, datetime.min.time()),
        metadata={
            "start": tomorrow.isoformat(),
            "end": (tomorrow + timedelta(days=1)).isoformat(),
            "all_day": True,
        },
    )

    result = _format_apple_calendar(doc)
    assert "Ongoing:" not in result
    assert tomorrow.strftime("%B %-d") in result
    assert "Future appointment" in result

@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "Hello body, table, td { font-family: Arial; } Important message",
            "Hello  Important message",
        ),
        (
            "*{box-sizing:border-box}body{margin:0;padding:0}Actual email text",
            "Actual email text",
        ),
        (
            "#outlook @media only screen and (min-width:480px) "
            "{ .column { width:100%; } }Order confirmation",
            "Order confirmation",
        ),
        (
            "Word Genius @font-face { font-family: Helvetica; "
            "src: url(font.woff); }Daily quiz",
            "Word Genius Daily quiz",
        ),
    ],
)
def test_remove_email_css(content, expected):
    from openjarvis.tools.digest_collect import _remove_email_css

    assert _remove_email_css(content) == expected
