"""Tests for deterministic local mail classification."""

import dataclasses
import re
from datetime import datetime

from openjarvis.connectors._stubs import Attachment, Document
from openjarvis.tools.mail_classifier import (
    classify_mail_document,
    deduplicate_shipments,
)

from unittest.mock import MagicMock

from openjarvis.tools.mail_classifier import (
    classify_mail_document_with_local_llm,
)

NOW = datetime(2026, 11, 7, 8, 0)


def make_mail(subject: str, body: str, doc_id: str = "mail-1") -> Document:
    return Document(
        doc_id=doc_id,
        source="apple_mail",
        doc_type="email",
        title=subject,
        author="Synthetic Sender",
        content=body,
        timestamp=NOW,
        metadata={"account": "Synthetic"},
    )

def test_escrow_shortage_separates_deadline_from_effective_date():
    item = classify_mail_document(
        make_mail(
            "Escrow shortage notice",
            "The full shortage can be paid before November 30, 2026. "
            "Take No Action and the shortage will be divided evenly over "
            "the next 12 monthly mortgage payments. New payment starts "
            "December 1, 2026.",
        ),
        NOW,
    )

    assert item.category == "REVIEW"
    assert item.status == "optional_decision"
    assert item.deadline == "2026-11-30"
    assert item.effective_date == "2026-12-01"
    assert "not establish that a separate payment is mandatory" in item.safe_summary

def old_test_escrow_shortage_with_no_action_alternative_is_not_mandatory():
    item = classify_mail_document(
        make_mail(
            "Escrow shortage notice",
            "Take No Action and the shortage will be divided evenly over "
            "the next 12 monthly mortgage payments. New payment starts "
            "December 1, 2026.",
        ),
        NOW,
    )
    assert item.category == "REVIEW"
    assert item.status == "optional_decision"
    assert item.deadline == "2026-12-01"
    assert "not establish that a separate payment is mandatory" in item.safe_summary

def test_payment_authorization_is_not_completion():
    item = classify_mail_document(
        make_mail(
            "Payment authorization",
            "Your payment was authorized successfully. "
            "The transaction has not yet been completed.",
        ),
        NOW,
    )
    assert item.category != "COMPLETED"

def test_future_payment_due_date_is_not_overdue():
    item = classify_mail_document(
        make_mail(
            "Upcoming payment reminder",
            "Your payment is due on November 30, 2026. "
            "Please ensure sufficient funds are available.",
        ),
        NOW,
    )
    assert item.category != "ACTION_REQUIRED"

def test_explicit_overdue_payment_requires_review():
    item = classify_mail_document(
        make_mail(
            "Payment overdue notice",
            "Your payment is overdue. Please make the outstanding "
            "payment to bring your account current.",
        ),
        NOW,
    )
    assert item.category == "ACTION_REQUIRED"
    assert item.status == "payment_overdue"

def test_successful_payment_is_completed():
    item = classify_mail_document(
        make_mail(
            "Payment receipt",
            "Your payment was successfully completed. "
            "The transaction was successful.",
        ),
        NOW,
    )
    assert item.category == "COMPLETED"
    assert item.status == "payment_completed"

def test_payment_confirmation_is_processing_not_settled():
    item = classify_mail_document(
        make_mail(
            "Automatic payment confirmation",
            "Your ACH payment is being processed. Allow 24-48 hours for "
            "your bank statement and 3-5 business days for credit.",
        ),
        NOW,
    )
    assert item.category == "REVIEW"
    assert item.status == "processing"
    assert "settled" in item.safe_summary

def test_successful_payment_followed_by_failed_payment_is_not_completed():
    item = classify_mail_document(
        make_mail(
            "Payment status update",
            "Your previous payment was successfully completed. "
            "Your latest payment failed and was declined.",
        ),
        NOW,
    )
    assert item.category != "COMPLETED"

def test_already_paid_overdue_notice_is_not_action_required():
    item = classify_mail_document(
        make_mail(
            "Payment status update",
            "Your account was previously overdue, but the outstanding "
            "payment has now been paid in full. No further action is required.",
        ),
        NOW,
    )
    assert item.category != "ACTION_REQUIRED"

def test_appointment_reminder_does_not_imply_mandatory_confirmation():
    item = classify_mail_document(
        make_mail(
            "Appointment reminder",
            "Your appointment is November 9, 2026 at 12:00 PM. "
            "You may reschedule if needed.",
        ),
        NOW,
    )
    assert item.category == "TIME_SENSITIVE"
    assert item.status == "scheduled"
    assert item.event_date == "2026-11-09"
    assert "12:00 PM" in item.safe_summary
    assert "mandatory action" in item.reason

def test_suspicious_sign_in_is_time_sensitive():
    item = classify_mail_document(
        make_mail(
            "New sign-in to your account",
            "We detected a sign-in from a new device. "
            "If this was not you, secure your account immediately.",
        ),
        NOW,
    )

    assert item.category == "TIME_SENSITIVE"
    assert item.status == "security_alert"

def test_payment_still_processing_is_not_completed():
    item = classify_mail_document(
        make_mail(
            "Payment update",
            "Your payment was successfully submitted, "
            "but the transaction is still processing.",
        ),
        NOW,
    )
    assert item.category == "REVIEW"
    assert item.status == "processing"


def test_security_alert_is_not_triggered_when_sign_in_is_verified():
    item = classify_mail_document(
        make_mail(
            "New sign-in verification",
            "We detected a sign-in from a new device. "
            "We verified this activity was yours. "
            "No action is required.",
        ),
        NOW,
    )

    assert item.category != "TIME_SENSITIVE"
    assert item.status != "security_alert"


def test_explicitly_not_overdue_payment_is_not_action_required():
    item = classify_mail_document(
        make_mail(
            "Payment account status",
            "Your payment is not overdue. "
            "If a payment becomes overdue in the future, please pay it promptly.",
        ),
        NOW,
    )

    assert item.category != "ACTION_REQUIRED"
    assert item.status != "payment_overdue"


def test_shipping_label_created_is_not_in_transit():
    item = classify_mail_document(
        make_mail(
            "Shipping label created",
            "A shipping label has been created. "
            "The carrier has not received the package. "
            "It has not shipped and is not in transit.",
        ),
        NOW,
    )

    assert item.category != "TIME_SENSITIVE"
    assert item.status != "in_transit"


def test_not_out_for_delivery_is_not_time_sensitive():
    item = classify_mail_document(
        make_mail(
            "Shipment update",
            "Your package is not out for delivery. "
            "It is still awaiting carrier pickup.",
        ),
        NOW,
    )

    assert item.category != "TIME_SENSITIVE"
    assert item.status != "out_for_delivery"


def test_information_session_is_not_an_appointment():
    item = classify_mail_document(
        make_mail(
            "Annual information session",
            "You are invited to our annual information session. "
            "This is an informational event, not an appointment "
            "or a scheduled individual meeting.",
        ),
        NOW,
    )

    assert item.status == "unclassified"
    assert item.category == "REVIEW"

def test_failed_payment_is_not_completed():
    item = classify_mail_document(
        make_mail(
            "Payment failed",
            "Your payment transaction was unsuccessful and was declined.",
        ),
        NOW,
    )
    assert item.category != "COMPLETED"

def test_generic_security_notice_is_not_action_required():
    item = classify_mail_document(
        make_mail(
            "Your monthly security update",
            "We regularly review our security practices "
            "to help protect your account.",
        ),
        NOW,
    )

    assert item.category == "REVIEW"
    assert item.status == "unclassified"

def test_appointment_ignores_unrelated_footer_date():
    item = classify_mail_document(
        make_mail(
            "Appointment reminder",
            "Your appointment is November 9, 2026 at 12:00 PM. "
            "Office mailing address updated January 2, 2026.",
        ),
        NOW,
    )

    assert item.event_date == "2026-11-09"
    assert "12:00 PM" in item.safe_summary

def test_past_appointment_is_not_time_sensitive():
    item = classify_mail_document(
        make_mail(
            "Appointment reminder",
            "Your appointment is November 6, 2026 at 12:00 PM.",
        ),
        NOW,
    )
    assert item.category == "REVIEW"
    assert item.status == "past"
    assert item.event_date == "2026-11-06"

def test_tax_certificate_does_not_claim_attachment_without_metadata():
    item = classify_mail_document(
        make_mail(
            "Your TDS certificate",
            "Your TDS certificate is now available.",
        ),
        NOW,
    )
    assert item.category == "REVIEW"
    assert "Attachment details are not verified" in item.safe_summary


def test_unrelated_attachment_does_not_verify_tax_certificate():
    doc = make_mail(
        "Your TDS certificate",
        "Your TDS certificate is now available.",
    )
    doc.attachments.append(
        Attachment(
            filename="newsletter.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        )
    )

    item = classify_mail_document(doc, NOW)

    assert item.category == "REVIEW"
    assert "Attachment details are not verified" in item.safe_summary
    assert "newsletter.pdf" not in item.safe_summary

def test_same_tracking_id_keeps_newest_even_when_input_is_oldest_first():
    older = classify_mail_document(
        make_mail(
            "Shipment in transit",
            "Tracking number: 1Z999AA10123456784. In transit.",
            "older",
        ),
        NOW,
    )

    newer = classify_mail_document(
        make_mail(
            "Shipment out for delivery",
            "Tracking number: 1Z999AA10123456784. Out for delivery.",
            "newer",
        ),
        NOW,
    )

    # Deliberately supply oldest first.
    older = dataclasses.replace(
        older, timestamp=datetime(2026, 11, 8, 10, 0)
    )
    newer = dataclasses.replace(
        newer, timestamp=datetime(2026, 11, 9, 10, 0)
    )

    result = deduplicate_shipments([older, newer])

    assert len(result) == 1
    assert result[0].doc_id == "newer"
    assert result[0].status == "out_for_delivery"

def test_different_tracking_ids_are_not_deduplicated_for_same_order():
    first = classify_mail_document(
        make_mail(
            "Costco shipment out for delivery",
            "Order 123456. Tracking number: 1Z999AA10123456784. Out for delivery.",
            "shipment-1",
        ),
        NOW,
    )
    second = classify_mail_document(
        make_mail(
            "Costco shipment in transit",
            "Order 123456. Tracking number: 1Z888BB20123456785. In transit.",
            "shipment-2",
        ),
        NOW,
    )
    assert first.tracking_fingerprint != second.tracking_fingerprint
    assert len(deduplicate_shipments([first, second])) == 2


def test_same_tracking_id_keeps_newest_first_status():
    earlier = classify_mail_document(
        make_mail(
            "Shipment on its way",
            "Tracking number: 1Z999AA10123456784. Your shipment is on its way.",
            "earlier",
        ),
        NOW,
    )
    later = classify_mail_document(
        make_mail(
            "Shipment out for delivery",
            "Tracking number: 1Z999AA10123456784. Out for delivery.",
            "later",
        ),
        NOW,
    )
    # The collector supplies newest-first documents.
    result = deduplicate_shipments([later, earlier])
    assert len(result) == 1
    assert result[0].doc_id == "later"
    assert result[0].status == "out_for_delivery"


def test_delivery_without_tracking_is_not_merged_with_other_shipments():
    delivered = classify_mail_document(
        make_mail("Order delivered", "Your order has been delivered.", "delivered"),
        NOW,
    )
    moving = classify_mail_document(
        make_mail(
            "Shipment in transit",
            "Tracking number: 1Z999AA10123456784. In transit.",
            "moving",
        ),
        NOW,
    )
    assert len(deduplicate_shipments([delivered, moving])) == 2

def test_not_yet_delivered_is_not_classified_as_completed():
    item = classify_mail_document(
        make_mail(
            "Shipment update",
            "Your package has not yet been delivered. It is still in transit.",
        ),
        NOW,
    )

    assert item.category == "TIME_SENSITIVE"
    assert item.status == "in_transit"

def test_promotions_and_newsletters_are_excluded():
    promo = classify_mail_document(
        make_mail(
            "Special offer this week",
            "Shop now and save. Unsubscribe from these emails.",
        ),
        NOW,
    )
    newsletter = classify_mail_document(
        make_mail(
            "This week in AI: local inference trends",
            "Weekly newsletter. Unsubscribe here.",
        ),
        NOW,
    )
    assert promo.category == "EXCLUDE"
    assert newsletter.category == "EXCLUDE"
    assert promo.safe_summary == ""
    assert newsletter.safe_summary == ""


def test_sensitive_identifiers_and_raw_body_are_not_in_safe_summary():
    body = (
        "Your ACH payment is being processed. "
        "Confirmation number 987654321. "
        "Recipient avi@example.com."
    )
    item = classify_mail_document(
        make_mail("Payment confirmation", body),
        NOW,
    )
    assert "987654321" not in item.safe_summary
    assert "avi@example.com" not in item.safe_summary
    assert body not in item.safe_summary


def test_generic_amount_due_is_not_automatically_action_required():
    item = classify_mail_document(
        make_mail(
            "Payment update",
            "Amount due is shown on your account. No payment action is specified.",
        ),
        NOW,
    )
    assert item.category != "ACTION_REQUIRED"
    assert item.status == "unclassified"

def test_deduplication_preserves_order_of_tracked_and_untracked_messages():
    untracked_first = classify_mail_document(
        make_mail("Order delivered", "Your order has been delivered.", "untracked-1"),
        NOW,
    )
    tracked_newest = classify_mail_document(
        make_mail(
            "Shipment out for delivery",
            "Tracking number: 1Z999AA10123456784. Out for delivery.",
            "tracked-newest",
        ),
        NOW,
    )
    untracked_middle = classify_mail_document(
        make_mail(
            "Appointment reminder",
            "Your appointment is November 9, 2026 at 12:00 PM.",
            "untracked-2",
        ),
        NOW,
    )
    tracked_older = classify_mail_document(
        make_mail(
            "Shipment on its way",
            "Tracking number: 1Z999AA10123456784. Your shipment is on its way.",
            "tracked-older",
        ),
        NOW,
    )
    tracked_other = classify_mail_document(
        make_mail(
            "Another shipment",
            "Tracking number: 1Z888BB20123456785. In transit.",
            "tracked-other",
        ),
        NOW,
    )

    result = deduplicate_shipments(
        [
            untracked_first,
            tracked_newest,
            untracked_middle,
            tracked_older,
            tracked_other,
        ]
    )

    assert [item.doc_id for item in result] == [
        "untracked-1",
        "tracked-newest",
        "untracked-2",
        "tracked-other",
    ]


def test_local_llm_classifies_unclassified_mail():
    doc = make_mail(
        "Synthetic account notice",
        "Please review the account information in this synthetic message.",
    )
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"category":"REVIEW","status":"notice","confidence":0.95}'
        )
    }

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.category == "REVIEW"
    assert item.status == "notice"
    assert item.safe_summary == (
        "A notice was identified and may need review."
    )
    assert item.evidence == ("local_llm_classification",)
    assert "synthetic message" not in item.safe_summary.lower()
    engine.generate.assert_called_once()


def test_local_llm_does_not_override_deterministic_classification():
    doc = make_mail(
        "Payment overdue notice",
        "Your payment is overdue. Please pay the outstanding balance.",
    )
    engine = MagicMock()

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.category == "ACTION_REQUIRED"
    assert item.status == "payment_overdue"
    engine.generate.assert_not_called()


def test_local_llm_rejects_exclude_category():
    doc = make_mail("Synthetic notice", "Please review this message.")
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"category":"EXCLUDE","status":"excluded","confidence":0.99}'
        )
    }

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.status == "unclassified"
    assert item.evidence == ("conservative_fallback",)


def test_local_llm_rejects_invalid_category_status_pair():
    doc = make_mail("Synthetic notice", "Please review this message.")
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"category":"COMPLETED","status":"payment_overdue",'
            '"confidence":0.99}'
        )
    }

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.status == "unclassified"


def test_local_llm_rejects_low_confidence():
    doc = make_mail("Synthetic notice", "Please review this message.")
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"category":"REVIEW","status":"notice","confidence":0.55}'
        )
    }

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.status == "unclassified"


def test_local_llm_failure_preserves_deterministic_result():
    doc = make_mail("Synthetic notice", "Please review this message.")
    engine = MagicMock()
    engine.generate.side_effect = RuntimeError("Synthetic connection failure")

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.category == "REVIEW"
    assert item.status == "unclassified"
    assert item.evidence == ("conservative_fallback",)


def test_local_llm_timeout_is_marked_and_preserves_email():
    import httpx

    doc = make_mail("Synthetic notice", "Please review this message.")
    engine = MagicMock()
    engine.generate.side_effect = httpx.ReadTimeout("Synthetic timeout")

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.category == "REVIEW"
    assert item.status == "unclassified-timeout"
    assert "local_llm_timeout" in item.evidence
    assert item.doc_id == doc.doc_id


def test_local_llm_rejects_malformed_json():
    doc = make_mail("Synthetic notice", "Please review this message.")
    engine = MagicMock()
    engine.generate.return_value = {"content": "not valid json"}

    item = classify_mail_document_with_local_llm(doc, NOW, engine=engine)

    assert item.status == "unclassified"


def test_explicit_overdue_payment_survives_unrelated_future_condition():
    item = classify_mail_document(
        make_mail(
            "Overdue payment notice",
            "Your November payment is overdue. "
            "Please pay the outstanding balance immediately. "
            "If a future payment becomes overdue, we will notify you.",
        ),
        NOW,
    )

    assert item.category == "ACTION_REQUIRED"
    assert item.status == "payment_overdue"

def test_information_session_does_not_suppress_real_appointment():
    item = classify_mail_document(
        make_mail(
            "Appointment confirmation and event information",
            "Your dental appointment is confirmed for November 18, 2026 "
            "at 10:30 AM. Please arrive 10 minutes early. "
            "Separately, you are invited to our annual information session. "
            "This informational event is not an appointment.",
        ),
        NOW,
    )

    assert item.category == "TIME_SENSITIVE"
    assert item.status == "scheduled"

def test_unshipped_package_does_not_suppress_other_in_transit_package():
    item = classify_mail_document(
        make_mail(
            "Updates for two packages",
            "Package A, tracking number 123456789, is in transit "
            "and expected to arrive tomorrow. "
            "Package B, tracking number 987654321, has only had "
            "a shipping label created. The carrier has not received it.",
        ),
        NOW,
    )

    assert item.category == "TIME_SENSITIVE"
    assert item.status == "in_transit"
    assert item.tracking_fingerprint is not None
