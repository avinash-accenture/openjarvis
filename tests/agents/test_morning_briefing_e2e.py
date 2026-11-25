"""End to end: collector selection + MorningDigestAgent with fake connectors."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from openjarvis.connectors._stubs import Document
from openjarvis.core.registry import ConnectorRegistry, TTSRegistry

NOW = datetime.now().astimezone()

pytestmark = pytest.mark.usefixtures("real_tool_policy")


def _mail(i, subject, body="", sender=None, account="Gmail", **metadata):
    metadata.setdefault("account", account)
    metadata.setdefault("body_available", bool(body))
    return Document(
        doc_id=f"apple_mail:{account}:{i}",
        source="apple_mail",
        doc_type="email",
        title=subject,
        author=sender or f"person{i}@example.test",
        content=body,
        timestamp=NOW - timedelta(minutes=10 * i),
        metadata=metadata,
    )


def _mailbox():
    docs = [
        _mail(
            1,
            "Payment overdue",
            "Your payment is overdue. Please pay now.",
            account="iCloud",
        ),
        _mail(
            2,
            "Dinner Saturday?",
            "Are you free Saturday at 7?",
            sender="priya@example.test",
            sender_name="Priya Raman",
            mail_headers={"from_name": "Priya Raman"},
            account="Yahoo Avi",
        ),
        _mail(
            3,
            "Your statement is ready",
            "Your statement is ready. Payment due date: December 3, 2026.",
            sender="alerts@bank.test",
            account="Outlook",
        ),
    ]
    docs += [
        _mail(
            10 + i,
            "Weekly newsletter",
            "Sale! Unsubscribe here.",
            sender="news@shop.test",
            account="Yahoo Pavi",
        )
        for i in range(12)
    ]
    docs += [
        _mail(
            30 + i,
            f"Notification {i}",
            "Something happened.",
            sender="no-reply@app.test",
        )
        for i in range(20)
    ]
    return docs


class _FakeMail:
    def __init__(self, *a, **k):
        self.account_stats = {
            "Gmail": {"fetched": 20, "with_body": 20, "error": ""},
            "iCloud": {"fetched": 1, "with_body": 1, "error": ""},
            "Yahoo Avi": {"fetched": 1, "with_body": 1, "error": ""},
            "Yahoo Pavi": {"fetched": 12, "with_body": 12, "error": ""},
            "Outlook": {"fetched": 1, "with_body": 1, "error": ""},
        }

    def is_connected(self):
        return True

    def sync(self, **kwargs):
        return _mailbox()


class _FakeCalendar:
    def __init__(self, days_ahead=2, **k):
        self.days_ahead = days_ahead

    def is_connected(self):
        return True

    def sync(self, **kwargs):
        today = NOW.date()
        events = [
            Document(
                doc_id="cal:1",
                source="apple_calendar",
                doc_type="calendar_event",
                title="Dentist",
                content="",
                timestamp=NOW,
                metadata={
                    "start": NOW.replace(hour=14, minute=0).isoformat(),
                    "end": "",
                    "calendar": "Home",
                    "all_day": False,
                },
            ),
            Document(
                doc_id="cal:2",
                source="apple_calendar",
                doc_type="calendar_event",
                title="Mom's Birthday",
                content="",
                timestamp=NOW,
                metadata={
                    "start": (today + timedelta(days=1)).isoformat(),
                    "end": (today + timedelta(days=2)).isoformat(),
                    "calendar": "Birthdays",
                    "all_day": True,
                },
            ),
        ]
        if self.days_ahead > 2:
            events.append(
                Document(
                    doc_id="cal:3",
                    source="apple_calendar",
                    doc_type="calendar_event",
                    title="Wedding anniversary",
                    content="",
                    timestamp=NOW,
                    metadata={
                        "start": (today + timedelta(days=9)).isoformat(),
                        "end": (today + timedelta(days=10)).isoformat(),
                        "calendar": "Home",
                        "all_day": True,
                    },
                )
            )
        return events


@pytest.fixture
def fake_sources(monkeypatch):
    import openjarvis.connectors  # noqa: F401  (registers real connectors first)

    ConnectorRegistry.register_value("apple_mail", _FakeMail)
    ConnectorRegistry.register_value("apple_calendar", _FakeCalendar)


def test_collector_selects_top_mail_across_accounts(fake_sources) -> None:
    from openjarvis.tools.digest_collect import DigestCollectTool

    result = DigestCollectTool().execute(
        sources=["apple_mail", "apple_calendar"], mail_limit=15
    )
    payload = result.metadata["briefing"]
    mail = payload["mail"]
    assert len(mail) == 15
    assert mail[0]["subject"] == "Payment overdue"
    assert {m["status"] for m in mail[:3]} == {
        "payment_overdue",
        "personal",
        "statement",
    }
    assert not any(m["subject"] == "Weekly newsletter" for m in mail)
    stats = payload["mail_stats"]
    assert stats["in_window"] == 35
    assert stats["excluded_marketing"] == 12
    assert stats["omitted_by_category"] == {"LOW_PRIORITY": 8}
    assert set(stats["accounts"]) == {
        "Gmail",
        "iCloud",
        "Yahoo Avi",
        "Yahoo Pavi",
        "Outlook",
    }
    # Birthdays come from the 14-day lookup and are kept out of the agenda.
    assert [e["title"] for e in payload["calendar"]] == ["Dentist"]
    assert {e["title"] for e in payload["special_events"]} == {
        "Mom's Birthday",
        "Wedding anniversary",
    }


def test_agent_produces_briefing_report_and_audio(fake_sources, tmp_path, monkeypatch):
    from openjarvis.agents.morning_digest import MorningDigestAgent
    from openjarvis.core.config import load_config
    from openjarvis.tools.digest_collect import DigestCollectTool
    from openjarvis.tools.text_to_speech import TextToSpeechTool

    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    load_config(cfg)  # installs local-only and the tool allowlist

    from openjarvis.speech.tts import TTSResult

    class _FakeSay:
        backend_id = "macos_say"

        def synthesize(self, text, **kwargs):
            _FakeSay.spoken = text
            return TTSResult(
                audio=b"AIFF",
                format="aiff",
                duration_seconds=1.0,
                voice_id="Samantha",
                metadata={},
            )

        def health(self):
            return True

    real_get = TTSRegistry.get
    monkeypatch.setattr(
        TTSRegistry,
        "get",
        classmethod(lambda cls, key: _FakeSay if key == "macos_say" else real_get(key)),
    )
    monkeypatch.setattr(
        TTSRegistry,
        "contains",
        classmethod(lambda cls, key: True if key == "macos_say" else False),
    )
    import openjarvis.agents.morning_digest as md

    monkeypatch.setattr(md, "get_config_dir", lambda: tmp_path)
    import openjarvis.agents.digest_store as store_mod

    monkeypatch.setattr(store_mod, "get_config_dir", lambda: tmp_path, raising=False)

    engine = MagicMock()
    engine.generate.side_effect = AssertionError("no LLM for messages/calendar")
    agent = MorningDigestAgent(
        engine,
        "qwen3.5:2b",
        tools=[DigestCollectTool(), TextToSpeechTool()],
        sections=["messages", "calendar"],
        section_sources={"messages": ["apple_mail"], "calendar": ["apple_calendar"]},
        honorific="sir",
        timezone="America/New_York",
        max_summaries=0,
        digest_store_path=str(tmp_path / "digest.db"),
    )
    result = agent.run("Generate my morning digest")

    meta = result.metadata
    assert meta["collect_ok"] and meta["tts_ok"], result.tool_results
    assert meta["word_count"] <= 630
    assert meta["emails_spoken"] + meta["emails_report_only"] == 15
    assert result.content == _FakeSay.spoken
    assert "From Priya Raman" in result.content
    assert "Coming up: Mom's Birthday tomorrow" in result.content
    assert "12 promotional emails skipped" in result.content
    report = (tmp_path / "digests").glob("briefing-*.md")
    report_path = next(report)
    assert str(report_path) == meta["report_path"]
    text = report_path.read_text()
    assert "## Mail coverage" in text and "| Yahoo Pavi | 12 | 12 | ok |" in text
    assert "Saved to" in text
    assert oct(report_path.stat().st_mode)[-3:] == "600"
