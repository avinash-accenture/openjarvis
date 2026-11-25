"""Tests for MorningDigestAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openjarvis.agents._stubs import AgentResult
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult


def test_morning_digest_registered():
    from openjarvis.agents.morning_digest import MorningDigestAgent

    AgentRegistry.register_value("morning_digest", MorningDigestAgent)
    assert AgentRegistry.contains("morning_digest")


def test_morning_digest_run(tmp_path):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    mock_engine = MagicMock()
    mock_engine.generate.return_value = {
        "content": "Good morning sir. AtlasDB 1.0 was released.",
        "finish_reason": "stop",
        "usage": {},
    }

    # Mock collect result
    mock_collect_result = ToolResult(
        tool_name="digest_collect",
        content="=== WORLD ===\n[hackernews] AtlasDB 1.0 Released — 241 points\n",
        success=True,
        metadata={"total_items": 2},
    )

    # Mock TTS result
    mock_tts_result = ToolResult(
        tool_name="text_to_speech",
        content=str(tmp_path / "digest.mp3"),
        success=True,
        metadata={"audio_path": str(tmp_path / "digest.mp3")},
    )

    agent = MorningDigestAgent(
        mock_engine,
        "test-model",
        tools=[],
        persona="jarvis",
        sections=["world"],
        section_sources={"world": ["hackernews", "news_rss"]},
        digest_store_path=str(tmp_path / "digest.db"),
    )

    with patch.object(
        agent._executor,
        "execute",
        side_effect=[mock_collect_result, mock_tts_result],
    ):
        result = agent.run("Generate morning digest")

    assert isinstance(result, AgentResult)
    assert "Good morning" in result.content
    assert result.turns == 1
    assert len(result.tool_results) == 2
    assert set(result.metadata["sources_used"]) == {"hackernews", "news_rss"}
    prompt = "\n".join(
        message.text for message in mock_engine.generate.call_args.args[0]
    ).casefold()
    assert "world —" in prompt
    for forbidden in (
        "messages —|calendar —|health —|rebuttal|dinner at|group chat|"
        "slack|next meeting|readiness|hrv|weather"
    ).split("|"):
        assert forbidden not in prompt


def test_messages_calendar_digest_uses_deterministic_formatting(tmp_path):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    mock_engine = MagicMock()

    collected_data = """=== MESSAGES ===
[apple_mail][REVIEW][unclassified] An email needs review.
[apple_mail][REVIEW][unclassified] An email needs review.

=== CALENDAR ===
[apple_calendar] Tuesday, November 10 at 6:30 PM — Grandview Paint with Me Night
[apple_calendar] Wednesday, November 11 (all day) — Veterans Day
"""

    mock_collect_result = ToolResult(
        tool_name="digest_collect",
        content=collected_data,
        success=True,
        metadata={"total_items": 4},
    )

    mock_tts_result = ToolResult(
        tool_name="text_to_speech",
        content=str(tmp_path / "digest.mp3"),
        success=True,
        metadata={"audio_path": str(tmp_path / "digest.mp3")},
    )

    agent = MorningDigestAgent(
        mock_engine,
        "test-model",
        tools=[],
        persona="jarvis",
        sections=["messages", "calendar"],
        section_sources={
            "messages": ["apple_mail"],
            "calendar": ["apple_calendar"],
        },
        digest_store_path=str(tmp_path / "digest.db"),
    )

    with patch.object(
        agent._executor,
        "execute",
        side_effect=[mock_collect_result, mock_tts_result],
    ):
        result = agent.run("Generate morning digest")

    assert "2 returned" in result.content
    assert "REVIEW / unclassified" in result.content
    assert "Tuesday, November 10 at 6:30 PM" in result.content
    assert "Grandview Paint with Me Night" in result.content
    assert "Wednesday, November 11 (all day)" in result.content
    assert "Veterans Day" in result.content
    mock_engine.generate.assert_not_called()


def test_partial_collection_failure_preserves_successful_calendar(tmp_path):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    mock_engine = MagicMock()

    mock_collect_result = ToolResult(
        tool_name="digest_collect",
        content="""=== CALENDAR ===
[apple_calendar] Tuesday, November 10 at 6:30 PM — Grandview Paint with Me Night
""",
        success=True,
        metadata={
            "sources_queried": ["apple_mail", "apple_calendar"],
            "sources_ok": ["apple_calendar"],
            "sources_failed": ["apple_mail"],
            "errors": ["Connector 'apple_mail' failed"],
            "total_items": 1,
        },
    )

    mock_tts_result = ToolResult(
        tool_name="text_to_speech",
        content=str(tmp_path / "digest.mp3"),
        success=True,
        metadata={"audio_path": str(tmp_path / "digest.mp3")},
    )

    agent = MorningDigestAgent(
        mock_engine,
        "test-model",
        tools=[],
        persona="jarvis",
        sections=["messages", "calendar"],
        section_sources={
            "messages": ["apple_mail"],
            "calendar": ["apple_calendar"],
        },
        digest_store_path=str(tmp_path / "digest.db"),
    )

    with patch.object(
        agent._executor,
        "execute",
        side_effect=[mock_collect_result, mock_tts_result],
    ):
        result = agent.run("Generate morning digest")

    assert "Message retrieval failed" in result.content
    assert "No messages were returned" not in result.content
    assert "Grandview Paint with Me Night" in result.content
    assert "Tuesday, November 10 at 6:30 PM" in result.content

    mock_engine.generate.assert_not_called()


def test_collection_failure_does_not_call_llm(tmp_path):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    mock_engine = MagicMock()

    failed_collect_result = ToolResult(
        tool_name="digest_collect",
        content="Tool 'digest_collect' timed out after 60s.",
        success=False,
        metadata={},
    )

    mock_tts_result = ToolResult(
        tool_name="text_to_speech",
        content=str(tmp_path / "digest.mp3"),
        success=True,
        metadata={"audio_path": str(tmp_path / "digest.mp3")},
    )

    agent = MorningDigestAgent(
        mock_engine,
        "test-model",
        tools=[],
        persona="jarvis",
        sections=["messages", "calendar"],
        digest_store_path=str(tmp_path / "digest.db"),
    )

    with patch.object(
        agent._executor,
        "execute",
        side_effect=[failed_collect_result, mock_tts_result],
    ):
        result = agent.run("Generate morning digest")

    assert "couldn't complete data collection" in result.content
    mock_engine.generate.assert_not_called()


def test_load_persona():
    from openjarvis.agents.morning_digest import _load_persona

    # Nonexistent persona returns empty string
    result = _load_persona("nonexistent_persona_xyz")
    assert result == ""
