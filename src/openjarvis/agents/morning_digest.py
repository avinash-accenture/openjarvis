"""Morning Digest Agent — synthesizes a daily briefing from multiple sources.

Thin orchestrator that delegates to digest_collect (data fetching),
the LLM (narrative synthesis), and text_to_speech (audio generation).
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.agents.digest_store import DigestArtifact, DigestStore
from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall

_SECTION_PROMPTS = {
    "messages": "MESSAGES — Prioritize provided messages or tasks needing action.",
    "calendar": "CALENDAR — Cover only provided upcoming events.",
    "health": "HEALTH — Describe only supported trends; omit raw measurements.",
    "world": "WORLD — Summarize only provided world items.",
    "music": "MUSIC — Summarize only provided listening information.",
}


def _load_persona(persona_name: str) -> str:
    """Load a persona prompt file by name."""
    search_paths = [
        Path("configs/openjarvis/prompts/personas") / f"{persona_name}.md",
        get_config_dir() / "prompts" / "personas" / f"{persona_name}.md",
    ]
    for p in search_paths:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""


def _extract_section_lines(data: str, section: str) -> list[str]:
    """Extract non-empty lines from a named digest section."""
    marker = f"=== {section.upper()} ==="
    lines: list[str] = []
    collecting = False

    for line in data.splitlines():
        stripped = line.strip()

        if stripped.startswith("===") and stripped.endswith("==="):
            collecting = stripped == marker
            continue

        if collecting and stripped:
            lines.append(stripped)

    return lines


def _build_deterministic_narrative(
    collected_data: str,
    sections: list[str],
    honorific: str = "sir",
    failed_sections: Optional[set[str]] = None,
) -> str:
    """Build a factual Messages/Calendar briefing without LLM synthesis."""
    import re

    parts: list[str] = []
    greeting = f"{honorific.strip().capitalize()}, here is your briefing." if honorific.strip() else "Here is your briefing."
    parts.append(greeting)
    failed_sections = failed_sections or set()

    for section in sections:
        section = section.strip().casefold()

        if section == "messages":
            lines = _extract_section_lines(collected_data, "messages")

            if not lines:
                if section in failed_sections:
                    parts.append("Message retrieval failed.")
                else:
                    parts.append("No messages were returned in this collection.")
                continue

            classifications: dict[tuple[str, str], int] = {}
            total = 0
            unclassified_count = 0

            for line in lines:
                match = re.match(
                    r"^\[[^\]]+\]\[([^\]]+)\]\[([^\]]+)\]\s*(.*)$",
                    line,
                )

                if not match:
                    total += 1
                    continue

                category, status, summary = match.groups()
                grouped = re.match(r"(\d+)\s+messages?\b", summary, re.IGNORECASE)
                count = int(grouped.group(1)) if grouped else 1

                total += count
                key = (category, status)
                classifications[key] = classifications.get(key, 0) + count

                if status.casefold() == "unclassified":
                    unclassified_count += count

            message = f"Messages: {total} returned."

            if classifications:
                details = [
                    f"{count} marked {category} / {status}"
                    for (category, status), count in classifications.items()
                ]
                message += " " + "; ".join(details) + "."

            if unclassified_count:
                message += (
                    " Their contents were not classified into a more specific category."
                )

            parts.append(message)

        elif section == "calendar":
            lines = _extract_section_lines(collected_data, "calendar")

            if not lines:
                if section in failed_sections:
                    parts.append("Calendar retrieval failed.")
                else:
                    parts.append("No calendar entries were returned in this collection.")
                continue

            # Remove only the connector prefix. Preserve the event text verbatim.
            events = [
                re.sub(r"^\[[^\]]+\]\s*", "", line)
                for line in lines
            ]

            parts.append(
                f"Calendar: {len(events)} entries. "
                + " ".join(events)
            )

    return " ".join(parts)


@AgentRegistry.register("morning_digest")
class MorningDigestAgent(ToolUsingAgent):
    """Pre-compute a daily digest from configured data sources."""

    agent_id = "morning_digest"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract digest-specific kwargs before passing to parent
        self._persona = kwargs.pop("persona", "jarvis")
        self._sections = kwargs.pop(
            "sections", ["messages", "calendar", "health", "world"]
        )
        self._section_sources = kwargs.pop("section_sources", {})
        self._timezone = kwargs.pop("timezone", "America/New_York")
        self._voice_id = kwargs.pop("voice_id", "Samantha")
        self._voice_speed = kwargs.pop("voice_speed", 0.72)
        self._tts_backend = kwargs.pop("tts_backend", "macos_say")
        self._digest_store_path = kwargs.pop("digest_store_path", "")
        self._honorific = kwargs.pop("honorific", "sir")
        super().__init__(*args, **kwargs)

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt from persona + briefing structure."""
        persona_text = _load_persona(self._persona)
        now = datetime.now(ZoneInfo(self._timezone))
        honorific = getattr(self, "_honorific", "sir")
        sections = dict.fromkeys(
            str(section).strip().casefold()
            for section in self._sections
            if str(section).strip()
        )
        section_block = "\n".join(
            f"- {_SECTION_PROMPTS.get(section, section.upper())}"
            for section in sections
        )

        return (
            f"{persona_text}\n\n"
            f"Today is {now.strftime('%A, %B %d, %Y')}. "
            f"The time is {now.strftime('%I:%M %p')} in {self._timezone}.\n"
            f"The user's preferred honorific is: {honorific}\n\n"
            "You receive structured data from the user's connected services. "
            "The data has ALREADY been collected — it appears in the user "
            "message. You do NOT fetch anything yourself.\n\n"
            "Produce a concise spoken briefing in decreasing order of importance. "
            "Cover only the configured sections below and only when the collected "
            "data supports them. Silently omit absent data and sources.\n\n"
            f"CONFIGURED SECTIONS:\n{section_block or '- None'}\n\n"
            "Open briefly with the honorific and end after the last supported item. "
            "Do not add conversational offers or personal asides.\n\n"
            "ABSOLUTE RULES (violations are unacceptable):\n"
            "- ONLY facts from the data. Zero hallucination.\n"
            "- NEVER mention disconnected or unavailable sources.\n"
            "- NEVER invent personal context or claim, offer, or suggest actions.\n"
            "- Acknowledge each configured section that returned meaningful data.\n"
            "- No markdown, emojis, bullets, or headers.\n"
            "- Keep the briefing concise: usually 60–150 words, with an absolute maximum of 180. There is no minimum word count. Never pad or repeat information.\n"
            "- For calendar entries, preserve the exact weekday and date shown in the data. Never describe a date as 'next week', 'tomorrow', or 'later today' unless that relative description is unambiguously correct. Preserve all-day status; do not invent times, locations, or event details.\n"
            "- For grouped messages, report the stated number of messages as a group. A group is not evidence of duplicates. Do not infer message contents, urgency, or required actions beyond the supplied classification and summary.\n"
            "- Do not claim that no other events or messages exist; describe only the items returned by the collection.\n"
        )

    def _resolve_sources(self) -> List[str]:
        """Get the list of connector IDs to query."""
        default_source_map = {
            "messages": [
                "apple_mail",
                "slack",
                "google_tasks",
                "imessage",
                "github_notifications",
            ],
            "calendar": ["apple_calendar"],
            "health": ["oura", "apple_health"],
            "world": ["weather", "hackernews", "news_rss"],
            "music": ["spotify", "apple_music"],
        }
        sources = dict.fromkeys([])
        for section in self._sections:
            section_sources = self._section_sources.get(
                section, default_source_map.get(section, [])
            )
            for source in section_sources:
                sources.setdefault(source, None)
        return list(sources)

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        # Step 1: Collect data from connectors
        sources = self._resolve_sources()
        collect_call = ToolCall(
            id="digest-collect-1",
            name="digest_collect",
            arguments=json.dumps({"sources": sources, "hours_back": 24}),
        )
        collect_result = self._executor.execute(collect_call)
        collected_data = collect_result.content

        
        # Step 2: Generate the briefing.
        configured_sections = [
            str(section).strip().casefold()
            for section in self._sections
            if str(section).strip()
        ]

        if not collect_result.success:
            # Never ask the LLM to turn failed collection data into a briefing.
            prefix = (
                f"{self._honorific.strip().capitalize()}, "
                if self._honorific.strip()
                else ""
            )
            narrative = (
                f"{prefix}I couldn't complete data collection, "
                "so I can't provide a reliable briefing."
            )

        elif configured_sections and all(
            section in {"messages", "calendar"}
            for section in configured_sections
        ):
            # Messages and Calendar are formatted deterministically.
            # No LLM interpretation of dates, titles, or classifications.
            failed_sources = set(
                collect_result.metadata.get("sources_failed", [])
            )

            default_source_map = {
                "messages": [
                    "apple_mail",
                    "slack",
                    "google_tasks",
                    "imessage",
                    "github_notifications",
                ],
                "calendar": ["apple_calendar"],
            }

            failed_sections = {
                section
                for section in configured_sections
                if any(
                    source in failed_sources
                    for source in self._section_sources.get(
                        section, default_source_map.get(section, [])
                    )
                )
            }

            narrative = _build_deterministic_narrative(
                collected_data,
                configured_sections,
                self._honorific,
                failed_sections,
            )

        else:
            # Other digest types retain the existing LLM synthesis path.
            system_prompt = self._build_system_prompt()
            messages = [
                Message(role=Role.SYSTEM, content=system_prompt),
                Message(
                    role=Role.USER,
                    content=(
                        "The following collected data is the only factual evidence for "
                        "the briefing:\n\n<collected_data>\n"
                        f"{collected_data}\n"
                        "</collected_data>\n\n"
                        "Use configured sections only. Omit missing "
                        "data and sources. Do not add personal context or activities. "
                        "Use the honorific no more than once. Keep the briefing concise, "
                        "usually 60–150 words, with an absolute maximum of 180 words. "
                        "Never pad, repeat, infer, or invent information."
                    ),
                ),
            ]

            result = self._generate(messages)
            narrative = self._strip_think_tags(result.get("content", ""))
 
        quality_score = 0.0
        evaluator_feedback = ""
 
        # Step 3: Generate audio via TTS
        # Strip any markdown that slipped through (##, *, -, etc.)
        import re

        tts_text = re.sub(r"^#{1,6}\s+", "", narrative, flags=re.MULTILINE)
        tts_text = re.sub(r"^\s*[-*•]\s+", "", tts_text, flags=re.MULTILINE)
        tts_text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", tts_text)
        tts_text = tts_text.strip()

        output_dir = str(get_config_dir() / "digests")
        tts_call = ToolCall(
            id="digest-tts-1",
            name="text_to_speech",
            arguments=json.dumps(
                {
                    "text": tts_text,
                    "voice_id": self._voice_id,
                    "backend": self._tts_backend,
                    "speed": self._voice_speed,
                    "output_dir": output_dir,
                }
            ),
        )
        tts_result = self._executor.execute(tts_call)
        audio_path = (
            tts_result.metadata.get("audio_path", "") if tts_result.success else ""
        )

        # Step 4: Store the artifact
        artifact = DigestArtifact(
            text=narrative,
            audio_path=Path(audio_path) if audio_path else Path(""),
            sections={},
            sources_used=sources,
            generated_at=datetime.now(ZoneInfo(self._timezone)),
            model_used=self._model,
            voice_used=self._voice_id,
            quality_score=quality_score,
            evaluator_feedback=evaluator_feedback,
        )

        store = DigestStore(db_path=self._digest_store_path)
        store.save(artifact)
        store.close()

        self._emit_turn_end(turns=1)
        return AgentResult(
            content=narrative,
            tool_results=[collect_result, tts_result],
            turns=1,
            metadata={
                "audio_path": audio_path,
                "sources_used": sources,
            },
        )
