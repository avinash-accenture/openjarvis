"""Live integration tests for the skills system.

These tests require a running Ollama instance with qwen3.5:4b.
Mark: live
"""

from __future__ import annotations

import pytest
import textwrap

from openjarvis.core.events import EventBus, EventType
from openjarvis.skills.manager import SkillManager
from openjarvis.skills.tool_adapter import SkillTool
from openjarvis.system import SystemBuilder


class TestSkillSystemIntegration:
    """Integration tests verifying skills flow end-to-end with a real engine."""


    def test_system_builder_discovers_skills(self):
        """SystemBuilder discovers installed skills without a live inference engine."""
        from unittest.mock import patch

        from openjarvis.core.config import JarvisConfig

        config = JarvisConfig()
        config.tools.enabled = []
        config.agent.tools = []
        config.telemetry.enabled = False

        builder = SystemBuilder()
        builder._config = config

        # Skill discovery does not require inference. Supply a placeholder
        # engine and bypass security setup, which normally wraps the engine.
        with (
            patch.object(
                builder,
                "_resolve_engine",
                return_value=(object(), "test"),
            ),
            patch(
                "openjarvis.security.setup_security",
                return_value=type(
                    "SecurityResult",
                    (),
                    {
                        "engine": object(),
                        "capability_policy": None,
                        "rate_limiter": None,
                        "audit_logger": None,
                    },
                )(),
            ),
        ):
            system = builder.model("test-model").build()

        try:
            assert system.skill_manager is not None, "SkillManager should be created"
            skill_names = system.skill_manager.skill_names()
            assert len(skill_names) > 0, f"Should discover skills, got: {skill_names}"

            skill_tools = [t for t in system.tools if isinstance(t, SkillTool)]
            assert len(skill_tools) > 0, "Skill tools should be in system tools"
        finally:
            if hasattr(system, "close"):
                system.close()


    
    def test_skill_catalog_in_system_prompt(self):
        from pathlib import Path

        bus = EventBus()
        mgr = SkillManager(bus=bus)

        bundled_skills = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "openjarvis"
            / "skills"
            / "data"
        )
        mgr.discover(paths=[bundled_skills])

        catalog = mgr.get_catalog_xml()

        assert "<available_skills>" in catalog
        assert "topic-research" in catalog
        assert "code-test-gen" in catalog
        assert "security-scan" in catalog

   

    def test_skill_tool_invocation_returns_content(self, tmp_path):
        skill_dir = tmp_path / "code-explainer"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
             textwrap.dedent("""\
             ---
             name: code-explainer
             description: Explains code in plain language
             version: "0.1.0"
             ---

             Explain programming concepts and code in plain language.
             When given a task, provide a clear explanation suitable for the reader.
         """),
         encoding="utf-8",
        )

        bus = EventBus(record_history=True)
        mgr = SkillManager(bus=bus)
        mgr.discover(paths=[tmp_path])

        tools = mgr.get_skill_tools()
        code_explainer = next(
           (tool for tool in tools if tool.spec.name == "skill_code-explainer"),
           None,
        )

        assert code_explainer is not None

        result = code_explainer.execute(task="explain a for loop")

        assert result.success
        assert "programming concepts" in result.content.lower()
        assert "plain language" in result.content.lower()
        assert result.metadata["skill_kind"] == "instructional"
        assert result.metadata["steps"] == 0
        assert bus.history is not None


    def old_test_skill_tool_invocation_returns_content(self):
        """Invoking a skill tool returns meaningful content."""
        bus = EventBus(record_history=True)
        mgr = SkillManager(bus=bus)
        from pathlib import Path

        mgr.discover(paths=[Path("~/.openjarvis/skills/").expanduser()])

        # Test instruction-only skill
        tools = mgr.get_skill_tools()
        code_explainer = next(
            (t for t in tools if "code-explainer" in t.spec.name), None
        )
        assert code_explainer is not None, "code-explainer skill should exist"

        result = code_explainer.execute(task="explain a for loop")
        assert result.success
        assert (
            "programming language" in result.content.lower()
            or "plain language" in result.content.lower()
        ), f"Should return instructions, got: {result.content[:200]}"
        print(f"  code-explainer returned: {result.content[:200]}...")

        # Verify the bus is reachable (instruction-only skills don't run
        # a pipeline, so no SKILL_EXECUTE_* events here — tool-mode skills
        # emit them, covered in TestSkillEventsAndTracing below)
        assert bus.history is not None

    @pytest.mark.live
    def test_agent_ask_with_skills_available(self):
        """Agent can answer a query with skills available in the tool list."""
        import openjarvis.agents  # noqa: F401
        from openjarvis.agents.simple import SimpleAgent
        from openjarvis.core.registry import AgentRegistry

        if not AgentRegistry.contains("simple"):
           AgentRegistry.register("simple")(SimpleAgent)

        system = (
            SystemBuilder()
            .engine("ollama")
            .model("qwen3.5:4b")
            .tools(
                [
                    "think",
                    "calculator",
                ]
            )
            .build()
        )
        try:
            # Skills should be auto-discovered and added
            assert system.skill_manager is not None
            skill_count = len(system.skill_manager.skill_names())
            total_tools = len(system.tools)
            print(f"  Skills: {skill_count}, Total tools: {total_tools}")

            # The tools list should include both regular tools and skill tools
            tool_names = [t.spec.name for t in system.tools]
            print(f"  Tool names: {tool_names}")

            # Ask a simple question — agent should work normally
            result = system.ask("What is 2 + 2?")
            assert "content" in result
            content = result["content"]
            assert "4" in content, f"Expected '4' in response, got: {content[:200]}"
            print(f"  Agent response: {content[:200]}")
        finally:
            if hasattr(system, "close"):
                try:
                    system.close()
                except Exception:
                    pass


class TestSkillEventsAndTracing:
    """Verify skills emit proper events for the Learning pipeline."""

    def test_skill_execution_emits_events(self):
        """Running a structured skill emits SKILL_EXECUTE_START/END events."""
        from openjarvis.core.types import ToolResult
        from openjarvis.skills.executor import SkillExecutor
        from openjarvis.skills.types import SkillManifest, SkillStep
        from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec

        class EchoTool(BaseTool):
            tool_id = "echo"

            @property
            def spec(self):
                return ToolSpec(name="echo", description="Echo")

            def execute(self, **params):
                return ToolResult(
                    tool_name="echo",
                    content=params.get("text", "echoed"),
                    success=True,
                )

        bus = EventBus(record_history=True)
        te = ToolExecutor([EchoTool()])
        executor = SkillExecutor(te, bus=bus)

        manifest = SkillManifest(
            name="test_traced",
            steps=[
                SkillStep(
                    tool_name="echo",
                    arguments_template='{"text": "hello"}',
                    output_key="result",
                ),
            ],
        )
        result = executor.run(manifest)
        assert result.success

        events = [(e.event_type, e.data) for e in bus.history]
        event_types = [e[0] for e in events]
        assert EventType.SKILL_EXECUTE_START in event_types
        assert EventType.SKILL_EXECUTE_END in event_types

        # Verify event data
        start_event = next(e for e in events if e[0] == EventType.SKILL_EXECUTE_START)
        assert start_event[1]["skill"] == "test_traced"
        end_event = next(e for e in events if e[0] == EventType.SKILL_EXECUTE_END)
        assert end_event[1]["success"] is True
        print("  Events correctly emitted with skill metadata")
