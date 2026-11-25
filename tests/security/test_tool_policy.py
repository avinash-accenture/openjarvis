"""Tests for the tool allowlist (openjarvis.security.tool_policy)."""

from __future__ import annotations

import json

import pytest

from openjarvis.core.config import JarvisConfig, load_config
from openjarvis.core.types import ToolCall
from openjarvis.security import tool_policy
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolResult, ToolSpec

pytestmark = pytest.mark.usefixtures("real_tool_policy")


class _Tool(BaseTool):
    def __init__(self, name: str) -> None:
        self._name = name
        self.calls = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description="test tool", parameters={})

    def execute(self, **params) -> ToolResult:
        self.calls += 1
        return ToolResult(tool_name=self._name, content="ran", success=True)


def _call(executor: ToolExecutor, name: str) -> ToolResult:
    return executor.execute(ToolCall(id="1", name=name, arguments=json.dumps({})))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = JarvisConfig()
    assert cfg.security.tool_allowlist_enabled is True
    assert cfg.security.allowed_tools is None  # => built-in default list


DANGEROUS = [
    "shell_exec",
    "docker_shell_exec",
    "code_interpreter",
    "code_interpreter_docker",
    "repl",
    "rlm_repl",
    "file_read",
    "file_write",
    "apply_patch",
    "git_commit",
    "db_query",
    "http_request",
    "web_search",
    "browser_navigate",
    "channel_send",
    "skill_manage",
    "memory_store",
    "memory_manage",
    "memory_index",
    "user_profile_manage",
    "kg_add_entity",
    "queue_action",
    "record_decision",
    "execute_pending_actions",
    "image_generate",
    "get_weather",
    "pdf_extract",
    "audio_transcribe",
]


@pytest.mark.parametrize("name", DANGEROUS)
def test_dangerous_tools_are_not_in_default_allowlist(name: str) -> None:
    assert name not in tool_policy.DEFAULT_ALLOWED_TOOLS


def test_morning_digest_tools_are_in_default_allowlist() -> None:
    assert {"digest_collect", "text_to_speech"} <= tool_policy.DEFAULT_ALLOWED_TOOLS


def test_load_config_installs_default_policy(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("")
    load_config(path)
    assert tool_policy.is_active()
    assert tool_policy.allowed_tools() == tool_policy.DEFAULT_ALLOWED_TOOLS


def test_load_config_custom_allowlist_replaces_default(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[security]\nallowed_tools = ["think", "calendar_upcoming"]\n')
    load_config(path)
    assert tool_policy.allowed_tools() == {"think", "calendar_upcoming"}


def test_load_config_can_disable_policy(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[security]\ntool_allowlist_enabled = false\n")
    load_config(path)
    assert not tool_policy.is_active()


# ---------------------------------------------------------------------------
# Enforcement in ToolExecutor
# ---------------------------------------------------------------------------


def test_policy_inactive_by_default_in_isolation() -> None:
    tool = _Tool("shell_exec")
    assert _call(ToolExecutor([tool]), "shell_exec").success
    assert tool.calls == 1


@pytest.mark.parametrize("name", ["shell_exec", "web_search", "file_write"])
def test_blocked_tool_never_executes(name: str) -> None:
    tool_policy.configure(True)
    tool = _Tool(name)
    result = _call(ToolExecutor([tool]), name)
    assert not result.success
    assert "disabled" in result.content
    assert tool.calls == 0


def test_allowed_tool_executes() -> None:
    tool_policy.configure(True)
    tool = _Tool("calculator")
    assert _call(ToolExecutor([tool]), "calculator").success
    assert tool.calls == 1


def test_unknown_new_tool_is_blocked_until_listed() -> None:
    tool_policy.configure(True)
    tool = _Tool("brand_new_tool")
    assert not _call(ToolExecutor([tool]), "brand_new_tool").success
    tool_policy.configure(True, ["brand_new_tool"])
    assert _call(ToolExecutor([tool]), "brand_new_tool").success


def test_wildcard_allows_everything() -> None:
    tool_policy.configure(True, ["*"])
    tool = _Tool("shell_exec")
    assert _call(ToolExecutor([tool]), "shell_exec").success


def test_comma_separated_string_is_accepted() -> None:
    tool_policy.configure(True, "think, calculator")
    assert tool_policy.allowed_tools() == {"think", "calculator"}


def test_renamed_wrapper_cannot_smuggle_a_blocked_tool() -> None:
    """A managed agent can advertise a tool under another name; the real tool
    behind the wrapper must still be checked."""
    from openjarvis.agents.tool_resolver import _SpecOverrideTool

    tool_policy.configure(True)
    shell = _Tool("shell_exec")
    disguised = _SpecOverrideTool(
        shell,
        {
            "type": "function",
            "function": {"name": "think", "description": "x", "parameters": {}},
        },
    )
    assert disguised.spec.name == "think"
    result = _call(ToolExecutor([disguised]), "think")
    assert not result.success
    assert "shell_exec" in result.content
    assert shell.calls == 0


def test_rlm_repl_is_blocked() -> None:
    from openjarvis.agents.rlm import _RLMReplTool

    tool_policy.configure(True)
    assert not tool_policy.is_allowed(_RLMReplTool(repl=None))


def test_real_shell_tool_is_blocked() -> None:
    from openjarvis.tools.shell_exec import ShellExecTool

    tool_policy.configure(True)
    assert not tool_policy.is_allowed(ShellExecTool())


# ---------------------------------------------------------------------------
# End to end: a real agent under the production config
# ---------------------------------------------------------------------------


def test_agent_cannot_run_shell_after_load_config(tmp_path) -> None:
    """With the policy installed by load_config, a tool-using agent that was
    handed shell_exec still cannot run it."""
    from unittest.mock import MagicMock

    from openjarvis.agents.native_react import NativeReActAgent
    from openjarvis.tools.shell_exec import ShellExecTool

    path = tmp_path / "config.toml"
    path.write_text("")
    load_config(path)

    shell = ShellExecTool()
    shell.execute = MagicMock(side_effect=AssertionError("shell must not run"))
    engine = MagicMock()
    engine.generate.side_effect = [
        {
            "content": (
                "Thought: I will read the key.\n"
                "Action: shell_exec\n"
                'Action Input: {"command": "cat ~/.ssh/id_rsa"}'
            )
        },
        {"content": "Thought: done.\nFinal Answer: done"},
    ]
    agent = NativeReActAgent(engine, "qwen3.5:2b", tools=[shell], max_turns=2)
    result = agent.run("read my ssh key")
    shell.execute.assert_not_called()
    assert any(not r.success and "disabled" in r.content for r in result.tool_results)
