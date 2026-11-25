"""Tests for the local-only inference policy (openjarvis.security.local_only).

These prove the policy is enforced in code: cloud engines cannot be created,
discovered or selected, cloud models are rejected by the server, and API keys
in the environment do not re-enable any of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openjarvis.core.config import JarvisConfig, load_config
from openjarvis.core.registry import EngineRegistry
from openjarvis.security import local_only
from openjarvis.security.local_only import CloudInferenceBlocked

CLOUD_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture
def cloud_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a shell that has cloud provider API keys exported."""
    for name in CLOUD_KEYS:
        monkeypatch.setenv(name, "sk-test-not-a-real-key")
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test-not-a-real-key")


# ---------------------------------------------------------------------------
# Policy switch
# ---------------------------------------------------------------------------


def test_policy_is_on_by_default_in_config() -> None:
    assert JarvisConfig().security.local_only_inference is True
    assert local_only.is_enabled(JarvisConfig()) is True


def test_policy_can_only_be_turned_off_explicitly() -> None:
    cfg = JarvisConfig()
    cfg.security.local_only_inference = False
    assert local_only.is_enabled(cfg) is False


def test_env_var_turns_policy_on_even_if_config_says_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = JarvisConfig()
    cfg.security.local_only_inference = False
    monkeypatch.setenv("OPENJARVIS_LOCAL_ONLY", "1")
    assert local_only.is_enabled(cfg) is True


def test_enable_is_a_one_way_latch() -> None:
    cfg = JarvisConfig()
    cfg.security.local_only_inference = False
    local_only.enable()
    assert local_only.is_enabled() is True
    assert local_only.is_enabled(cfg) is True


def test_load_config_latches_policy(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[engine]\ndefault = "ollama"\n')
    assert local_only.is_enabled() is False
    cfg = load_config(path)
    assert cfg.security.local_only_inference is True
    assert local_only.is_enabled() is True


def test_load_config_respects_explicit_opt_out(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[security]\nlocal_only_inference = false\n")
    cfg = load_config(path)
    assert cfg.security.local_only_inference is False
    assert local_only.is_enabled() is False


# ---------------------------------------------------------------------------
# Loopback detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://127.0.0.2:8000/v1",
        "http://[::1]:11434",
        "http://0.0.0.0:11434",
        "localhost:11434",
        "http://LOCALHOST:11434",
    ],
)
def test_loopback_urls_are_local(url: str) -> None:
    assert local_only.is_loopback_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://integrate.api.nvidia.com",
        "http://192.168.1.20:11434",
        "http://100.64.0.5:11434",  # a Tailscale address is still remote
        "http://localhost.evil.com:11434",
        "http://127.0.0.1.nip.io:11434",
        "",
        None,
    ],
)
def test_remote_urls_are_not_local(url) -> None:
    assert not local_only.is_loopback_url(url)


# ---------------------------------------------------------------------------
# Engine checks
# ---------------------------------------------------------------------------


def test_local_ollama_engine_is_accepted() -> None:
    engine = SimpleNamespace(is_cloud=False, _host="http://localhost:11434")
    local_only.check_engine("ollama", engine, JarvisConfig())


def test_remote_ollama_host_is_rejected() -> None:
    engine = SimpleNamespace(is_cloud=False, _host="http://10.0.0.7:11434")
    with pytest.raises(CloudInferenceBlocked, match="non-local host"):
        local_only.check_engine("ollama", engine, JarvisConfig())


@pytest.mark.parametrize("key", ["cloud", "litellm", "nim", "brand-new-engine"])
def test_non_allowlisted_engine_keys_are_rejected(key: str) -> None:
    with pytest.raises(CloudInferenceBlocked):
        local_only.check_engine_key(key, JarvisConfig())


def test_engine_flagged_cloud_is_rejected_even_with_local_key() -> None:
    engine = SimpleNamespace(is_cloud=True, _host="http://localhost:11434")
    with pytest.raises(CloudInferenceBlocked, match="cloud engine"):
        local_only.check_engine("ollama", engine, JarvisConfig())


def test_checks_are_noops_when_policy_is_off() -> None:
    cfg = JarvisConfig()
    cfg.security.local_only_inference = False
    local_only.check_engine_key("cloud", cfg)
    local_only.check_engine("cloud", SimpleNamespace(is_cloud=True), cfg)
    local_only.ensure_cloud_allowed(config=cfg)


# ---------------------------------------------------------------------------
# Cloud engine constructors
# ---------------------------------------------------------------------------


def test_cloud_engine_cannot_be_constructed(cloud_keys) -> None:
    from openjarvis.engine.cloud import CloudEngine

    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        CloudEngine()


def test_litellm_engine_cannot_be_constructed(cloud_keys) -> None:
    litellm_mod = pytest.importorskip("openjarvis.engine.litellm")
    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        litellm_mod.LiteLLMEngine(default_model="gpt-4o")


def test_nim_cloud_host_cannot_be_constructed(cloud_keys) -> None:
    from openjarvis.engine.nim import NIMEngine

    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        NIMEngine()


def test_nim_on_localhost_is_still_constructible() -> None:
    from openjarvis.engine.nim import NIMEngine

    local_only.enable()
    NIMEngine(host="http://localhost:8000")


# ---------------------------------------------------------------------------
# Discovery and selection
# ---------------------------------------------------------------------------


class _HealthyRemote:
    """A registered engine that is healthy and claims to serve any model."""

    engine_id = "fake-remote"
    is_cloud = True

    def __init__(self, *a, **k) -> None:
        self._host = "https://api.example.com"

    def health(self) -> bool:
        return True

    def list_models(self):
        return ["gpt-4o"]

    def can_serve(self, model: str) -> bool:
        return True


class _HealthyLocal(_HealthyRemote):
    engine_id = "ollama"
    is_cloud = False

    def __init__(self, host: str | None = None, *a, **k) -> None:
        self._host = host or "http://localhost:11434"

    def list_models(self):
        return ["qwen3.5:2b"]


def _register_fakes() -> None:
    EngineRegistry.register_value("ollama", _HealthyLocal)
    EngineRegistry.register_value("cloud", _HealthyRemote)
    EngineRegistry.register_value("litellm", _HealthyRemote)


def test_discovery_never_returns_cloud_engines(cloud_keys) -> None:
    from openjarvis.engine._discovery import discover_engines

    _register_fakes()
    keys = [key for key, _ in discover_engines(JarvisConfig())]
    assert keys == ["ollama"]


def test_discovery_still_includes_cloud_when_policy_is_off(cloud_keys) -> None:
    from openjarvis.engine._discovery import discover_engines

    _register_fakes()
    cfg = JarvisConfig()
    cfg.security.local_only_inference = False
    keys = {key for key, _ in discover_engines(cfg)}
    assert {"ollama", "cloud", "litellm"} <= keys


def test_explicit_cloud_engine_request_is_refused(cloud_keys) -> None:
    from openjarvis.engine._discovery import get_engine

    _register_fakes()
    assert get_engine(JarvisConfig(), "cloud") is None


def test_fallback_never_crosses_to_cloud_when_ollama_is_down(cloud_keys) -> None:
    from openjarvis.engine._discovery import get_engine

    class _DownLocal(_HealthyLocal):
        def health(self) -> bool:
            return False

    EngineRegistry.register_value("ollama", _DownLocal)
    EngineRegistry.register_value("cloud", _HealthyRemote)
    cfg = JarvisConfig()
    cfg.engine.default = "ollama"
    assert get_engine(cfg, model="gpt-4o") is None


def test_ollama_pointed_at_remote_host_is_not_discovered() -> None:
    from openjarvis.engine._discovery import discover_engines

    EngineRegistry.register_value("ollama", _HealthyLocal)
    cfg = JarvisConfig()
    cfg.engine.ollama_host = "http://192.168.1.50:11434"
    assert discover_engines(cfg) == []


# ---------------------------------------------------------------------------
# Server: direct cloud router and HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_cloud_is_blocked(cloud_keys) -> None:
    from openjarvis.core.types import Message, Role
    from openjarvis.server.cloud_router import stream_cloud

    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        async for _ in stream_cloud("gpt-4o", [Message(role=Role.USER, content="hi")]):
            pass


def _client(config: JarvisConfig):
    pytest.importorskip("fastapi")
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from openjarvis.server.app import create_app

    engine = MagicMock()
    engine.engine_id = "ollama"
    engine.list_models.return_value = ["qwen3.5:2b"]
    engine.health.return_value = True
    engine.generate.return_value = {"content": "local answer", "usage": {}}
    config.analytics.enabled = False
    config.traces.enabled = False
    config.security.enabled = False
    app = create_app(engine, "qwen3.5:2b", engine_name="ollama", config=config)
    return TestClient(app), engine


@pytest.mark.parametrize(
    "model", ["gpt-4o", "claude-sonnet-4-5", "gemini-2.5-pro", "openrouter/auto"]
)
@pytest.mark.parametrize("stream", [False, True])
def test_chat_endpoint_rejects_cloud_models(cloud_keys, model: str, stream) -> None:
    client, engine = _client(JarvisConfig())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "stream": stream,
            "messages": [{"role": "user", "content": "summarise my inbox"}],
        },
    )
    assert resp.status_code == 403
    assert "local-only" in resp.json()["detail"]
    engine.generate.assert_not_called()


def test_chat_endpoint_still_serves_local_model() -> None:
    client, engine = _client(JarvisConfig())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5:2b",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200, resp.text


def test_cloud_reload_endpoint_is_refused(cloud_keys) -> None:
    client, _ = _client(JarvisConfig())
    resp = client.post("/v1/cloud/reload", json={"keys": {"OPENAI_API_KEY": "x"}})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Other paths that would send personal content off the machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("openjarvis.speech.openai_tts", "OpenAITTSBackend"),
        ("openjarvis.speech.elevenlabs_tts", "ElevenLabsTTSBackend"),
        ("openjarvis.speech.cartesia_tts", "CartesiaTTSBackend"),
        ("openjarvis.speech.openai_whisper", "OpenAIWhisperBackend"),
        ("openjarvis.speech.deepgram", "DeepgramSpeechBackend"),
    ],
)
def test_cloud_speech_backends_are_blocked(cloud_keys, module: str, cls: str) -> None:
    import importlib

    backend_cls = getattr(importlib.import_module(module), cls)
    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        backend_cls(api_key="test-key")


def test_tts_discovery_skips_blocked_cloud_backends(cloud_keys) -> None:
    import openjarvis.speech  # noqa: F401  (registers backends)
    from openjarvis.core.registry import TTSRegistry
    from openjarvis.speech._tts_discovery import get_tts_backend
    from openjarvis.speech.openai_tts import OpenAITTSBackend

    TTSRegistry.register_value("openai_tts", OpenAITTSBackend)
    local_only.enable()
    backend = get_tts_backend("openai_tts")
    assert not isinstance(backend, OpenAITTSBackend)


def test_claude_code_agent_is_blocked() -> None:
    from unittest.mock import MagicMock

    from openjarvis.agents.claude_code import ClaudeCodeAgent

    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        ClaudeCodeAgent(MagicMock(), "claude-sonnet-4-5")


def test_hybrid_local_cloud_agent_is_blocked() -> None:
    from unittest.mock import MagicMock

    from openjarvis.agents.hybrid._base import LocalCloudAgent

    class _Concrete(LocalCloudAgent):
        def _run_paradigm(self, *a, **k):  # pragma: no cover
            raise AssertionError("must not run")

    local_only.enable()
    with pytest.raises(CloudInferenceBlocked):
        _Concrete(MagicMock(), "claude-sonnet-4-5")


def test_external_analytics_is_off_under_local_only() -> None:
    from openjarvis.analytics.identity import is_analytics_enabled
    from openjarvis.core.config import AnalyticsConfig

    cfg = AnalyticsConfig(enabled=True)
    assert is_analytics_enabled(cfg) is True
    local_only.enable()
    assert is_analytics_enabled(cfg) is False
