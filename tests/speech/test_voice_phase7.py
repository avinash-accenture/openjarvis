"""Voice: Samantha output formats, whisper.cpp speech-to-text, /api/ask."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from openjarvis.speech import macos_say, whisper_cpp
from openjarvis.speech.whisper_cpp import WhisperCppBackend, sniff_format


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


# ---------------------------------------------------------------------------
# macOS say (Samantha): AIFF, WAV and M4A
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_macos(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    # say -v VOICE -r RATE -o OUT TEXT  -> write an AIFF-looking file
    _script(
        bindir / "say",
        f'echo "say $@" >> {log}\n'
        'while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; esac; shift; done\n'
        'printf "FORM....AIFF" > "$out"\n',
    )
    # afconvert ARGS SRC DST -> copy with a marker
    _script(
        bindir / "afconvert",
        f'echo "afconvert $@" >> {log}\n'
        "for last; do :; done\n"
        'printf "CONVERTED" > "$last"\n',
    )
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(macos_say.sys, "platform", "darwin")
    return log


def test_aiff_is_default(fake_macos) -> None:
    result = macos_say.MacOSSayBackend().synthesize("Hello", voice_id="Samantha")
    assert result.audio.startswith(b"FORM")
    assert "afconvert" not in fake_macos.read_text()


@pytest.mark.parametrize(
    ("fmt", "args"),
    [("wav", "-f WAVE -d LEI16"), ("m4a", "-f m4af -d aac")],
)
def test_wav_and_m4a_are_converted(fake_macos, fmt, args) -> None:
    result = macos_say.MacOSSayBackend().synthesize(
        "Hello", voice_id="Samantha", speed=0.72, output_format=fmt
    )
    assert result.audio == b"CONVERTED"
    log = fake_macos.read_text()
    assert "-v Samantha -r 126" in log  # 175 wpm x 0.72
    assert args in log


def test_unknown_format_is_refused(fake_macos) -> None:
    with pytest.raises(ValueError, match="m4a"):
        macos_say.MacOSSayBackend().synthesize("Hello", output_format="flac")


def test_samantha_is_the_default_server_voice() -> None:
    from openjarvis.core.config import JarvisConfig
    from openjarvis.speech._tts_discovery import TTS_BACKEND_ORDER, default_voice_for

    speech = JarvisConfig().speech
    assert (speech.tts_backend, speech.voice_id, speech.voice_speed) == (
        "macos_say",
        "Samantha",
        0.72,
    )
    assert TTS_BACKEND_ORDER[0] == "macos_say"
    assert default_voice_for("macos_say") == "Samantha"


# ---------------------------------------------------------------------------
# whisper.cpp speech-to-text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "fmt"),
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"\x1a\x45\xdf\xa3rest", "webm"),
        (b"\x00\x00\x00\x1cftypM4A ", "m4a"),
        (b"OggS\x00", "ogg"),
        (b"FORM\x00\x00\x00\x00AIFF", "aiff"),
        (b"ID3\x04", "mp3"),
    ],
)
def test_sniff_format_ignores_misleading_names(data, fmt) -> None:
    assert sniff_format(data, "webm") == fmt


@pytest.fixture
def fake_whisper(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    _script(
        bindir / "afconvert",
        f'echo "afconvert $@" >> {log}\nfor last; do :; done\n'
        'printf "RIFF0000WAVEdata" > "$last"\n',
    )
    _script(
        bindir / "whisper-cli",
        f'echo "whisper $@" >> {log}\n'
        'while [ $# -gt 0 ]; do case "$1" in -of) of="$2"; shift;; esac; shift; done\n'
        'printf "  What is on my\\n calendar tomorrow?  " > "$of.txt"\n',
    )
    monkeypatch.setenv("PATH", str(bindir))
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"model")
    return log, model


@pytest.mark.skipif(sys.platform == "win32", reason="shell script fakes")
def test_transcribes_iphone_recording_with_afconvert(fake_whisper) -> None:
    log, model = fake_whisper
    backend = WhisperCppBackend(model_path=str(model))
    assert backend.health()
    result = backend.transcribe(b"\x00\x00\x00\x1cftypM4A data", format="webm")
    assert result.text == "What is on my calendar tomorrow?"
    calls = log.read_text()
    assert "LEI16@16000 -c 1" in calls and "input.m4a" in calls
    assert "-l en" in calls and "-nt" in calls


@pytest.mark.skipif(sys.platform == "win32", reason="shell script fakes")
def test_webm_without_ffmpeg_gives_a_clear_error(fake_whisper) -> None:
    _, model = fake_whisper
    with pytest.raises(RuntimeError, match="ffmpeg"):
        WhisperCppBackend(model_path=str(model)).transcribe(b"\x1a\x45\xdf\xa3x")


def test_missing_binary_or_model_is_unhealthy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not WhisperCppBackend(model_path=str(tmp_path / "none.bin")).health()
    with pytest.raises(RuntimeError, match="not installed"):
        WhisperCppBackend().transcribe(b"RIFF")


def test_discovery_offers_whisper_cpp(fake_whisper, monkeypatch) -> None:
    from openjarvis.core.config import JarvisConfig
    from openjarvis.speech import _discovery

    _, model = fake_whisper
    cfg = JarvisConfig()
    cfg.speech.whisper_cpp_model = str(model)
    from openjarvis.core.registry import SpeechRegistry

    if not SpeechRegistry.contains("whisper_cpp"):
        SpeechRegistry.register_value("whisper_cpp", WhisperCppBackend)
    monkeypatch.setattr(_discovery, "DISCOVERY_ORDER", ["whisper_cpp"])
    backend = _discovery.get_speech_backend(cfg)
    assert isinstance(backend, WhisperCppBackend)
    assert whisper_cpp.default_model_path().name == "ggml-base.en.bin"


# ---------------------------------------------------------------------------
# /api/ask for Shortcuts, and briefing audio type
# ---------------------------------------------------------------------------


def test_spoken_text_strips_citations_markdown_and_links() -> None:
    from openjarvis.server.assistant_routes import spoken_text

    answer = (
        "**Tomorrow** you have:\n- Dentist at 2:00 PM [e1a2b3c4]\n"
        "- Email from Priya [m12345] see https://x.test/y"
    )
    assert spoken_text(answer) == (
        "Tomorrow you have:. Dentist at 2:00 PM. Email from Priya see"
    )
    assert "http" not in spoken_text(answer)
    assert "[m12345]" not in spoken_text(answer)


def _app(agent=None, key="oj_sk_" + "k" * 40):
    pytest.importorskip("fastapi")
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from openjarvis.core.config import JarvisConfig
    from openjarvis.server.app import create_app

    engine = MagicMock()
    engine.list_models.return_value = ["m"]
    cfg = JarvisConfig()
    cfg.analytics.enabled = False
    cfg.traces.enabled = False
    cfg.security.enabled = False
    app = create_app(engine, "m", agent=agent, config=cfg, api_key=key)
    return TestClient(app), {"Authorization": f"Bearer {key}"}


def test_ask_endpoint_runs_the_agent_and_needs_the_key() -> None:
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.run.return_value = MagicMock(content="Dentist at 2 PM [e1a2b3c4d].")
    client, auth = _app(agent)
    assert client.post("/api/ask", json={"question": "Tomorrow?"}).status_code == 401
    resp = client.post("/api/ask", json={"question": "Tomorrow?"}, headers=auth)
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": "Dentist at 2 PM [e1a2b3c4d].",
        "spoken": "Dentist at 2 PM.",
    }
    agent.run.assert_called_once_with("Tomorrow?")


def test_ask_rejects_empty_questions() -> None:
    from unittest.mock import MagicMock

    client, auth = _app(MagicMock())
    assert (
        client.post("/api/ask", json={"question": ""}, headers=auth).status_code == 422
    )
