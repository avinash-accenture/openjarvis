"""Local speech-to-text with whisper.cpp (no onnxruntime, works on Intel macOS 12).

``faster-whisper`` depends on onnxruntime, which no longer publishes wheels
for Intel Macs older than macOS 13. whisper.cpp is a native binary that runs
well on CPU. Audio is converted to 16 kHz mono WAV first, with ``ffmpeg`` if
installed, otherwise macOS's built-in ``afconvert`` (which reads iPhone/Safari
AAC recordings but not Chrome's WebM).

Setup on the Mac: build or install whisper.cpp (it provides ``whisper-cli``)
and download a model such as ``ggml-base.en.bin`` into
``~/.openjarvis/models/``. See docs/personal/voice.md.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import SpeechBackend, TranscriptionResult

_BINARY_NAMES = ("whisper-cli", "whisper-cpp", "whisper")
_DEFAULT_MODEL = "ggml-base.en.bin"


def sniff_format(audio: bytes, hint: str = "") -> str:
    """Audio container from its first bytes (browsers mislabel uploads)."""
    head = audio[:16]
    if head.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    if audio[4:8] == b"ftyp":
        return "m4a"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"FORM") and audio[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return "mp3"
    return (hint or "wav").lower().lstrip(".")


def find_binary(configured: str = "") -> Optional[str]:
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.exists() else shutil.which(configured)
    for name in _BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def default_model_path() -> Path:
    return get_config_dir() / "models" / _DEFAULT_MODEL


def to_wav16k(source: Path, target: Path) -> None:
    """Convert any supported recording to 16 kHz mono 16-bit WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    else:
        afconvert = shutil.which("afconvert")
        if not afconvert:
            raise RuntimeError("Install ffmpeg to convert recordings for whisper.cpp.")
        if source.suffix == ".webm" or source.suffix == ".ogg":
            raise RuntimeError(
                "This browser records WebM/Ogg, which afconvert cannot read. "
                "Install ffmpeg (brew install ffmpeg) or use Safari."
            )
        cmd = [
            afconvert,
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            str(source),
            str(target),
        ]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if done.returncode != 0 or not target.exists():
        raise RuntimeError(f"Audio conversion failed: {done.stderr.strip()[:300]}")


@SpeechRegistry.register("whisper_cpp")
class WhisperCppBackend(SpeechBackend):
    """Speech-to-text through a local whisper.cpp binary."""

    backend_id = "whisper_cpp"

    def __init__(
        self, binary: str = "", model_path: str = "", threads: int = 4
    ) -> None:
        self._binary = find_binary(binary)
        self._model = (
            Path(model_path).expanduser() if model_path else default_model_path()
        )
        self._threads = max(1, int(threads))

    def health(self) -> bool:
        return bool(self._binary) and self._model.is_file()

    def supported_formats(self) -> List[str]:
        return ["wav", "m4a", "mp4", "aac", "aiff", "caf", "mp3", "webm", "ogg"]

    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if not self._binary:
            raise RuntimeError("whisper.cpp (whisper-cli) is not installed.")
        if not self._model.is_file():
            raise RuntimeError(f"whisper.cpp model not found at {self._model}.")
        if not audio:
            raise ValueError("No audio received.")

        with tempfile.TemporaryDirectory(prefix="jarvis-stt-") as tmp:
            source = Path(tmp) / f"input.{sniff_format(audio, format)}"
            source.write_bytes(audio)
            wav = Path(tmp) / "audio16k.wav"
            to_wav16k(source, wav)
            out_base = Path(tmp) / "transcript"
            lang = language or ("en" if ".en." in self._model.name else "auto")
            cmd = [
                self._binary,
                "-m",
                str(self._model),
                "-f",
                str(wav),
                "-l",
                lang,
                "-t",
                str(self._threads),
                "-nt",
                "-np",
                "-otxt",
                "-of",
                str(out_base),
            ]
            done = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False
            )
            text_file = out_base.with_suffix(".txt")
            if done.returncode != 0 or not text_file.exists():
                raise RuntimeError(
                    f"whisper.cpp failed: {(done.stderr or done.stdout).strip()[:300]}"
                )
            text = " ".join(text_file.read_text(errors="replace").split())
            duration = max(0.0, (wav.stat().st_size - 44) / (16000 * 2))
        return TranscriptionResult(
            text=text,
            language=lang if lang != "auto" else None,
            duration_seconds=duration,
        )
