"""Native macOS text-to-speech backend using the built-in say command."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_DEFAULT_VOICE = "Samantha"
_DEFAULT_RATE = 175

#: Output formats. ``say`` writes AIFF; others are converted with macOS's
#: built-in ``afconvert``. WAV plays in every browser; M4A (AAC) is small and
#: plays on iPhone, in Safari and in Shortcuts.
_AFCONVERT_ARGS = {
    "wav": ["-f", "WAVE", "-d", "LEI16"],
    "m4a": ["-f", "m4af", "-d", "aac"],
}
_FORMATS = {"aiff", "aif", *_AFCONVERT_ARGS}


def _convert(source: Path, output_format: str) -> Path:
    """Convert an AIFF file with macOS's afconvert."""
    afconvert = shutil.which("afconvert")
    if not afconvert:
        raise RuntimeError("macOS 'afconvert' command was not found.")
    target = source.with_suffix(f".{output_format}")
    completed = subprocess.run(
        [afconvert, *_AFCONVERT_ARGS[output_format], str(source), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"Audio conversion to {output_format} failed: {completed.stderr.strip()}"
        )
    return target


@TTSRegistry.register("macos_say")
class MacOSSayBackend(TTSBackend):
    """Local text-to-speech using Apple's native speech synthesizer."""

    backend_id = "macos_say"

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = _DEFAULT_VOICE,
        speed: float = 1.0,
        output_format: str = "aiff",
    ) -> TTSResult:

        if sys.platform != "darwin":
            raise RuntimeError("macos_say is only available on macOS.")

        say_path = shutil.which("say")
        if not say_path:
            raise RuntimeError("macOS 'say' command was not found.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        if speed <= 0:
            raise ValueError("Speech speed must be greater than zero.")

        output_format = output_format.lower()
        if output_format not in _FORMATS:
            raise ValueError(
                f"macos_say supports {', '.join(sorted(_FORMATS))} output."
            )

        voice = voice_id or _DEFAULT_VOICE
        rate = round(_DEFAULT_RATE * speed)

        with tempfile.TemporaryDirectory(prefix="jarvis-say-") as temp_dir:
            audio_path = Path(temp_dir) / "speech.aiff"

            completed = subprocess.run(
                [
                    say_path,
                    "-v", voice,
                    "-r", str(rate),
                    "-o", str(audio_path),
                    text,
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    f"macOS speech synthesis failed: "
                    f"{completed.stderr.strip()}"
                )

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise RuntimeError("macOS generated an empty audio file.")

            if output_format in _AFCONVERT_ARGS:
                audio_path = _convert(audio_path, output_format)

            audio = audio_path.read_bytes()

        estimated_duration = (
            len(text.split()) * 60.0 / rate
        )

        return TTSResult(
            audio=audio,
            format=output_format,
            duration_seconds=estimated_duration,
            voice_id=voice,
            metadata={
                "backend": "macos_say",
                "local": True,
                "rate": rate,
            },
        )

    def available_voices(self) -> List[str]:
        say_path = shutil.which("say")
        if not say_path:
            return []

        try:
            result = subprocess.run(
                [say_path, "-v", "?"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return [
                line.split()[0]
                for line in result.stdout.splitlines()
                if line.strip()
            ]
        except (subprocess.SubprocessError, OSError):
            return []

    def health(self) -> bool:
        return sys.platform == "darwin" and shutil.which("say") is not None
