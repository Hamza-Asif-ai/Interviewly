"""Speech-to-text: local Whisper with fallback to the OpenAI Whisper API."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None

FALLBACK_STT_PROMPT = """SAMPLE_TRANSCRIPT_RESPONSE_FOR_TESTING"""

_FFMPEG_EXE: str | None = None


def _get_ffmpeg_exe() -> str:
    """Resolve an ffmpeg executable, falling back to imageio-ffmpeg's bundle."""
    global _FFMPEG_EXE
    if _FFMPEG_EXE is not None:
        return _FFMPEG_EXE
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _FFMPEG_EXE = system_ffmpeg
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        _FFMPEG_EXE = bundled
        logger.info("Using bundled imageio-ffmpeg binary: %s", bundled)
        return bundled
    except Exception as exc:
        _FFMPEG_EXE = ""
        raise RuntimeError(
            f"ffmpeg not found. Install it (apt-get install ffmpeg) and retry: {exc}"
        ) from exc


def ffmpeg_available() -> bool:
    """Return True if an ffmpeg executable can be resolved (system or bundled)."""
    try:
        return bool(_get_ffmpeg_exe())
    except RuntimeError:
        return False


def extract_audio_from_video(video_path: str, target_wav: str) -> str:
    """Extract 16 kHz mono audio from a video file using ffmpeg.

    Raises RuntimeError if ffmpeg is unavailable or the extraction fails.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found. Install it (apt-get install ffmpeg) and retry."
        )
    Path(target_wav).parent.mkdir(parents=True, exist_ok=True)
    command = [
        _get_ffmpeg_exe(),
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        target_wav,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0 or not os.path.exists(target_wav):
        logger.error("ffmpeg failed: %s", result.stderr[-2000:])
        raise RuntimeError("Failed to extract audio with ffmpeg.")
    return target_wav


def _load_local_model():
    """Load and cache the local Whisper model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper

        _WHISPER_MODEL = whisper.load_model(settings.whisper_model)
    return _WHISPER_MODEL


def _ensure_ffmpeg_on_path() -> None:
    """Make an ffmpeg executable name discoverable by subprocess consumers.

    openai-whisper 20231117 invokes "ffmpeg" by literal name, so when the system
    binary is missing we stage imageio-ffmpeg's bundled binary as `ffmpeg.exe`
    in a local dir and prepend that dir to PATH.
    """
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        staging_dir = Path(settings.upload_dir) / "ffmpeg_bin"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / "ffmpeg.exe"
        if not staged.exists():
            import shutil as _shutil

            _shutil.copyfile(bundled, staged)

        os.environ["AUDIO_TOOLS_FFMPEG"] = str(staged)
        bindir = str(staging_dir)
        if bindir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not stage bundled ffmpeg: %s", exc)


def _transcribe_local(audio_path: str) -> str:
    """Transcribe using local openai-whisper."""
    _ensure_ffmpeg_on_path()  # whisper reads AUDIO_TOOLS_FFMPEG at import time
    model = _load_local_model()
    result = model.transcribe(audio_path, language="ur")
    return (result.get("text") or "").strip()


def _transcribe_api(audio_path: str) -> str:
    """Transcribe using the OpenAI Whisper API."""
    from .llm import get_client

    client = get_client()
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return getattr(response, "text", "") or ""


def urdu_to_english(text: str) -> str:
    """Translate Urdu script text to clean English using the configured LLM.

    Whisper transcription of Urdu speech is often garbled, so the LLM is asked
    to understand the intended meaning from context and produce a clean,
    natural English translation. Falls back to the original text if the LLM is
    unavailable or fails.
    """
    if not text or not text.strip():
        return text
    if not settings.has_llm():
        logger.warning("No LLM configured; returning Urdu script as-is.")
        return text

    from .llm import chat_completion

    prompt = (
        "The following text was transcribed from an Urdu speech recording by "
        "Whisper and may contain misheard or garbled words. Using context, "
        "understand what the speaker actually meant and translate it into clean, "
        "natural, grammatically correct English. Return ONLY the English "
        f"translation, nothing else. Text: {text}"
    )
    try:
        translated = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
        )
        translated = (translated or "").strip()
        return translated or text
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Urdu->English translation failed (%s); using original.", exc)
        return text


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text.

    Uses local Whisper by default; falls back to the OpenAI Whisper API
    when the local model is unavailable. When no model source is available,
    returns an empty transcript instead of crashing.

    The Whisper output (Urdu script) is post-processed into clean English via
    the configured LLM when available.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    text = ""
    if not settings.whisper_api_enabled:
        try:
            text = _transcribe_local(audio_path)
            if not text:
                logger.warning("Local Whisper returned empty transcript.")
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Local Whisper failed (%s).", exc)

    if not text and settings.has_llm():
        try:
            text = _transcribe_api(audio_path)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Whisper API failed (%s).", exc)

    if not text:
        logger.warning("No working transcription source; returning empty transcript.")
        return ""

    return urdu_to_english(text)