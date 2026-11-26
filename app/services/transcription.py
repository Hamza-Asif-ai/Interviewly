"""Speech-to-text: local Whisper with fallback to the OpenAI Whisper API."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from ..config import settings

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """No usable transcription source was available for the audio file."""


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


def _get_ffprobe_exe() -> str:
    """Resolve an ffprobe executable, or return an empty string if unavailable."""
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe
    try:
        ffmpeg = Path(_get_ffmpeg_exe())
        sibling = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix else "ffprobe")
        if sibling.exists():
            return str(sibling)
    except RuntimeError:
        pass
    return ""


def probe_media(path: str) -> Dict[str, Any]:
    """Validate a media file with ffprobe before any expensive processing."""
    if not os.path.exists(path):
        return {"valid": False, "error": "File not found."}
    executable = _get_ffprobe_exe()
    if not executable:
        logger.warning("ffprobe unavailable; skipping media validation.")
        return {"valid": True, "skipped": True, "has_video": True, "has_audio": True}
    try:
        result = subprocess.run(
            [
                executable, "-v", "error",
                "-show_entries", "format=format_name,duration:stream=codec_type",
                "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return {
                "valid": False,
                "error": "Invalid or corrupt media file: ffprobe could not read it.",
            }
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        error = None
        if not has_video:
            error = "File has no video stream."
        elif not has_audio:
            error = "Video has no audio track."
        return {
            "valid": error is None,
            "error": error,
            "has_video": has_video,
            "has_audio": has_audio,
            "duration": float(fmt.get("duration") or 0.0),
            "format_name": str(fmt.get("format_name") or ""),
        }
    except Exception as exc:
        logger.warning("ffprobe validation failed for %s: %s", path, exc)
        return {"valid": False, "error": f"Invalid or corrupt media file: {exc}"}


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


def _transcribe_local(audio_path: str, language: str | None = None) -> tuple[str, str]:
    """Transcribe with local Whisper and return (text, detected language)."""
    _ensure_ffmpeg_on_path()
    model = _load_local_model()
    lang = (language or settings.whisper_lang or "").strip()
    kwargs = {"language": lang} if lang else {}
    result = model.transcribe(audio_path, **kwargs)
    text = (result.get("text") or "").strip()
    detected = (result.get("language") or lang or "").strip()
    return text, detected


def _transcribe_api(audio_path: str, language: str | None = None) -> str:
    """Transcribe using the OpenAI Whisper API."""
    from .llm import get_client

    client = get_client()
    lang = (language or settings.whisper_lang or "").strip()
    kwargs = {"language": lang} if lang else {}
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            **kwargs,
        )
    return getattr(response, "text", "") or ""


def translate_to_english(text: str, source_lang: str = "") -> str:
    """Translate a non-English transcript into clean English for analysis."""
    if not text or not text.strip():
        return text
    lang = (source_lang or "").strip()
    if lang.lower().startswith("en"):
        return text
    if not settings.has_llm():
        logger.warning("No LLM configured; returning transcript in source language.")
        return text

    from .llm import chat_completion

    label = lang or "Urdu"
    prompt = (
        f"The following text was transcribed from {label} speech by Whisper and "
        "may contain misheard or garbled words. Using context, understand what "
        "the speaker actually meant and translate it into clean, natural, "
        "grammatically correct English. Return ONLY the English translation, "
        f"nothing else. Text: {text}"
    )
    try:
        translated = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=settings.prep_max_tokens,
        )
        translated = (translated or "").strip()
        return translated or text
    except Exception as exc:
        logger.warning("Translation from %s failed (%s); using original.", label, exc)
        return text


def urdu_to_english(text: str) -> str:
    return translate_to_english(text, "ur")


def transcribe_audio(audio_path: str, language: str | None = None) -> str:
    """Transcribe audio, translate non-English output, and return English text."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start = time.perf_counter()
    text = ""
    detected_language = (language or settings.whisper_lang or "").strip()
    if not settings.whisper_api_enabled:
        try:
            text, local_language = _transcribe_local(audio_path, language)
            detected_language = detected_language or local_language
            if not text:
                logger.warning("Local Whisper returned empty transcript.")
        except Exception as exc:
            logger.warning("Local Whisper failed (%s).", exc)

    if not text and settings.has_llm():
        try:
            text = _transcribe_api(audio_path, language)
        except Exception as exc:
            logger.warning("Whisper API failed (%s).", exc)

    if not text:
        logger.warning("No working transcription source for %s.", audio_path)
        logger.info("Whisper transcription took %.2fs", time.perf_counter() - start)
        raise TranscriptionError(
            "Transcription unavailable: no Whisper model or API could transcribe the audio."
        )

    has_urdu_script = any("\u0600" <= ch <= "\u06ff" for ch in text)
    if detected_language and not detected_language.lower().startswith("en"):
        text = translate_to_english(text, detected_language)
    elif has_urdu_script:
        text = translate_to_english(text, "ur")
    logger.info("Whisper transcription took %.2fs", time.perf_counter() - start)
    return text
