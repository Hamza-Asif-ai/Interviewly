"""Language-pipeline regression tests (section G)."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from app import config  # noqa: E402
from app.services import text_analysis, transcription  # noqa: E402


def test_urdu_fillers_are_counted(monkeypatch) -> None:
    monkeypatch.setattr(config.settings, "openai_api_key", "")
    metrics = asyncio.run(
        text_analysis.analyze_text("یعنی مطلب I led the project yaani matlab", 60)
    )
    counts = metrics["filler_word_counts"]
    assert counts["yaani"] == 2
    assert counts["matlab"] == 2


def test_language_override_is_passed_to_whisper(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    recorded: dict = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            recorded.update(kwargs)
            return {"text": "hello world", "language": "en"}

    monkeypatch.setattr(transcription, "_load_local_model", lambda: FakeModel())
    monkeypatch.setattr(transcription, "_ensure_ffmpeg_on_path", lambda: None)
    monkeypatch.setattr(transcription.settings, "whisper_api_enabled", False)
    monkeypatch.setattr(transcription.settings, "whisper_lang", "")
    result = transcription.transcribe_audio(str(audio), language="ur")
    assert recorded["language"] == "ur"
    assert result == "hello world"


def test_detected_urdu_is_translated(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return {"text": "سلام دنیا", "language": "ur"}

    monkeypatch.setattr(transcription, "_load_local_model", lambda: FakeModel())
    monkeypatch.setattr(transcription, "_ensure_ffmpeg_on_path", lambda: None)
    monkeypatch.setattr(transcription.settings, "whisper_api_enabled", False)
    monkeypatch.setattr(transcription.settings, "whisper_lang", "")
    monkeypatch.setattr(
        transcription, "translate_to_english", lambda text, lang="": "hello world"
    )
    assert transcription.transcribe_audio(str(audio)) == "hello world"


def test_bundled_ffmpeg_is_staged_with_platform_name(monkeypatch, tmp_path) -> None:
    bundled = tmp_path / "imageio_ffmpeg_binary"
    bundled.write_bytes(b"ffmpeg")
    monkeypatch.setattr(transcription.shutil, "which", lambda name: None)
    monkeypatch.setattr(transcription, "_bundled_ffmpeg_name", lambda: "ffmpeg")
    monkeypatch.setattr(config.settings, "upload_dir", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    import imageio_ffmpeg

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(bundled))
    transcription._ensure_ffmpeg_on_path()
    staged = tmp_path / "ffmpeg_bin" / "ffmpeg"
    assert staged.exists()
    assert config.settings.upload_dir in str(staged)


def test_api_rejects_unsupported_language() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import practice

    app = FastAPI()
    app.include_router(practice.router)
    with TestClient(app) as client:
        response = client.post(
            "/practice/upload",
            files={"video_file": ("test.mp4", b"fake", "video/mp4")},
            data={"language": "xx"},
        )
    assert response.status_code == 422
