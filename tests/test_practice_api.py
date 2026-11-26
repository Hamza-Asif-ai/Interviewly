"""API-level tests for practice upload validation and error mapping."""

from __future__ import annotations

import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, ".")

from app.database import Base, get_db  # noqa: E402
from app.routers import practice  # noqa: E402
from app.services.transcription import TranscriptionError  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    async def fake_save(file, extension):
        path = tmp_path / f"video{extension}"
        path.write_bytes(b"fake")
        return str(path)

    def fake_extract(video_path, audio_path):
        with open(audio_path, "wb") as fh:
            fh.write(b"RIFF")

    async def fake_probe(audio_path):
        return 60.0

    async def fake_analysis(transcript, audio_path, video_path, duration, question):
        return {}, {}, {}

    async def fake_review(scorecard):
        return {}

    monkeypatch.setattr(practice, "_save_video", fake_save)
    monkeypatch.setattr(
        practice,
        "probe_media",
        lambda path: {
            "valid": True,
            "has_video": True,
            "has_audio": True,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    )
    monkeypatch.setattr(practice, "extract_audio_from_video", fake_extract)
    monkeypatch.setattr(practice, "_probe_duration", fake_probe)
    monkeypatch.setattr(practice, "transcribe_audio", lambda path: "A long enough transcript.")
    monkeypatch.setattr(practice, "_run_analysis", fake_analysis)
    monkeypatch.setattr(practice, "build_scorecard", lambda *args: {"overall_score": 7.0})
    monkeypatch.setattr(practice, "analyze_report_with_llm", fake_review)

    app = FastAPI()
    app.include_router(practice.router)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _upload(client):
    return client.post(
        "/practice/upload",
        files={"video_file": ("test.mp4", b"fake", "video/mp4")},
        data={"session_id": "0", "question_prompt": ""},
    )


def test_short_transcript_is_rejected(client, monkeypatch) -> None:
    monkeypatch.setattr(practice, "transcribe_audio", lambda path: "hello")
    response = _upload(client)
    assert response.status_code == 422
    assert "No speech detected" in response.json()["detail"]


def test_unavailable_transcription_returns_503(client, monkeypatch) -> None:
    def fail(path):
        raise TranscriptionError("No working transcription source.")

    monkeypatch.setattr(practice, "transcribe_audio", fail)
    response = _upload(client)
    assert response.status_code == 503
    assert "Transcription unavailable" in response.json()["detail"]
