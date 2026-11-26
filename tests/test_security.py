"""Security and privacy regression tests (section D)."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, ".")

from app.database import Base, get_db  # noqa: E402
from app.routers import practice, prep  # noqa: E402
from app.services import pdf_export, prep_engine  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    counter = {"n": 0}

    async def fake_save(file, extension):
        counter["n"] += 1
        path = tmp_path / f"video{counter['n']}{extension}"
        path.write_bytes(b"fake")
        return str(path)

    async def fake_prep_save(file, extension):
        counter["n"] += 1
        path = tmp_path / f"cv{counter['n']}{extension}"
        path.write_bytes(b"cv")
        return str(path)

    monkeypatch.setattr(practice, "_save_video", fake_save)
    monkeypatch.setattr(prep, "_save_upload", fake_prep_save)
    monkeypatch.setattr(prep, "extract_text_from_file", lambda path: "CV text")
    monkeypatch.setattr(prep_engine.PrepEngine, "index_documents", lambda self, *a: None)
    monkeypatch.setattr(prep_engine.PrepEngine, "cleanup", lambda self: None)
    monkeypatch.setattr(
        prep_engine,
        "load_fallback_guide",
        lambda: {
            "technical_questions": [{"question": "Q", "answer": "A"}],
            "behavioral_questions": [{"question": "Q", "answer": "A"}],
            "positioning_tips": ["tip"],
            "skill_gaps": ["gap"],
            "_fallback": True,
        },
    )
    monkeypatch.setattr(
        practice,
        "probe_media",
        lambda path: {"valid": True, "has_video": True, "has_audio": True,
                      "format_name": "mov,mp4"},
    )
    monkeypatch.setattr(practice, "extract_audio_from_video", lambda v, a: None)

    async def fake_probe(audio_path):
        return 60.0

    async def fake_analysis(*args):
        return {}, {}, {}

    async def fake_review(scorecard):
        return {}

    monkeypatch.setattr(practice, "_probe_duration", fake_probe)
    monkeypatch.setattr(practice, "transcribe_audio", lambda path: "A long enough transcript.")
    monkeypatch.setattr(practice, "_run_analysis", fake_analysis)
    monkeypatch.setattr(practice, "build_scorecard", lambda *a: {"overall_score": 7.0})
    monkeypatch.setattr(practice, "analyze_report_with_llm", fake_review)

    app = FastAPI()
    app.include_router(prep.router)
    app.include_router(practice.router)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_prep_guide_requires_access_token(client) -> None:
    created = client.post(
        "/prep/generate",
        files={"cv_file": ("cv.txt", b"CV", "text/plain")},
        data={"job_description": "Backend engineer", "fast_mode": "true"},
    )
    assert created.status_code == 200
    guide_id = created.json()["id"]
    token = created.json().get("access_token")
    assert token and len(token) >= 20

    assert client.get(f"/prep/{guide_id}").status_code == 404
    assert client.get(
        f"/prep/{guide_id}", headers={"X-Access-Token": "wrong"}
    ).status_code == 404
    ok = client.get(
        f"/prep/{guide_id}", headers={"X-Access-Token": token}
    )
    assert ok.status_code == 200
    assert "access_token" not in ok.json()


def test_practice_report_requires_access_token(client) -> None:
    created = client.post(
        "/practice/upload",
        files={"video_file": ("test.mp4", b"fake", "video/mp4")},
        data={"question_prompt": ""},
    )
    assert created.status_code == 200
    video_id = created.json()["video_id"]
    token = created.json().get("access_token")
    assert token and len(token) >= 20

    assert client.get(f"/practice/{video_id}/report").status_code == 404
    ok = client.get(
        f"/practice/{video_id}/report", headers={"X-Access-Token": token}
    )
    assert ok.status_code == 200
    assert "access_token" not in ok.json()


def test_session_id_form_field_is_ignored(client) -> None:
    response = client.post(
        "/practice/upload",
        files={"video_file": ("test.mp4", b"fake", "video/mp4")},
        data={"session_id": "999999", "question_prompt": ""},
    )
    assert response.status_code == 200


def test_chroma_collection_is_ephemeral(monkeypatch) -> None:
    called = {}

    class FakeEphemeral:
        def get_or_create_collection(self, name):
            called["name"] = name
            return object()

    monkeypatch.setattr(prep_engine.chromadb, "EphemeralClient", FakeEphemeral)
    engine = prep_engine.PrepEngine("session")
    engine._collection()
    assert called["name"] == engine.collection_name


def test_old_pdfs_are_removed(tmp_path) -> None:
    stale = tmp_path / "prep_guide_stale.pdf"
    fresh = tmp_path / "prep_guide_fresh.pdf"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    old_time = time.time() - 49 * 3600
    os.utime(stale, (old_time, old_time))
    pdf_export.cleanup_old_pdfs(str(tmp_path), ttl_hours=48)
    assert not stale.exists()
    assert fresh.exists()


def test_cors_origins_are_env_configurable(monkeypatch) -> None:
    from app import config

    monkeypatch.setenv("CORS_ORIGINS", "https://example.com,https://app.example.com")
    settings = config.Settings()
    assert settings.cors_origins == [
        "https://example.com", "https://app.example.com"
    ]
