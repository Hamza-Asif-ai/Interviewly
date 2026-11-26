"""Regression tests for resilient video input handling (section C)."""

from __future__ import annotations

import subprocess
import sys

import cv2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, ".")

from app.database import Base, get_db  # noqa: E402
from app.routers import practice  # noqa: E402
from app.services import video_analysis  # noqa: E402
from app.services.transcription import _get_ffmpeg_exe  # noqa: E402


def _make_webm(path) -> None:
    command = [
        _get_ffmpeg_exe(), "-y", "-f", "lavfi",
        "-i", "testsrc=duration=2:size=64x48:rate=5",
        "-c:v", "libvpx", "-b:v", "200k", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _patched_capture(monkeypatch, *, frame_count=-1.0, fps=0.0):
    real_capture = cv2.VideoCapture

    class CaptureProxy:
        def __init__(self, capture):
            self._capture = capture

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return frame_count
            if property_id == cv2.CAP_PROP_FPS:
                return fps
            return self._capture.get(property_id)

        def __getattr__(self, name):
            return getattr(self._capture, name)

    def factory(path):
        return CaptureProxy(real_capture(path))

    monkeypatch.setattr(cv2, "VideoCapture", factory)


class TestWebMWithoutDuration:
    def test_negative_frame_count_and_zero_fps_still_sample(self, tmp_path, monkeypatch) -> None:
        webm = tmp_path / "no_duration.webm"
        _make_webm(webm)
        _patched_capture(monkeypatch, frame_count=-1.0, fps=0.0)
        metrics = video_analysis.analyze_video(str(webm))
        assert metrics["frames_analyzed"] >= 1

    @pytest.mark.parametrize("fps", [1000.0, float("nan")])
    def test_implausible_fps_is_ignored(self, tmp_path, monkeypatch, fps) -> None:
        video = tmp_path / "odd_fps.mp4"
        command = [
            _get_ffmpeg_exe(), "-y", "-f", "lavfi",
            "-i", "testsrc=duration=2:size=64x48:rate=5", str(video),
        ]
        assert subprocess.run(command, capture_output=True).returncode == 0
        _patched_capture(monkeypatch, frame_count=10, fps=fps)
        metrics = video_analysis.analyze_video(str(video))
        assert metrics["frames_analyzed"] >= 1


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    async def fake_save(file, extension):
        path = tmp_path / f"video{extension}"
        path.write_bytes(b"fake")
        return str(path)

    monkeypatch.setattr(practice, "_save_video", fake_save)
    monkeypatch.setattr(
        practice, "probe_media", lambda path: {
            "valid": True, "has_video": True, "has_audio": True,
            "duration": 2.0, "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        }
    )

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


@pytest.mark.parametrize(
    ("probe", "message"),
    [
        ({"valid": False, "error": "corrupt"}, "corrupt"),
        ({"valid": True, "has_video": True, "has_audio": False}, "no audio"),
        (
            {"valid": True, "has_video": True, "has_audio": True,
             "format_name": "image2"},
            "extension",
        ),
    ],
)
def test_bad_media_is_rejected_before_analysis(client, monkeypatch, probe, message) -> None:
    monkeypatch.setattr(practice, "probe_media", lambda path: probe)
    response = _upload(client)
    assert response.status_code == 422
    assert message in response.json()["detail"].lower()
