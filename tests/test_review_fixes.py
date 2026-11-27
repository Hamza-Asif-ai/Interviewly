"""Independent-review regression tests: each item is proved by a named test."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, ".")

from app import config, main  # noqa: E402
from app.database import Base, get_db  # noqa: E402
from app.models import (  # noqa: E402
    AnalysisReport,
    PrepGuide,
    PracticeVideo,
    Session as DBSession,
)
from app.routers import prep as prep_router  # noqa: E402
from app.services import (  # noqa: E402
    cv_parser,
    pdf_export,
    prep_engine,
    scoring,
    text_analysis,
    transcription,
    video_analysis,
)
from app.services.audio_analysis import analyze_audio  # noqa: E402
from app.services.text_analysis import _count_filler_words  # noqa: E402
from app.services.transcription import (  # noqa: E402
    TranscriptionError,
    transcribe_audio,
)
from app.ui import gradio_app  # noqa: E402


VALID_GUIDE = json.loads(
    Path("data/prep_guide_fallback.json").read_text(encoding="utf-8")
)


def _valid_guide() -> dict:
    return {
        "technical_questions": [
            {"question": f"Technical question {i}", "answer": f"STAR answer {i}"}
            for i in range(5)
        ],
        "behavioral_questions": [
            {"question": f"Behavioral question {i}", "answer": f"STAR answer {i}"}
            for i in range(3)
        ],
        "positioning_tips": [f"Positioning tip {i}" for i in range(5)],
        "skill_gaps": [f"Skill gap {i}" for i in range(5)],
    }


def _rate_limit_app():
    """A tiny FastAPI app that reuses the production guard middleware."""
    test_app = FastAPI()
    test_app.middleware("http")(main.guard_requests)

    @test_app.get("/")
    def ui():
        return {"ok": True}

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    @test_app.post("/prep/generate")
    def prep():
        return {"ok": True}

    @test_app.post("/gradio_api/run/_generate_prep_handler")
    def gradio_run():
        return {"ok": True}

    @test_app.post("/gradio_api/queue/join")
    def gradio_queue():
        return {"ok": True}

    return test_app


class TestItem1RateLimiting:
    def test_loading_ui_never_returns_429(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "rate_limit_per_minute", 2)
        main._request_windows.clear()
        with TestClient(_rate_limit_app()) as client:
            responses = [client.get("/") for _ in range(12)]
            responses += [client.get("/health") for _ in range(12)]
        assert all(response.status_code != 429 for response in responses)

    def test_only_expensive_actions_consume_the_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "rate_limit_per_minute", 2)
        main._request_windows.clear()
        with TestClient(_rate_limit_app()) as client:
            assert [client.get("/").status_code for _ in range(5)] == [200] * 5
            codes = [
                client.post("/gradio_api/queue/join", json={}).status_code
                for _ in range(5)
            ]
        assert codes[:2] == [200, 200]
        assert codes[2:] == [429, 429, 429]

    def test_forwarded_ip_only_trusted_when_peer_is_a_proxy(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "rate_limit_per_minute", 1)
        main._request_windows.clear()
        monkeypatch.setattr(main.settings, "trusted_proxy_ips", ["testclient"])
        with TestClient(_rate_limit_app()) as client:
            first = client.post(
                "/prep/generate", headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
            )
            second = client.post(
                "/prep/generate", headers={"X-Forwarded-For": "203.0.113.8"}
            )
        assert first.status_code == 200
        assert second.status_code == 200

        main._request_windows.clear()
        monkeypatch.setattr(main.settings, "trusted_proxy_ips", [])
        with TestClient(_rate_limit_app()) as client:
            first = client.post(
                "/prep/generate", headers={"X-Forwarded-For": "203.0.113.7"}
            )
            second = client.post(
                "/prep/generate", headers={"X-Forwarded-For": "203.0.113.8"}
            )
        assert first.status_code == 200
        assert second.status_code == 429

    def test_gradio_handlers_respect_the_shared_analysis_cap(self, monkeypatch) -> None:
        class FullSlots:
            def try_acquire(self):
                return False

            def release(self):
                raise AssertionError("release must not run when no slot was acquired")

        monkeypatch.setattr(gradio_app, "analysis_slots", FullSlots())
        cv_path = Path("data/prep_guide_fallback.json")
        with pytest.raises(gr.Error, match="Server busy"):
            gradio_app._generate_prep_handler(str(cv_path), "JD", "", False)


class TestItem2StarHeuristic:
    def test_word_boundaries_and_meaningful_star_evidence(self) -> None:
        check = text_analysis._simple_star_check
        assert check("i") == 0.0
        assert check("a pig in a big field") == 0.0
        assert check("i was happy when the result") <= 5.0
        strong = (
            "When I joined the payments team the reconciliation project was late. "
            "My responsibility was to rebuild the pipeline. I implemented the new "
            "job and led the rollout. As a result we reduced errors by 40%."
        )
        assert check(strong) >= 7.5


class TestItem3OriginalAndTranslation:
    def test_transcription_keeps_both_texts_and_metrics_use_original(
        self, monkeypatch, tmp_path
    ) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        original = "یعنی میں نے پروجیکٹ کیا اور نتیجہ بہتر ہوا"
        translation = "I led the project and improved the result"
        seen: dict = {}

        def fake_translate(text, lang=""):
            seen["source"] = text
            return translation

        monkeypatch.setattr(transcription, "_ensure_ffmpeg_on_path", lambda: None)
        monkeypatch.setattr(
            transcription,
            "_transcribe_local",
            lambda path, language=None: (original, "ur"),
        )
        monkeypatch.setattr(transcription, "translate_to_english", fake_translate)

        result = transcribe_audio(str(audio))
        assert result.original == original
        assert result.translation == translation
        assert result.display_text == translation
        assert seen["source"] == original

        scored: dict = {}

        async def fake_llm_scores(text, question_prompt):
            scored["text"] = text
            return 8.0, 9.0, 7.0, "llm"

        monkeypatch.setattr(text_analysis, "_llm_scores", fake_llm_scores)
        metrics = asyncio.run(
            text_analysis.analyze_text(result.original, 60, "Tell me about a project",
                                       scoring_text=result.display_text)
        )
        assert scored["text"] == translation
        assert metrics["word_count"] == len(original.split())
        assert metrics["words_per_minute"] == round(len(original.split()), 1)

    def test_translation_prompt_does_not_ask_for_cleanup(self) -> None:
        source = Path("app/services/transcription.py").read_text(encoding="utf-8")
        assert "clean, natural" not in source
        assert "faithfully" in source.lower()


class TestItem4GradioPrepTransaction:
    def test_generate_precedes_db_writes_and_uses_uuid_engine_id(
        self, monkeypatch, tmp_path
    ) -> None:
        events: list[tuple] = []

        class FakeEngine:
            def __init__(self, session_id):
                events.append(("engine", session_id))

            def index_documents(self, *args):
                events.append(("index",))

            async def generate(self):
                events.append(("generate",))
                return _valid_guide()

            def cleanup(self):
                events.append(("cleanup",))

        class FakeDB:
            def add(self, _obj):
                events.append(("add",))

            def flush(self):
                events.append(("flush",))

            def commit(self):
                events.append(("commit",))

            def close(self):
                events.append(("close",))

        monkeypatch.setattr(gradio_app, "PrepEngine", FakeEngine)
        monkeypatch.setattr(gradio_app, "SessionLocal", FakeDB)
        monkeypatch.setattr(gradio_app, "extract_text_from_file", lambda path: "CV text")
        monkeypatch.setattr(gradio_app, "build_prep_guide_pdf", lambda *a, **k: None)
        cv_path = tmp_path / "cv.txt"
        cv_path.write_text("CV text", encoding="utf-8")

        output = gradio_app._generate_prep_handler(str(cv_path), "Backend JD", "", False)
        assert output[0].startswith("# Interview Preparation Guide")
        assert events[0][0] == "engine"
        uuid.UUID(events[0][1])
        assert [name for name, *_ in events].index("generate") < [
            name for name, *_ in events
        ].index("add")
        assert sum(name == "commit" for name, *_ in events) == 1


class TestItem5NoSpeech:
    def test_empty_output_from_working_engine_is_422(self, monkeypatch) -> None:
        monkeypatch.setattr(
            transcription, "_transcribe_local", lambda path, language=None: ("", "en")
        )
        monkeypatch.setattr(config.settings, "whisper_api_enabled", False)
        with pytest.raises(transcription.NoSpeechError, match="No speech detected"):
            transcribe_audio(str(Path("data/prep_guide_fallback.json")))

    def test_all_engines_failing_is_503(self, monkeypatch) -> None:
        def fail(*args, **kwargs):
            raise RuntimeError("engine exploded")

        monkeypatch.setattr(transcription, "_transcribe_local", fail)
        monkeypatch.setattr(transcription, "_transcribe_api", fail)
        monkeypatch.setattr(config.settings, "whisper_api_enabled", False)
        monkeypatch.setattr(config.settings, "has_llm", lambda: True)
        with pytest.raises(TranscriptionError):
            transcribe_audio(str(Path("data/prep_guide_fallback.json")))

    def test_gradio_shows_no_speech_and_deletes_video(self, monkeypatch, tmp_path) -> None:
        video = tmp_path / "practice.mp4"
        video.write_bytes(b"video")
        monkeypatch.setattr(gradio_app, "extract_audio_from_video", lambda v, a: None)
        monkeypatch.setattr(gradio_app, "_probe_duration", lambda path: 3.0)
        monkeypatch.setattr(
            gradio_app,
            "transcribe_audio",
            lambda path, lang="": SimpleNamespace(original="", translation="", language="en"),
        )
        with pytest.raises(gr.Error, match="No speech detected"):
            gradio_app._analyze_practice_handler(str(video), "Tell me about yourself", "Auto", False)
        assert not video.exists()


class TestItem6Privacy:
    def test_old_database_records_and_media_are_deleted(self, tmp_path) -> None:
        from app import retention

        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        db = factory()
        old_time = time.time() - 10 * 86400
        stale_media = tmp_path / "old.mp4"
        fresh_media = tmp_path / "new.mp4"
        stale_media.write_bytes(b"old")
        fresh_media.write_bytes(b"new")

        old_session = DBSession(title="old")
        new_session = DBSession(title="new")
        db.add_all([old_session, new_session])
        db.flush()
        old_guide = PrepGuide(
            session_id=old_session.id, cv_filename="cv.pdf", job_description="JD",
            guide_json="{}", created_at=datetime.utcfromtimestamp(old_time),
        )
        old_video = PracticeVideo(
            session_id=old_session.id, filename="old.mp4", stored_path=str(stale_media),
            audio_path=str(tmp_path / "old.wav"), created_at=datetime.utcfromtimestamp(old_time),
        )
        db.add_all([old_guide, old_video])
        db.flush()
        db.add(
            AnalysisReport(
                video_id=old_video.id, transcript="t", report_json="{}",
                created_at=datetime.utcfromtimestamp(old_time),
            )
        )
        db.add(
            PrepGuide(
                session_id=new_session.id, cv_filename="cv.pdf", job_description="JD",
                guide_json="{}",
            )
        )
        db.commit()
        db.close()

        removed = retention.cleanup_old_records(session_factory=factory, hours=24)
        assert removed["prep_guides"] == 1
        assert removed["practice_videos"] == 1
        assert not stale_media.exists()
        assert fresh_media.exists()
        check = factory()
        assert check.query(DBSession).count() == 1
        check.close()

    def test_pdf_cleanup_uses_configured_ttl(self, tmp_path, monkeypatch) -> None:
        guide = json.loads(json.dumps(VALID_GUIDE))
        stale = tmp_path / "prep_guide_stale.pdf"
        stale.write_bytes(b"old")
        old_time = time.time() - 100 * 3600
        os.utime(stale, (old_time, old_time))
        monkeypatch.setattr(pdf_export.settings, "pdf_ttl_hours", 1)
        pdf_export.build_prep_guide_pdf(guide, str(tmp_path))
        assert not stale.exists()

    def test_fast_mode_and_validation_errors_delete_uploads(self, monkeypatch, tmp_path) -> None:
        cv = tmp_path / "cv.txt"
        cv.write_text("CV", encoding="utf-8")
        monkeypatch.setattr(gradio_app, "build_prep_guide_pdf", lambda *a, **k: None)
        gradio_app._generate_prep_handler(str(cv), "JD", "", True)
        assert not cv.exists()

        cv2 = tmp_path / "cv2.txt"
        cv2.write_text("CV", encoding="utf-8")
        with pytest.raises(gr.Error, match="job description"):
            gradio_app._generate_prep_handler(str(cv2), "", "", False)
        assert not cv2.exists()

        video = tmp_path / "demo.mp4"
        video.write_bytes(b"video")
        gradio_app._analyze_practice_handler(str(video), "Q", "Auto", True)
        assert not video.exists()

    def test_footer_and_readme_are_truthful(self) -> None:
        footer = gradio_app.FOOTER.lower()
        assert "Nothing is stored long-term" not in gradio_app.FOOTER
        assert "llm provider" in footer
        assert "retained" in footer
        readme = Path("README.md").read_text(encoding="utf-8")
        assert "Noto Naskh" not in readme
        assert "fallback JSON" not in readme.replace("no silent fallback JSON", "")
        assert "LLM provider" in readme


class TestItem7GuideValidation:
    def test_real_answer_with_cant_and_extra_tip_gap_is_accepted(self) -> None:
        guide = _valid_guide()
        guide["technical_questions"][0]["answer"] = (
            "I can't claim sole credit, but I designed the service and shipped it."
        )
        guide["positioning_tips"] = guide["positioning_tips"][:4]
        guide["skill_gaps"] = guide["skill_gaps"] + ["One more gap"]
        assert prep_engine.validate_guide(guide)

    def test_tbd_inside_a_real_answer_is_not_a_placeholder(self) -> None:
        guide = _valid_guide()
        guide["behavioral_questions"][0]["answer"] = (
            "I negotiated the TBDs in the contract and closed the gap."
        )
        assert prep_engine.validate_guide(guide)

    def test_bare_tbd_is_still_rejected(self) -> None:
        guide = _valid_guide()
        guide["positioning_tips"][0] = "TBD"
        with pytest.raises(ValueError, match="placeholder"):
            prep_engine.validate_guide(guide)


class TestItem8UIData:
    def test_metrics_table_has_no_duplicate_header_row(self) -> None:
        rows = gradio_app._metrics_table(
            {"text_metrics": {"word_count": 10}, "speech_metrics": {}, "visual_metrics": {}}
        )
        assert rows[0] != ["Category", "Metric", "Value"]
        assert not any(row == ["Category", "Metric", "Value"] for row in rows)

    def test_transcript_is_escaped_without_stray_chevron(self) -> None:
        rendered = gradio_app._render_practice(
            {"visual_score": 7.0}, {"strengths": ["ok"]},
            "<script>alert('x')</script> I led the project", demo=False,
        )
        transcript_md = rendered[6]
        assert "&lt;script&gt;" in transcript_md
        assert "<script>" not in transcript_md
        assert ">>" not in transcript_md

    def test_demo_scorecard_matches_scorecard_builder_and_transcript(self) -> None:
        data = json.loads(
            Path("data/practice_scorecard_fallback.json").read_text(encoding="utf-8")
        )
        card = data["scorecard"]
        expected = scoring.build_scorecard(
            card["text_metrics"], card["speech_metrics"], card["visual_metrics"]
        )
        for key in ("content_score", "speech_score", "visual_score", "overall_score"):
            assert card[key] == expected[key], key

        transcript = data["transcript"]
        text = card["text_metrics"]
        speech = card["speech_metrics"]
        assert text["word_count"] == len(transcript.split())
        assert text["words_per_minute"] == speech["words_per_minute"]
        assert text["words_per_minute"] == round(
            text["word_count"] / (data["duration_seconds"] / 60), 1
        )
        _, counts = _count_filler_words(transcript)
        for key, value in text["filler_word_counts"].items():
            assert counts.get(key, 0) == value, key
        assert text["filler_word_count"] == sum(text["filler_word_counts"].values())
        assert text["filler_words_per_minute"] == round(
            text["filler_word_count"] / (data["duration_seconds"] / 60), 2
        )


class TestItem9FaceFrames:
    def test_visual_score_uses_real_face_frame_count(self) -> None:
        card = scoring.build_scorecard(
            {"star_score": 8.0}, {},
            {"face_detection_percent": 100.0, "eye_contact_percent": 80.0,
             "posture_stability_score": 8.0, "head_movement_score": 8.0,
             "frames_analyzed": 50, "face_frames": 0},
        )
        assert card["visual_score"] is None
        card2 = scoring.build_scorecard(
            {"star_score": 8.0}, {},
            {"face_detection_percent": 100.0, "eye_contact_percent": 80.0,
             "posture_stability_score": 8.0, "head_movement_score": 8.0,
             "frames_analyzed": 1, "face_frames": 20},
        )
        assert card2["visual_score"] is not None

    def test_missing_mediapipe_reports_analysis_unavailable(self, tmp_path, monkeypatch) -> None:
        import cv2
        import numpy as np

        video = tmp_path / "blank.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        monkeypatch.setitem(sys.modules, "mediapipe", None)
        metrics = video_analysis.analyze_video(str(video))
        assert metrics["face_frames"] == 0
        assert metrics["analysis_available"] is False
        rendered = gradio_app._render_practice(
            {**scoring.build_scorecard({}, {}, metrics), "visual_metrics": metrics},
            {"strengths": []},
            "I led the project and improved the result",
        )
        assert "analysis unavailable" in rendered[6].lower()


class TestItem10ErrorMessages:
    def test_gradio_errors_do_not_expose_raw_exception_text(self, monkeypatch, tmp_path) -> None:
        cv = tmp_path / "cv.txt"
        cv.write_text("CV", encoding="utf-8")
        monkeypatch.setattr(
            gradio_app,
            "extract_text_from_file",
            lambda path: (_ for _ in ()).throw(RuntimeError("LEAK_ME_SECRET")),
        )
        with pytest.raises(gr.Error) as caught:
            gradio_app._generate_prep_handler(str(cv), "JD", "", False)
        assert "LEAK_ME_SECRET" not in str(caught.value)

        monkeypatch.setattr(gradio_app, "extract_text_from_file", lambda path: "CV text")
        monkeypatch.setattr(
            gradio_app.PrepEngine,
            "generate",
            lambda self: (_ for _ in ()).throw(
                prep_engine.PrepGuideError("SECRET_MODEL_ENDPOINT")),
        )
        with pytest.raises(gr.Error) as caught:
            gradio_app._generate_prep_handler(str(cv), "JD", "", False)
        assert "SECRET_MODEL_ENDPOINT" not in str(caught.value)

    def test_api_503_detail_does_not_expose_raw_exception_text(self, monkeypatch) -> None:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        app = FastAPI()
        app.include_router(prep_router.router)

        def override_get_db():
            db = factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        monkeypatch.setattr(prep_router, "extract_text_from_file", lambda path: "CV text")
        monkeypatch.setattr(
            prep_router.PrepEngine,
            "generate",
            lambda self: (_ for _ in ()).throw(
                prep_engine.PrepGuideError("SECRET_MODEL_ENDPOINT")),
        )
        with TestClient(app) as client:
            response = client.post(
                "/prep/generate",
                files={"cv_file": ("cv.txt", b"CV", "text/plain")},
                data={"job_description": "Backend engineer"},
            )
        assert response.status_code == 503
        assert "SECRET_MODEL_ENDPOINT" not in response.json()["detail"]


class TestItem11RetrievalParsingPauses:
    def test_docx_merged_cells_are_deduplicated(self, tmp_path) -> None:
        from docx import Document

        path = tmp_path / "cv.docx"
        doc = Document()
        table = doc.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Shared"
        table.cell(0, 0).merge(table.cell(0, 1))
        doc.save(path)
        assert cv_parser.extract_text_from_file(str(path)).count("Shared") == 1

    def test_retrieve_all_uses_query_and_keeps_full_job_description(self, monkeypatch) -> None:
        engine = prep_engine.PrepEngine("query")
        calls: dict = {}
        docs = [f"JD chunk {i} " + "x" * 100 for i in range(25)]
        docs[-1] += " UNIQUE_TAIL_MARKER"

        class FakeCollection:
            def query(self, **kwargs):
                calls.update(kwargs)
                return {
                    "documents": [docs],
                    "metadatas": [[{"source": "jd"} for _ in docs]],
                }

            def get(self, **kwargs):
                raise AssertionError("_retrieve_all must use the query API")

        monkeypatch.setattr(engine, "_collection", lambda: FakeCollection())
        monkeypatch.setattr(
            prep_engine,
            "get_embedder",
            lambda: SimpleNamespace(
                encode=lambda texts, **kwargs: SimpleNamespace(
                    tolist=lambda: [[0.1, 0.2] for _ in texts]
                )
            ),
        )
        monkeypatch.setattr(config.settings, "rag_context_chars", 500)
        monkeypatch.setattr(config.settings, "rag_max_chunks", 4)
        context = engine._retrieve_all("backend fastapi", source="jd", bounded=False)
        assert "query_embeddings" in calls
        assert calls["n_results"] >= len(docs)
        assert "UNIQUE_TAIL_MARKER" in context

    def test_pause_ratio_counts_only_pauses_over_0_3s(self, tmp_path) -> None:
        def wav_with_gap(name, gap):
            import numpy as np
            import soundfile as sf

            path = tmp_path / name
            sr = 16000
            y = 0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 4, sr * 4, endpoint=False))
            start, end = sr, sr + int(sr * gap)
            y[start:end] = 0.0
            sf.write(str(path), y, sr)
            return str(path)

        short = analyze_audio(wav_with_gap("short.wav", 0.2))
        long = analyze_audio(wav_with_gap("long.wav", 0.5))
        assert short["pause_ratio"] <= 0.01
        assert long["pause_ratio"] >= 0.1


class TestItem12Documentation:
    def test_readme_is_current_and_documents_new_environment(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        for stale in ("on timeout/failure → prep_guide_fallback.json", "Noto Naskh",
                      "Running **without** a key still works"):
            assert stale not in readme
        for setting in (
            "RATE_LIMIT_PER_MINUTE", "MAX_CONCURRENT_ANALYSES", "TRUSTED_PROXY_IPS",
            "RETENTION_HOURS", "RETENTION_INTERVAL_HOURS", "PDF_TTL_HOURS",
            "CORS_ORIGINS", "PDF_FONT_PATH",
        ):
            assert setting in readme, setting
