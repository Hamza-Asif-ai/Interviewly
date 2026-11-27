"""Reliability and LLM-configuration regression tests (section E)."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from app import config  # noqa: E402
from app.services import llm, prep_engine  # noqa: E402


def test_fallback_models_are_current_and_env_configurable(monkeypatch) -> None:
    monkeypatch.setenv(
        "LLM_MODEL_FALLBACKS", "gemini-3.8-flash,gemini-3.6-flash"
    )
    settings = config.Settings()
    assert settings.llm_model_fallbacks == [
        "gemini-3.8-flash", "gemini-3.6-flash"
    ]
    monkeypatch.setattr(
        llm.settings, "llm_model_fallbacks", settings.llm_model_fallbacks
    )
    candidates = llm._model_candidates("primary")
    assert "gemini-3.8-flash" in candidates
    assert "gemini-2.0-flash" not in candidates
    assert "gemini-1.5-flash-latest" not in candidates


def test_total_timeout_stops_fallback_chain(monkeypatch) -> None:
    clock = iter([0.0, 0.0, 10.0, 10.0, 20.0, 20.0])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(llm, "get_client", lambda: object())
    monkeypatch.setattr(llm.settings, "llm_model_fallbacks", ["a", "b", "c"])
    monkeypatch.setattr(llm.settings, "llm_total_timeout", 5.0)
    with pytest.raises(TimeoutError):
        llm.chat_completion([{"role": "user", "content": "hi"}])


def test_async_client_is_created_per_event_loop(monkeypatch) -> None:
    instances: list[object] = []

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            instances.append(self)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(llm, "_model_candidates", lambda model: ["test-model"])
    asyncio.run(llm.achat_completion([{"role": "user", "content": "hi"}]))
    asyncio.run(llm.achat_completion([{"role": "user", "content": "hi"}]))
    assert len(instances) == 2


def test_json_repair_uses_prep_max_tokens(monkeypatch) -> None:
    captured: dict = {}

    async def fake_completion(messages, **kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(llm, "achat_completion", fake_completion)
    monkeypatch.setattr(llm.settings, "prep_max_tokens", 4321)
    assert asyncio.run(llm.extract_faithful_json("{bad json")) == {"ok": True}
    assert captured["max_tokens"] == 4321


def test_retrieval_runs_off_event_loop(monkeypatch) -> None:
    calls: list[str] = []
    original_to_thread = asyncio.to_thread

    async def tracked_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", str(func)))
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(prep_engine.asyncio, "to_thread", tracked_to_thread)
    monkeypatch.setattr(prep_engine.settings, "has_llm", lambda: False)
    engine = prep_engine.PrepEngine("test")
    monkeypatch.setattr(engine, "_retrieve_all", lambda query, source, bounded=True: "")
    with pytest.raises(prep_engine.PrepGuideError):
        asyncio.run(engine.generate())
    assert len(calls) >= 3


def test_prep_commit_happens_after_generation(tmp_path, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base, get_db
    from app.routers import prep

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    events: list[str] = []

    def override_get_db():
        db = testing_session()
        original_commit = db.commit

        def tracked_commit():
            events.append("commit")
            return original_commit()

        db.commit = tracked_commit
        try:
            yield db
        finally:
            db.close()

    async def fake_save(file, extension):
        path = tmp_path / f"cv{extension}"
        path.write_bytes(b"CV")
        return str(path)

    async def fake_generate(self):
        events.append("generate")
        return prep_engine.load_fallback_guide()

    app = FastAPI()
    app.include_router(prep.router)
    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(prep, "_save_upload", fake_save)
    monkeypatch.setattr(prep, "extract_text_from_file", lambda path: "CV text")
    monkeypatch.setattr(prep.PrepEngine, "index_documents", lambda *a, **k: None)
    monkeypatch.setattr(prep.PrepEngine, "generate", fake_generate)
    monkeypatch.setattr(prep.PrepEngine, "cleanup", lambda self: None)

    with TestClient(app) as client:
        response = client.post(
            "/prep/generate",
            files={"cv_file": ("cv.txt", b"CV", "text/plain")},
            data={"job_description": "Backend engineer"},
        )
    assert response.status_code == 200
    assert events == ["generate", "commit"]
