"""Regression tests for honest prep generation: no silent demo data (section B)."""

from __future__ import annotations

import asyncio
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fpdf import FPDF
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, ".")

from app.database import Base, get_db  # noqa: E402
from app.routers import prep as prep_router  # noqa: E402
from app.services import pdf_export  # noqa: E402
from app.services.prep_engine import (  # noqa: E402
    PrepEngine,
    PrepGuideError,
    load_fallback_guide,
)
from app.ui.gradio_app import _guide_to_markdown  # noqa: E402


@pytest.fixture
def prep_client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    monkeypatch.setattr(prep_router, "extract_text_from_file", lambda path: "CV text")
    monkeypatch.setattr(PrepEngine, "index_documents", lambda self, *args: None)
    monkeypatch.setattr(PrepEngine, "cleanup", lambda self: None)

    app = FastAPI()
    app.include_router(prep_router.router)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _post_prep(client, fast: bool = False):
    data = {"job_description": "Backend engineer", "company_overview": ""}
    if fast:
        data["fast_mode"] = "true"
    return client.post(
        "/prep/generate",
        files={"cv_file": ("cv.txt", b"CV text", "text/plain")},
        data=data,
    )


def test_normal_mode_returns_503_when_llm_unavailable(prep_client, monkeypatch) -> None:
    def fail(self):
        raise PrepGuideError("LLM unavailable: rate limited.")

    monkeypatch.setattr(PrepEngine, "generate", fail)
    response = _post_prep(prep_client)
    assert response.status_code == 503
    assert "LLM unavailable" in response.json()["detail"]
    assert "Fast Mode" in response.json()["detail"]


def test_fast_mode_returns_demo_guide(prep_client, monkeypatch) -> None:
    monkeypatch.setattr(
        PrepEngine,
        "generate",
        lambda self: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    response = _post_prep(prep_client, fast=True)
    assert response.status_code == 200
    assert response.json()["guide"]["technical_questions"]


def test_no_private_fallback_method() -> None:
    assert not hasattr(PrepEngine, "_fallback_guide")


def test_generate_raises_without_llm(monkeypatch) -> None:
    from app.services import prep_engine

    monkeypatch.setattr(prep_engine.settings, "has_llm", lambda: False)
    engine = PrepEngine("test")
    monkeypatch.setattr(engine, "_retrieve_all", lambda query, source: "")
    with pytest.raises(PrepGuideError):
        asyncio.run(engine.generate())


def test_markdown_labels_demo_data() -> None:
    markdown = _guide_to_markdown(load_fallback_guide())
    assert "DEMO DATA" in markdown
    assert "not based on your CV" in markdown


def test_pdf_labels_demo_data(tmp_path, monkeypatch) -> None:
    written: list[str] = []
    original_cell = FPDF.cell
    original_multi_cell = FPDF.multi_cell

    def cell(self, w, h, text="", *args, **kwargs):
        written.append(str(text))
        return original_cell(self, w, h, text, *args, **kwargs)

    def multi_cell(self, w, h, text="", *args, **kwargs):
        written.append(str(text))
        return original_multi_cell(self, w, h, text, *args, **kwargs)

    monkeypatch.setattr(FPDF, "cell", cell)
    monkeypatch.setattr(FPDF, "multi_cell", multi_cell)
    pdf_export.build_prep_guide_pdf(load_fallback_guide(), str(tmp_path), demo=True)
    assert any("DEMO DATA" in text for text in written)
