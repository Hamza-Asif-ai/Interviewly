"""Regression tests for prep generation quality, parsing and PDF export (section F)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from app import config  # noqa: E402
from app.services import cv_parser, pdf_export, prep_engine  # noqa: E402


def _guide(technical: int = 5, behavioral: int = 3, tips: int = 5, gaps: int = 5):
    return {
        "technical_questions": [
            {"question": f"TQ{i}", "answer": f"STAR answer {i}"}
            for i in range(technical)
        ],
        "behavioral_questions": [
            {"question": f"BQ{i}", "answer": f"STAR answer {i}"}
            for i in range(behavioral)
        ],
        "positioning_tips": [f"Tip {i}" for i in range(tips)],
        "skill_gaps": [f"Gap {i}" for i in range(gaps)],
    }


def test_validate_guide_uses_pydantic_exact_counts() -> None:
    assert prep_engine.validate_guide(_guide()) == _guide()
    with pytest.raises(ValueError, match="4 technical"):
        prep_engine.validate_guide(_guide(technical=4))


def test_validate_guide_rejects_placeholders() -> None:
    guide = _guide()
    guide["technical_questions"][0]["answer"] = "Unable to generate an answer"
    with pytest.raises(ValueError, match="placeholder"):
        prep_engine.validate_guide(guide)


def test_retrieval_context_is_budgeted(monkeypatch) -> None:
    engine = prep_engine.PrepEngine("budget")
    docs = [f"chunk {i} " + "x" * 200 for i in range(50)]

    class FakeCollection:
        def query(self, **kwargs):
            return {
                "documents": [docs],
                "metadatas": [[{"source": "cv"} for _ in docs]],
            }

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
    monkeypatch.setattr(config.settings, "rag_max_chunks", 50)
    context = engine._retrieve_all("query", source="cv")
    assert len(context) <= 600
    assert "chunk 0" in context


def test_docx_tables_are_extracted(tmp_path) -> None:
    from docx import Document

    path = tmp_path / "cv.docx"
    doc = Document()
    doc.add_paragraph("Summary paragraph")
    table = doc.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Table skill"
    doc.save(path)
    text = cv_parser.extract_text_from_file(str(path))
    assert "Summary paragraph" in text
    assert "Table skill" in text


def test_legacy_doc_has_clear_error(tmp_path) -> None:
    path = tmp_path / "cv.doc"
    path.write_bytes(b"old binary")
    with pytest.raises(cv_parser.CVParseError, match="save as .docx or .pdf"):
        cv_parser.extract_text_from_file(str(path))


def test_whitespace_only_cv_is_rejected(tmp_path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("   \n\t  ", encoding="utf-8")
    with pytest.raises(cv_parser.CVParseError, match="No extractable text"):
        cv_parser.extract_text_from_file(str(path))


def test_text_inputs_are_capped() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import prep

    app = FastAPI()
    app.include_router(prep.router)
    with TestClient(app) as client:
        response = client.post(
            "/prep/generate",
            files={"cv_file": ("cv.txt", b"CV", "text/plain")},
            data={"job_description": "x" * (config.settings.max_input_chars + 1)},
        )
    assert response.status_code == 422


def test_pdf_export_handles_unicode_and_rtl(tmp_path) -> None:
    guide = _guide()
    text = "مہارت: Python — triglyceride"
    guide["positioning_tips"][0] = text
    shaped = pdf_export.sanitize_pdf_text(text)
    assert any(0xFB50 <= ord(ch) <= 0xFEFF for ch in shaped)
    assert "triglyceride" in shaped
    path = pdf_export.build_prep_guide_pdf(guide, str(tmp_path))
    assert path.endswith(".pdf")
    assert len(open(path, "rb").read()) > 1000
