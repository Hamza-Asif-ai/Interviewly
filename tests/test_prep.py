"""Tests for the Prep Engine (CV parsing, chunking, guide generation)."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import pytest

sys.path.insert(0, ".")

from app.services.cv_parser import CVParseError, extract_text_from_file  # noqa: E402
from app.services.pdf_export import (  # noqa: E402
    build_prep_guide_pdf,
    sanitize_pdf_text,
)
from app.services.prep_engine import (  # noqa: E402
    PREP_SYSTEM_PROMPT,
    PrepEngine,
    _chunk_text,
    _facts_sheet,
    load_fallback_guide,
)


class TestChunkText:
    """Tests for text chunking."""

    def test_short_text_single_chunk(self) -> None:
        assert _chunk_text("hello world") == ["hello world"]

    def test_long_text_is_split(self) -> None:
        text = "word " * 5000
        chunks = _chunk_text(text, chunk_size=1200, overlap=150)
        assert len(chunks) > 1
        assert all(len(c) <= 1200 for c in chunks)

    def test_empty_text_returns_empty(self) -> None:
        assert _chunk_text("   ") == []


class TestCVParser:
    """Tests for CV text extraction."""

    def test_extract_txt(self, tmp_path) -> None:
        cv = tmp_path / "cv.txt"
        cv.write_text("Experienced Python developer with FastAPI skills.", encoding="utf-8")
        assert "FastAPI" in extract_text_from_file(str(cv))

    def test_unsupported_extension(self, tmp_path) -> None:
        bad = tmp_path / "cv.rst"
        bad.write_text("x", encoding="utf-8")
        with pytest.raises(CVParseError):
            extract_text_from_file(str(bad))

    def test_missing_file(self) -> None:
        with pytest.raises(CVParseError):
            extract_text_from_file("does_not_exist.pdf")


class TestPrepEngine:
    """Tests for the generation pipeline (offline mode)."""

    def test_system_prompt_forbids_hallucination(self) -> None:
        rules = [
            "You must ONLY use the information explicitly provided",
            "Do NOT invent, assume, or hallucinate any experience, tools, metrics, or projects",
            "I don't have direct experience with this, but I would approach it by",
            "mention it as a skill gap",
            "Do not over-exaggerate",
            "Do not mix up the dates of different jobs",
            "Use the exact dates provided in the CV for each specific role",
            "If a date is missing, do not invent it",
            "A certification is proof of learning",
        ]
        for rule in rules:
            assert rule in PREP_SYSTEM_PROMPT
        assert "5 technical questions, 3 behavioral questions" in PREP_SYSTEM_PROMPT
        assert "at most 3 sentences" in PREP_SYSTEM_PROMPT

    def test_fallback_guide_json_shape(self) -> None:
        guide = load_fallback_guide()
        assert len(guide["technical_questions"]) == 5
        assert len(guide["behavioral_questions"]) == 3
        assert guide.get("_fallback") is True
        for key in ("positioning_tips", "skill_gaps"):
            assert isinstance(guide[key], list) and guide[key]

    def test_facts_sheet_labels_sections(self) -> None:
        cv = (
            "Faizan Asif\n"
            "Education\n"
            "BBA in Business Analytics\n"
            "Skills\n"
            "Excel, Communication\n"
            "Certification\n"
            "SEO training\n"
            "Experience\n"
            "Customer Service at TCS\n"
        )
        sheet = _facts_sheet(cv)
        assert "[SECTION: EDUCATION]" in sheet
        assert "[SECTION: SKILLS]" in sheet
        assert "[SECTION: CERTIFICATIONS]" in sheet
        assert "[SECTION: EXPERIENCE]" in sheet

    def test_facts_sheet_splits_crammed_experience_rows(self) -> None:
        cv = (
            "Experience\n"
            "Trax Courier Company InternshipatCodeAlpha\n"
            "CustomerSrviceDept.(3Months) FrontendDeveloper\n"
            "June2023-August2023 August2024\n"
        )
        sheet = _facts_sheet(cv)
        assert "- Trax Courier Company" in sheet
        assert "Internship" in sheet
        assert "CustomerSrviceDept.(3Months)" in sheet
        assert "FrontendDeveloper" in sheet
        assert "August2024" in sheet
        assert "Trax Courier Company || Internship at CodeAlpha" in sheet
        assert "CustomerSrviceDept.(3Months) || FrontendDeveloper" in sheet
        assert "June2023-August2023 || August2024" in sheet

    def test_fallback_guide_shape(self) -> None:
        ctx = "Some context text"
        guide = PrepEngine._fallback_guide(ctx)
        assert len(guide["technical_questions"]) == 5
        assert len(guide["behavioral_questions"]) == 3
        assert len(guide["positioning_tips"]) == 5
        assert isinstance(guide["skill_gaps"], list)

    def test_fallback_answers_are_star(self) -> None:
        guide = PrepEngine._fallback_guide("ctx")
        answer = guide["technical_questions"][0]["answer"]
        assert "S:" in answer and "R:" in answer

    def test_index_and_generate_offline(self, tmp_path) -> None:
        """Index documents and generate a guide without an LLM API key."""
        try:
            session_id = uuid.uuid4().hex[:8]
            engine = PrepEngine(session_id)
            engine.index_documents(
                "Software engineer with expertise in Python, FastAPI and machine learning.",
                "We need a backend engineer who knows FastAPI and SQL.",
                "A startup building AI-powered interview tools.",
            )
            guide = asyncio.run(engine.generate())
        except Exception as exc:  # model download / chroma unavailable offline
            pytest.skip(f"Skipping index test: {exc}")

        assert isinstance(guide, dict)
        assert guide.get("_fallback") is True
        assert "technical_questions" in guide

    def test_collection_is_session_scoped(self) -> None:
        engine = PrepEngine("some_session")
        assert engine.collection_name.startswith("prep_some_session_")


class TestPDFExport:
    """Tests for prep-guide PDF rendering."""

    def _guide(self) -> dict:
        return {
            "technical_questions": [
                {
                    "question": "How would you design a FastAPI service?",
                    "answer": "S: We scaled \u2014 fast. T: Ship on time. A: I rebuilt it. R: 40% faster",
                }
            ],
            "behavioral_questions": [
                {
                    "question": "Tell me about a tough deadline.",
                    "answer": "S: Crash week \u2014 T: \u201cMUST\u201d ship. A: Cut scope. R: Shipped.",
                }
            ],
            "positioning_tips": [
                "Point at the RAG experience \u2014 it matches.",
                "Emoji \U0001f600 and star \u2605 characters must not crash the render.",
            ],
            "skill_gaps": ["\u00c9l\u00e9ments Docker are missing \u2026"],
        }

    def test_sanitize_keeps_latin1_within_bounds(self) -> None:
        text = "caf\u00e9 \u266f \u2014 \u2018quotes\u2019 \u2022 bullet \u2605 \U0001f600"
        out = sanitize_pdf_text(text)
        assert "caf" in out
        assert all(ord(ch) <= 0xFF for ch in out)

    def test_sanitize_maps_smart_punctuation(self) -> None:
        assert sanitize_pdf_text("\u201cquoted\u201d") == '"quoted"'
        assert sanitize_pdf_text("a \u2013 b") == "a - b"
        assert sanitize_pdf_text("\u2026") == "..."

    def test_build_pdf_creates_nonempty_file(self, tmp_path) -> None:
        out = tmp_path / "pdfs"
        path = build_prep_guide_pdf(self._guide(), str(out))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 500
        assert path.startswith(str(out))

    def test_build_pdf_with_hostile_unicode(self, tmp_path) -> None:
        guide = self._guide()
        guide["positioning_tips"].append("emoji \U0001f680 rocket and \u2603 snowman")
        path = build_prep_guide_pdf(guide, str(tmp_path))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 500