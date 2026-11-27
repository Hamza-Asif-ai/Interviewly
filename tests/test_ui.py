"""UI and demo-data consistency tests (section H)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import pytest

sys.path.insert(0, ".")

from app.config import settings  # noqa: E402
from app.ui import gradio_app  # noqa: E402


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_demo_json_matches_current_schemas() -> None:
    prep = _load("data/prep_guide_fallback.json")
    assert len(prep["technical_questions"]) == 5
    assert len(prep["behavioral_questions"]) == 3
    assert len(prep["positioning_tips"]) == 5
    assert len(prep["skill_gaps"]) == 5
    assert prep["scoring_mode"] == "demo"

    practice = _load("data/practice_scorecard_fallback.json")
    text = practice["scorecard"]["text_metrics"]
    speech = practice["scorecard"]["speech_metrics"]
    assert "words_per_minute" in text
    assert "pause_count" in speech
    assert practice["review"]["scoring_mode"] == "demo"
    assert practice["transcript"].strip()
    assert practice["_fast"] is True


def test_metrics_table_flattens_nested_values_and_skips_none() -> None:
    rows = gradio_app._metrics_table(
        {
            "text_metrics": {
                "word_count": 10,
                "filler_word_counts": {"um": 4, "uh": None},
            },
            "speech_metrics": {"pause_count": None},
            "visual_metrics": {},
            "visual_score": None,
        }
    )
    flat = [[str(cell) for cell in row] for row in rows]
    assert ["Text", "Filler Word Counts: Um", "4"] in flat
    assert not any("None" in row for row in flat)
    assert ["Score", "Visual", "Not available"] in flat


def test_transcript_direction_and_demo_banner() -> None:
    rendered = gradio_app._render_practice(
        {"visual_score": 8.0},
        {"strengths": ["Clear answer"]},
        "یعنی میں نے پروجیکٹ کیا",
        demo=True,
    )
    review_md = rendered[5]
    transcript_md = rendered[6]
    assert "DEMO DATA" in review_md
    assert "dir='rtl'" in transcript_md


def test_prep_handler_rejects_oversized_cv_text(tmp_path, monkeypatch) -> None:
    cv_path = tmp_path / "cv.txt"
    cv_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        gradio_app,
        "extract_text_from_file",
        lambda _path: "x" * (settings.max_input_chars + 1),
    )

    with pytest.raises(gr.Error, match="Extracted CV text exceeds"):
        gradio_app._generate_prep_handler(str(cv_path), "Backend engineer", "", False)
