"""Regression tests for honest, metric-backed practice results (section B)."""

from __future__ import annotations

import asyncio
import sys

import pytest

sys.path.insert(0, ".")

from app.services import scoring, text_analysis  # noqa: E402
from app.services.audio_analysis import analyze_audio  # noqa: E402
from app.services.scoring import _compute_visual_score  # noqa: E402
from app.services.text_analysis import _count_filler_words  # noqa: E402


def _make_wav(path, duration: float = 4.0, gap: float = 0.0) -> None:
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 220 * t)
    if gap:
        start = int(sr * 1.0)
        end = min(len(y), start + int(sr * gap))
        y[start:end] = 0.0
    sf.write(path, y, sr)


class TestFillers:
    def test_like_as_verb_is_not_a_filler(self) -> None:
        total, counts = _count_filler_words("I like this project and I like Python.")
        assert total == 0
        assert counts["like"] == 0

    def test_like_as_discourse_marker_is_a_filler(self) -> None:
        total, counts = _count_filler_words(
            "Like, I basically feel, um, you know what I mean?"
        )
        assert counts["like"] == 1
        assert counts["um"] == 1
        assert counts["basically"] == 1
        assert counts["you_know"] == 1
        assert total >= 4

    def test_you_know_variants_share_one_key(self) -> None:
        _, counts = _count_filler_words("you know you_know you-know")
        assert counts["you_know"] == 3
        assert "you know" not in counts


class TestHeuristicScores:
    def test_no_fixed_relevance_or_conciseness(self, monkeypatch) -> None:
        monkeypatch.setattr(text_analysis, "settings_has_llm", lambda: False)
        star, relevance, conciseness, _mode = asyncio.run(
            text_analysis._llm_scores("Tell me about yourself.", None)
        )
        assert 0 <= star <= 10
        assert relevance is None
        assert conciseness is None

    def test_fallback_review_uses_measured_metrics(self) -> None:
        card = {
            "text_metrics": {"word_count": 120, "filler_word_count": 4,
                             "words_per_minute": 140.0},
            "speech_metrics": {"pause_ratio": 0.3, "speaking_rate_wpm": 140.0},
            "visual_metrics": {"eye_contact_percent": 75.0,
                               "face_detection_percent": 90.0},
            "content_score": 7.0,
            "speech_score": 6.0,
            "visual_score": 6.5,
            "overall_score": 6.7,
        }
        review = scoring._fallback_review(card)
        text = " ".join(review["strengths"] + review["weaknesses"] +
                        review["improvement_tips"]).lower()
        assert "120 words" in text
        assert "4 filler" in text
        assert "75%" in text

    def test_llm_failure_marks_heuristic_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(text_analysis, "settings_has_llm", lambda: True)

        def boom(*args, **kwargs):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(text_analysis, "achat_completion", boom)
        metrics = asyncio.run(
            text_analysis.analyze_text(
                "I built a FastAPI service that cut latency by 40%.", 60.0
            )
        )
        assert metrics["scoring_mode"] == "heuristic"
        assert metrics["relevance_score"] is None
        assert metrics["conciseness_score"] is None


class TestVisualScoring:
    def test_no_face_data_is_reported_as_unavailable(self) -> None:
        assert _compute_visual_score(0.0, 0.0, 0.0, 0.0, face_frames=0) is None
        assert _compute_visual_score(5.0, 0.0, 0.0, 0.0, face_frames=2) is None

    def test_visual_score_scales_with_face_presence(self) -> None:
        full = _compute_visual_score(90.0, 80.0, 8.0, 7.0, face_frames=90)
        partial = _compute_visual_score(40.0, 80.0, 8.0, 7.0, face_frames=40)
        assert full is not None and partial is not None
        assert full > partial > 0

    def test_scorecard_accepts_missing_visual_and_relevance(self) -> None:
        card = scoring.build_scorecard(
            {"word_count": 50, "words_per_minute": 120.0,
             "star_score": 5.0, "conciseness_score": 5.0},
            {},
            {"face_detection_percent": 0.0, "frames_analyzed": 0},
        )
        assert card["visual_score"] is None
        assert 0 <= card["overall_score"] <= 10


class TestAudioMetrics:
    def test_wpm_uses_transcript_word_count(self, tmp_path) -> None:
        wav = tmp_path / "speech.wav"
        _make_wav(wav, duration=4.0)
        metrics = analyze_audio(str(wav), word_count=80)
        assert metrics["words_per_minute"] == pytest.approx(1200.0, rel=0.1)

    def test_only_long_pauses_count(self, tmp_path) -> None:
        long_gap = tmp_path / "long_gap.wav"
        short_gap = tmp_path / "short_gap.wav"
        _make_wav(long_gap, gap=0.4)
        _make_wav(short_gap, gap=0.2)
        assert analyze_audio(str(long_gap))["pause_count"] >= 1
        assert analyze_audio(str(short_gap))["pause_count"] == 0
