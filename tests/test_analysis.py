"""Tests for the Analysis Engine (scoring, text/audio/video analysis)."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from app.services.audio_analysis import analyze_audio  # noqa: E402
from app.services.scoring import build_scorecard  # noqa: E402
from app.services.text_analysis import _count_filler_words  # noqa: E402
from app.services.transcription import transcribe_audio  # noqa: E402
from app.services.video_analysis import analyze_video  # noqa: E402


class TestTextAnalysis:
    """Tests for transcript text metrics."""

    def test_filler_word_counting(self) -> None:
        text = "Um, like, I basically feel, um, you know what I mean?"
        total, counts = _count_filler_words(text)
        assert total >= 4
        assert counts["um"] == 2
        assert counts["like"] >= 1

    def test_no_fillers(self) -> None:
        total, counts = _count_filler_words("I led a project and shipped results.")
        assert total == 0
        assert counts["um"] == 0


class TestScoring:
    """Tests for scorecard combination logic."""

    def _sample_metrics(self) -> dict:
        return {
            "text_metrics": {
                "star_score": 8.0,
                "relevance_score": 9.0,
                "conciseness_score": 7.0,
                "filler_words_per_minute": 3.0,
                "word_count": 200,
                "words_per_minute": 140.0,
                "filler_word_count": 4,
                "filler_word_counts": {"um": 2, "uh": 2, "like": 0,
                                       "basically": 0, "you know": 0,
                                       "matlab": 0, "yaani": 0},
                "filler_words_per_minute": 3.0,
            },
            "speech_metrics": {
                "speaking_rate_wpm": 150.0,
                "pause_ratio": 0.25,
                "pitch_mean_hz": 180.0,
                "pitch_std_hz": 30.0,
                "energy_variance": 0.002,
                "clarity_score": 8.0,
            },
            "visual_metrics": {
                "face_detection_percent": 100.0,
                "eye_contact_percent": 80.0,
                "posture_stability_score": 8.0,
                "head_movement_score": 7.0,
                "frames_analyzed": 10,
            },
        }

    def test_weighted_combination(self) -> None:
        card = build_scorecard(**self._sample_metrics())
        expected_content = round(max(0, min(10, (0.40 * 8 + 0.35 * 9 + 0.25 * 7) - min(3 / 12, 2.5))), 2)
        assert card["content_score"] == expected_content
        assert text_scorecard_within_range(card["content_score"])
        assert card["overall_score"] <= 10 and card["overall_score"] >= 0

    def test_max_visual(self) -> None:
        metrics = self._sample_metrics()
        metrics["visual_metrics"] = {
            "face_detection_percent": 100.0,
            "eye_contact_percent": 100.0,
            "posture_stability_score": 10.0,
            "head_movement_score": 10.0,
            "frames_analyzed": 5,
        }
        card = build_scorecard(**metrics)
        assert card["visual_score"] == 10.0

    def test_empty_metrics_do_not_crash(self) -> None:
        card = build_scorecard({}, {}, {})
        assert isinstance(card["overall_score"], float)
        assert 0.0 <= card["overall_score"] <= 10.0
        assert 0.0 <= card["content_score"] <= 10.0


def text_scorecard_within_range(value: float) -> bool:
    """Assert a value lies within the closed interval [0, 10]."""
    return 0.0 <= value <= 10.0


def _make_test_wav(path: str, duration: float = 2.0) -> None:
    """Generate a synthetic WAV: 1s tone + 1s silence using soundfile."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone_end = sr
    y = np.sin(2 * np.pi * 220 * t)
    y[tone_end:] = 0.0
    y[0:int(0.05 * sr)] = 0.0
    sf.write(path, y, sr)


class TestAudioAnalysis:
    """Tests for librosa-based audio metrics."""

    def test_synthetic_wav(self, tmp_path) -> None:
        wav = tmp_path / "test.wav"
        _make_test_wav(str(wav))
        metrics = analyze_audio(str(wav))
        assert metrics["speaking_rate_wpm"] >= 0
        assert 0.0 <= metrics["pause_ratio"] <= 1.0
        assert metrics["clarity_score"] >= 0 and metrics["clarity_score"] <= 10
        assert metrics["pitch_mean_hz"] > 0


class TestVideoAnalysis:
    """Tests for mediapipe/opencv visual metrics."""

    def test_blank_video_does_not_crash(self, tmp_path) -> None:
        video = tmp_path / "blank.mp4"
        _make_test_video(str(video), seconds=2)
        metrics = analyze_video(str(video))
        # Blank video → no face / pose detections → all zeros allowed.
        assert metrics["face_detection_percent"] >= 0
        assert metrics["face_detection_percent"] <= 100
        assert metrics["frames_analyzed"] >= 1


def _make_test_video(path: str, seconds: int = 2, fps: int = 5) -> None:
    """Generate a synthetic blank video with OpenCV."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (320, 240),
    )
    for _ in range(seconds * fps):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestTranscription:
    """Tests for transcription fallbacks (offline)."""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            transcribe_audio("no_such_file.wav")