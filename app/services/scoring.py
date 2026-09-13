"""Score combining + LLM strengths/weaknesses/tips generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .llm import achat_completion, extract_faithful_json
from ..config import settings

logger = logging.getLogger(__name__)

SCORE_REVIEW_PROMPT = """You are an elite interview coach reviewing a candidate's performance metrics.
The candidate practised answering an interview question and the system computed these metrics:
{metrics}

Return ONLY valid JSON with exactly these keys (no other text):
{{
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "improvement_tips": ["...", "..."]
}}
Use 3 of each. Tie every weakness to at least one concrete metric and give a specific,
actionable improvement tip. Be kind but honest."""


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    """Clamp a value into the inclusive [low, high] range."""
    return max(low, min(high, value))


def build_scorecard(
    text_metrics: Dict[str, Any],
    speech_metrics: Dict[str, Any],
    visual_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine text/speech/visual metrics into a weighted scorecard.

    Weights: Content 40%, Speech 30%, Visual 30%.
    """
    # ---- Content (0-10) ----
    star = float(text_metrics.get("star_score", 0))
    relevance = float(text_metrics.get("relevance_score", 0))
    conciseness = float(text_metrics.get("conciseness_score", 0))
    filler_fpm = float(text_metrics.get("filler_words_per_minute", 0))
    content_base = 0.40 * star + 0.35 * relevance + 0.25 * conciseness
    filler_penalty = min(filler_fpm / 12.0, 2.5)
    content_score = round(_clamp(content_base - filler_penalty), 2)

    # ---- Speech (0-10) ----
    clarity = float(speech_metrics.get("clarity_score", 0))
    wpm = float(speech_metrics.get("speaking_rate_wpm", 0))
    pause_ratio = float(speech_metrics.get("pause_ratio", 0))
    rate_score = _clamp(10 - abs(wpm - 150) / 20)
    pause_score = _clamp(10 - (abs(pause_ratio - 0.25) / 0.15) * 10)
    speech_score = round(
        _clamp(0.5 * clarity + 0.3 * rate_score + 0.2 * pause_score), 2
    )

    # ---- Visual (0-10) ----
    face_pct = float(visual_metrics.get("face_detection_percent", 0))
    eye_pct = float(visual_metrics.get("eye_contact_percent", 0))
    posture = float(visual_metrics.get("posture_stability_score", 0))
    head = float(visual_metrics.get("head_movement_score", 0))
    visual_score = round(
        _clamp(
            0.3 * (face_pct / 10)
            + 0.3 * (eye_pct / 10)
            + 0.2 * posture
            + 0.2 * head
        ),
        2,
    )

    overall_score = round(
        _clamp(0.4 * content_score + 0.3 * speech_score + 0.3 * visual_score), 2
    )

    return {
        "content_score": content_score,
        "speech_score": speech_score,
        "visual_score": visual_score,
        "overall_score": overall_score,
        "text_metrics": text_metrics,
        "speech_metrics": speech_metrics,
        "visual_metrics": visual_metrics,
    }


def _fallback_review() -> Dict[str, List[str]]:
    """Deterministic review used when no LLM is configured."""
    return {
        "strengths": [
            "The answer has a coherent narrative arc worth building on.",
            "You are engaging with the question topic.",
            "Practice shows commitment to interview preparation.",
        ],
        "weaknesses": [
            "Watch filler words (um/uh/like) — aim for under 4 per minute.",
            "Keep answers to 60–120 seconds and stay on the STAR structure.",
            "Maintain a steady speaking pace near 120–160 words per minute.",
        ],
        "improvement_tips": [
            "Rehearse each answer aloud with a timer until you hit 90–120 seconds.",
            "Record yourself weekly and review the eye-contact and posture scores.",
            "Structure every answer as Situation → Task → Action → Result.",
        ],
    }


async def analyze_report_with_llm(scorecard: Dict[str, Any]) -> Dict[str, str]:
    """Generate strengths, weaknesses, and improvement tips from metrics via LLM."""
    text_metrics = scorecard.get("text_metrics", {})
    speech_metrics = scorecard.get("speech_metrics", {})
    visual_metrics = scorecard.get("visual_metrics", {})

    metrics_text = "\n".join(
        f"- {key}: {value}" for key, value in {
            **{f"text.{k}": v for k, v in text_metrics.items()},
            **{f"speech.{k}": v for k, v in speech_metrics.items()},
            **{f"visual.{k}": v for k, v in visual_metrics.items()},
            "content_score": scorecard.get("content_score"),
            "speech_score": scorecard.get("speech_score"),
            "visual_score": scorecard.get("visual_score"),
            "overall_score": scorecard.get("overall_score"),
        }.items()
    )

    if not settings.has_llm():
        return _fallback_review()

    try:
        content = await achat_completion(
            [
                {"role": "system", "content": "You return only valid JSON."},
                {"role": "user", "content": SCORE_REVIEW_PROMPT.format(metrics=metrics_text)},
            ],
            temperature=0.3,
        )
        data = await extract_faithful_json(content)
        return {
            "strengths": data.get("strengths") or _fallback_review()["strengths"],
            "weaknesses": data.get("weaknesses") or _fallback_review()["weaknesses"],
            "improvement_tips": data.get("improvement_tips")
            or _fallback_review()["improvement_tips"],
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM review failed (%s); using fallback.", exc)
        return _fallback_review()