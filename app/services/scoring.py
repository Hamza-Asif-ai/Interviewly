"""Score combining + LLM strengths/weaknesses/tips generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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


MIN_FACE_FRAMES = 5
MIN_FACE_DETECTION_PERCENT = 20.0


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    """Clamp a value into the inclusive [low, high] range."""
    return max(low, min(high, value))


def _number(value: Any, default: float = 0.0) -> float:
    """Convert a metric to float, treating missing/unknown values as default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_visual_score(
    face_detection_percent: float,
    eye_contact_percent: float,
    posture_stability_score: float,
    head_movement_score: float,
    face_frames: int = 0,
) -> Optional[float]:
    """Return a 0-10 visual score, or None when face data is too sparse."""
    if face_frames < MIN_FACE_FRAMES or face_detection_percent < MIN_FACE_DETECTION_PERCENT:
        return None
    eye_contact = _clamp(eye_contact_percent / 10.0)
    posture = _clamp(posture_stability_score)
    movement = _clamp(head_movement_score)
    base = 0.40 * eye_contact + 0.30 * posture + 0.30 * movement
    presence = _clamp(face_detection_percent / 100.0, 0.0, 1.0)
    return round(_clamp(base * presence), 2)


def build_scorecard(
    text_metrics: Dict[str, Any],
    speech_metrics: Dict[str, Any],
    visual_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine text/speech/visual metrics into a weighted scorecard.

    Weights: Content 40%, Speech 30%, Visual 30%.
    """
    # ---- Content (0-10); renormalise when a signal is genuinely unknown ----
    star = _number(text_metrics.get("star_score"))
    relevance = text_metrics.get("relevance_score")
    conciseness = text_metrics.get("conciseness_score")
    available = [value for value in (star, relevance, conciseness) if value is not None]
    content_base = (
        sum(float(value) for value in available) / len(available) if available else 0.0
    )
    filler_fpm = _number(text_metrics.get("filler_words_per_minute"))
    filler_penalty = min(filler_fpm / 12.0, 2.5)
    content_score = round(_clamp(content_base - filler_penalty), 2)

    # ---- Speech (0-10) ----
    clarity = _number(speech_metrics.get("clarity_score"))
    wpm = _number(speech_metrics.get("speaking_rate_wpm"))
    pause_ratio = _number(speech_metrics.get("pause_ratio"), 0.25)
    rate_score = _clamp(10 - abs(wpm - 150) / 20)
    pause_score = _clamp(10 - (abs(pause_ratio - 0.25) / 0.15) * 10)
    speech_score = round(
        _clamp(0.5 * clarity + 0.3 * rate_score + 0.2 * pause_score), 2
    )

    # ---- Visual (0-10); unavailable when face tracking had too little data ----
    visual_score = _compute_visual_score(
        _number(visual_metrics.get("face_detection_percent")),
        _number(visual_metrics.get("eye_contact_percent")),
        _number(visual_metrics.get("posture_stability_score")),
        _number(visual_metrics.get("head_movement_score")),
        int(_number(visual_metrics.get("frames_analyzed"))),
    )

    if visual_score is None:
        overall_raw = 0.6 * content_score + 0.4 * speech_score
    else:
        overall_raw = 0.4 * content_score + 0.3 * speech_score + 0.3 * visual_score
    overall_score = round(_clamp(overall_raw), 2)

    return {
        "content_score": content_score,
        "speech_score": speech_score,
        "visual_score": visual_score,
        "overall_score": overall_score,
        "text_metrics": text_metrics,
        "speech_metrics": speech_metrics,
        "visual_metrics": visual_metrics,
    }


def _fallback_review(scorecard: Dict[str, Any]) -> Dict[str, List[str]]:
    """Deterministic review derived from the measured metrics, not generic praise."""
    text = scorecard.get("text_metrics") or {}
    speech = scorecard.get("speech_metrics") or {}
    visual = scorecard.get("visual_metrics") or {}

    word_count = int(_number(text.get("word_count")))
    words_per_minute = _number(text.get("words_per_minute"))
    filler_count = int(_number(text.get("filler_word_count")))
    filler_counts = text.get("filler_word_counts") or {}
    top_fillers = sorted(
        ((str(key), int(_number(value))) for key, value in filler_counts.items()),
        key=lambda item: -item[1],
    )
    top_fillers = [f"{key} ({count})" for key, count in top_fillers if count > 0][:3]
    pause_count = int(_number(speech.get("pause_count")))
    eye_contact = _number(visual.get("eye_contact_percent"))
    has_visual = scorecard.get("visual_score") is not None

    strengths = [
        f"You delivered {word_count} words at about {words_per_minute:.0f} words per minute.",
        f"Overall score: {scorecard.get('overall_score')}/10, content "
        f"{scorecard.get('content_score')}/10, speech {scorecard.get('speech_score')}/10.",
    ]
    if has_visual:
        strengths.append(f"Eye contact was detected for {eye_contact:.0f}% of analysed frames.")

    weaknesses = [
        f"{filler_count} filler words were detected"
        + (f" (most common: {', '.join(top_fillers)})." if top_fillers else "."),
        f"{pause_count} pauses longer than 0.3 seconds were detected.",
    ]
    if not has_visual:
        weaknesses.append(
            "Face tracking had too little data, so no visual score was calculated."
        )

    tips = [
        "Pause deliberately instead of using fillers, then check the filler count again.",
        f"Aim for 120-160 words per minute; your measured pace was "
        f"{words_per_minute:.0f} words per minute.",
        "Rehearse a STAR story aloud and compare the new metrics with this run.",
    ]
    return {"strengths": strengths, "weaknesses": weaknesses, "improvement_tips": tips}


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
        return {**_fallback_review(scorecard), "scoring_mode": "heuristic"}

    try:
        content = await achat_completion(
            [
                {"role": "system", "content": "You return only valid JSON."},
                {"role": "user", "content": SCORE_REVIEW_PROMPT.format(metrics=metrics_text)},
            ],
            temperature=0.3,
        )
        data = await extract_faithful_json(content)
        fallback = _fallback_review(scorecard)
        return {
            "strengths": data.get("strengths") or fallback["strengths"],
            "weaknesses": data.get("weaknesses") or fallback["weaknesses"],
            "improvement_tips": data.get("improvement_tips") or fallback["improvement_tips"],
            "scoring_mode": "llm",
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM review failed (%s); using metric-based fallback.", exc)
        return {**_fallback_review(scorecard), "scoring_mode": "heuristic"}
