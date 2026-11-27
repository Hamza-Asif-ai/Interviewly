"""Text analysis of interview transcripts: filler words, WPM, STAR, relevance."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from .llm import achat_completion, extract_faithful_json

logger = logging.getLogger(__name__)

FILLER_WORDS: Dict[str, int] = {
    "um": 0,
    "uh": 0,
    "erm": 0,
    "hmm": 0,
    "like": 0,
    "basically": 0,
    "literally": 0,
    "actually": 0,
    "you_know": 0,
    "matlab": 0,
    "yaani": 0,
}

_PLAIN_FILLER_RE = re.compile(
    r"(?<!\w)(?:um|uh|erm|hmm|basically|literally|actually|matlab|yaani|"
    r"مطلب|یعنی|you[\s_-]+know)(?!\w)",
    re.IGNORECASE,
)
_LIKE_FILLER_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+|(?<=,)\s+)like\b(?=[,!?]|\s+(?:I mean|you know|I guess))",
    re.IGNORECASE,
)

_TEXT_REVIEW_PROMPT = """You are an interview coach scoring a candidate's transcribed answer.
TASK: return ONLY valid JSON with this shape (no commentary):
{
  "star_score": <float 0-10>,
  "relevance_score": <float 0-10>,
  "conciseness_score": <float 0-10>
}
SCORING GUIDELINES:
- star_score: Does the response naturally follow Situation → Task → Action → Result?
  Give 0 for incoherent answers; 10 for perfectly structured STAR stories.
- relevance_score: How well does the answer relate to the question? 
  Give 10 for directly relevant answers; 0 for irrelevant ones.
- conciseness_score: Is the answer within a 60-150 second speaking window?
  Reward focused, tight answers. Penalise rambling. 60-120s = 8-10; <30s or >180s = 3-4.
"""


def _count_filler_words(text: str) -> Tuple[int, Dict[str, int]]:
    """Count fillers with word boundaries, one key for all 'you know' variants.

    'like' only counts as a discourse marker (sentence/comma-initial, or
    followed by 'I mean'/'you know'), not as the verb in 'I like Python'.
    """
    normalized = re.sub(r"you[\s_-]+know", "you_know", text, flags=re.IGNORECASE)
    normalized = re.sub(r"یعنی(?:\s+کہ)?", "yaani", normalized)
    normalized = re.sub(r"مطلب(?:\s+کی)?", "matlab", normalized)
    result = dict(FILLER_WORDS)
    for match in _PLAIN_FILLER_RE.finditer(normalized):
        word = re.sub(r"\s+", "_", match.group(0).lower())
        word = "you_know" if word.startswith("you") else word
        if word in result:
            result[word] += 1
    result["like"] += len(_LIKE_FILLER_RE.findall(normalized))
    return sum(result.values()), result


_STAGE_PATTERNS = {
    "situation": re.compile(
        r"\b(?:situation|context|background|challenge|project|team|company|role|"
        r"when|at my previous|in my last)\b",
        re.IGNORECASE,
    ),
    "task": re.compile(
        r"\b(?:task|responsibility|responsibilities|goal|objective|asked|needed|"
        r"required|had to|my role|charged with)\b",
        re.IGNORECASE,
    ),
    "action": re.compile(
        r"\b(?:i|we|my team)\s+(?:led|built|implemented|developed|created|"
        r"organized|managed|initiated|proposed|designed|automated|delivered|"
        r"wrote|owned|drove|shipped|rebuilt|coordinated)\b",
        re.IGNORECASE,
    ),
    "result": re.compile(
        r"\b(?:result|outcome|achieved|increased|reduced|improved|saved|"
        r"delivered|grew|impact|percent)\b",
        re.IGNORECASE,
    ),
}


def _simple_star_check(text: str) -> float:
    """Heuristic STAR structure check (0-10) requiring whole-word evidence.

    Each stage needs a real signal: a bare pronoun, copula or substring such as
    "i" in "pig" never counts as an action.
    """
    hits = sum(
        1 for pattern in _STAGE_PATTERNS.values() if pattern.search(text or "")
    )
    return round(10 * hits / len(_STAGE_PATTERNS), 1)


async def _llm_scores(
    text: str, question_prompt: str | None
) -> Tuple[float, float | None, float | None, str]:
    """Get STAR, relevance and conciseness scores via LLM.

    Returns `(star, relevance, conciseness, scoring_mode)`. Missing signals
    stay `None` rather than being silently replaced with a fixed 7.0/6.5.
    """
    heuristic_star = _simple_star_check(text)
    if not settings_has_llm():
        return heuristic_star, None, None, "heuristic"

    user_msg = (
        f"Question the candidate was answering: {question_prompt or '(none provided)'}\n\n"
        f"Candidate transcript (may contain speech-to-text artefacts):\n"
        f"{text[:3000]}\n\n"
        "Return ONLY the JSON described in the system prompt."
    )
    try:
        content = await achat_completion(
            [
                {"role": "system", "content": _TEXT_REVIEW_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        data = await extract_faithful_json(content)
    except Exception as exc:  # includes rate limits and transport errors
        logger.warning("LLM scoring unavailable (%s); using heuristic only.", exc)
        return heuristic_star, None, None, "heuristic"

    def _score(key: str) -> float | None:
        value = data.get(key)
        if value is None:
            return None
        try:
            return max(0.0, min(10.0, float(value)))
        except (TypeError, ValueError):
            return None

    star = _score("star_score")
    relevance = _score("relevance_score")
    conciseness = _score("conciseness_score")
    if star is None and relevance is None and conciseness is None:
        return heuristic_star, None, None, "heuristic"
    return (
        star if star is not None else heuristic_star,
        relevance,
        conciseness,
        "llm",
    )


def settings_has_llm() -> bool:
    """Local alias to avoid circular import."""
    from ..config import settings

    return settings.has_llm()


async def analyze_text(
    transcript: str,
    duration_seconds: float,
    question_prompt: str | None = None,
    scoring_text: str | None = None,
) -> Dict[str, Any]:
    """Analyse an interview transcript and return structured metrics.

    ``transcript`` is the original speech and drives word count, WPM and
    filler counts. ``scoring_text`` (typically a faithful translation) is used
    only for STAR/relevance/conciseness grading, never to clean up metrics.
    """
    words = transcript.split()
    word_count = len(words)
    minutes = max(duration_seconds / 60, 0.01)
    wpm = round(word_count / minutes, 1)

    filler_total, filler_counts = _count_filler_words(transcript)
    fpm = round(filler_total / minutes, 2)

    star_score, relevance_score, conciseness_score, scoring_mode = await _llm_scores(
        scoring_text if scoring_text is not None else transcript, question_prompt
    )

    if conciseness_score is not None and 60 <= duration_seconds <= 120:
        conciseness_score = max(0.0, min(10.0, conciseness_score + 1.0))

    return {
        "word_count": word_count,
        "words_per_minute": wpm,
        "filler_word_count": filler_total,
        "filler_word_counts": filler_counts,
        "filler_words_per_minute": fpm,
        "conciseness_score": (
            round(conciseness_score, 2) if conciseness_score is not None else None
        ),
        "star_score": round(max(0, min(10, star_score)), 2),
        "relevance_score": (
            round(relevance_score, 2) if relevance_score is not None else None
        ),
        "scoring_mode": scoring_mode,
    }
