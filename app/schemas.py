"""Pydantic schemas for request/response validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PrepGenerateRequest(BaseModel):
    """Request body for generating a prep guide."""

    job_description: str = Field(..., min_length=1)
    company_overview: str = Field(default="")


class Question(BaseModel):
    """A single interview question with its STAR-format model answer."""

    question: str
    answer: str


class PrepGuide(BaseModel):
    """Generated preparation guide."""

    technical_questions: List[Question]
    behavioral_questions: List[Question]
    positioning_tips: List[str]
    skill_gaps: List[str]


class PrepGuideResponse(BaseModel):
    """Response payload for a prep guide."""

    id: int
    cv_filename: str
    job_description: str
    company_overview: str
    guide: PrepGuide
    created_at: Optional[str] = None


class PracticeUploadResponse(BaseModel):
    """Response after a practice video is analyzed."""

    report_id: int
    video_id: int
    duration_seconds: float
    transcript: str
    scorecard: Dict[str, Any]


class FillerWordCounts(BaseModel):
    """Counts of detected filler words."""

    um: int = 0
    uh: int = 0
    like: int = 0
    basically: int = 0
    you_know: int = 0
    matlab: int = 0
    yaani: int = 0


class TextMetrics(BaseModel):
    """Metrics from transcript text analysis."""

    word_count: int
    words_per_minute: float
    filler_word_count: int
    filler_word_counts: FillerWordCounts
    filler_words_per_minute: float
    conciseness_score: float = Field(ge=0, le=10)
    star_score: float = Field(ge=0, le=10)
    relevance_score: float = Field(ge=0, le=10)


class SpeechMetrics(BaseModel):
    """Metrics from audio signal analysis."""

    speaking_rate_wpm: float
    pause_ratio: float
    pitch_mean_hz: float
    pitch_std_hz: float
    energy_variance: float
    clarity_score: float = Field(ge=0, le=10)


class VisualMetrics(BaseModel):
    """Metrics from video frame analysis."""

    face_detection_percent: float = Field(ge=0, le=100)
    eye_contact_percent: float = Field(ge=0, le=100)
    posture_stability_score: float = Field(ge=0, le=10)
    head_movement_score: float = Field(ge=0, le=10)
    frames_analyzed: int


class AnalysisReportResponse(BaseModel):
    """Full diagnostic scorecard for a practice session."""

    id: int
    video_id: int
    transcript: str
    content_score: float
    speech_score: float
    visual_score: float
    overall_score: float
    text_metrics: TextMetrics
    speech_metrics: SpeechMetrics
    visual_metrics: VisualMetrics
    strengths: List[str]
    weaknesses: List[str]
    improvement_tips: List[str]