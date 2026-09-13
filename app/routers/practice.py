"""Practice Engine API: upload a practice video and get a diagnostic scorecard."""

from __future__ import annotations

import json
import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import AnalysisReport, PracticeVideo, Session as DBSession
from ..services import (
    analyze_audio,
    analyze_report_with_llm,
    analyze_text,
    analyze_video,
    build_scorecard,
    transcribe_audio,
    extract_audio_from_video,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/practice", tags=["practice"])

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


async def _save_video(file: UploadFile, extension: str) -> str:
    """Save an uploaded video to the uploads directory with size enforcement."""
    filename = f"{uuid.uuid4().hex}{extension}"
    target = os.path.join(settings.upload_dir, filename)
    size = 0
    with open(target, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                out.close()
                os.remove(target)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds maximum size of {settings.max_upload_mb} MB.",
                )
            out.write(chunk)
    return target


def _cleanup(paths) -> None:
    """Best-effort removal of temporary files."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not remove %s: %s", path, exc)


@router.post("/upload")
async def upload_practice_video(
    video_file: UploadFile = File(...),
    session_id: int = Form(default=0),
    question_prompt: str = Form(default=""),
    db: Session = Depends(get_db),
) -> dict:
    """Analyze a practice video and return a full diagnostic scorecard."""
    extension = os.path.splitext(video_file.filename or "")[1].lower()
    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Video must be MP4, MOV, AVI, MKV or WEBM.",
        )

    video_path = await _save_video(video_file, extension)
    audio_path = os.path.join(
        settings.upload_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.wav"
    )

    try:
        await run_in_threadpool(extract_audio_from_video, video_path, audio_path)
        duration = await _probe_duration(audio_path)

        transcript = await run_in_threadpool(transcribe_audio, audio_path)

        text_metrics, speech_metrics, visual_metrics = await _run_analysis(
            transcript, audio_path, video_path, duration, question_prompt
        )
        scorecard = build_scorecard(text_metrics, speech_metrics, visual_metrics)
        review = await analyze_report_with_llm(scorecard)

        report = AnalysisReport(
            video_id=0,
            transcript=transcript,
            report_json=json.dumps({**scorecard, **review}),
        )

        saved_session = None
        if session_id and session_id > 0:
            saved_session = db.get(DBSession, session_id)
        if saved_session is None:
            saved_session = DBSession(title="Practice session")
            db.add(saved_session)
            db.flush()

        video = PracticeVideo(
            session_id=saved_session.id,
            filename=video_file.filename or "practice.mp4",
            stored_path=video_path,
            audio_path=audio_path,
            duration_seconds=duration,
            question_prompt=question_prompt,
            trash=False,
        )
        db.add(video)
        db.flush()
        report.video_id = video.id
        db.add(report)
        db.commit()
        db.refresh(report)
        db.refresh(video)
    except Exception as exc:
        logger.exception("Practice analysis failed.")
        _cleanup([video_path, audio_path])
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    # Privacy: local video/audio deleted immediately after processing.
    _cleanup([video_path, audio_path])

    result = json.loads(report.report_json)
    return {
        "report_id": report.id,
        "video_id": video.id,
        "duration_seconds": video.duration_seconds,
        "transcript": transcript,
        "scorecard": result,
    }


async def _probe_duration(wav_path: str) -> float:
    """Return audio duration in seconds using librosa."""
    import librosa

    return float(librosa.get_duration(path=wav_path, sr=16000))


async def _run_analysis(
    transcript: str,
    audio_path: str,
    video_path: str,
    duration: float,
    question_prompt: str,
):
    """Run text, audio and video analysis concurrently."""
    import asyncio

    text_task = asyncio.create_task(analyze_text(transcript, duration, question_prompt))
    audio_task = run_in_threadpool(analyze_audio, audio_path)
    video_task = run_in_threadpool(analyze_video, video_path)
    text_metrics, speech_metrics, visual_metrics = await asyncio.gather(
        text_task, audio_task, video_task
    )
    return text_metrics, speech_metrics, visual_metrics


@router.get("/{video_id}/report")
def get_practice_report(
    video_id: int, db: Session = Depends(get_db)
) -> dict:
    """Return the saved analysis report for a practice video."""
    video = db.get(PracticeVideo, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Practice video not found.")
    if video.report is None:
        raise HTTPException(status_code=404, detail="No report generated for this video.")
    report = video.report
    data = json.loads(report.report_json)
    return {
        "id": report.id,
        "video_id": video.id,
        "duration_seconds": video.duration_seconds,
        "transcript": report.transcript,
        "question_prompt": video.question_prompt,
        "created_at": str(report.created_at.isoformat()) if report.created_at else None,
        "scorecard": data,
    }