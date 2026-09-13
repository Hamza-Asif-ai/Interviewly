"""Gradio Blocks UI: Prep tab + Practice tab, mounted inside FastAPI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List

import gradio as gr

from ..config import settings
from ..database import SessionLocal
from ..models import AnalysisReport, PracticeVideo, Session as DBSession
from ..models import PrepGuide
from ..services import (
    analyze_audio,
    analyze_report_with_llm,
    analyze_text,
    analyze_video,
    build_scorecard,
    extract_audio_from_video,
    transcribe_audio,
)
from ..services.cv_parser import CVParseError, extract_text_from_file
from ..services.pdf_export import build_prep_guide_pdf
from ..services.prep_engine import PrepEngine, load_fallback_guide, validate_guide

logger = logging.getLogger(__name__)

APP_TITLE = "Interviewly — AI Interview Buddy & Performance Analyzer"
FOOTER = (
    "**Privacy:** Your videos and CVs are processed locally on this server, analysed in "
    "memory, and deleted immediately after processing. Nothing is stored long-term."
)


def _run(coro) -> Any:
    """Run an async coroutine to completion from a sync Gradio handler."""
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  PREP TAB
# --------------------------------------------------------------------------- #
def _item_to_qa(item: Any) -> tuple:
    """Normalize a question item (dict or bare string) into (question, answer)."""
    if isinstance(item, dict):
        return item.get("question", ""), item.get("answer", "")
    if isinstance(item, str):
        return item, ""
    return "", ""


def _guide_to_markdown(guide: Dict[str, Any]) -> str:
    """Render a prep guide dict as Markdown."""
    if not isinstance(guide, dict):
        return (
            "# Interview Preparation Guide\n\n"
            "**Error:** the model did not return a guide. Please try again.\n"
        )

    tech_count = len(guide.get("technical_questions", []))
    beh_count = len(guide.get("behavioral_questions", []))
    tips_count = len(guide.get("positioning_tips", []))
    gaps_count = len(guide.get("skill_gaps", []))
    if tech_count + beh_count + tips_count + gaps_count == 0:
        return (
            "# Interview Preparation Guide\n\n"
            "**Error:** the model returned an empty guide (no questions, answers, tips "
            "or skill gaps). Please try generating again.\n"
        )

    lines = ["# Interview Preparation Guide", ""]

    lines.append("## Technical Questions")
    for i, item in enumerate(guide.get("technical_questions", []), start=1):
        question, answer = _item_to_qa(item)
        lines.append(f"### {i}. {question or '_No question text._'}")
        lines.append(f"**STAR model answer:**\n\n{answer or '_No answer text._'}")
        lines.append("")

    lines.append("## Behavioral Questions")
    for i, item in enumerate(guide.get("behavioral_questions", []), start=1):
        question, answer = _item_to_qa(item)
        lines.append(f"### {i}. {question or '_No question text._'}")
        lines.append(f"**STAR model answer:**\n\n{answer or '_No answer text._'}")
        lines.append("")

    lines.append("## Company-Specific Positioning Tips")
    tips = [
        t for t in guide.get("positioning_tips", [])
        if isinstance(t, str) and t.strip()
    ]
    if tips:
        for i, tip in enumerate(tips, start=1):
            lines.append(f"{i}. {tip}")
    else:
        lines.append("_No positioning tips returned._")
    lines.append("")

    lines.append("## Skill-Gap Analysis")
    gaps = [
        g for g in guide.get("skill_gaps", [])
        if isinstance(g, str) and g.strip()
    ]
    if gaps:
        for gap in gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("_No skill gaps returned._")

    return "\n".join(lines)


def _generate_prep_handler(
    cv_file: str | None,
    job_description: str,
    company_overview: str,
    fast_mode: bool = False,
):
    """Generate a prep guide and return markdown output and a PDF download path."""
    import traceback

    if fast_mode:
        logger.info("Prep Engine: Fast Mode ON - serving cached fallback guide.")
        guide = load_fallback_guide()
        pdf_path = None
        try:
            pdf_path = build_prep_guide_pdf(guide, settings.pdf_dir)
        except Exception as exc:  # pragma: no cover - never fail the guide over PDF
            logger.warning("PDF export failed in Fast Mode: %s", exc)
        return _guide_to_markdown(guide), pdf_path

    if cv_file is None or not os.path.exists(cv_file):
        raise gr.Error("Please upload a CV file (PDF, DOCX or TXT).")
    if not job_description.strip():
        raise gr.Error("Please paste the target job description.")

    try:
        try:
            cv_text = extract_text_from_file(cv_file)
        except CVParseError as exc:
            raise gr.Error(f"Could not parse CV: {exc}") from exc

        db = SessionLocal()
        try:
            session = DBSession(title="Prep session")
            db.add(session)
            db.flush()

            engine = PrepEngine(str(session.id))
            engine.index_documents(cv_text, job_description, company_overview)
            guide = _run(engine.generate())

            if not isinstance(guide, dict):
                raise gr.Error("The model returned an unexpected guide shape.")

            try:
                validate_guide(guide)
            except ValueError as exc:
                logger.error(
                    "LLM returned an unusable guide: %s", traceback.format_exc()
                )
                raise gr.Error(
                    f"The model did not produce usable content: {exc}"
                ) from exc

            logger.info(
                "Prep guide generated and validated: %d technical, %d behavioral, "
                "%d tips, %d skill gaps.",
                len(guide.get("technical_questions", [])),
                len(guide.get("behavioral_questions", [])),
                len(guide.get("positioning_tips", [])),
                len(guide.get("skill_gaps", [])),
            )

            row = PrepGuide(
                session_id=session.id,
                cv_filename=os.path.basename(cv_file),
                job_description=job_description,
                company_overview=company_overview,
                guide_json=json.dumps(guide),
            )
            db.add(row)
            db.commit()
        finally:
            db.close()

        pdf_path = None
        try:
            pdf_path = build_prep_guide_pdf(guide, settings.pdf_dir)
            logger.info("Prep guide PDF saved to: %s", pdf_path)
        except Exception as exc:  # pragma: no cover - never fail the guide over PDF
            logger.warning(
                "PDF export failed; continuing without download: %s", exc
            )
        return _guide_to_markdown(guide), pdf_path
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("Prep generation failed: %s", traceback.format_exc())
        raise gr.Error(f"Prep generation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
#  PRACTICE TAB
# --------------------------------------------------------------------------- #
def _metrics_table(scorecard: Dict[str, Any]) -> List[List[Any]]:
    """Flatten metrics into a rows table for display."""
    rows: List[List[Any]] = [["Category", "Metric", "Value"]]
    groups: Dict[str, Dict[str, Any]] = {
        "Text": scorecard.get("text_metrics", {}),
        "Speech": scorecard.get("speech_metrics", {}),
        "Visual": scorecard.get("visual_metrics", {}),
    }
    for category, metrics in groups.items():
        for key, value in metrics.items():
            rows.append([category, key.replace("_", " ").title(), value])
    rows.extend(
        [
            ["Score", "Content", scorecard.get("content_score")],
            ["Score", "Speech", scorecard.get("speech_score")],
            ["Score", "Visual", scorecard.get("visual_score")],
        ]
    )
    return rows


@lru_cache(maxsize=1)
def _load_practice_fallback() -> Dict[str, Any]:
    """Load the cached demo scorecard used in Practice Fast Mode."""
    logger.info("Loading practice demo scorecard from %s", settings.practice_fallback_path)
    with open(settings.practice_fallback_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _render_practice(
    scorecard: Dict[str, Any],
    review: Dict[str, Any],
    transcript: str,
) -> tuple:
    """Render a scorecard into the Practice tab's output widgets."""
    content_score = scorecard.get("content_score", 0)
    speech_score = scorecard.get("speech_score", 0)
    visual_score = scorecard.get("visual_score", 0)
    overall_score = scorecard.get("overall_score", 0)

    review_md = "## Strengths\n" + "\n".join(
        f"- {s}" for s in review.get("strengths", [])
    )
    review_md += "\n\n## Weaknesses\n" + "\n".join(
        f"- {w}" for w in review.get("weaknesses", [])
    )
    review_md += "\n\n## Improvement Tips\n" + "\n".join(
        f"- {t}" for t in review.get("improvement_tips", [])
    )

    transcript_md = (
        f"## Transcript\n\n<div dir='ltr' style='text-align:left'>> {transcript}</div>"
        if transcript.strip() else
        "## Transcript\n\n_No transcript detected — is audio present in the video?_"
    )

    return (
        content_score,
        speech_score,
        visual_score,
        overall_score,
        _metrics_table(scorecard),
        review_md,
        transcript_md,
    )


def _analyze_practice_handler(
    video: str | None, question_prompt: str, fast_mode: bool = False
) -> tuple:
    """Analyze a practice video end to end and return display outputs."""
    import traceback

    if fast_mode:
        logger.info("Practice Engine: Fast Mode ON - serving cached scorecard.")
        data = _load_practice_fallback()
        return _render_practice(data["scorecard"], data["review"], data["transcript"])

    if video is None or not os.path.exists(video):
        raise gr.Error("Please upload or record a practice video.")

    video_path = ""
    audio_path = ""
    try:
        video_path = video
        base = os.path.splitext(video)[0]
        audio_path = base + ".wav"
        os.makedirs(settings.upload_dir, exist_ok=True)

        extract_audio_from_video(video_path, audio_path)
        duration = _probe_duration(audio_path)
        transcript = transcribe_audio(audio_path)
        text_metrics = _run(analyze_text(transcript, duration, question_prompt))
        speech_metrics = analyze_audio(audio_path)
        visual_metrics = analyze_video(video_path)
        scorecard = build_scorecard(text_metrics, speech_metrics, visual_metrics)
        review = _run(analyze_report_with_llm(scorecard))

        db = SessionLocal()
        try:
            session = DBSession(title="Practice session")
            db.add(session)
            db.flush()
            video_row = PracticeVideo(
                session_id=session.id,
                filename=os.path.basename(video_path),
                stored_path=video_path,
                audio_path=audio_path,
                duration_seconds=duration,
                question_prompt=question_prompt,
                trash=True,
            )
            db.add(video_row)
            db.flush()
            report = AnalysisReport(
                video_id=video_row.id,
                transcript=transcript,
                report_json=json.dumps({**scorecard, **review}),
            )
            db.add(report)
            db.commit()
        finally:
            db.close()

        for path_ in (audio_path, video_path):
            try:
                if path_ and os.path.exists(path_) and path_ != video:
                    os.remove(path_)
            except OSError:
                logger.warning("Could not delete temp file: %s", path_)

        return _render_practice(scorecard, review, transcript)
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("Practice analysis failed: %s", traceback.format_exc())
        raise gr.Error(f"Analysis failed: {exc}") from exc


def _probe_duration(wav_path: str) -> float:
    """Return audio duration in seconds via librosa."""
    import librosa

    return float(librosa.get_duration(path=wav_path, sr=16000))


# --------------------------------------------------------------------------- #
#  APP
# --------------------------------------------------------------------------- #
def build_gradio_app() -> gr.Blocks:
    """Construct the Gradio Blocks application."""
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(
            "Prepare for interviews with tailored questions and STAR answers, then "
            "practice by recording a video and get a performance scorecard."
        )

        with gr.Tab("Prep Engine"):
            with gr.Row():
                with gr.Column(scale=1):
                    cv_file = gr.File(
                        label="Upload CV (PDF / DOCX / TXT)",
                        file_types=[".pdf", ".docx", ".txt"],
                    )
                    job_desc = gr.Textbox(
                        label="Target Job Description",
                        lines=6,
                        placeholder="Paste the job description here...",
                    )
                    company_overview = gr.Textbox(
                        label="Company Overview",
                        lines=4,
                        placeholder="Paste the company overview / about page (optional)...",
                    )
                    generate_btn = gr.Button(
                        "Generate Guide", variant="primary"
                    )
                    fast_mode_prep = gr.Checkbox(
                        label="Fast Mode (instant demo)",
                        value=False,
                        info="Skip the LLM and serve the cached demo guide in ~0.5s.",
                    )
                with gr.Column(scale=2):
                    guide_output = gr.Markdown(
                        label="Preparation Guide", value=""
                    )
                    download_pdf = gr.File(
                        label="Download as PDF", interactive=False
                    )

            generate_btn.click(
                _generate_prep_handler,
                inputs=[cv_file, job_desc, company_overview, fast_mode_prep],
                outputs=[guide_output, download_pdf],
            )

        with gr.Tab("Practice Engine"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="Record or upload a practice video",
                        sources=["upload", "webcam"],
                    )
                    question_input = gr.Textbox(
                        label="Question you were answering (optional)",
                        lines=3,
                        placeholder="e.g. Tell me about a time you led a project...",
                    )
                    analyze_btn = gr.Button("Analyze Performance", variant="primary")
                    fast_mode_practice = gr.Checkbox(
                        label="Fast Mode (instant demo)",
                        value=False,
                        info="Skip Whisper/MediaPipe and show the cached demo scorecard instantly.",
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### Scorecard")
                    content_gauge = gr.Slider(
                        label="Content Score (40% weight)",
                        minimum=0, maximum=10, value=0, interactive=False,
                    )
                    speech_gauge = gr.Slider(
                        label="Speech Score (30% weight)",
                        minimum=0, maximum=10, value=0, interactive=False,
                    )
                    visual_gauge = gr.Slider(
                        label="Visual Score (30% weight)",
                        minimum=0, maximum=10, value=0, interactive=False,
                    )
                    overall_gauge = gr.Slider(
                        label="Overall Score",
                        minimum=0, maximum=10, value=0, interactive=False,
                    )
                    metrics_table = gr.Dataframe(
                        headers=["Category", "Metric", "Value"],
                        label="Metrics Breakdown",
                        interactive=False,
                    )
                    review_output = gr.Markdown(label="Review")
                    transcript_output = gr.Markdown(label="Transcript")

            analyze_btn.click(
                _analyze_practice_handler,
                inputs=[video_input, question_input, fast_mode_practice],
                outputs=[
                    content_gauge,
                    speech_gauge,
                    visual_gauge,
                    overall_gauge,
                    metrics_table,
                    review_output,
                    transcript_output,
                ],
            )

        gr.Markdown(FOOTER)

    return demo