"""Prep Engine API: generate and fetch preparation guides."""

from __future__ import annotations

import json
import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import PrepGuide as PrepGuideModel
from ..models import Session as DBSession
from ..services.cv_parser import extract_text_from_file, CVParseError
from ..services.prep_engine import PrepEngine
from ..schemas import PrepGuide as PrepGuideSchema
from ..schemas import PrepGuideResponse, Question
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prep", tags=["prep"])


async def _save_upload(file: UploadFile, extension: str) -> str:
    """Save an upload to disk guarding against path and size abuse."""
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
                    detail=f"File exceeds maximum size of {settings.max_upload_mb} MB.",
                )
            out.write(chunk)
    return target


@router.post("/generate", response_model=PrepGuideResponse)
async def generate_prep_guide(
    cv_file: UploadFile = File(...),
    job_description: str = Form(..., min_length=1),
    company_overview: str = Form(default=""),
    db: Session = Depends(get_db),
) -> PrepGuideResponse:
    """Generate a preparation guide from a CV, job description, and company overview."""
    extension = os.path.splitext(cv_file.filename or "")[1].lower()
    if extension not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(
            status_code=400,
            detail="CV must be a PDF, DOCX, DOC or TXT file.",
        )

    saved_path = await _save_upload(cv_file, extension)
    try:
        cv_text = await run_in_threadpool(extract_text_from_file, saved_path)
    except CVParseError as exc:
        logger.error("CV parse failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)  # CV text is embedded; raw file not retained

    session = DBSession(title="Prep session")
    db.add(session)
    db.flush()  # obtain session.id

    engine = PrepEngine(str(session.id))
    try:
        await run_in_threadpool(
            engine.index_documents, cv_text, job_description, company_overview
        )
        guide_data = await engine.generate()
    except Exception as exc:
        logger.exception("Prep generation failed.")
        raise HTTPException(status_code=500, detail=f"Prep generation failed: {exc}") from exc

    if not isinstance(guide_data, dict):
        raise HTTPException(status_code=500, detail="Model returned an unexpected guide shape.")

    guide_json = json.dumps(guide_data)
    guide = PrepGuideModel(
        session_id=session.id,
        cv_filename=cv_file.filename or "cv",
        job_description=job_description,
        company_overview=company_overview,
        guide_json=guide_json,
    )
    db.add(guide)
    db.commit()
    db.refresh(guide)

    return _to_response(guide)


def _to_response(guide: PrepGuideModel) -> PrepGuideResponse:
    """Convert an ORM PrepGuide into the response schema."""
    data = json.loads(guide.guide_json)
    technical = [Question(**item) for item in data.get("technical_questions", [])]
    behavioral = [Question(**item) for item in data.get("behavioral_questions", [])]
    return PrepGuideResponse(
        id=guide.id,
        cv_filename=guide.cv_filename,
        job_description=guide.job_description,
        company_overview=guide.company_overview,
        guide=PrepGuideSchema(
            technical_questions=technical,
            behavioral_questions=behavioral,
            positioning_tips=data.get("positioning_tips", []),
            skill_gaps=data.get("skill_gaps", []),
        ),
        created_at=str(guide.created_at.isoformat()) if guide.created_at else None,
    )


@router.get("/{guide_id}", response_model=PrepGuideResponse)
def get_prep_guide(guide_id: int, db: Session = Depends(get_db)) -> PrepGuideResponse:
    """Return a previously generated prep guide by id."""
    guide = db.get(PrepGuideModel, guide_id)
    if guide is None:
        raise HTTPException(status_code=404, detail="Prep guide not found.")
    return _to_response(guide)