"""Prep Engine API: generate and fetch preparation guides."""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import PrepGuide as PrepGuideModel
from ..models import Session as DBSession
from ..services.cv_parser import extract_text_from_file, CVParseError
from ..services.prep_engine import PrepEngine, PrepGuideError, load_fallback_guide
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
    job_description: str = Form(
        ..., min_length=1, max_length=settings.max_input_chars
    ),
    company_overview: str = Form(
        default="", max_length=settings.max_input_chars
    ),
    fast_mode: bool = Form(default=False),
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

    if len(cv_text) > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Extracted CV text exceeds {settings.max_input_chars} characters.",
        )

    engine_session_id = uuid.uuid4().hex
    engine = PrepEngine(engine_session_id)
    try:
        if fast_mode:
            logger.info("Fast Mode: serving clearly labelled demo prep guide.")
            guide_data = load_fallback_guide()
        else:
            await run_in_threadpool(
                engine.index_documents, cv_text, job_description, company_overview
            )
            guide_data = await engine.generate()
    except PrepGuideError as exc:
        logger.warning("Prep generation unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM unavailable. Retry later, or enable Fast Mode for a demo "
                "guide that is explicitly not based on your CV."
            ),
        ) from exc
    except Exception:
        logger.exception("Prep generation failed.")
        raise HTTPException(
            status_code=500, detail="Prep generation failed. Please retry shortly."
        )
    finally:
        await run_in_threadpool(engine.cleanup)

    if not isinstance(guide_data, dict):
        raise HTTPException(status_code=500, detail="Model returned an unexpected guide shape.")

    guide_json = json.dumps(guide_data)
    access_token = secrets.token_urlsafe(24)
    # One short transaction after generation: no DB row is held open while
    # embeddings or the LLM are running.
    session = DBSession(title="Prep session")
    db.add(session)
    db.flush()  # obtain session.id
    guide = PrepGuideModel(
        session_id=session.id,
        cv_filename=cv_file.filename or "cv",
        job_description=job_description,
        company_overview=company_overview,
        guide_json=guide_json,
        access_token=access_token,
    )
    db.add(guide)
    db.commit()
    db.refresh(guide)

    return _to_response(guide, include_token=access_token)


def _to_response(
    guide: PrepGuideModel, include_token: str | None = None
) -> PrepGuideResponse:
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
        access_token=include_token,
    )


@router.get(
    "/{guide_id}", response_model=PrepGuideResponse, response_model_exclude_none=True
)
def get_prep_guide(
    guide_id: int,
    x_access_token: str | None = Header(default=None, alias="X-Access-Token"),
    db: Session = Depends(get_db),
) -> PrepGuideResponse:
    """Return a previously generated prep guide by id, if the token matches."""
    guide = db.get(PrepGuideModel, guide_id)
    if guide is None or not guide.access_token:
        raise HTTPException(status_code=404, detail="Prep guide not found.")
    if not x_access_token or not hmac.compare_digest(
        guide.access_token, x_access_token
    ):
        raise HTTPException(status_code=404, detail="Prep guide not found.")
    return _to_response(guide)
