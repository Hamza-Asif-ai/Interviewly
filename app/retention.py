"""Privacy retention: prune expired database records, media and PDFs."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import create_engine, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import AnalysisReport, PrepGuide, PracticeVideo, Session as DBSession
from .services.pdf_export import cleanup_old_pdfs

logger = logging.getLogger(__name__)


def _delete_file(path: str | None) -> None:
    """Remove a staged artifact, ignoring missing files and permission issues."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - best effort cleanup
        logger.warning("Could not remove expired file %s: %s", path, exc)


def cleanup_old_records(
    hours: int | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> dict[str, int]:
    """Delete records older than the retention window and their media files."""
    hours = settings.retention_hours if hours is None else hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    factory = session_factory or sessionmaker(bind=create_engine(settings.database_url))
    removed = {"prep_guides": 0, "practice_videos": 0, "analysis_reports": 0, "sessions": 0}
    affected_sessions: set[int] = set()
    db = factory()
    try:
        old_reports = db.scalars(
            select(AnalysisReport).where(AnalysisReport.created_at < cutoff)
        ).all()
        for report in old_reports:
            db.delete(report)
            removed["analysis_reports"] += 1

        old_videos = db.scalars(
            select(PracticeVideo).where(PracticeVideo.created_at < cutoff)
        ).all()
        for video in old_videos:
            _delete_file(video.stored_path)
            _delete_file(video.audio_path)
            if video.session_id:
                affected_sessions.add(video.session_id)
            db.delete(video)
            removed["practice_videos"] += 1

        old_guides = db.scalars(
            select(PrepGuide).where(PrepGuide.created_at < cutoff)
        ).all()
        for guide in old_guides:
            if guide.session_id:
                affected_sessions.add(guide.session_id)
            db.delete(guide)
            removed["prep_guides"] += 1

        old_sessions = db.scalars(
            select(DBSession).where(
                or_(
                    DBSession.created_at < cutoff,
                    DBSession.id.in_(affected_sessions),
                )
            )
        ).all()
        db.flush()
        for session in old_sessions:
            remaining_guides = db.scalar(
                select(func.count())
                .select_from(PrepGuide)
                .where(PrepGuide.session_id == session.id)
            )
            remaining_videos = db.scalar(
                select(func.count())
                .select_from(PracticeVideo)
                .where(PracticeVideo.session_id == session.id)
            )
            if not remaining_guides and not remaining_videos:
                db.delete(session)
                removed["sessions"] += 1

        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Retention cleanup failed")
        raise
    finally:
        db.close()
    if any(removed.values()):
        logger.info("Retention cleanup removed %s", removed)
    return removed


def cleanup_old_uploads(hours: int | None = None) -> int:
    """Remove orphaned uploads left behind by interrupted requests."""
    hours = settings.retention_hours if hours is None else hours
    cutoff = time.time() - hours * 3600
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.is_dir():
        return 0
    removed = 0
    for path in upload_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove expired upload %s: %s", path, exc)
    return removed


def cleanup_all() -> dict[str, Any]:
    """Run every retention task once."""
    result: dict[str, Any] = {
        "records": cleanup_old_records(hours=settings.retention_hours),
        "uploads": cleanup_old_uploads(hours=settings.retention_hours),
    }
    try:
        cleanup_old_pdfs(settings.pdf_dir, ttl_hours=settings.pdf_ttl_hours)
    except OSError as exc:  # pragma: no cover - best effort cleanup
        logger.warning("PDF retention cleanup failed: %s", exc)
    return result


async def retention_loop() -> None:
    """Periodically run retention cleanup for the lifetime of the process."""
    interval = max(settings.retention_interval_hours, 0.1) * 3600
    while True:
        await asyncio.sleep(interval)
        await asyncio.to_thread(cleanup_all)
