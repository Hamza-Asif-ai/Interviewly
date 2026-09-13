"""ORM models: User, Session, PrepGuide, PracticeVideo, AnalysisReport."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


class User(Base):
    """A platform user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(Base):
    """A user session grouping prep and practice activity."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="Interview session")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="sessions")
    prep_guides: Mapped[list["PrepGuide"]] = relationship(back_populates="session")
    videos: Mapped[list["PracticeVideo"]] = relationship(back_populates="session")


class PrepGuide(Base):
    """A generated preparation guide for a CV + JD + company."""

    __tablename__ = "prep_guides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id"), nullable=True, index=True
    )
    cv_filename: Mapped[str] = mapped_column(String(255))
    job_description: Mapped[str] = mapped_column(Text)
    company_overview: Mapped[str] = mapped_column(Text, default="")
    guide_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped["Session"] = relationship(back_populates="prep_guides")


class PracticeVideo(Base):
    """A recorded practice interview video."""

    __tablename__ = "practice_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id"), nullable=True, index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    audio_path: Mapped[str] = mapped_column(String(500), default="")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    question_prompt: Mapped[str] = mapped_column(Text, default="")
    trash: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped["Session"] = relationship(back_populates="videos")
    report: Mapped["AnalysisReport"] = relationship(
        back_populates="video", uselist=False
    )


class AnalysisReport(Base):
    """A diagnostic scorecard produced for a practice video."""

    __tablename__ = "analysis_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("practice_videos.id"), unique=True, index=True
    )
    transcript: Mapped[str] = mapped_column(Text, default="")
    report_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    video: Mapped["PracticeVideo"] = relationship(back_populates="report")