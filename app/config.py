"""Environment configuration and application settings for Interviewly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
PDFS_DIR = DATA_DIR / "pdfs"
DB_PATH = DATA_DIR / "interviewly.db"

for _dir in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR, PDFS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url: str = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "25"))
        self.prep_max_tokens: int = int(os.getenv("PREP_MAX_TOKENS", "8192"))
        self.prep_retries: int = int(os.getenv("PREP_RETRIES", "2"))
        self.prep_fallback_path: str = os.getenv(
            "PREP_FALLBACK_PATH", str(BASE_DIR / "data" / "prep_guide_fallback.json")
        )
        self.practice_fallback_path: str = os.getenv(
            "PRACTICE_FALLBACK_PATH",
            str(BASE_DIR / "data" / "practice_scorecard_fallback.json"),
        )
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "base")
        # Empty WHISPER_LANG => auto-detect the spoken language.
        self.whisper_lang: str = os.getenv("WHISPER_LANG", "").strip()
        self.max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
        self.max_upload_bytes: int = self.max_upload_mb * 1024 * 1024
        self.pdf_ttl_hours: int = int(os.getenv("PDF_TTL_HOURS", "48"))
        self.rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
        self.max_concurrent_analyses: int = int(
            os.getenv("MAX_CONCURRENT_ANALYSES", "2")
        )
        self.cors_origins: List[str] = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "").split(",")
            if origin.strip()
        ]

        self.database_url: str = os.getenv(
            "DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}"
        )
        self.chroma_dir: str = os.getenv("CHROMA_DIR", CHROMA_DIR.as_posix())
        self.upload_dir: str = os.getenv("UPLOAD_DIR", UPLOAD_DIR.as_posix())
        self.pdf_dir: str = os.getenv("PDF_DIR", PDFS_DIR.as_posix())

        self.whisper_api_enabled: bool = (
            os.getenv("WHISPER_API_ENABLED", "false").lower() == "true"
        )

        self.allowed_extensions: List[str] = [
            ".pdf",
            ".docx",
            ".doc",
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
        ]

    def has_llm(self) -> bool:
        """Whether an OpenAI-compatible API key is configured."""
        return bool(self.openai_api_key)


settings = Settings()