"""Extract plain text from CV files (PDF, DOCX, DOC, TXT)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CVParseError(Exception):
    """Raised when a CV file cannot be parsed."""


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using pdfplumber with PyPDF2 fallback."""
    text: str = ""
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("pdfplumber failed (%s), trying PyPDF2.", exc)
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as inner_exc:
            raise CVParseError(f"Could not parse PDF: {inner_exc}") from inner_exc
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document

        document = Document(file_path)
        paragraphs = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.append(cell.text)
        return "\n".join(p for p in paragraphs if p.strip()).strip()
    except Exception as exc:
        raise CVParseError(f"Could not parse DOCX file: {exc}") from exc


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a CV file based on its extension.

    Supports: .pdf, .docx, .txt. Unsupported extensions raise CVParseError.
    """
    suffix = Path(file_path).suffix.lower()
    if not os.path.exists(file_path):
        raise CVParseError(f"File not found: {file_path}")

    if suffix == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif suffix == ".docx":
        text = extract_text_from_docx(file_path)
    elif suffix == ".txt":
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    else:
        raise CVParseError(f"Unsupported CV file type: {suffix}")

    if not text:
        raise CVParseError("No extractable text found in the CV file.")
    return text