"""PDF export for interview prep guides (fpdf2)."""

from __future__ import annotations

import logging
import os
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict

import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF
from fpdf.enums import XPos, YPos

logger = logging.getLogger(__name__)

_FONT_FAMILY = "helvetica"

# Common smart/typographic characters that are NOT representable in the
# latin-1 range used by fpdf2's built-in core fonts. It is helpful to map
# them to portable equivalents instead of crashing the render.
_TYPOGRAPHIC_MAP = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201a": "'",   # single low-9 quote
    "\u201b": "'",   # single high-reversed-9 quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u201e": '"',   # double low-9 quote
    "\u201f": '"',   # double high-reversed-9 quote
    "\u2032": "'",   # prime
    "\u2033": '"',   # double prime
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u2009": " ",   # thin space
    "\u200a": " ",   # hair space
    "\u200b": "",    # zero-width space
    "\u202f": " ",   # narrow no-break space
    "\u2022": "-",   # bullet
    "\u2011": "-",   # non-breaking hyphen
    "\u2192": "->",  # rightwards arrow
    "\u27a1": "->",  # heavy rightwards arrow
    "\u2713": "OK",  # check mark
    "\u2714": "OK",  # heavy check mark
    "\u2717": "x",   # ballot X
    "\u2718": "x",   # heavy ballot X
    "\u2605": "*",   # black star
    "\u2606": "*",   # white star
    "\u00ad": "",    # soft hyphen
}


def sanitize_pdf_text(text: Any) -> str:
    """Normalize text and apply Arabic shaping/RTL display order for PDFs."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    for src, dst in _TYPOGRAPHIC_MAP.items():
        s = s.replace(src, dst)
    try:
        s = get_display(arabic_reshaper.reshape(s))
    except Exception:
        logger.warning("Arabic shaping failed for PDF text; using plain text.")
    return s


def _find_font(filename: str) -> Path | None:
    candidates = [
        os.getenv("PDF_FONT_PATH", ""),
        str(Path(__file__).resolve().parents[2] / "assets" / "fonts" / filename),
        "/usr/share/fonts/truetype/dejavu/" + filename,
        "C:/Windows/Fonts/" + filename.replace("DejaVuSans", "DejaVuSans"),
    ]
    try:
        import matplotlib

        candidates.append(
            str(
                Path(matplotlib.__file__).resolve().parent
                / "mpl-data/fonts/ttf"
                / filename
            )
        )
    except ImportError:  # pragma: no cover - optional font source
        pass
    for raw in candidates:
        if raw and Path(raw).is_file():
            return Path(raw)
    return None


def _register_unicode_font() -> str:
    global _FONT_FAMILY
    if _find_font("DejaVuSans.ttf") is None:
        logger.warning("Unicode PDF font not found; falling back to core fonts.")
        return _FONT_FAMILY
    _FONT_FAMILY = "Unicode"
    return _FONT_FAMILY


def _add_unicode_fonts(pdf: FPDF) -> None:
    regular = _find_font("DejaVuSans.ttf")
    if regular is None:
        return
    styles = {
        "": regular,
        "B": _find_font("DejaVuSans-Bold.ttf") or regular,
        "I": _find_font("DejaVuSans-Oblique.ttf") or regular,
        "BI": _find_font("DejaVuSans-BoldOblique.ttf") or regular,
    }
    for style, path in styles.items():
        pdf.add_font("Unicode", style, str(path))


class _PrepGuidePDF(FPDF):
    """A4 PDF with a footer page counter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _FONT_FAMILY == "Unicode":
            _add_unicode_fonts(self)

    def footer(self) -> None:
        if self.page_no() == 0:
            return
        self.set_y(-15)
        self.set_font(_FONT_FAMILY, "I", 8)
        self.set_text_color(130, 140, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _qa_pairs(items: Any) -> list[tuple[str, str]]:
    """Normalize a list of question items into (question, answer) tuples."""
    pairs: list[tuple[str, str]] = []
    for item in items or []:
        if isinstance(item, dict):
            pairs.append((item.get("question", ""), item.get("answer", "")))
        elif isinstance(item, str):
            pairs.append((item, ""))
    return pairs


def cleanup_old_pdfs(output_dir: str, ttl_hours: int = 48) -> None:
    """Delete generated PDFs older than their time-to-live."""
    if not os.path.isdir(output_dir):
        return
    cutoff = time.time() - ttl_hours * 3600
    for name in os.listdir(output_dir):
        if not name.startswith("prep_guide_") or not name.endswith(".pdf"):
            continue
        path = os.path.join(output_dir, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                logger.info("Removed expired PDF %s", path)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove expired PDF %s: %s", path, exc)


def build_prep_guide_pdf(
    guide: Dict[str, Any], output_dir: str, demo: bool = False
) -> str:
    """Render a prep guide dict into a PDF file and return its absolute path."""
    cleanup_old_pdfs(output_dir)
    _register_unicode_font()
    pdf = _PrepGuidePDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()

    # Header
    pdf.set_font(_FONT_FAMILY, "B", 18)
    pdf.set_text_color(26, 42, 74)
    pdf.multi_cell(
        0, 10, sanitize_pdf_text("Interviewly - Interview Preparation Guide"),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_draw_color(26, 42, 74)
    pdf.set_line_width(0.6)
    pdf.line(12, pdf.get_y(), pdf.w - 12, pdf.get_y())
    pdf.ln(6)
    pdf.set_font(_FONT_FAMILY, "I", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0, 6,
        sanitize_pdf_text(
            "Tailored questions with full STAR answers, positioning tips and "
            "skill-gap analysis."
        ),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)
    if demo or guide.get("_fallback"):
        pdf.set_font(_FONT_FAMILY, "B", 11)
        pdf.set_text_color(180, 40, 40)
        pdf.multi_cell(
            0, 7,
            sanitize_pdf_text(
                "DEMO DATA - not based on your CV or the job description."
            ),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.ln(2)
    pdf.set_text_color(0, 0, 0)

    def section(title: str) -> None:
        pdf.ln(2)
        pdf.set_font(_FONT_FAMILY, "B", 13)
        pdf.set_fill_color(26, 42, 74)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(
            0, 9, sanitize_pdf_text(title), fill=True,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.ln(3)
        pdf.set_text_color(0, 0, 0)

    def question_block(index: int, question: str, answer: str) -> None:
        question = sanitize_pdf_text(question)
        answer = sanitize_pdf_text(answer)
        pdf.set_font(_FONT_FAMILY, "B", 11)
        pdf.set_text_color(26, 42, 74)
        pdf.multi_cell(
            0, 6, f"{index}. {question}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.set_text_color(0, 0, 0)
        if answer:
            pdf.set_font(_FONT_FAMILY, "I", 10)
            pdf.set_text_color(120, 110, 90)
            pdf.multi_cell(
                0, 5, "STAR model answer:", new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            pdf.set_text_color(0, 0, 0)
            pdf.set_font(_FONT_FAMILY, "", 10)
            pdf.multi_cell(
                0, 5, answer, new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
        pdf.ln(3)

    section("Technical Questions")
    for i, (question, answer) in enumerate(
        _qa_pairs(guide.get("technical_questions")), start=1
    ):
        question_block(i, question, answer)

    section("Behavioral Questions")
    for i, (question, answer) in enumerate(
        _qa_pairs(guide.get("behavioral_questions")), start=1
    ):
        question_block(i, question, answer)

    section("Company-Specific Positioning Tips")
    pdf.set_font(_FONT_FAMILY, "", 10)
    for i, tip in enumerate(guide.get("positioning_tips") or [], start=1):
        if isinstance(tip, str) and tip.strip():
            pdf.multi_cell(
                0, 5, sanitize_pdf_text(f"{i}. {tip}"),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
    pdf.ln(3)

    section("Skill-Gap Analysis")
    pdf.set_font(_FONT_FAMILY, "", 10)
    for gap in guide.get("skill_gaps") or []:
        if isinstance(gap, str) and gap.strip():
            pdf.multi_cell(
                0, 5, sanitize_pdf_text(f"- {gap}"),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
    pdf.ln(3)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"prep_guide_{uuid.uuid4().hex[:8]}.pdf")
    pdf.output(out_path)
    logger.info("Prep guide PDF written to: %s", out_path)
    return out_path