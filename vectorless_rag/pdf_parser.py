"""
pdf_parser.py
-------------
Extracts text from a PDF file page-by-page using PyMuPDF (fitz).

Returns a list of PageText objects — one per page — with whitespace
normalised but no further processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageText:
    """Raw text extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    char_count: int


def parse_pdf(path: str | Path) -> list[PageText]:
    """
    Open *path* and extract cleaned text from every page.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        Ordered list of PageText, one per page (empty pages included).

    Raises:
        FileNotFoundError: If *path* does not exist.
        fitz.FileDataError: If the file is not a valid PDF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[PageText] = []

    with fitz.open(str(path)) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            raw = page.get_text("text")  # plain text extraction
            clean = _clean_text(raw)
            pages.append(
                PageText(
                    page_number=page_index + 1,
                    text=clean,
                    char_count=len(clean),
                )
            )

    return pages


def infer_document_title(pages: list[PageText], max_chars: int = 200) -> str:
    """
    Best-effort document title: first non-empty line of page 1.

    Falls back to 'Untitled Document' if page 1 is empty.
    """
    if not pages:
        return "Untitled Document"
    first_line = pages[0].text.strip().splitlines()[0] if pages[0].text.strip() else ""
    return first_line[:max_chars].strip() or "Untitled Document"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_text(raw: str) -> str:
    """Normalise whitespace from raw PDF extraction."""
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", raw)
    # Normalise multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
