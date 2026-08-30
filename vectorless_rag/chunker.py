"""
chunker.py
----------
Converts a list of PageText objects into overlapping Chunk objects
suitable for LLM processing.

Chunking strategy (in priority order):
  1. Hard cap at ChunkConfig.max_chars characters
  2. Split at paragraph boundary (\\n\\n) when possible
  3. Fall back to sentence boundary (regex) if paragraph too long
  4. Last resort: hard cut at max_chars
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vectorless_rag.config import ChunkConfig
from vectorless_rag.pdf_parser import PageText


@dataclass
class Chunk:
    """A text slice ready to be sent to the LLM."""

    chunk_id: str
    """Unique identifier: 'p{first_page}_c{chunk_index_within_doc}'."""

    page_numbers: list[int]
    """1-indexed page numbers that contributed text to this chunk."""

    text: str
    """The actual text content."""

    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_pages(pages: list[PageText], config: ChunkConfig | None = None) -> list[Chunk]:
    """
    Split *pages* into overlapping chunks.

    Args:
        pages: Output of pdf_parser.parse_pdf().
        config: Chunking configuration; uses defaults if None.

    Returns:
        Ordered list of Chunk objects.
    """
    if config is None:
        config = ChunkConfig()

    # Concatenate all page text, keeping track of page boundaries
    # so each chunk knows which pages it came from.
    segments: list[tuple[str, int]] = []  # (text_segment, page_number)
    for page in pages:
        if page.text:
            segments.append((page.text, page.page_number))

    if not segments:
        return []

    # Flat text and per-character page mapping
    flat_text, char_page_map = _flatten_segments(segments)

    # Split into non-overlapping windows first
    raw_chunks = _split_with_boundaries(flat_text, config.max_chars, config.respect_paragraphs)

    # Add overlap prefix from previous chunk
    overlap_len = int(config.max_chars * config.overlap_ratio)
    chunks: list[Chunk] = []

    for i, (start, end) in enumerate(raw_chunks):
        if i > 0 and overlap_len > 0:
            prefix_start = max(raw_chunks[i - 1][1] - overlap_len, raw_chunks[i - 1][0])
            text = flat_text[prefix_start:end]
        else:
            text = flat_text[start:end]

        # Determine which pages are covered
        first_page = char_page_map[start]
        last_page = char_page_map[min(end - 1, len(char_page_map) - 1)]
        page_nums = list(range(first_page, last_page + 1))

        chunks.append(
            Chunk(
                chunk_id=f"p{first_page}_c{i}",
                page_numbers=page_nums,
                text=text.strip(),
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_segments(segments: list[tuple[str, int]]) -> tuple[str, list[int]]:
    """
    Concatenate all text segments into one string and build a
    character → page-number lookup array.
    """
    buffer: list[str] = []
    char_page: list[int] = []

    for text, page_num in segments:
        blob = text + "\n\n"  # explicit paragraph break between pages
        buffer.append(blob)
        char_page.extend([page_num] * len(blob))

    return "".join(buffer), char_page


def _split_with_boundaries(text: str, max_chars: int, respect_paragraphs: bool) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) byte ranges for non-overlapping chunks.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    total = len(text)

    while cursor < total:
        end = min(cursor + max_chars, total)

        if end >= total:
            spans.append((cursor, total))
            break

        # Try to split at paragraph boundary
        cut = end
        if respect_paragraphs:
            para_cut = text.rfind("\n\n", cursor, end)
            if para_cut != -1 and para_cut > cursor:
                cut = para_cut + 2  # include the blank line
            else:
                # Fall back to sentence boundary
                sentence_pattern = re.compile(r"(?<=[.!?])\s+")
                matches = list(sentence_pattern.finditer(text, cursor, end))
                if matches:
                    cut = matches[-1].end()

        spans.append((cursor, cut))
        cursor = cut

    return spans
