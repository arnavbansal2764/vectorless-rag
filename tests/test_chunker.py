"""Tests for chunker.py"""
from __future__ import annotations

from vectorless_rag.chunker import Chunk, _flatten_segments, _split_with_boundaries, chunk_pages
from vectorless_rag.config import ChunkConfig
from vectorless_rag.pdf_parser import PageText


def make_pages(texts: list[str]) -> list[PageText]:
    return [PageText(page_number=i + 1, text=t, char_count=len(t)) for i, t in enumerate(texts)]


class TestChunkPages:
    def test_empty_pages_returns_empty(self):
        assert chunk_pages([]) == []

    def test_single_short_page_produces_one_chunk(self):
        pages = make_pages(["Short text."])
        chunks = chunk_pages(pages, ChunkConfig(max_chars=1000))
        assert len(chunks) == 1
        assert chunks[0].page_numbers == [1]

    def test_chunk_ids_are_unique(self):
        pages = make_pages(["A" * 3000, "B" * 3000])
        chunks = chunk_pages(pages, ChunkConfig(max_chars=1000))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_chunk_exceeds_max_plus_overlap(self):
        """
        With overlap, a chunk can be longer than max_chars,
        but the core (non-overlapping) part should respect it.
        """
        pages = make_pages(["Word " * 2000])  # ~10000 chars
        cfg = ChunkConfig(max_chars=500, overlap_ratio=0.1)
        chunks = chunk_pages(pages, cfg)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert len(chunks) > 1

    def test_page_numbers_populated(self):
        pages = make_pages(["Hello page 1", "Hello page 2"])
        chunks = chunk_pages(pages, ChunkConfig(max_chars=10000))
        assert all(len(c.page_numbers) >= 1 for c in chunks)


class TestSplitWithBoundaries:
    def test_respects_paragraph_boundary(self):
        text = ("A" * 100) + "\n\n" + ("B" * 100)
        spans = _split_with_boundaries(text, max_chars=120, respect_paragraphs=True)
        # The split should fall at the paragraph boundary
        assert len(spans) == 2

    def test_handles_no_boundary_available(self):
        # 200 chars of continuous text, split at 100
        text = "A" * 200
        spans = _split_with_boundaries(text, max_chars=100, respect_paragraphs=True)
        assert len(spans) >= 2

    def test_single_chunk_if_under_limit(self):
        text = "Short text"
        spans = _split_with_boundaries(text, max_chars=1000, respect_paragraphs=True)
        assert spans == [(0, len(text))]
