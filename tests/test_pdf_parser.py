"""Tests for pdf_parser.py"""
from __future__ import annotations

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from vectorless_rag.pdf_parser import PageText, _clean_text, infer_document_title, parse_pdf


class TestCleanText:
    def test_collapses_tabs_and_spaces(self):
        assert _clean_text("hello   \t world") == "hello world"

    def test_collapses_triple_newlines(self):
        result = _clean_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strips_leading_trailing(self):
        assert _clean_text("  hi  ") == "hi"

    def test_empty_string(self):
        assert _clean_text("") == ""


class TestInferDocumentTitle:
    def test_returns_first_line(self):
        pages = [PageText(page_number=1, text="Introduction\nSome body text", char_count=30)]
        assert infer_document_title(pages) == "Introduction"

    def test_empty_pages(self):
        assert infer_document_title([]) == "Untitled Document"

    def test_empty_first_page(self):
        pages = [PageText(page_number=1, text="", char_count=0)]
        assert infer_document_title(pages) == "Untitled Document"

    def test_truncates_long_title(self):
        long_title = "A" * 300
        pages = [PageText(page_number=1, text=long_title, char_count=300)]
        result = infer_document_title(pages)
        assert len(result) <= 200


class TestParsePdf:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path/file.pdf")

    def test_returns_list_of_page_texts(self, tmp_path):
        """Test with a real minimal fitz-generated PDF."""
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Hello World")
        doc.save(str(pdf_path))
        doc.close()

        pages = parse_pdf(pdf_path)
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert "Hello" in pages[0].text

    def test_page_numbers_are_1_indexed(self, tmp_path):
        import fitz

        pdf_path = tmp_path / "two_pages.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 72), f"Page {i + 1}")
        doc.save(str(pdf_path))
        doc.close()

        pages = parse_pdf(pdf_path)
        assert [p.page_number for p in pages] == [1, 2, 3]
