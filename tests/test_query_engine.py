"""Tests for query_engine.py — BFS structural retrieval."""
from __future__ import annotations

import pytest
from vectorless_rag.tree_model import HierarchyTree, TreeNode
from vectorless_rag.query_engine import QueryEngine, _score_node, _tokenise


def build_test_tree() -> HierarchyTree:
    """
    Builds a simple 3-level tree:
      root
        ├── Chapter 1: Machine Learning  [p1_c0]
        │     └── Section 1.1: Supervised Learning  [p2_c0]
        └── Chapter 2: Risk Management  [p3_c0]
    """
    root = TreeNode(node_id="root", title="Test Document", summary="", level=0)

    ch1 = TreeNode(node_id="ch1", title="Chapter 1: Machine Learning",
                   summary="Introduction to machine learning paradigms.", level=1,
                   source_refs=["p1_c0"])
    sec11 = TreeNode(node_id="sec11", title="Section 1.1: Supervised Learning",
                     summary="Supervised learning requires labelled data.", level=2,
                     source_refs=["p2_c0"])
    ch1.child_nodes.append(sec11)

    ch2 = TreeNode(node_id="ch2", title="Chapter 2: Risk Management",
                   summary="Covers enterprise risk frameworks and controls.", level=1,
                   source_refs=["p3_c0"])

    root.child_nodes.extend([ch1, ch2])

    return HierarchyTree(
        document_title="Test Document",
        source_file="test.pdf",
        created_at="2026-01-01T00:00:00Z",
        total_pages=10,
        root=root,
    )


class TestTokenise:
    def test_filters_short_tokens(self):
        tokens = _tokenise("hi an a")
        assert "hi" not in tokens  # too short

    def test_removes_stopwords(self):
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens

    def test_lowercases(self):
        tokens = _tokenise("Machine Learning")
        assert "machine" in tokens
        assert "learning" in tokens


class TestScoreNode:
    def test_exact_title_match_scores_high(self):
        node = TreeNode(node_id="n1", title="Machine Learning", summary="", level=1)
        tokens = _tokenise("machine learning")
        score = _score_node(node, tokens)
        assert score > 0.5

    def test_no_match_scores_zero(self):
        node = TreeNode(node_id="n1", title="Unrelated Content", summary="", level=1)
        tokens = _tokenise("machine learning")
        score = _score_node(node, tokens)
        assert score == 0.0

    def test_summary_match_contributes(self):
        node = TreeNode(node_id="n1", title="Chapter 5", summary="machine learning basics", level=1)
        tokens = _tokenise("machine learning")
        score = _score_node(node, tokens)
        assert score > 0.0


class TestQueryEngine:
    def setup_method(self):
        self.tree = build_test_tree()
        self.engine = QueryEngine(self.tree)

    def test_search_returns_relevant_nodes(self):
        results = self.engine.search("machine learning", top_k=3)
        assert len(results) > 0
        titles = [r.node.title for r in results]
        assert any("Machine Learning" in t or "Supervised" in t for t in titles)

    def test_results_sorted_by_score_descending(self):
        results = self.engine.search("risk management", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        results = self.engine.search("learning", top_k=1)
        assert len(results) <= 1

    def test_min_score_filters_results(self):
        results = self.engine.search("machine", min_score=0.9)
        assert all(r.score >= 0.9 for r in results)

    def test_path_includes_ancestors(self):
        results = self.engine.search("supervised learning", top_k=5)
        deep = next((r for r in results if "Supervised" in r.node.title), None)
        if deep:
            assert len(deep.path) >= 2  # at least root → chapter → section

    def test_get_node_by_id(self):
        node = self.engine.get_node_by_id("ch1")
        assert node is not None
        assert node.title == "Chapter 1: Machine Learning"

    def test_get_node_by_id_missing(self):
        assert self.engine.get_node_by_id("nonexistent") is None

    def test_get_path_to(self):
        path = self.engine.get_path_to("sec11")
        assert path is not None
        assert path[-1].node_id == "sec11"
        assert path[0].node_id == "root"

    def test_summarise_path(self):
        summary = self.engine.summarise_path("sec11")
        assert "Section 1.1" in summary
        assert "root" in summary.lower() or "Test Document" in summary

    def test_empty_query_returns_empty(self):
        results = self.engine.search("")
        assert results == []

    def test_no_results_for_unrelated_query(self):
        results = self.engine.search("quantum physics blockchain", min_score=0.8)
        assert len(results) == 0
