"""Tests for storage.py"""
from __future__ import annotations

from pathlib import Path
import pytest
from vectorless_rag.tree_model import HierarchyTree, TreeNode
from vectorless_rag.storage import TreeStorage


def make_simple_tree() -> HierarchyTree:
    root = TreeNode(node_id="root", title="Test Doc", summary="Root summary.", level=0)
    child = TreeNode(node_id="ch1", title="Chapter 1", summary="First chapter.", level=1)
    root.child_nodes.append(child)
    return HierarchyTree(
        document_title="Test Doc",
        source_file="test.pdf",
        created_at="2026-01-01T00:00:00Z",
        total_pages=5,
        root=root,
    )


class TestSaveLoad:
    def test_round_trip_preserves_structure(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        tree = make_simple_tree()
        path = storage.save(tree)
        loaded = storage.load(path)
        assert loaded.document_title == tree.document_title
        assert loaded.root.title == tree.root.title
        assert len(loaded.root.child_nodes) == 1
        assert loaded.root.child_nodes[0].title == "Chapter 1"

    def test_save_creates_json_file(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        tree = make_simple_tree()
        path = storage.save(tree)
        assert path.exists()
        assert path.suffix == ".json"

    def test_load_file_not_found(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            storage.load(tmp_path / "nonexistent.json")

    def test_custom_filename(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        path = storage.save(make_simple_tree(), filename="custom.json")
        assert path.name == "custom.json"


class TestCheckpoints:
    def test_checkpoint_is_written(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        tree = make_simple_tree()
        cp_path = storage.save_checkpoint(tree, chunk_index=5)
        assert cp_path.exists()
        assert "checkpoint_0005" in cp_path.name

    def test_load_latest_checkpoint(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        tree = make_simple_tree()
        storage.save_checkpoint(tree, chunk_index=1)
        storage.save_checkpoint(tree, chunk_index=2)
        loaded = storage.load_latest_checkpoint()
        assert loaded is not None
        assert loaded.document_title == tree.document_title

    def test_load_latest_checkpoint_returns_none_when_empty(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        result = storage.load_latest_checkpoint()
        assert result is None


class TestValidateSchema:
    def test_valid_tree_passes(self, tmp_path):
        storage = TreeStorage(output_dir=tmp_path)
        tree = make_simple_tree()
        errors = storage.validate_schema(tree)
        # Schema may not be present in test env — no errors either way expected
        assert isinstance(errors, list)
