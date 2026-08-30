"""Tests for tree_operations.py"""
from __future__ import annotations

import pytest
from vectorless_rag.config import TreeConfig
from vectorless_rag.tree_model import HierarchyTree, LLMTreeResponse, OpType, TreeNode, TreeOperation
from vectorless_rag.tree_operations import apply_operation, build_node_index


def make_tree() -> tuple[HierarchyTree, dict]:
    root = TreeNode(node_id="root", title="Document Root", summary="", level=0)
    tree = HierarchyTree(
        document_title="Test Doc",
        source_file="test.pdf",
        created_at="2026-01-01T00:00:00Z",
        total_pages=10,
        root=root,
    )
    index = build_node_index(tree.root)
    return tree, index


class TestCreateChild:
    def test_creates_child_node(self):
        tree, index = make_tree()
        op = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                           payload={"title": "Chapter 1", "summary": "Intro chapter."})
        result = apply_operation(tree, index, op, "p1_c0")
        assert result is True
        assert len(tree.root.child_nodes) == 1
        assert tree.root.child_nodes[0].title == "Chapter 1"

    def test_child_gets_correct_level(self):
        tree, index = make_tree()
        op = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                           payload={"title": "Ch1", "summary": "..."})
        apply_operation(tree, index, op, "p1_c0")
        assert tree.root.child_nodes[0].level == 1

    def test_rejects_missing_title(self):
        tree, index = make_tree()
        op = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                           payload={"title": "", "summary": "No title op."})
        result = apply_operation(tree, index, op, "p1_c0")
        assert result is False

    def test_deduplicates_similar_titles(self):
        tree, index = make_tree()
        op1 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                            payload={"title": "Introduction", "summary": "First."})
        op2 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                            payload={"title": "Introduction", "summary": "Second."})
        apply_operation(tree, index, op1, "p1_c0")
        apply_operation(tree, index, op2, "p1_c1")
        # Should merge, not create duplicate
        assert len(tree.root.child_nodes) == 1

    def test_flattens_beyond_max_depth(self):
        tree, index = make_tree()
        cfg = TreeConfig(max_depth=1)
        # Create a child at level 1
        op1 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                            payload={"title": "Ch1", "summary": "..."})
        apply_operation(tree, index, op1, "p1_c0", cfg)
        child_id = tree.root.child_nodes[0].node_id
        # Try to go deeper — should be clamped
        op2 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id=child_id,
                            payload={"title": "Sub1", "summary": "..."})
        apply_operation(tree, index, op2, "p1_c1", cfg)
        grandchild = tree.root.child_nodes[0].child_nodes[0]
        assert grandchild.level == cfg.max_depth


class TestUpdateSummary:
    def test_appends_to_existing_summary(self):
        tree, index = make_tree()
        tree.root.summary = "Original."
        op = TreeOperation(op=OpType.UPDATE_SUMMARY, target_node_id="root",
                           payload={"summary": "Additional."})
        apply_operation(tree, index, op, "p1_c0")
        assert "Original." in tree.root.summary
        assert "Additional." in tree.root.summary

    def test_rejects_empty_summary(self):
        tree, index = make_tree()
        op = TreeOperation(op=OpType.UPDATE_SUMMARY, target_node_id="root",
                           payload={"summary": ""})
        result = apply_operation(tree, index, op, "p1_c0")
        assert result is False


class TestCreateSibling:
    def test_creates_sibling_node(self):
        tree, index = make_tree()
        # First create a child so we have something to sibling
        op1 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                            payload={"title": "Ch1", "summary": "..."})
        apply_operation(tree, index, op1, "p1_c0")
        ch1_id = tree.root.child_nodes[0].node_id

        op2 = TreeOperation(op=OpType.CREATE_SIBLING, target_node_id=ch1_id,
                            payload={"title": "Ch2", "summary": "Second chapter."})
        result = apply_operation(tree, index, op2, "p2_c0")
        assert result is True
        assert len(tree.root.child_nodes) == 2

    def test_rejects_sibling_of_root(self):
        tree, index = make_tree()
        op = TreeOperation(op=OpType.CREATE_SIBLING, target_node_id="root",
                           payload={"title": "Sibling", "summary": "..."})
        result = apply_operation(tree, index, op, "p1_c0")
        assert result is False


class TestMergeNodes:
    def test_merges_source_into_target(self):
        tree, index = make_tree()
        op1 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                            payload={"title": "Ch1", "summary": "Chapter one."})
        op2 = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="root",
                            payload={"title": "Ch2", "summary": "Chapter two."})
        apply_operation(tree, index, op1, "p1_c0")
        apply_operation(tree, index, op2, "p2_c0")

        ch1_id = tree.root.child_nodes[0].node_id
        ch2_id = tree.root.child_nodes[1].node_id

        op_merge = TreeOperation(op=OpType.MERGE_NODES, target_node_id=ch1_id,
                                 payload={"merge_with_node_id": ch2_id})
        result = apply_operation(tree, index, op_merge, "p3_c0")
        assert result is True
        assert len(tree.root.child_nodes) == 1
        assert "Chapter two." in tree.root.child_nodes[0].summary


class TestUnknownTargetId:
    def test_rejects_unknown_node_id(self):
        tree, index = make_tree()
        op = TreeOperation(op=OpType.CREATE_CHILD, target_node_id="nonexistent",
                           payload={"title": "X", "summary": "Y"})
        result = apply_operation(tree, index, op, "p1_c0")
        assert result is False
        assert any("REJECTED" in log for log in tree.build_log)
