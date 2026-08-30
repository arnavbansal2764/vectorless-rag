"""
tree_model.py
-------------
Pydantic data models for the Vectorless RAG hierarchy tree.

- TreeNode   : a single node in the hierarchy
- HierarchyTree : the complete in-memory document tree
- TreeOperation  : an atomic mutation proposed by the LLM
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core node model
# ---------------------------------------------------------------------------


class TreeNode(BaseModel):
    """A single node in the document hierarchy."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    """Short unique ID used as reference in LLM prompts."""

    title: str
    """Human-readable section title."""

    summary: str = ""
    """Concise, information-dense summary (2-4 sentences)."""

    source_refs: list[str] = Field(default_factory=list)
    """Chunk IDs that contributed to this node, e.g. ['p3_c1', 'p4_c0']."""

    level: int = 0
    """Tree depth: 0 = root, 1 = major section, 2 = subsection, …"""

    child_nodes: list[TreeNode] = Field(default_factory=list)
    """Ordered list of child nodes."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Extensible bag: page ranges, keywords, confidence scores, etc."""

    model_config = {"arbitrary_types_allowed": True}

    # ---- helpers -----------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.child_nodes) == 0

    def skeleton_dict(self) -> dict:
        """Return a compact dict (no summaries) for use in LLM prompts."""
        return {
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "children": [c.skeleton_dict() for c in self.child_nodes],
        }


# ---------------------------------------------------------------------------
# Full tree model
# ---------------------------------------------------------------------------


class HierarchyTree(BaseModel):
    """Root container for the document hierarchy."""

    document_title: str
    source_file: str
    created_at: str
    total_pages: int
    root: TreeNode
    build_log: list[str] = Field(default_factory=list)
    """Audit trail: rejected ops, merge decisions, warnings."""

    def add_log(self, message: str) -> None:
        self.build_log.append(message)


# ---------------------------------------------------------------------------
# Tree operation model (LLM output)
# ---------------------------------------------------------------------------


class OpType(str, Enum):
    CREATE_CHILD = "CREATE_CHILD"
    UPDATE_SUMMARY = "UPDATE_SUMMARY"
    MERGE_NODES = "MERGE_NODES"
    CREATE_SIBLING = "CREATE_SIBLING"


class TreeOperation(BaseModel):
    """One atomic mutation proposed by the LLM for a given chunk."""

    op: OpType
    """The type of mutation to apply."""

    target_node_id: str
    """node_id of the node to act on."""

    payload: dict[str, Any]
    """Op-specific data:
        - CREATE_CHILD / CREATE_SIBLING: {title, summary}
        - UPDATE_SUMMARY: {summary}
        - MERGE_NODES: {merge_with_node_id}
    """


class LLMTreeResponse(BaseModel):
    """Parsed structure of the LLM's JSON reply."""

    reasoning: str = ""
    operations: list[TreeOperation] = Field(default_factory=list)
