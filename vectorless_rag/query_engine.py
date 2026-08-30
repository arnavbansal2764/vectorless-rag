"""
query_engine.py
---------------
Structural retrieval over a HierarchyTree using Breadth-First Search (BFS).

The QueryEngine traverses the tree level by level, scoring each node by how
well its title + summary match the query string.  It returns the top-K most
relevant nodes along with their ancestry path (for context).

No vector embeddings, no external models — just text matching + BFS.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from vectorless_rag.tree_model import HierarchyTree, TreeNode
from vectorless_rag.tree_operations import build_node_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """A single matching node returned by the query engine."""

    node: TreeNode
    score: float
    """Relevance score in [0, 1]. Higher = more relevant."""

    path: list[str] = field(default_factory=list)
    """Title breadcrumb from root → this node, e.g. ['Document Root', 'Chapter 1', 'Section 1.2']"""

    source_refs: list[str] = field(default_factory=list)
    """Chunk IDs that contributed to this node (for downstream retrieval)."""


class BFSEntry(NamedTuple):
    node: TreeNode
    path: list[str]  # breadcrumb of titles


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------


class QueryEngine:
    """
    BFS-based structural query engine over a HierarchyTree.

    Usage
    -----
    >>> engine = QueryEngine(tree)
    >>> results = engine.search("risk management framework", top_k=5)
    >>> for r in results:
    ...     print(r.score, " / ".join(r.path))
    """

    def __init__(self, tree: HierarchyTree) -> None:
        self.tree = tree
        self._node_index = build_node_index(tree.root)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[QueryResult]:
        """
        Return the top-K nodes most relevant to *query* via BFS traversal.

        BFS processes nodes level by level. Each node is scored by keyword
        overlap between the query and the node's title + summary. Results
        are sorted by score descending.

        Args:
            query:     Free-text question or keyword phrase.
            top_k:     Maximum number of results to return.
            min_score: Discard results with score below this threshold.

        Returns:
            List of QueryResult ordered by descending relevance score.
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        results: list[QueryResult] = []

        # BFS queue: (node, breadcrumb_path)
        queue: list[BFSEntry] = [BFSEntry(self.tree.root, [self.tree.root.title])]

        while queue:
            entry = queue.pop(0)
            node, path = entry.node, entry.path

            score = _score_node(node, query_tokens)

            if score >= min_score and node.node_id != "root":
                results.append(
                    QueryResult(
                        node=node,
                        score=score,
                        path=list(path),
                        source_refs=list(node.source_refs),
                    )
                )

            # Enqueue children with extended breadcrumb
            for child in node.child_nodes:
                queue.append(BFSEntry(child, path + [child.title]))

        # Sort by score descending, cap at top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get_node_by_id(self, node_id: str) -> TreeNode | None:
        """Direct O(1) lookup of a node by its ID."""
        return self._node_index.get(node_id)

    def get_path_to(self, node_id: str) -> list[TreeNode] | None:
        """
        Return the ancestor chain from root to the node with *node_id*
        (inclusive), or None if not found.
        """
        return _bfs_path(self.tree.root, node_id)

    def summarise_path(self, node_id: str) -> str:
        """
        Return a formatted breadcrumb + summary for a node, including its
        full ancestor chain. Useful for providing LLM context during Q&A.
        """
        path = self.get_path_to(node_id)
        if path is None:
            return ""

        lines: list[str] = []
        for i, node in enumerate(path):
            indent = "  " * i
            lines.append(f"{indent}[{node.node_id}] {node.title}")
            if node.summary:
                lines.append(f"{indent}  → {node.summary}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> set[str]:
    """Lowercase alphabetic tokens, length >= 3, with common stopwords removed."""
    _STOPWORDS = {
        "the", "and", "for", "are", "was", "that", "with", "this",
        "from", "have", "has", "its", "not", "but", "can", "into",
    }
    tokens = set(re.findall(r"[a-z]{3,}", text.lower()))
    return tokens - _STOPWORDS


def _score_node(node: TreeNode, query_tokens: set[str]) -> float:
    """
    Score a node on keyword overlap:
      - title tokens weighted 2×
      - summary tokens weighted 1×
    Normalised to [0, 1] by dividing by the number of query tokens.
    """
    if not query_tokens:
        return 0.0

    title_tokens = _tokenise(node.title)
    summary_tokens = _tokenise(node.summary)

    title_hits = len(query_tokens & title_tokens)
    summary_hits = len(query_tokens & summary_tokens)

    raw = title_hits * 2 + summary_hits
    max_possible = len(query_tokens) * 2  # all tokens matching title
    return min(raw / max_possible, 1.0)


# ---------------------------------------------------------------------------
# BFS path helper
# ---------------------------------------------------------------------------


def _bfs_path(root: TreeNode, target_id: str) -> list[TreeNode] | None:
    """
    BFS returning the full path from *root* to the node with *target_id*,
    or None if not found.
    """
    # Queue entries: (node, path_so_far)
    queue: list[tuple[TreeNode, list[TreeNode]]] = [(root, [root])]

    while queue:
        node, path = queue.pop(0)
        if node.node_id == target_id:
            return path
        for child in node.child_nodes:
            queue.append((child, path + [child]))

    return None
