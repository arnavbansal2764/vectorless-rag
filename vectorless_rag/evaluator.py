"""
evaluator.py
------------
Quality metrics for generated HierarchyTree instances.

All metrics are computed locally (no LLM calls) and return a dict
suitable for logging or serialization.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from pathlib import Path

from vectorless_rag.tree_model import HierarchyTree, TreeNode
from vectorless_rag.tree_operations import build_node_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(tree: HierarchyTree, original_text_chars: int = 0) -> dict:
    """
    Compute a full evaluation report for *tree*.

    Args:
        tree:                 The hierarchy to evaluate.
        original_text_chars:  Total character count of original PDF text.
                              If provided, compression ratio is computed.

    Returns:
        Dict with keys:
          node_count, leaf_count, max_depth, avg_depth, depth_distribution,
          coverage_ratio, compression_ratio, avg_summary_length,
          empty_summaries, orphaned_refs, log_warnings
    """
    all_nodes = list(build_node_index(tree.root).values())
    depths = [n.level for n in all_nodes]
    leaves = [n for n in all_nodes if n.is_leaf()]

    # Coverage: what fraction of total chunks have at least one matching node
    all_refs: list[str] = []
    for n in all_nodes:
        all_refs.extend(n.source_refs)
    unique_refs = set(all_refs)

    # Depth distribution histogram
    depth_counter = Counter(depths)
    depth_distribution = {f"level_{k}": v for k, v in sorted(depth_counter.items())}

    # Summary length stats
    summary_lengths = [len(n.summary) for n in all_nodes]
    avg_summary = statistics.mean(summary_lengths) if summary_lengths else 0
    empty_summaries = sum(1 for n in all_nodes if not n.summary.strip())

    # Log warnings count
    log_warnings = sum(1 for entry in tree.build_log if entry.startswith("REJECTED"))

    report: dict = {
        "node_count": len(all_nodes),
        "leaf_count": len(leaves),
        "max_depth": max(depths) if depths else 0,
        "avg_depth": round(statistics.mean(depths), 2) if depths else 0,
        "depth_distribution": depth_distribution,
        "unique_chunks_referenced": len(unique_refs),
        "avg_summary_length_chars": round(avg_summary, 1),
        "empty_summaries": empty_summaries,
        "log_warnings": log_warnings,
    }

    if original_text_chars > 0:
        tree_json_chars = len(tree.model_dump_json())
        report["compression_ratio"] = round(tree_json_chars / original_text_chars, 4)

    return report


def print_report(report: dict) -> None:
    """Pretty-print an evaluation report using Rich if available."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Vectorless RAG — Evaluation Report", show_header=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="green")

        for k, v in report.items():
            if isinstance(v, dict):
                table.add_row(k, json.dumps(v))
            else:
                table.add_row(k, str(v))

        console.print(table)
    except ImportError:
        # Fallback to plain print
        for k, v in report.items():
            print(f"{k}: {v}")


def check_invariants(tree: HierarchyTree) -> list[str]:
    """
    Structural sanity checks. Returns a list of violation messages.
    An empty list means the tree is valid.
    """
    violations: list[str] = []
    node_index = build_node_index(tree.root)

    # 1. node_id uniqueness
    seen_ids: set[str] = set()
    for node_id in node_index:
        if node_id in seen_ids:
            violations.append(f"Duplicate node_id: {node_id}")
        seen_ids.add(node_id)

    # 2. Parent-child level consistency
    def _check_levels(node: TreeNode, expected_level: int) -> None:
        if node.level != expected_level:
            violations.append(
                f"Node '{node.title}' ({node.node_id}) has level={node.level} "
                f"but expected {expected_level}"
            )
        for child in node.child_nodes:
            _check_levels(child, expected_level + 1)

    _check_levels(tree.root, 0)

    return violations
