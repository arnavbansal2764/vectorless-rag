"""
tree_operations.py
------------------
Validates and applies atomic tree mutations (TreeOperation) to the live tree.

Each operation type has its own validator + applicator:

  CREATE_CHILD    → add a new child node under target
  CREATE_SIBLING  → add a node at the same level as target
  UPDATE_SUMMARY  → append to / replace an existing node's summary
  MERGE_NODES     → copy source_refs from one node into another, then delete
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from vectorless_rag.config import TreeConfig
from vectorless_rag.tree_model import HierarchyTree, OpType, TreeNode, TreeOperation

logger = logging.getLogger(__name__)

NodeIndex = dict[str, TreeNode]  # node_id → node


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_operation(
    tree: HierarchyTree,
    node_index: NodeIndex,
    op: TreeOperation,
    chunk_id: str,
    config: TreeConfig | None = None,
) -> bool:
    """
    Validate and apply *op* to *tree*.

    Returns True if the operation was applied, False if it was rejected.
    Rejection details are appended to tree.build_log.
    """
    if config is None:
        config = TreeConfig()

    target = node_index.get(op.target_node_id)
    if target is None:
        msg = f"REJECTED [{op.op}] @ chunk={chunk_id}: unknown target_node_id={op.target_node_id!r}"
        tree.add_log(msg)
        logger.warning(msg)
        return False

    try:
        if op.op == OpType.CREATE_CHILD:
            return _create_child(tree, node_index, target, op, chunk_id, config)
        elif op.op == OpType.CREATE_SIBLING:
            return _create_sibling(tree, node_index, target, op, chunk_id, config)
        elif op.op == OpType.UPDATE_SUMMARY:
            return _update_summary(tree, target, op, chunk_id)
        elif op.op == OpType.MERGE_NODES:
            return _merge_nodes(tree, node_index, target, op, chunk_id)
        else:
            tree.add_log(f"REJECTED unknown op type: {op.op}")
            return False
    except Exception:
        logger.exception("Unexpected error applying op %s", op)
        tree.add_log(f"ERROR applying {op.op} @ chunk={chunk_id}")
        return False


def build_node_index(root: TreeNode) -> NodeIndex:
    """BFS walk to build a flat {node_id: node} index from the full tree."""
    index: NodeIndex = {}
    queue = [root]
    while queue:
        node = queue.pop(0)
        index[node.node_id] = node
        queue.extend(node.child_nodes)
    return index


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------


def _create_child(
    tree: HierarchyTree,
    node_index: NodeIndex,
    target: TreeNode,
    op: TreeOperation,
    chunk_id: str,
    config: TreeConfig,
) -> bool:
    title = op.payload.get("title", "").strip()
    summary = op.payload.get("summary", "").strip()

    if not title:
        tree.add_log(f"REJECTED CREATE_CHILD @ {chunk_id}: missing title")
        return False

    new_level = target.level + 1
    if new_level > config.max_depth:
        tree.add_log(
            f"DEPTH LIMIT: CREATE_CHILD '{title}' would be level {new_level} "
            f"(max {config.max_depth}). Flattening to level {config.max_depth}."
        )
        new_level = config.max_depth

    # Check for near-duplicate sibling — prefer merge over duplicate creation
    existing = _find_similar_sibling(target.child_nodes, title, config.merge_threshold)
    if existing is not None:
        tree.add_log(
            f"DEDUP: CREATE_CHILD '{title}' merged into existing '{existing.title}' "
            f"@ chunk={chunk_id}"
        )
        if summary:
            existing.summary = f"{existing.summary} {summary}".strip()
        existing.source_refs.append(chunk_id)
        return True

    new_node = TreeNode(
        title=title,
        summary=summary,
        level=new_level,
        source_refs=[chunk_id],
    )
    target.child_nodes.append(new_node)
    node_index[new_node.node_id] = new_node
    logger.debug("CREATE_CHILD '%s' under '%s'", title, target.title)
    return True


def _create_sibling(
    tree: HierarchyTree,
    node_index: NodeIndex,
    target: TreeNode,
    op: TreeOperation,
    chunk_id: str,
    config: TreeConfig,
) -> bool:
    title = op.payload.get("title", "").strip()
    summary = op.payload.get("summary", "").strip()

    if not title:
        tree.add_log(f"REJECTED CREATE_SIBLING @ {chunk_id}: missing title")
        return False

    # A sibling lives in the same parent — find the parent via BFS
    parent = _find_parent(tree.root, target.node_id)
    if parent is None:
        # Target is root — nowhere to add sibling
        tree.add_log(f"REJECTED CREATE_SIBLING @ {chunk_id}: target is root node")
        return False

    # Duplicate check
    existing = _find_similar_sibling(parent.child_nodes, title, config.merge_threshold)
    if existing is not None:
        tree.add_log(
            f"DEDUP: CREATE_SIBLING '{title}' merged into '{existing.title}' @ chunk={chunk_id}"
        )
        if summary:
            existing.summary = f"{existing.summary} {summary}".strip()
        existing.source_refs.append(chunk_id)
        return True

    new_node = TreeNode(
        title=title,
        summary=summary,
        level=target.level,
        source_refs=[chunk_id],
    )
    parent.child_nodes.append(new_node)
    node_index[new_node.node_id] = new_node
    logger.debug("CREATE_SIBLING '%s' beside '%s'", title, target.title)
    return True


def _update_summary(
    tree: HierarchyTree,
    target: TreeNode,
    op: TreeOperation,
    chunk_id: str,
) -> bool:
    addition = op.payload.get("summary", "").strip()
    if not addition:
        tree.add_log(f"REJECTED UPDATE_SUMMARY @ {chunk_id}: empty summary")
        return False

    if target.summary:
        target.summary = f"{target.summary} {addition}"
    else:
        target.summary = addition

    if chunk_id not in target.source_refs:
        target.source_refs.append(chunk_id)

    logger.debug("UPDATE_SUMMARY for '%s'", target.title)
    return True


def _merge_nodes(
    tree: HierarchyTree,
    node_index: NodeIndex,
    target: TreeNode,
    op: TreeOperation,
    chunk_id: str,
) -> bool:
    merge_id = op.payload.get("merge_with_node_id", "").strip()
    if not merge_id:
        tree.add_log(f"REJECTED MERGE_NODES @ {chunk_id}: missing merge_with_node_id")
        return False

    source = node_index.get(merge_id)
    if source is None:
        tree.add_log(f"REJECTED MERGE_NODES @ {chunk_id}: merge_with_node_id={merge_id!r} not found")
        return False

    if source.node_id == target.node_id:
        tree.add_log(f"REJECTED MERGE_NODES @ {chunk_id}: cannot merge node with itself")
        return False

    # Absorb source's children, refs, and summary extension into target
    target.child_nodes.extend(source.child_nodes)
    for ref in source.source_refs:
        if ref not in target.source_refs:
            target.source_refs.append(ref)
    if source.summary and source.summary not in target.summary:
        target.summary = f"{target.summary} {source.summary}".strip()

    # Remove source from its parent
    parent = _find_parent(tree.root, source.node_id)
    if parent is not None:
        parent.child_nodes = [c for c in parent.child_nodes if c.node_id != source.node_id]
    node_index.pop(source.node_id, None)

    tree.add_log(f"MERGED '{source.title}' → '{target.title}' @ chunk={chunk_id}")
    logger.debug("MERGE_NODES: '%s' absorbed into '%s'", source.title, target.title)
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_similar_sibling(
    siblings: list[TreeNode],
    new_title: str,
    threshold: float,
) -> TreeNode | None:
    """Return the first sibling whose title is similar to *new_title*, or None."""
    nl = new_title.lower()
    for sibling in siblings:
        ratio = SequenceMatcher(None, sibling.title.lower(), nl).ratio()
        if ratio >= threshold:
            return sibling
    return None


def _find_parent(root: TreeNode, target_id: str) -> TreeNode | None:
    """BFS to find the parent of the node with *target_id*."""
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in node.child_nodes:
            if child.node_id == target_id:
                return node
        queue.extend(node.child_nodes)
    return None
