"""
prompts.py
----------
All LLM prompt templates for Vectorless RAG v1.

Keeping prompts in one file makes them easy to iterate on without
touching business logic.
"""

from __future__ import annotations

import json

from vectorless_rag.chunker import Chunk
from vectorless_rag.tree_model import TreeNode


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a document structure analyst. Your task is to incrementally build a
hierarchical outline (tree) of a document by processing it one text chunk at a
time.

RULES:
1. You receive the current tree skeleton (node IDs, titles, and levels only)
   and a new text chunk from the document.
2. Decide one or more operations from the list below:
   - CREATE_CHILD    : Add a new section or subsection UNDER an existing node.
   - UPDATE_SUMMARY  : Enrich the summary of an EXISTING node with new information.
   - MERGE_NODES     : Combine an existing node with a very similar sibling.
   - CREATE_SIBLING  : Add a new section AT THE SAME LEVEL as an existing node.
3. Always output VALID JSON matching the schema shown below. No markdown fences.
4. DO NOT include any conversational text before or after the JSON block. Do NOT start with phrases like "Here is the JSON...".
5. Do NOT invent information that is not present in the chunk.
5. Summaries must be factual, concise (2-4 sentences), and information-dense.
6. Prefer attaching to existing sections over creating duplicates.
7. Never exceed tree depth 5 (root = 0). Flatten anything deeper.
8. Think step by step in the "reasoning" field before proposing operations.

OUTPUT SCHEMA (strict JSON, no extra keys):
{
  "reasoning": "<1-3 sentences of step-by-step thinking>",
  "operations": [
    {
      "op": "CREATE_CHILD | UPDATE_SUMMARY | MERGE_NODES | CREATE_SIBLING",
      "target_node_id": "<node_id to act on>",
      "payload": {
        "title": "<only for CREATE_CHILD, CREATE_SIBLING>",
        "summary": "<for CREATE_CHILD, CREATE_SIBLING, UPDATE_SUMMARY>",
        "merge_with_node_id": "<only for MERGE_NODES>"
      }
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Few-shot example (prepended to the FIRST chunk's user prompt only)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLE = """\
=== EXAMPLE (for guidance only, not part of this document) ===

CURRENT TREE SKELETON:
{"node_id": "root", "title": "Document Root", "level": 0, "children": []}

RUNNING SUMMARY: (none yet)

CHUNK (pages 1-2):
Chapter 1: Introduction to Machine Learning
Machine learning is a subset of artificial intelligence concerned with giving
computers the ability to learn from data without being explicitly programmed.
This chapter covers foundational definitions, the three main learning paradigms
(supervised, unsupervised, reinforcement), and the scope of the field.

RESPONSE:
{
  "reasoning": "The chunk introduces Chapter 1 about Machine Learning. There are
   no existing children under root, so I will create a new child node.",
  "operations": [
    {
      "op": "CREATE_CHILD",
      "target_node_id": "root",
      "payload": {
        "title": "Chapter 1: Introduction to Machine Learning",
        "summary": "Introduces machine learning as a branch of AI focused on
         learning from data. Covers the three learning paradigms: supervised,
         unsupervised, and reinforcement learning."
      }
    }
  ]
}
=== END EXAMPLE ===

"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(
    skeleton: dict,
    chunk: Chunk,
    running_summary: str,
    is_first: bool = False,
) -> str:
    """
    Build the user-turn portion of the LLM prompt.

    Args:
        skeleton:        Compact tree dict from TreeNode.skeleton_dict().
        chunk:           The current Chunk being processed.
        running_summary: Evolving 2-sentence document summary.
        is_first:        If True, prepend the few-shot example.

    Returns:
        Formatted prompt string.
    """
    pages_label = (
        f"page {chunk.page_numbers[0]}"
        if len(chunk.page_numbers) == 1
        else f"pages {chunk.page_numbers[0]}–{chunk.page_numbers[-1]}"
    )

    parts: list[str] = []

    if is_first:
        parts.append(FEW_SHOT_EXAMPLE)

    parts.append("CURRENT TREE SKELETON:")
    parts.append(json.dumps(skeleton, indent=2))
    parts.append("")

    parts.append(f"RUNNING SUMMARY: {running_summary or '(none yet)'}")
    parts.append("")

    parts.append(f"CHUNK [{chunk.chunk_id}] ({pages_label}):")
    parts.append(chunk.text)
    parts.append("")

    parts.append("Respond with JSON only (no markdown, no code fences).")

    return "\n".join(parts)


def update_running_summary(current: str, chunk: Chunk, reasoning: str) -> str:
    """
    Keep a rolling ~2-sentence summary of what has been processed so far.
    Simple concatenation with truncation — does not make an extra LLM call.
    """
    addition = reasoning.split(".")[0].strip()  # first sentence of reasoning
    combined = f"{current} {addition}".strip() if current else addition
    # Keep at most ~400 chars (2 sentences)
    if len(combined) > 400:
        combined = combined[-400:]
    return combined
