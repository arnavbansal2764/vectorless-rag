"""
Vectorless RAG v1
=================
A Python library that builds a hierarchical reasoning tree from a PDF
using a local LLM (via Ollama) instead of vector embeddings.

Retrieval = structural navigation, not cosine similarity.
"""

from vectorless_rag.tree_model import HierarchyTree, TreeNode
from vectorless_rag.tree_builder import TreeBuilder
from vectorless_rag.storage import TreeStorage
from vectorless_rag.query_engine import QueryEngine

__all__ = ["HierarchyTree", "TreeNode", "TreeBuilder", "TreeStorage", "QueryEngine"]
__version__ = "0.1.0"
