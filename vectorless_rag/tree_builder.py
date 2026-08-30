"""
tree_builder.py
---------------
Orchestrates the full PDF → HierarchyTree pipeline.

Main class: TreeBuilder
  .build(pdf_path)  →  HierarchyTree

Internally:
  1. Parse PDF into pages
  2. Chunk pages
  3. For each chunk: prompt LLM, parse ops, apply to tree, checkpoint
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from vectorless_rag.chunker import Chunk, chunk_pages
from vectorless_rag.config import VRAGConfig
from vectorless_rag.llm_client import LLMClient
from vectorless_rag.pdf_parser import infer_document_title, parse_pdf
from vectorless_rag.prompts import SYSTEM_PROMPT, build_user_prompt, update_running_summary
from vectorless_rag.storage import TreeStorage
from vectorless_rag.tree_model import HierarchyTree, TreeNode
from vectorless_rag.tree_operations import apply_operation, build_node_index

logger = logging.getLogger(__name__)


class TreeBuilder:
    """
    High-level orchestrator that drives the PDF → hierarchy pipeline.

    Parameters
    ----------
    config:
        Full configuration object. Uses sensible defaults if not provided.
    llm_client:
        Optionally inject a custom LLMClient (useful for testing with mocks).
    """

    def __init__(
        self,
        config: VRAGConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or VRAGConfig()
        self.llm = llm_client or LLMClient(self.config.llm)
        self.storage = TreeStorage(output_dir=self.config.output_dir)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self, pdf_path: str | Path) -> HierarchyTree:
        """
        Main entry point: parse *pdf_path* and return the completed tree.

        Checkpoints are saved automatically every N chunks if configured.
        """
        pdf_path = Path(pdf_path)
        logger.info("Starting tree build for: %s", pdf_path)

        # 1 · Parse PDF
        pages = parse_pdf(pdf_path)
        logger.info("Parsed %d pages.", len(pages))

        # 2 · Chunk
        chunks = chunk_pages(pages, config=self.config.chunk)
        logger.info("Created %d chunks.", len(chunks))

        # 3 · Initialise tree
        doc_title = infer_document_title(pages)
        tree = HierarchyTree(
            document_title=doc_title,
            source_file=str(pdf_path.resolve()),
            created_at=datetime.now(timezone.utc).isoformat(),
            total_pages=len(pages),
            root=TreeNode(title=doc_title, summary="", level=0, node_id="root"),
        )
        node_index = build_node_index(tree.root)

        # 4 · Incremental build loop
        self._run_build_loop(tree, node_index, chunks)

        logger.info(
            "Tree built: %d nodes, %d log entries.",
            len(node_index),
            len(tree.build_log),
        )
        return tree

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run_build_loop(
        self,
        tree: HierarchyTree,
        node_index: dict,
        chunks: list[Chunk],
    ) -> None:
        """
        Iterate over chunks, call LLM for each, apply resulting operations.
        Shows a rich progress bar in the terminal.
        """
        running_summary = ""
        checkpoint_every = self.config.tree.checkpoint_every

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Processing chunk {task.fields[chunk_id]}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("build", total=len(chunks), chunk_id="…")

            for i, chunk in enumerate(chunks):
                progress.update(task, chunk_id=chunk.chunk_id, advance=0)

                # Build prompt
                skeleton = tree.root.skeleton_dict()
                user_prompt = build_user_prompt(
                    skeleton=skeleton,
                    chunk=chunk,
                    running_summary=running_summary,
                    is_first=(i == 0),
                )

                # Call LLM
                llm_response = self.llm.generate(
                    system=SYSTEM_PROMPT,
                    prompt=user_prompt,
                )

                # Apply all proposed operations
                ops_applied = 0
                for op in llm_response.operations:
                    applied = apply_operation(
                        tree=tree,
                        node_index=node_index,
                        op=op,
                        chunk_id=chunk.chunk_id,
                        config=self.config.tree,
                    )
                    if applied:
                        ops_applied += 1

                logger.debug(
                    "Chunk %s: %d/%d ops applied. reasoning=%r",
                    chunk.chunk_id,
                    ops_applied,
                    len(llm_response.operations),
                    llm_response.reasoning[:80],
                )

                # Update rolling summary
                running_summary = update_running_summary(
                    running_summary, chunk, llm_response.reasoning
                )

                # Checkpoint
                if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
                    self.storage.save_checkpoint(tree, chunk_index=i + 1)
                    logger.info("Checkpoint saved at chunk %d.", i + 1)

                progress.advance(task)
