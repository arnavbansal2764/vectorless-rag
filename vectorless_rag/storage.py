"""
storage.py
----------
Serialization, deserialization, and checkpoint management for HierarchyTree.

Responsibilities:
- Save the final tree to a versioned JSON file
- Save incremental checkpoints
- Load a tree from a JSON file (with Pydantic validation)
- Validate the JSON against the bundled schema
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jsonschema

from vectorless_rag.tree_model import HierarchyTree

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "tree_schema.json"


class TreeStorage:
    """
    Handles reading and writing of HierarchyTree to disk.

    Parameters
    ----------
    output_dir:
        Directory where tree files and checkpoints are written.
        Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str | Path = ".") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load JSON Schema once
        self._schema: dict | None = None
        if _SCHEMA_PATH.exists():
            with _SCHEMA_PATH.open() as f:
                self._schema = json.load(f)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #

    def save(self, tree: HierarchyTree, filename: str | None = None) -> Path:
        """
        Serialize *tree* to a JSON file and return the written path.

        Args:
            tree:     The tree to save.
            filename: Optional explicit filename. Defaults to
                      ``<stem_of_source_file>_tree.json``.
        """
        if filename is None:
            stem = Path(tree.source_file).stem
            filename = f"{stem}_tree.json"

        out_path = self.output_dir / filename
        out_path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Tree saved → %s", out_path)
        return out_path

    def save_checkpoint(self, tree: HierarchyTree, chunk_index: int) -> Path:
        """Save a checkpoint named ``checkpoint_{chunk_index:04d}.json``."""
        checkpoints_dir = self.output_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        filename = f"checkpoint_{chunk_index:04d}.json"
        path = checkpoints_dir / filename
        path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Checkpoint written: %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #

    def load(self, path: str | Path) -> HierarchyTree:
        """
        Deserialize a tree from *path*.

        Performs Pydantic schema validation automatically.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If the JSON does not match the model.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tree file not found: {path}")

        text = path.read_text(encoding="utf-8")
        tree = HierarchyTree.model_validate_json(text)
        logger.info("Tree loaded from %s (%d pages, root='%s')",
                    path, tree.total_pages, tree.root.title)
        return tree

    def load_latest_checkpoint(self) -> HierarchyTree | None:
        """
        Load the most recent checkpoint from the checkpoints sub-directory.
        Returns None if no checkpoints exist.
        """
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        files = sorted(checkpoints_dir.glob("checkpoint_*.json"))
        if not files:
            return None

        latest = files[-1]
        logger.info("Resuming from checkpoint: %s", latest)
        return self.load(latest)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate_schema(self, tree: HierarchyTree) -> list[str]:
        """
        Validate *tree* against the bundled JSON Schema.

        Returns a list of validation error messages (empty = valid).
        """
        if self._schema is None:
            logger.warning("No JSON Schema found at %s — skipping validation.", _SCHEMA_PATH)
            return []

        data = json.loads(tree.model_dump_json())
        errors = []
        validator = jsonschema.Draft202012Validator(self._schema)
        for error in validator.iter_errors(data):
            errors.append(f"{' > '.join(str(p) for p in error.absolute_path)}: {error.message}")

        if errors:
            logger.warning("Schema validation: %d error(s) found.", len(errors))
        else:
            logger.info("Schema validation passed.")

        return errors
