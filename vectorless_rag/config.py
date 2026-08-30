"""
config.py
---------
Central configuration dataclass for Vectorless RAG v1.
All tunable knobs live here so callers never need to touch internals.
"""

from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    """Controls how PDF text is split into LLM-digestible pieces."""

    max_chars: int = 6000
    """Approximate character limit per chunk (~1500 tokens for most models)."""

    overlap_ratio: float = 0.15
    """Fraction of chunk size to repeat as prefix of the next chunk."""

    respect_paragraphs: bool = True
    """If True, never split mid-paragraph; prefer paragraph boundaries."""


@dataclass
class LLMConfig:
    """Controls the Ollama integration."""

    model: str = "gpt-oss-20b"
    """Exact model name as shown in `ollama list`."""

    temperature: float = 0.3
    """Low temperature for deterministic structure extraction."""

    num_ctx: int = 8192
    """Context window size in tokens. Increase if model supports more."""

    max_retries: int = 3
    """How many times to retry a failed LLM call."""

    retry_min_wait: float = 2.0
    """Minimum seconds to wait between retries (exponential backoff)."""

    retry_max_wait: float = 30.0
    """Maximum seconds to wait between retries."""


@dataclass
class TreeConfig:
    """Controls hierarchy constraints."""

    max_depth: int = 5
    """Maximum nesting level (0 = root). Nodes beyond this are flattened."""

    merge_threshold: float = 0.85
    """SequenceMatcher ratio above which sibling titles are considered duplicates."""

    checkpoint_every: int = 10
    """Save a checkpoint every N chunks processed. 0 = disabled."""


@dataclass
class VRAGConfig:
    """Top-level configuration object passed through the entire pipeline."""

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)

    # Output paths
    output_dir: str = "."
    """Directory where the final tree JSON and checkpoints are written."""
