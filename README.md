# Vectorless RAG v1

> **Hierarchical reasoning trees from PDFs — no vector embeddings required.**

Vectorless RAG replaces cosine-similarity retrieval with **structural navigation**. A local LLM (via [Ollama](https://ollama.ai)) reasons over your PDF chunk-by-chunk and builds a JSON hierarchy tree mirroring how humans read documents: *chapter → section → subsection → detail*.

---

## How It Works

```
PDF
 └─ parse pages (PyMuPDF)
     └─ chunk with overlap
         └─ LLM reasons over each chunk (Ollama, local)
             └─ incremental JSON tree built
                 └─ BFS query engine for structural retrieval
```

### Traditional RAG vs Vectorless RAG

| | Traditional RAG | Vectorless RAG |
|---|---|---|
| **Retrieval** | Cosine similarity (embeddings) | BFS structural navigation |
| **Infrastructure** | Vector DB | Single JSON file |
| **Interpretability** | Low | High (human-readable tree) |
| **Offline** | Depends on API | ✅ Fully local |

---

## Prerequisites

1. **Python 3.9+**
2. **Ollama** running locally — [install](https://ollama.ai)
3. **uv** — [install](https://github.com/astral-sh/uv)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
4. Pull your local model:
   ```bash
   ollama pull <your-model-name>
   ```

---

## Installation

```bash
git clone https://github.com/arnavbansal2764/vectorless-rag
cd vectorless-rag
uv venv
uv pip install -e ".[dev]"
```

---

## Quick Start

### Build a tree from a PDF

```bash
uv run vrag-build path/to/document.pdf --model <your-model> --output-dir ./output
```

Or in Python:

```python
from vectorless_rag import TreeBuilder
from vectorless_rag.config import VRAGConfig, LLMConfig

config = VRAGConfig(llm=LLMConfig(model="your-model-name"))
builder = TreeBuilder(config=config)
tree = builder.build("path/to/document.pdf")

# Save
from vectorless_rag import TreeStorage
TreeStorage(output_dir="./output").save(tree)
```

### Query the tree

```bash
uv run vrag-query output/document_tree.json --top-k 5
```

Or in Python:

```python
from vectorless_rag import QueryEngine, TreeStorage

tree = TreeStorage().load("output/document_tree.json")
engine = QueryEngine(tree)

results = engine.search("risk management framework", top_k=5)
for r in results:
    print(f"{r.score:.2f}  {' > '.join(r.path)}")
    print(f"       {r.node.summary}\n")
```

---

## Tree Node Schema

```json
{
  "node_id": "a1b2c3d4",
  "title": "Chapter 3: Risk Management",
  "summary": "Covers enterprise risk frameworks and internal controls.",
  "source_refs": ["p12_c0", "p13_c1"],
  "level": 1,
  "child_nodes": [],
  "metadata": {}
}
```

---

## Configuration

```python
from vectorless_rag.config import VRAGConfig, ChunkConfig, LLMConfig, TreeConfig

config = VRAGConfig(
    chunk=ChunkConfig(max_chars=6000, overlap_ratio=0.15),
    llm=LLMConfig(model="your-model", temperature=0.3, num_ctx=8192),
    tree=TreeConfig(max_depth=5, checkpoint_every=10),
    output_dir="./output",
)
```

---

## Running Tests

```bash
uv run pytest tests/ -v --tb=short
```

---

## Project Structure

```
vectorless_rag/
├── config.py           # All configuration dataclasses
├── pdf_parser.py       # PDF → List[PageText]
├── chunker.py          # PageText → List[Chunk] with overlap
├── prompts.py          # LLM system prompt + user prompt builder
├── llm_client.py       # Ollama Python library wrapper + retries
├── tree_model.py       # Pydantic models: TreeNode, HierarchyTree, TreeOperation
├── tree_operations.py  # Apply/validate tree mutations
├── tree_builder.py     # Main orchestrator loop
├── storage.py          # JSON save/load + checkpoints
├── query_engine.py     # BFS structural retrieval
├── evaluator.py        # Quality metrics + invariant checks
└── cli.py              # vrag-build and vrag-query CLI commands
```

---

## License

MIT
