"""
cli.py
------
Click-based CLI entry points registered in pyproject.toml:

  vrag-build <pdf>   → build tree from a PDF
  vrag-query <tree>  → interactive query over a saved tree
"""

from __future__ import annotations

import logging

import click
from rich.console import Console

console = Console()


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--model", default="gpt-oss-20b", show_default=True, help="Ollama model name")
@click.option("--output-dir", default=".", show_default=True, help="Directory for output files")
@click.option("--checkpoint-every", default=10, show_default=True, help="Checkpoint every N chunks (0=off)")
@click.option("--max-chars", default=6000, show_default=True, help="Max chars per chunk")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def build(pdf_path: str, model: str, output_dir: str, checkpoint_every: int, max_chars: int, debug: bool) -> None:
    """Build a hierarchical tree from a PDF file and save it as JSON."""
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    from vectorless_rag.config import ChunkConfig, LLMConfig, TreeConfig, VRAGConfig
    from vectorless_rag.evaluator import check_invariants, evaluate, print_report
    from vectorless_rag.storage import TreeStorage
    from vectorless_rag.tree_builder import TreeBuilder

    config = VRAGConfig(
        chunk=ChunkConfig(max_chars=max_chars),
        llm=LLMConfig(model=model),
        tree=TreeConfig(checkpoint_every=checkpoint_every),
        output_dir=output_dir,
    )

    console.rule("[bold cyan]Vectorless RAG v1 — Build")

    builder = TreeBuilder(config=config)
    tree = builder.build(pdf_path)

    storage = TreeStorage(output_dir=output_dir)
    out_path = storage.save(tree)

    # Evaluate
    violations = check_invariants(tree)
    if violations:
        console.print(f"[yellow]⚠ {len(violations)} invariant violation(s):[/yellow]")
        for v in violations:
            console.print(f"  {v}")

    report = evaluate(tree)
    print_report(report)

    console.print(f"\n[bold green]✓ Tree saved → {out_path}[/bold green]")


@click.command()
@click.argument("tree_path", type=click.Path(exists=True))
@click.option("--top-k", default=5, show_default=True, help="Number of results to return")
@click.option("--min-score", default=0.0, show_default=True, help="Minimum relevance score")
def query(tree_path: str, top_k: int, min_score: float) -> None:
    """Interactively query a saved tree file using BFS structural retrieval."""
    from vectorless_rag.query_engine import QueryEngine
    from vectorless_rag.storage import TreeStorage

    storage = TreeStorage()
    tree = storage.load(tree_path)
    engine = QueryEngine(tree)

    console.rule(f"[bold cyan]Querying: {tree.document_title}")
    console.print(f"[dim]Tree has {len(tree.build_log)} build log entries. Type 'exit' to quit.[/dim]\n")

    while True:
        try:
            q = console.input("[bold yellow]Query:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue

        results = engine.search(q, top_k=top_k, min_score=min_score)
        if not results:
            console.print("[dim]No results found.[/dim]")
            continue

        for i, r in enumerate(results, 1):
            path_str = " › ".join(r.path)
            console.print(f"\n[bold]{i}. [cyan]{r.node.title}[/cyan][/bold]  [dim](score={r.score:.2f})[/dim]")
            console.print(f"   [dim]Path: {path_str}[/dim]")
            if r.node.summary:
                console.print(f"   {r.node.summary}")
            if r.source_refs:
                console.print(f"   [dim]Sources: {', '.join(r.source_refs)}[/dim]")
