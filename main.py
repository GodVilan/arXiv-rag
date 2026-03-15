"""
main.py – Interactive RAG question-answering demo.

Lets you pick an embedding model and ask questions about the downloaded papers.

Usage:
    python main.py [--model MiniLM|MPNet|BGE] [--top_k 5] [--chunk_size 512]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag import config
from rag.processing.chunker import process_papers, save_chunks, load_chunks
from rag.retrieval.dense import Retriever
from rag.generation.generator import Generator
logging.basicConfig(level=logging.WARNING)   # quiet in interactive mode

# ──────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   RAG Demo – ML Research Paper Q&A                          ║
║   CS5720 Neural Networks and Deep Learning · Spring 2026    ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive RAG Q&A")
    p.add_argument("--model",      default="MiniLM",
                   choices=list(config.EMBEDDING_MODELS.keys()),
                   help="Embedding model to use")
    p.add_argument("--top_k",      type=int, default=config.DEFAULT_TOP_K,
                   help="Number of retrieved chunks")
    p.add_argument("--chunk_size", type=int, default=config.DEFAULT_CHUNK,
                   choices=config.CHUNK_SIZES,
                   help="Chunk size used during indexing")
    p.add_argument("--list", action="store_true", help="List all available papers and exit")
    return p.parse_args()


def load_or_build_chunks(chunk_size: int) -> list:
    cache_path = config.DATA_DIR / f"chunks_{chunk_size}.json"
    if cache_path.exists():
        print(f"  Loading chunks from cache ({cache_path.name}) …")
        return load_chunks(cache_path)

    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("  No papers found. Run: python data_collection.py")
        sys.exit(1)

    with open(meta_path) as f:
        papers = json.load(f)

    print(f"  Processing {len(papers)} papers (chunk_size={chunk_size}) …")
    chunks = process_papers(papers, chunk_size=chunk_size)
    save_chunks(chunks, cache_path)
    return chunks


def format_sources(results: list) -> str:
    lines = []
    for i, (chunk, score) in enumerate(results, start=1):
        lines.append(f"  [{i}] {chunk.title}  (score: {score:.3f})")
        lines.append(f"      chunk_id: {chunk.chunk_id}")
    return "\n".join(lines)

def list_papers() -> None:
    """Print all available papers in the corpus."""
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("No papers found. Run: python data_collection.py")
        return
    with open(meta_path) as f:
        papers = json.load(f)
    print(f"\n📚  {len(papers)} papers in corpus:\n")
    for i, p in enumerate(papers, 1):
        print(f"  {i:3d}. {p['title']}")
        print(f"       {p['paper_id']}  |  {p['published'][:10]}")
    print()

def main() -> None:
    args = parse_args()
    if args.list:
        list_papers()
        return
    print(BANNER)

    print(f"Model      : {args.model}  ({config.EMBEDDING_MODELS[args.model]})")
    print(f"Top-K      : {args.top_k}")
    print(f"Chunk size : {args.chunk_size} tokens")
    print()

    # ── Load corpus ───────────────────────────────────────────────────────────
    print("Loading corpus …")
    chunks = load_or_build_chunks(args.chunk_size)
    print(f"  {len(chunks)} chunks ready.")

    # ── Build retriever ───────────────────────────────────────────────────────
    print(f"\nBuilding retriever [{args.model}] …")
    retriever = Retriever.build(
        model_key=args.model,
        chunks=chunks,
        chunk_size=args.chunk_size,
        index_dir=config.RESULTS_DIR / "indices",
    )
    print("  Retriever ready.\n")

    # ── Init generator ────────────────────────────────────────────────────────
    print("Loading generator …")
    generator = Generator()
    print("  Generator ready.\n")

    # ── Interactive loop ──────────────────────────────────────────────────────
    print("Ask anything about machine learning — the system finds relevant papers for you.")
    print("Examples:")
    print("  - What are recent approaches to continual learning?")
    print("  - How do papers handle reinforcement learning for robotics?")
    print("  - What methods are used for model reliability?")
    print("  - Run 'python main.py --list' to see all available papers")
    print("\nType 'quit' to exit.\n")
    print("-" * 65)

    while True:
        try:
            query = input("\n❓  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        # ── Retrieve ───────────────────────────────────────────────────────
        results = retriever.retrieve(query, top_k=args.top_k)
        context = retriever.format_context(query, top_k=args.top_k)

        # ── Generate ───────────────────────────────────────────────────────
        answer = generator.generate(query, context)

        print(f"\n💬  Answer:\n{answer}")
        print(f"\n📄  Sources:\n{format_sources(results)}")
        print("-" * 65)

    print("\nExperiment results (if run): ", config.RESULTS_DIR)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
