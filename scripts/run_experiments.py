"""
run_experiments.py – Full experiment pipeline.

Compares all three embedding models + BM25 baseline across:
  • Chunk sizes : 256, 512, 1024
  • Top-K values: 3, 5, 10

Outputs:
  • results/retrieval_metrics.csv / .json
  • results/generation_metrics.csv / .json
  • results/plots/ (bar charts, heat-maps, latency)

Run:
    python3 run_experiments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # adds project root to path

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from rag import config
from rag.data.collector import download_papers
from rag.processing.chunker import process_papers, save_chunks, load_chunks
from rag.retrieval.dense import Retriever
from rag.generation.generator import Generator
from rag.evaluation.metrics import (
    RetrievalEvaluator, GenerationEvaluator,
    create_synthetic_qa_pairs, load_manual_qa_pairs,
)
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.embeddings import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
log.info("Device: %s", config.DEVICE)

PLOTS_DIR = config.RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Data
# ──────────────────────────────────────────────────────────────────────────────

def step_data() -> list[dict]:
    log.info("=" * 60)
    log.info("STEP 1 – Data collection")
    log.info("=" * 60)
    papers = download_papers()
    log.info("Papers available: %d", len(papers))
    return papers


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Chunking
# ──────────────────────────────────────────────────────────────────────────────

def step_chunking(papers: list[dict]) -> dict[int, list]:
    log.info("=" * 60)
    log.info("STEP 2 – Text extraction & chunking")
    log.info("=" * 60)
    chunks_by_size = {}
    for cs in config.CHUNK_SIZES:
        cache_path = config.DATA_DIR / f"chunks_{cs}.json"
        if cache_path.exists():
            log.info("Loading cached chunks for chunk_size=%d …", cs)
            chunks = load_chunks(cache_path)
        else:
            chunks = process_papers(papers, chunk_size=cs)
            save_chunks(chunks, cache_path)
        chunks_by_size[cs] = chunks
        log.info("  chunk_size=%d → %d chunks", cs, len(chunks))
    return chunks_by_size


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_qa_pairs(chunks: list, n: int) -> tuple[list, str]:
    """
    Return (qa_pairs, source_label).
    Uses manual QA pairs if data/manual_qa.json exists, else synthetic.
    """
    manual = load_manual_qa_pairs(chunks)
    if manual:
        log.info("  Using %d manual QA pairs", len(manual))
        return manual, "manual"
    pairs = create_synthetic_qa_pairs(chunks, n=n)
    log.info("  Using %d synthetic QA pairs", len(pairs))
    return pairs, "synthetic"


def _measure_retrieval_latency(retriever, qa_pairs: list, top_k: int, n: int = 20) -> float:
    """Average retrieval latency in ms over first n queries."""
    sample = qa_pairs[:n]
    t0 = time.time()
    for qa in sample:
        retriever.retrieve(qa.question, top_k)
    return round((time.time() - t0) / len(sample) * 1000, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Retrieval experiments  (Dense + BM25 baseline)
# ──────────────────────────────────────────────────────────────────────────────

def step_retrieval(chunks_by_size: dict[int, list]) -> list[dict]:
    log.info("=" * 60)
    log.info("STEP 3 – Retrieval evaluation  (Dense + BM25 baseline)")
    log.info("=" * 60)

    evaluator = RetrievalEvaluator()
    all_results: list[dict] = []
    index_dir = config.RESULTS_DIR / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    for cs, chunks in chunks_by_size.items():
        qa_pairs, qa_source = _get_qa_pairs(chunks, n=config.RETRIEVAL_EVAL_SAMPLES)

        # ── BM25 baseline ──────────────────────────────────────────────────
        log.info("  [BM25 baseline | chunk_size=%d] Building …", cs)
        bm25       = BM25Retriever(chunks)
        latency_ms = _measure_retrieval_latency(bm25, qa_pairs, config.DEFAULT_TOP_K)

        row_bm25 = {
            "model_key":  "BM25",
            "chunk_size": cs,
            "build_time": bm25.build_time,
            "n_chunks":   len(chunks),
            "latency_ms": latency_ms,
            "qa_source":  qa_source,
        }
        for k in config.TOP_K_VALUES:
            metrics = evaluator.evaluate(qa_pairs, retrieval_fn=bm25.retrieve, k=k)
            row_bm25.update(metrics)

        all_results.append(row_bm25)
        log.info("    BM25 → latency=%.1fms  MRR=%.4f",
                 latency_ms, row_bm25.get("MRR", 0))

        # ── Dense embedding models ─────────────────────────────────────────
        for model_key in config.EMBEDDING_MODELS:
            log.info("  [%s | chunk_size=%d] Building retriever …", model_key, cs)
            t0        = time.time()
            retriever = Retriever.build(
                model_key=model_key,
                chunks=chunks,
                chunk_size=cs,
                index_dir=index_dir,
            )
            build_time = round(time.time() - t0, 2)
            latency_ms = _measure_retrieval_latency(retriever, qa_pairs, config.DEFAULT_TOP_K)

            row = {
                "model_key":  model_key,
                "chunk_size": cs,
                "build_time": build_time,
                "n_chunks":   len(chunks),
                "latency_ms": latency_ms,
                "qa_source":  qa_source,
            }
            for k in config.TOP_K_VALUES:
                metrics = evaluator.evaluate(qa_pairs, retrieval_fn=retriever.retrieve, k=k)
                row.update(metrics)

            all_results.append(row)
            log.info("    → latency=%.1fms  MRR=%.4f",
                     latency_ms, row.get("MRR", 0))

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Generation experiments  (Dense + BM25 baseline)
# ──────────────────────────────────────────────────────────────────────────────

def step_generation(
    chunks_by_size: dict[int, list],
) -> list[dict]:
    log.info("=" * 60)
    log.info("STEP 4 – Generation evaluation  (Dense + BM25 baseline)")
    log.info("=" * 60)

    generator       = Generator()
    all_gen_results = []
    index_dir       = config.RESULTS_DIR / "indices"
    cs              = config.DEFAULT_CHUNK
    chunks          = chunks_by_size[cs]
    qa_pairs, qa_source = _get_qa_pairs(chunks, n=config.GENERATION_EVAL_SAMPLES)

    # ── Dense embedding models ─────────────────────────────────────────────
    for model_key in config.EMBEDDING_MODELS:
        log.info("  [%s | chunk_size=%d] Generation eval …", model_key, cs)

        emb_model     = EmbeddingModel(model_key)
        gen_evaluator = GenerationEvaluator(embedding_model=emb_model)
        retriever     = Retriever.build(
            model_key=model_key,
            chunks=chunks,
            chunk_size=cs,
            index_dir=index_dir,
        )

        # Measure end-to-end latency (retrieve + generate) on one sample
        sample_ctx = retriever.format_context(qa_pairs[0].question)
        t0         = time.time()
        generator.generate(qa_pairs[0].question, sample_ctx)
        gen_latency = round((time.time() - t0) * 1000, 2)

        gen_metrics = gen_evaluator.evaluate(
            qa_pairs,
            retrieval_fn=retriever.retrieve,
            generation_fn=generator.generate,
            top_k=config.DEFAULT_TOP_K,
        )
        row = {
            "model_key":  model_key,
            "chunk_size": cs,
            "latency_ms": gen_latency,
            "qa_source":  qa_source,
            **gen_metrics,
        }
        all_gen_results.append(row)
        log.info("    → latency=%.1fms  %s", gen_latency, gen_metrics)

    # ── BM25 generation baseline ───────────────────────────────────────────
    log.info("  [BM25 baseline | chunk_size=%d] Generation eval …", cs)
    bm25          = BM25Retriever(chunks)
    gen_evaluator = GenerationEvaluator()   # Jaccard fallback — no embedding model

    sample_ctx  = bm25.format_context(qa_pairs[0].question)
    t0          = time.time()
    generator.generate(qa_pairs[0].question, sample_ctx)
    gen_latency = round((time.time() - t0) * 1000, 2)

    gen_metrics = gen_evaluator.evaluate(
        qa_pairs,
        retrieval_fn=bm25.retrieve,
        generation_fn=generator.generate,
        top_k=config.DEFAULT_TOP_K,
    )
    row = {
        "model_key":  "BM25",
        "chunk_size": cs,
        "latency_ms": gen_latency,
        "qa_source":  qa_source,
        **gen_metrics,
    }
    all_gen_results.append(row)
    log.info("    → latency=%.1fms  %s", gen_latency, gen_metrics)

    return all_gen_results


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Save results & plots
# ──────────────────────────────────────────────────────────────────────────────

def step_save_and_plot(
    retrieval_results: list[dict],
    generation_results: list[dict],
) -> None:
    log.info("=" * 60)
    log.info("STEP 5 – Saving results and generating plots")
    log.info("=" * 60)

    ret_df = pd.DataFrame(retrieval_results)
    gen_df = pd.DataFrame(generation_results)

    # ── CSV + JSON ────────────────────────────────────────────────────────────
    ret_df.to_csv(config.RESULTS_DIR / "retrieval_metrics.csv", index=False)
    gen_df.to_csv(config.RESULTS_DIR / "generation_metrics.csv", index=False)
    with open(config.RESULTS_DIR / "retrieval_metrics.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)
    with open(config.RESULTS_DIR / "generation_metrics.json", "w") as f:
        json.dump(generation_results, f, indent=2)
    log.info("Saved CSVs + JSON to %s", config.RESULTS_DIR)

    # ── Retrieval bar charts ──────────────────────────────────────────────────
    _plot_metric_bar(ret_df, metric="Recall@5",    title="Recall@5 by Model & Chunk Size")
    _plot_metric_bar(ret_df, metric="Precision@5", title="Precision@5 by Model & Chunk Size")
    _plot_metric_bar(ret_df, metric="MRR",         title="MRR by Model & Chunk Size")

    # ── Generation metrics ────────────────────────────────────────────────────
    _plot_gen_metrics(gen_df)

    # ── Heat-map ──────────────────────────────────────────────────────────────
    _plot_heatmap(ret_df, metric="MRR")

    # ── Latency chart (new) ───────────────────────────────────────────────────
    _plot_latency(ret_df)

    # ── BM25 vs Dense comparison (new) ───────────────────────────────────────
    _plot_bm25_vs_dense(ret_df)

    log.info("Plots saved to %s", PLOTS_DIR)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_metric_bar(df: pd.DataFrame, metric: str, title: str) -> None:
    if metric not in df.columns:
        return
    fig, ax     = plt.subplots(figsize=(10, 5))
    models      = df["model_key"].unique()
    chunk_sizes = sorted(df["chunk_size"].unique())
    x           = np.arange(len(chunk_sizes))
    n_models    = len(models)
    width       = 0.8 / n_models

    color_map = {"BM25": "#888888", "MiniLM": "#4C72B0",
                 "MPNet": "#DD8452", "BGE": "#55A868"}

    for i, model in enumerate(models):
        vals = [
            df[(df["model_key"] == model) & (df["chunk_size"] == cs)][metric].values
            for cs in chunk_sizes
        ]
        vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model, color=color_map.get(model, "#aaa"))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_sizes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{metric.replace('@', '_at_')}.png", dpi=150)
    plt.close()


def _plot_gen_metrics(gen_df: pd.DataFrame) -> None:
    metrics = ["Answer Relevance", "Faithfulness", "Context Precision"]
    metrics = [m for m in metrics if m in gen_df.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    color_map = {"BM25": "#888888", "MiniLM": "#4C72B0",
                 "MPNet": "#DD8452", "BGE": "#55A868"}

    for ax, metric in zip(axes, metrics):
        labels = gen_df["model_key"].tolist()
        vals   = gen_df[metric].tolist()
        colors = [color_map.get(l, "#aaa") for l in labels]
        bars   = ax.bar(labels, vals, color=colors)
        ax.set_title(metric)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Generation Metrics by Model (chunk_size=512)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generation_metrics.png", dpi=150)
    plt.close()


def _plot_heatmap(df: pd.DataFrame, metric: str = "MRR") -> None:
    if metric not in df.columns:
        return
    pivot = df.pivot_table(index="model_key", columns="chunk_size", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="YlOrRd", linewidths=0.5,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(f"Heat-map: {metric} (Model × Chunk Size)")
    ax.set_ylabel("Model")
    ax.set_xlabel("Chunk Size (tokens)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"heatmap_{metric.replace('@','_at_')}.png", dpi=150)
    plt.close()


def _plot_latency(df: pd.DataFrame) -> None:
    """Latency vs MRR scatter — the accuracy/speed trade-off plot."""
    if "latency_ms" not in df.columns or "MRR" not in df.columns:
        return

    sub = df[df["chunk_size"] == config.DEFAULT_CHUNK].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {"BM25": "#888888", "MiniLM": "#4C72B0",
                 "MPNet": "#DD8452", "BGE": "#55A868"}

    for _, row in sub.iterrows():
        model = row["model_key"]
        color = color_map.get(model, "#aaa")
        ax.scatter(row["latency_ms"], row["MRR"],
                   s=180, color=color, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(model,
                    xy=(row["latency_ms"], row["MRR"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=10, color=color)

    ax.set_xlabel("Retrieval Latency per Query (ms)")
    ax.set_ylabel("MRR")
    ax.set_title(f"Accuracy vs Speed Trade-off (chunk_size={config.DEFAULT_CHUNK})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_vs_mrr.png", dpi=150)
    plt.close()

    # Also plain latency bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    models  = sub["model_key"].tolist()
    latency = sub["latency_ms"].tolist()
    colors  = [color_map.get(m, "#aaa") for m in models]
    bars    = ax.bar(models, latency, color=colors, width=0.5)
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f"{val:.1f}ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Retrieval Latency per Query (chunk_size={config.DEFAULT_CHUNK})")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency.png", dpi=150)
    plt.close()


def _plot_bm25_vs_dense(df: pd.DataFrame) -> None:
    """Side-by-side MRR: BM25 vs best dense model at each chunk size."""
    if "MRR" not in df.columns:
        return

    chunk_sizes = sorted(df["chunk_size"].unique())
    bm25_mrr, best_dense_mrr, best_dense_name = [], [], []

    for cs in chunk_sizes:
        sub  = df[df["chunk_size"] == cs]
        bm25 = sub[sub["model_key"] == "BM25"]["MRR"].values
        bm25_mrr.append(bm25[0] if len(bm25) > 0 else 0.0)

        dense = sub[sub["model_key"] != "BM25"]
        if not dense.empty:
            best_row = dense.loc[dense["MRR"].idxmax()]
            best_dense_mrr.append(best_row["MRR"])
            best_dense_name.append(best_row["model_key"])
        else:
            best_dense_mrr.append(0.0)
            best_dense_name.append("N/A")

    x     = np.arange(len(chunk_sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, bm25_mrr,       width, label="BM25 (baseline)", color="#888888")
    ax.bar(x + width/2, best_dense_mrr, width, label="Best dense model",  color="#55A868")

    for i, (bval, dval, dname) in enumerate(zip(bm25_mrr, best_dense_mrr, best_dense_name)):
        gain = ((dval - bval) / bval * 100) if bval > 0 else 0
        ax.text(i + width/2, dval + 0.01, f"+{gain:.0f}%\n({dname})",
                ha="center", va="bottom", fontsize=8, color="#2d6a4f")

    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("MRR")
    ax.set_title("BM25 Baseline vs Best Dense Model — MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_sizes)
    ax.legend()
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bm25_vs_dense.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    start = time.time()

    papers         = step_data()
    chunks_by_size = step_chunking(papers)
    ret_results    = step_retrieval(chunks_by_size)
    gen_results    = step_generation(chunks_by_size, ret_results)
    step_save_and_plot(ret_results, gen_results)

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("Experiment complete in %.1f minutes.", elapsed / 60)
    log.info("Results saved to: %s", config.RESULTS_DIR)
    log.info("=" * 60)

    ret_df = pd.DataFrame(ret_results)
    print("\n📊  RETRIEVAL RESULTS SUMMARY")
    print(ret_df.to_string(index=False))

    gen_df = pd.DataFrame(gen_results)
    print("\n📊  GENERATION RESULTS SUMMARY")
    print(gen_df.to_string(index=False))


if __name__ == "__main__":
    main()