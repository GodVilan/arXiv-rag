# arXiv-rag

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0078D4?style=flat-square)](https://faiss.ai)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com)
[![Apple MPS](https://img.shields.io/badge/Apple-MPS_Accelerated-000000?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Semantic question answering over 120 arXiv ML papers.**  
**Ask anything. Get grounded answers from real papers.**

[Demo](#demo) · [Quick Start](#quick-start) · [Results](#benchmark-results) · [Architecture](#architecture) · [Structure](#project-structure)

</div>

---

![Demo](docs/screenshot_qa.png)

---

## Overview

`arXiv-rag` is a production-structured Retrieval-Augmented Generation pipeline that answers
questions about machine learning research by retrieving semantically relevant passages from
a 120-paper arXiv corpus and generating grounded answers via Google Gemini.

The project also serves as a **benchmark** — empirically comparing three transformer embedding
models against a BM25 sparse retrieval baseline across chunk sizes, retrieval depths, and
generation quality metrics using a manually curated evaluation dataset.

```
You ask → BGE encodes query → FAISS finds top-5 passages → Gemini answers with citations
```

---

## Demo

| Welcome screen | Live answer with sources |
|---|---|
| ![Welcome](docs/screenshot_welcome.png) | ![QA](docs/screenshot_qa.png) |

```bash
streamlit run app.py
```

Switch between BM25, MiniLM, MPNet, and BGE live. Every answer shows retrieved paper titles
and similarity scores. Browse all 120 papers from the sidebar.

---

## Benchmark Results

Evaluated on **20 manually curated domain-expert QA pairs** — questions written from paper
titles, abstracts, and content. Relevance defined at paper level across all chunk sizes.

### Retrieval · chunk size 512

| Model | MRR | Precision@5 | Recall@10 | Latency |
|-------|-----|-------------|-----------|---------|
| **BGE** ⭐ | **1.000** | **0.975** | 0.316 | 37 ms |
| MPNet | 0.975 | 0.890 | 0.270 | 18 ms |
| MiniLM | 0.975 | 0.885 | 0.257 | **8 ms** ⚡ |
| BM25 (baseline) | 0.946 | 0.900 | 0.257 | 9 ms |

### Generation · chunk size 512

| Model | Answer Relevance | Faithfulness | Context Precision |
|-------|-----------------|--------------|------------------|
| **BGE** ⭐ | **0.910** | 0.978 | **1.000** |
| MPNet | 0.724 | **0.992** | 0.985 |
| MiniLM | 0.709 | 0.974 | 0.990 |
| BM25 (baseline) | 0.118 | 0.967 | 1.000 |

**Key findings:**
- BGE achieves perfect MRR (1.000) — dense semantic retrieval consistently outperforms BM25
- BM25 vs BGE Answer Relevance gap: **0.118 vs 0.910 — 7.7×** — BM25 retrieves the right paper but misses semantic intent
- MiniLM at 8 ms matches BM25 speed at **97.5% of BGE accuracy** — optimal for latency-sensitive deployments
- Faithfulness near-perfect across all dense models (0.97–0.99) — Gemini stays grounded regardless of retriever

---

## Architecture

```
arXiv API (120 papers)
       │
       ▼
  rag/data/collector.py
  PyMuPDF  →  plain text  →  cleaning
       │
       ▼
  rag/processing/chunker.py
  Recursive chunker (256 / 512 / 1024 tokens, 64-token overlap)
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
  rag/retrieval/embeddings.py          rag/retrieval/bm25.py
  SentenceTransformer                  Okapi BM25
  MPS / CUDA / CPU auto-detect         log-normalised scores
  L2-normalised vectors
       │
       ▼
  rag/retrieval/vector_store.py
  FAISS IndexFlatIP  (exact cosine)
       │
       └──────────────┬───────────────────────────┘
                      ▼
               Top-K passages
                      │
                      ▼
          rag/generation/generator.py
          Gemini 2.5 Flash Lite
          Token-bucket rate limiter
                      │
                      ▼
            Grounded answer + citations
```

---

## Project Structure

```
arXiv-rag/
│
├── app.py                        # Streamlit UI — entry point
├── main.py                       # CLI demo   — entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
│
├── rag/                          # Core package
│   ├── __init__.py
│   ├── config.py                 # All settings: models, paths, device, API
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── collector.py          # arXiv API downloader with resume support
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   └── chunker.py            # PDF extraction + recursive chunker
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # SentenceTransformer wrapper + disk cache
│   │   ├── vector_store.py       # FAISS index (build / save / load / search)
│   │   ├── dense.py              # Dense retriever (build index + search)
│   │   └── bm25.py               # BM25 sparse baseline (same interface)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py          # Gemini generator + token-bucket rate limiter
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            # Recall@K, Precision@K, MRR, AR, Faithfulness
│       └── qa_generator.py       # Auto-generate QA pairs from paper content
│
├── scripts/
│   └── run_experiments.py        # Full ablation: 4 models × 3 chunk sizes → plots
│
├── data/
│   ├── metadata.json             # Paper metadata (committed — no PDFs)
│   └── manual_qa.json            # 20 curated evaluation QA pairs
│
├── results/
│   ├── retrieval_metrics.json
│   ├── generation_metrics.json
│   └── plots/                    # MRR, Precision, Recall, Latency, BM25 vs Dense
│
└── docs/
    ├── screenshot_welcome.png
    └── screenshot_qa.png
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Google AI Studio API key](https://aistudio.google.com/app/apikey) — free tier works
- Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

### Install

```bash
git clone https://github.com/GodVilan/arXiv-rag
cd arXiv-rag

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip3 install -r requirements.txt

cp .env.example .env
# open .env and set: GEMINI_API_KEY=AIza...
```

### Run

```bash
# 1 — Download 120 arXiv ML papers (~10 min)
python3 -c "from rag.data.collector import download_papers; download_papers()"

# 2 — Generate evaluation QA pairs via Gemini (~2 min)
python3 rag/evaluation/qa_generator.py --n 20

# 3 — Run full benchmark: BM25 + 3 models × 3 chunk sizes (~15 min)
python3 scripts/run_experiments.py

# 4 — Launch the UI
streamlit run app.py

# 5 — Or use the CLI
python3 main.py --model BGE --top_k 5
python3 main.py --list             # browse all 120 papers
```

---

## Configuration

All settings in `rag/config.py`:

```python
# Embedding models
EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",   # 384d — fast
    "MPNet":  "sentence-transformers/all-mpnet-base-v2",   # 768d — balanced
    "BGE":    "BAAI/bge-large-en",                         # 1024d — best accuracy
}

# Chunk sizes for ablation
CHUNK_SIZES  = [256, 512, 1024]    # tokens
CHUNK_OVERLAP = 64                 # overlap between consecutive chunks

# Generation
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_RPM   = 12                  # free tier: 15 RPM → 12 for safety margin

# Device — auto-detected: MPS → CUDA → CPU
DEVICE = _best_device()
```

---

## Evaluation

| | |
|---|---|
| **Corpus** | 120 arXiv cs.LG papers · early 2026 |
| **Chunks** | 1,792 (1024-token) to 8,131 (256-token) |
| **QA pairs** | 20 manually curated — from paper title + abstract + content |
| **Relevance** | Paper-level — all chunks from the same source paper are relevant |

| Metric | Definition |
|--------|-----------|
| MRR | Mean reciprocal rank of first relevant result |
| Recall@K | Fraction of all relevant (paper-level) chunks in top-K |
| Precision@K | Fraction of top-K results that are relevant |
| Latency | Avg retrieval time per query over 20 queries (ms) |
| Answer Relevance | Cosine similarity between question and answer embeddings |
| Faithfulness | Fraction of answer sentences supported by retrieved context |
| Context Precision | Fraction of retrieved chunks contributing to the answer |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF extraction | PyMuPDF 1.25.5 |
| Embeddings | sentence-transformers 3.0 (MiniLM · MPNet · BGE) |
| Sparse retrieval | rank-bm25 (Okapi BM25) |
| Vector index | FAISS IndexFlatIP |
| Generation | Google Gemini 2.5 Flash Lite |
| UI | Streamlit |
| Acceleration | Apple MPS · NVIDIA CUDA · CPU |

---

## License

MIT — see [LICENSE](LICENSE)
