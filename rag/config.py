"""
config.py – Central configuration for all hyperparameters and paths.
"""

import os
from pathlib import Path
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
CACHE_DIR       = BASE_DIR / "embeddings_cache"
RESULTS_DIR     = BASE_DIR / "results"

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# arXiv collection
# ──────────────────────────────────────────────
ARXIV_CATEGORY  = "cs.LG"
NUM_PAPERS      = 120          # papers to download (100-150 as per proposal)
ARXIV_SORT_BY   = "submittedDate"

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZES     = [256, 512, 1024]   # tokens (ablation)
DEFAULT_CHUNK   = 512
CHUNK_OVERLAP   = 64                 # token overlap between consecutive chunks

# ──────────────────────────────────────────────
# Embedding models to compare
# ──────────────────────────────────────────────
EMBEDDING_MODELS = {
    "MiniLM":   "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet":    "sentence-transformers/all-mpnet-base-v2",
    "BGE":      "BAAI/bge-large-en",
}

# ──────────────────────────────────────────────
# FAISS / retrieval
# ──────────────────────────────────────────────
TOP_K_VALUES    = [3, 5, 10]         # ablation over retrieval depth
DEFAULT_TOP_K   = 5
FAISS_INDEX_TYPE = "FlatIP"          # inner-product (cosine after normalisation)

# ──────────────────────────────────────────────
# Generation  –  Google Gemini via AI Studio
# ──────────────────────────────────────────────
# Set GEMINI_API_KEY in your .env file (copy from Google AI Studio)
# Models available on the free / AI-Pro tier:
#   gemini-2.0-flash          ← fast, generous rate limits  (recommended)
#   gemini-2.0-flash-thinking ← reasoning variant
#   gemini-1.5-pro            ← best quality, 1M token context
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TEMPERATURE  = 0.2
GEMINI_MAX_TOKENS   = 1024

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
EVAL_K_VALUES   = [1, 3, 5, 10]

# Match to your plan:
GEMINI_RPM = 12   # Free tier (15 RPM, use 12 for safety margin)
# GEMINI_RPM = 60    # AI Pro / paid tier

# How many QA pairs for retrieval eval (no API calls, can be large)
RETRIEVAL_EVAL_SAMPLES = 50

# How many QA pairs for generation eval (1 API call each, keep small)
GENERATION_EVAL_SAMPLES = 12

# ──────────────────────────────────────────────
# Secrets (loaded from .env if present)
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

def _best_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"       # ← Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()    # used everywhere automatically
