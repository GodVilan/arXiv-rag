"""
metrics.py – Retrieval and generation evaluation metrics.

Retrieval metrics:
  • Recall@K          – fraction of relevant docs found in top-K
  • Precision@K       – fraction of top-K results that are relevant
  • MRR               – Mean Reciprocal Rank

Generation metrics (reference-free, LLM-based approximations):
  • Answer Relevance  – cosine similarity between question and answer embeddings
  • Faithfulness      – fraction of answer sentences entailed by retrieved context
  • Context Precision – fraction of retrieved chunks that contributed to the answer
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from rag import config
from rag.processing.chunker import Chunk
from pathlib import Path

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QAPair:
    """A single ground-truth question-answer pair with known relevant chunks."""
    question:         str
    reference_answer: str
    relevant_chunk_ids: list[str]          # chunk_ids that are "ground truth relevant"


@dataclass
class RetrievalResult:
    question: str
    retrieved: list[tuple[Chunk, float]]   # (chunk, score) from retriever


@dataclass
class MetricResult:
    model_key:  str
    chunk_size: int
    top_k:      int
    recall:     float
    precision:  float
    mrr:        float
    # generation
    answer_relevance:  float = 0.0
    faithfulness:      float = 0.0
    context_precision: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Computes Recall@K, Precision@K, and MRR over a list of QAPairs.
    """

    @staticmethod
    def recall_at_k(
        retrieved: list[Chunk],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        if not relevant_ids:
            return 0.0
        top_ids = {c.chunk_id for c in retrieved[:k]}
        return len(top_ids & relevant_ids) / len(relevant_ids)

    @staticmethod
    def precision_at_k(
        retrieved: list[Chunk],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        if k == 0:
            return 0.0
        top_ids = [c.chunk_id for c in retrieved[:k]]
        hits = sum(1 for cid in top_ids if cid in relevant_ids)
        return hits / k

    @staticmethod
    def reciprocal_rank(
        retrieved: list[Chunk],
        relevant_ids: set[str],
    ) -> float:
        for rank, chunk in enumerate(retrieved, start=1):
            if chunk.chunk_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def evaluate(
        self,
        qa_pairs: list[QAPair],
        retrieval_fn: Callable[[str, int], list[tuple[Chunk, float]]],
        k: int = config.DEFAULT_TOP_K,
    ) -> dict:
        """
        Run *retrieval_fn(query, k)* for every QA pair and aggregate metrics.

        *retrieval_fn* should match the signature of Retriever.retrieve.
        """
        recalls, precisions, rrs = [], [], []

        for qa in qa_pairs:
            results   = retrieval_fn(qa.question, k)
            retrieved = [chunk for chunk, _ in results]
            rel_ids   = set(qa.relevant_chunk_ids)

            recalls.append(self.recall_at_k(retrieved, rel_ids, k))
            precisions.append(self.precision_at_k(retrieved, rel_ids, k))
            rrs.append(self.reciprocal_rank(retrieved, rel_ids))

        return {
            f"Recall@{k}":    round(float(np.mean(recalls)),    4),
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            "MRR":            round(float(np.mean(rrs)),         4),
        }

    def evaluate_k_sweep(
        self,
        qa_pairs: list[QAPair],
        retrieval_fn: Callable[[str, int], list[tuple[Chunk, float]]],
        k_values: list[int] = config.EVAL_K_VALUES,
    ) -> dict[int, dict]:
        return {k: self.evaluate(qa_pairs, retrieval_fn, k) for k in k_values}


# ──────────────────────────────────────────────────────────────────────────────
# Generation Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class GenerationEvaluator:
    """
    Reference-free generation evaluation.

    Answer Relevance  – how similar is the generated answer to the question?
                        Uses embedding cosine similarity.
    Faithfulness      – what fraction of answer sentences can be traced back
                        to the retrieved context? (string overlap heuristic)
    Context Precision – what fraction of retrieved chunks contributed to the
                        answer? (simple keyword overlap)
    """

    def __init__(self, embedding_model=None) -> None:
        self._emb = embedding_model   # optional EmbeddingModel for relevance

    # ── Answer Relevance ──────────────────────────────────────────────────────

    def answer_relevance(self, question: str, answer: str) -> float:
        """
        Cosine similarity via embeddings if available, else content-word Jaccard.
        Filters stop words so short answers don't score zero.
        """
        if self._emb is not None:
            q_vec = self._emb.encode_query(question)
            a_vec = self._emb.encode_query(answer)
            return float(np.dot(q_vec, a_vec.T).squeeze())

        # Stop-word filtered Jaccard
        stop = {
            "the","a","an","is","are","was","were","be","been","have","has",
            "had","do","does","did","will","would","could","should","may",
            "might","this","that","these","those","in","on","at","to","for",
            "of","and","or","but","not","with","from","by","as","it","its",
            "what","how","why","which","paper","propose","describe","method"
        }
        q_toks = {w.lower().strip("?.,'\"") for w in question.split()
                if w.lower().strip("?.,'\"") not in stop and len(w) > 2}
        a_toks = {w.lower().strip("?.,'\"") for w in answer.split()
                if w.lower().strip("?.,'\"") not in stop and len(w) > 2}

        if not q_toks or not a_toks:
            return 0.0
        return len(q_toks & a_toks) / len(q_toks | a_toks)

    # ── Faithfulness ──────────────────────────────────────────────────────────

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def faithfulness(self, answer: str, context: str) -> float:
        """
        For each sentence in the answer, check whether any bigram (2-word sequence)
        from that sentence appears in the context.
        Returns the fraction of answer sentences that are supported.
        """
        sentences = self._sentences(answer)
        if not sentences:
            return 0.0

        # Also check single important words (length > 5) as fallback
        stop = {
            "the","a","an","is","are","was","were","be","been","have","has",
            "had","do","does","did","will","would","could","should","may",
            "might","this","that","these","those","in","on","at","to","for",
            "of","and","or","but","not","with","from","by","as","it","its",
            "which","such","can","using","used","use","based","also","their",
            "source","context","cannot","find","reliable","provided","sources"
        }
        ctx_lower = context.lower()
        supported = 0

        for sent in sentences:
            words = [w.lower().strip(".,;:()\"'") for w in sent.split()]

            # Try bigrams first
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            if any(bg in ctx_lower for bg in bigrams):
                supported += 1
                continue

            # Fallback: any long content word appears in context
            content_words = [w for w in words if w not in stop and len(w) > 5]
            if any(w in ctx_lower for w in content_words):
                supported += 1

        return supported / len(sentences)

    # ── Context Precision ─────────────────────────────────────────────────────

    def context_precision(
        self,
        answer: str,
        retrieved_chunks: list[Chunk],
    ) -> float:
        """
        Fraction of retrieved chunks that share at least one content word
        with the generated answer.
        """
        if not retrieved_chunks:
            return 0.0

        # Remove stop words (simple list)
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "this", "that", "these",
            "those", "in", "on", "at", "to", "for", "of", "and", "or",
            "but", "not", "with", "from", "by", "as", "it", "its"
        }
        answer_words = {w for w in answer.lower().split() if w not in stop and len(w) > 2}

        used = 0
        for chunk in retrieved_chunks:
            chunk_words = {w for w in chunk.text.lower().split() if w not in stop and len(w) > 2}
            if answer_words & chunk_words:
                used += 1

        return used / len(retrieved_chunks)

    # ── Combined evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        qa_pairs: list[QAPair],
        retrieval_fn: Callable[[str, int], list[tuple[Chunk, float]]],
        generation_fn: Callable[[str, str], str],
        top_k: int = config.DEFAULT_TOP_K,
    ) -> dict:
        ar_scores, faith_scores, cp_scores = [], [], []

        for qa in qa_pairs:
            retrieved = retrieval_fn(qa.question, top_k)
            chunks    = [c for c, _ in retrieved]
            context   = "\n\n".join(
                f"[Source {i+1}] {c.text}" for i, c in enumerate(chunks)
            )
            answer = generation_fn(qa.question, context)

            ar_scores.append(self.answer_relevance(qa.question, answer))
            faith_scores.append(self.faithfulness(answer, context))
            cp_scores.append(self.context_precision(answer, chunks))

        return {
            "Answer Relevance":  round(float(np.mean(ar_scores)),    4),
            "Faithfulness":      round(float(np.mean(faith_scores)), 4),
            "Context Precision": round(float(np.mean(cp_scores)),    4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth QA creation helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_synthetic_qa_pairs(chunks: list[Chunk], n: int = 50) -> list[QAPair]:
    """
    Create synthetic evaluation QA pairs.
    Relevance is paper-level: ALL chunks from the same paper are considered
    relevant, making retrieval evaluation meaningful.
    """
    import random
    random.seed(42)

    # Group chunk_ids by paper_id
    paper_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        paper_to_chunks.setdefault(chunk.paper_id, []).append(chunk.chunk_id)

    # Sample one chunk per paper (up to n)
    sampled = random.sample(chunks, min(n, len(chunks)))

    templates = [
        "What does the paper '{}' propose?",
        "What method is described in '{}'?",
        "What are the main findings of '{}'?",
        "How does the approach in '{}' work?",
        "What problem does '{}' solve?",
    ]

    qa_pairs = []
    for chunk in sampled:
        template = random.choice(templates)
        question = template.format(chunk.title[:60])
        # All chunks from the same paper are relevant answers
        relevant_ids = paper_to_chunks.get(chunk.paper_id, [chunk.chunk_id])
        qa_pairs.append(QAPair(
            question            = question,
            reference_answer    = chunk.text[:300],
            relevant_chunk_ids  = relevant_ids,
        ))
    return qa_pairs


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ev = RetrievalEvaluator()

    def dummy_retriever(q, k):
        c = Chunk("p1_0000", "p1", "Title", "Some text.", 3, 0)
        return [(c, 0.9)]

    qa = [QAPair("What is attention?", "It is a mechanism.", ["p1_0000"])]
    metrics = ev.evaluate(qa, dummy_retriever, k=5)
    print("Retrieval metrics:", metrics)

def load_manual_qa_pairs(
    chunks: list[Chunk],
    path: str = None,
) -> list[QAPair]:
    """
    Load manually annotated QA pairs from data/manual_qa.json.
    Matches each entry to all chunks from the same paper_id.
    """
    import json
    qa_path = path or str(config.DATA_DIR / "manual_qa.json")
    if not Path(qa_path).exists():
        log.warning("manual_qa.json not found at %s", qa_path)
        return []

    with open(qa_path) as f:
        raw = json.load(f)

    # Build paper_id → chunk_ids map
    paper_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        paper_to_chunks.setdefault(chunk.paper_id, []).append(chunk.chunk_id)

    qa_pairs = []
    skipped  = 0
    for entry in raw:
        pid = entry["paper_id"]
        if pid not in paper_to_chunks:
            log.warning("paper_id '%s' not found in chunks — skipping", pid)
            skipped += 1
            continue
        qa_pairs.append(QAPair(
            question            = entry["question"],
            reference_answer    = entry["answer"],
            relevant_chunk_ids  = paper_to_chunks[pid],
        ))

    log.info("Loaded %d manual QA pairs (%d skipped)", len(qa_pairs), skipped)
    return qa_pairs