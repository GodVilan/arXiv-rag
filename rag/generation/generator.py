"""
generation.py – Answer generation using Google Gemini (google-genai SDK).
"""

import logging
import time
import threading

from google import genai
from google.genai import types

from rag import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research assistant with access to a collection of recent machine learning
research papers from arXiv (cs.LG category), published in early 2026.

The user does NOT know which papers are in the collection. They will ask general
questions about ML topics, methods, or concepts. Your job is to:
- Search the provided context passages for relevant information
- Answer based on what the retrieved papers say about the topic
- Always mention which paper(s) the information comes from (e.g. [Source 1]: <paper title>)
- If multiple papers touch the topic, synthesise their perspectives
- If the context is partially relevant, still extract and explain what is useful
- Only say you cannot answer if the passages are completely unrelated

The user is exploring ML research — treat every question as an opportunity to
connect them with relevant findings from the literature, even if their question
is broad (e.g. "What is reinforcement learning?" should be answered using
whatever the retrieved papers say about it, not refused).
"""

def _build_prompt(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


# ── Proactive rate limiter ────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.min_interval = 60.0 / rpm
        self._lock        = threading.Lock()
        self._last_call   = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            gap     = self.min_interval - elapsed
            if gap > 0:
                log.debug("Rate limiter sleeping %.2fs", gap)
                time.sleep(gap)
            self._last_call = time.monotonic()


# ── Gemini generator (new google-genai SDK) ───────────────────────────────────

class GeminiGenerator:
    def __init__(
        self,
        model_name: str    = config.GEMINI_MODEL,
        api_key: str       = config.GEMINI_API_KEY,
        temperature: float = config.GEMINI_TEMPERATURE,
        max_tokens: int    = config.GEMINI_MAX_TOKENS,
        rpm: int           = config.GEMINI_RPM,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Add it to .env:  GEMINI_API_KEY=AIza...")

        self._client     = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._gen_cfg    = types.GenerateContentConfig(
            system_instruction = SYSTEM_PROMPT,
            temperature        = temperature,
            max_output_tokens  = max_tokens,
        )
        self._limiter = _RateLimiter(rpm)
        log.info("Gemini ready  model=%s  rpm=%d  interval=%.1fs",
                 model_name, rpm, 60.0 / rpm)

    def generate(self, query: str, context: str, retries: int = 3) -> str:
        prompt = _build_prompt(query, context)

        for attempt in range(1, retries + 1):
            self._limiter.wait()          # proactive throttle — prevents 429s
            try:
                response = self._client.models.generate_content(
                    model    = self._model_name,
                    contents = prompt,
                    config   = self._gen_cfg,
                )
                return response.text.strip()
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    wait = 30 * attempt   # 30s, 60s, 90s
                    log.warning("429 on attempt %d/%d — waiting %ds", attempt, retries, wait)
                    time.sleep(wait)
                else:
                    log.error("Gemini error: %s", exc)
                    return f"[Generation error: {exc}]"

        return "[Generation error: max retries exceeded]"


# ── Public interface ──────────────────────────────────────────────────────────

class Generator:
    def __init__(self) -> None:
        self._backend = GeminiGenerator()

    def generate(self, query: str, context: str) -> str:
        return self._backend.generate(query=query, context=context)

    def generate_with_retriever(
        self, query: str, retriever, top_k: int = config.DEFAULT_TOP_K
    ) -> dict:
        results = retriever.retrieve(query, top_k=top_k)
        context = retriever.format_context(query, top_k=top_k)
        answer  = self.generate(query, context)
        sources = [
            {"title": c.title, "chunk_id": c.chunk_id, "score": round(s, 4)}
            for c, s in results
        ]
        return {"query": query, "context": context, "answer": answer, "sources": sources}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = Generator()
    ctx = (
        "[Source 1] Vaswani et al. (2017) – Attention Is All You Need\n"
        "Self-attention computes a weighted sum of value vectors where weights "
        "come from query-key dot-product similarity, allowing every token to "
        "attend to all positions simultaneously."
    )
    print(gen.generate("What is self-attention?", ctx))
