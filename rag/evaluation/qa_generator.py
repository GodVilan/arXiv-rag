"""
qa_generator.py – Auto-generate manual_qa.json using Gemini.

Reads each paper's abstract + first chunk and asks Gemini to produce
one high-quality, specific question-answer pair per paper.

Run:
    python3 qa_generator.py
"""

import json
import logging
import time

from google import genai
from google.genai import types

from rag import config
from rag.processing.chunker import load_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research expert creating evaluation data for a RAG system.
Given a machine learning paper's title, abstract, and a content excerpt,
generate exactly ONE question-answer pair that:
  1. Is specific to THIS paper (not generic ML knowledge)
  2. Can be answered directly from the provided text
  3. Tests understanding of the paper's core contribution or method
  4. Uses natural language a researcher would actually ask

Respond in this exact JSON format with no other text:
{
  "question": "...",
  "answer": "..."
}
"""


def build_prompt(title: str, abstract: str, excerpt: str) -> str:
    return f"""Paper title: {title}

Abstract:
{abstract[:800]}

Content excerpt:
{excerpt[:600]}

Generate one specific question-answer pair about this paper's contribution or method."""


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_qa_pairs(
    n_papers: int = 20,
    output_path: str = None,
) -> list[dict]:
    output_path = output_path or str(config.DATA_DIR / "manual_qa.json")

    # Load metadata and chunks
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        log.error("metadata.json not found. Run data_collection.py first.")
        return []

    with open(meta_path) as f:
        papers = json.load(f)

    chunks_path = config.DATA_DIR / f"chunks_{config.DEFAULT_CHUNK}.json"
    if not chunks_path.exists():
        log.error("chunks_%d.json not found. Run run_experiments.py first.", config.DEFAULT_CHUNK)
        return []

    chunks = load_chunks(chunks_path)

    # Build paper_id → first chunk text
    paper_first_chunk: dict[str, str] = {}
    for chunk in chunks:
        if chunk.paper_id not in paper_first_chunk:
            paper_first_chunk[chunk.paper_id] = chunk.text

    # Set up Gemini client
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    cfg    = types.GenerateContentConfig(
        system_instruction = SYSTEM_PROMPT,
        temperature        = 0.3,
        max_output_tokens  = 300,
    )

    interval = 60.0 / config.GEMINI_RPM   # rate limit spacing
    results  = []
    selected = [p for p in papers if p["paper_id"] in paper_first_chunk][:n_papers]

    log.info("Generating QA pairs for %d papers (%.1fs between calls) …",
             len(selected), interval)

    for i, paper in enumerate(selected, 1):
        pid      = paper["paper_id"]
        title    = paper["title"]
        abstract = paper.get("abstract", "")
        excerpt  = paper_first_chunk.get(pid, "")

        log.info("[%d/%d] %s", i, len(selected), title[:60])

        time.sleep(interval)   # proactive rate limiting

        try:
            prompt   = build_prompt(title, abstract, excerpt)
            response = client.models.generate_content(
                model    = config.GEMINI_MODEL,
                contents = prompt,
                config   = cfg,
            )
            raw = response.text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            question = parsed.get("question", "").strip()
            answer   = parsed.get("answer", "").strip()

            if not question or not answer:
                log.warning("  Empty Q/A — skipping")
                continue

            results.append({
                "paper_id": pid,
                "title":    title,
                "question": question,
                "answer":   answer,
            })
            log.info("  Q: %s", question[:80])

        except json.JSONDecodeError as e:
            log.warning("  JSON parse error: %s — raw: %s", e, raw[:100])
        except Exception as e:
            log.error("  Gemini error: %s", e)

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("Saved %d QA pairs to %s", len(results), output_path)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Auto-generate manual_qa.json using Gemini")
    p.add_argument("--n",      type=int, default=20,  help="Number of papers to process")
    p.add_argument("--output", type=str, default=None, help="Output path (default: data/manual_qa.json)")
    args = p.parse_args()

    pairs = generate_qa_pairs(n_papers=args.n, output_path=args.output)

    print(f"\n✅  Generated {len(pairs)} QA pairs\n")
    for i, pair in enumerate(pairs[:3], 1):
        print(f"  [{i}] {pair['title'][:50]}")
        print(f"       Q: {pair['question']}")
        print(f"       A: {pair['answer'][:100]}…")
        print()