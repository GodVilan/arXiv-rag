"""
Refresh only generation metrics/plots using cached data and the current config.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiments import step_chunking, step_data, step_generation, step_save_and_plot  # noqa: E402
from rag import config  # noqa: E402


def main() -> None:
    retrieval_path = config.RESULTS_DIR / "retrieval_metrics.json"
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Missing retrieval metrics: {retrieval_path}")

    with retrieval_path.open() as handle:
        retrieval_results = json.load(handle)

    papers = step_data()
    chunks_by_size = step_chunking(papers)
    generation_results = step_generation(chunks_by_size)
    step_save_and_plot(retrieval_results, generation_results)


if __name__ == "__main__":
    main()
