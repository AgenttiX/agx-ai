"""AI performance and quality benchmarks.

The modules use only the Python standard library (except ``embedding_retrieval_eval``, which needs NumPy),
so that they can be run with Python 3.11 or newer on the benchmarked computer with
``python -m agx_ai.benchmark.<module>`` from the repository root. With uv, use ``uv run -m agx_ai.benchmark.<module>``
in any directory of the repository.
"""

from pathlib import Path

from agx_ai.hostname import REPO_DIR

#: The results are stored in the repository and not in the package.
RESULTS_DIR: Path = REPO_DIR / "benchmark" / "results"
