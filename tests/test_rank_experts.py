"""Tests of vllm-rtx3090-flash-next/rvii_experts/rank_experts.py."""

from __future__ import annotations

import array
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from agx_ai.hostname import REPO_DIR

DIR = REPO_DIR / "vllm-rtx3090-flash-next" / "rvii_experts"
_spec = importlib.util.spec_from_file_location("rank_experts", DIR / "rank_experts.py")
assert _spec is not None
assert _spec.loader is not None
rank_experts: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rank_experts)

L: int = rank_experts.LAYERS
E: int = rank_experts.EXPERTS


def write_usage(path: Path, counts: dict[tuple[int, int], int]) -> Path:
    """A usage file with the given (layer, expert) counts and zeros elsewhere."""
    values = array.array("I", [0] * (L * E))
    for (layer, expert), n in counts.items():
        values[layer * E + expert] = n
    path.write_bytes(values.tobytes())
    return path


def test_reproduces_committed_ranking(tmp_path: Path) -> None:
    """The committed measured ranking is the general collection ranked with the static ranking as the fallback."""
    out = tmp_path / "r.json"
    assert rank_experts.main([str(out), str(DIR / "usage" / "general.bin")]) == 0
    assert out.read_bytes() == (DIR / "rankings-measured.json").read_bytes()


def test_weights_mix_normalized_collections(tmp_path: Path) -> None:
    """Collections are normalized per layer, so the weights decide between them regardless of their sizes."""
    big = write_usage(tmp_path / "big.bin", {(0, 1): 1000, (0, 2): 0})
    small = write_usage(tmp_path / "small.bin", {(0, 2): 1})
    fallback = [list(range(E)) for _ in range(L)]
    sources = [(rank_experts.load_usage(big), 1.0), (rank_experts.load_usage(small), 1.0)]
    assert rank_experts.build_ranking(sources, fallback)[0][:2] == [1, 2]  # tie: fallback order
    sources[1] = (sources[1][0], 2.0)
    assert rank_experts.build_ranking(sources, fallback)[0][:2] == [2, 1]


def test_fallback_orders_unseen_experts() -> None:
    """Unseen experts follow the fallback ranking, then the experts that it does not list by index."""
    counts = [[0] * E for _ in range(L)]
    counts[0][7] = 5
    fallback = [[3, 2] for _ in range(L)]
    ranking = rank_experts.build_ranking([(counts, 1.0)], fallback)
    assert ranking[0][:5] == [7, 3, 2, 0, 1]
    assert all(sorted(r) == list(range(E)) for r in ranking)


def test_split_and_parse_source() -> None:
    """The split of the requests by rank, and FILE:WEIGHT parsing."""
    counts = [[0] * E for _ in range(L)]
    counts[0][0], counts[0][1], counts[0][2] = 2, 1, 1
    ranking = [list(range(E)) for _ in range(L)]
    assert rank_experts.split(counts, ranking, 1, 1) == (0.5, 0.25, 0.25)
    assert rank_experts.parse_source("a/b.bin:0.7") == (Path("a/b.bin"), 0.7)
    assert rank_experts.parse_source("a/b.bin") == (Path("a/b.bin"), 1.0)


def test_rejects_wrong_size(tmp_path: Path) -> None:
    """A file of the wrong size is not a usage file."""
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\0" * 16)
    with pytest.raises(ValueError, match="expected"):
        rank_experts.load_usage(bad)
