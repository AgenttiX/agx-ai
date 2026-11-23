"""Build an expert ranking for Qwen3.8-Flash-Next from measured expert usage, e.g. to favor particular topics.

The ranking file lists the 512 experts of each of the 48 layers, most used first. The runtime keeps the first
`QWEN38_HOT_ONLY` (24) of each layer on the RTX 3090, the Radeon VII worker the next ones (`RVII_HOT`,
`RVII_PER_LAYER`: 135), and the CPU computes the rest. See README.md and ../README.md (Radeon VII experts).

Usage counts come from the runtime with `QWEN38_EXPERT_USAGE=<file>`: uint32 little-endian [48][512], the decode
requests of each routed expert, rewritten every 256 decode steps (so stop the collection only after the last write).
Each collection is normalized per layer to the fraction of that layer's requests and weighted, so that collections of
different sizes mix by their weights. Ties (e.g. experts never seen) keep the order of the fallback ranking (the
runtime's static ranking, rankings.json), and the experts that it does not list follow by index.

    python3 rank_experts.py rankings-code.json usage/general.bin:0.3 my-code.bin:0.7
    python3 rank_experts.py --report rankings-measured.json usage/general.bin my-code.bin

The first command writes a ranking that mixes 30 % of the general collection with 70 % of a code collection; the second
shows how the requests of each collection would be split between the RTX 3090, the Radeon VII and the CPU with a given
ranking. To use a new ranking, mount it in docker-compose.yml (over static_hot_cache_rankings.json) and set
RVII_RANKINGS of the rvii-experts service to it; the worker packs a new cache for it (~15 s, ~15 GB of disk).

Only uses the Python standard library.
"""

import argparse
import array
import json
from pathlib import Path
import sys

LAYERS = 48
EXPERTS = 512
HERE = Path(__file__).resolve().parent

Counts = list[list[int]]
Ranking = list[list[int]]


def load_usage(path: Path) -> Counts:
    """The counts of a usage file, [layer][expert]."""
    data = path.read_bytes()
    if len(data) != LAYERS * EXPERTS * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {LAYERS * EXPERTS * 4} (uint32 [{LAYERS}][{EXPERTS}])")
    values = array.array("I")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return [list(values[layer * EXPERTS:(layer + 1) * EXPERTS]) for layer in range(LAYERS)]


def load_ranking(path: Path) -> Ranking:
    """A ranking file ({"0": [expert, ...], ...}) as [layer][rank]; it may list only part of the experts."""
    with path.open() as fh:
        data = json.load(fh)
    return [[int(e) for e in data[str(layer)]] for layer in range(LAYERS)]


def save_ranking(path: Path, ranking: Ranking) -> None:
    """A ranking file in the runtime's format, one layer per line."""
    lines = [f'"{layer}": {json.dumps(experts)}' for layer, experts in enumerate(ranking)]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def parse_source(spec: str) -> tuple[Path, float]:
    """FILE or FILE:WEIGHT."""
    path, sep, weight = spec.rpartition(":")
    if sep and path:
        try:
            return Path(path), float(weight)
        except ValueError:
            pass
    return Path(spec), 1.0


def build_ranking(sources: list[tuple[Counts, float]], fallback: Ranking) -> Ranking:
    """Rank the experts of each layer by the weighted sum of their per-layer request fractions."""
    if not sources:
        raise ValueError("no usage counts")
    ranking: Ranking = []
    for layer in range(LAYERS):
        score = [0.0] * EXPERTS
        for counts, weight in sources:
            total = sum(counts[layer])
            if total:
                for e, n in enumerate(counts[layer]):
                    score[e] += weight * n / total
        order = {e: i for i, e in enumerate(fallback[layer])}
        ranking.append(sorted(range(EXPERTS), key=lambda e: (-score[e], order.get(e, EXPERTS + e))))
    return ranking


def split(counts: Counts, ranking: Ranking, hot: int, rvii: int) -> tuple[float, float, float]:
    """The fractions of the requests that the RTX 3090, the Radeon VII and the CPU would compute."""
    gpu = rv = total = 0
    for layer in range(LAYERS):
        order = ranking[layer]
        layer_counts = counts[layer]
        gpu += sum(layer_counts[e] for e in order[:hot])
        rv += sum(layer_counts[e] for e in order[hot:hot + rvii])
        total += sum(layer_counts)
    if not total:
        return 0.0, 0.0, 0.0
    return gpu / total, rv / total, (total - gpu - rv) / total


def main(argv: list[str] | None = None) -> int:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path,
                        help="the ranking file to write (or, with --report, an existing ranking to evaluate)")
    parser.add_argument("usage", nargs="+", help="usage files of QWEN38_EXPERT_USAGE, as FILE or FILE:WEIGHT")
    parser.add_argument("--fallback", type=Path, default=HERE / "rankings.json",
                        help="ranking for ties and unseen experts (default: the runtime's static ranking)")
    parser.add_argument("--report", action="store_true",
                        help="do not write; show the split of each usage file with the ranking OUTPUT and the fallback")
    parser.add_argument("--hot", type=int, default=24, help="experts per layer on the RTX 3090 (default 24)")
    parser.add_argument("--rvii", type=int, default=135, help="experts per layer on the Radeon VII (default 135)")
    args = parser.parse_args(argv)

    sources = [(load_usage(path), weight) for path, weight in map(parse_source, args.usage)]
    fallback = load_ranking(args.fallback)
    if args.report:
        rankings = {str(args.output): load_ranking(args.output), f"{args.fallback} (fallback)": fallback}
    else:
        ranking = build_ranking(sources, fallback)
        save_ranking(args.output, ranking)
        print(f"wrote {args.output}")
        rankings = {str(args.output): ranking, f"{args.fallback} (fallback)": fallback}

    print(f"Share of the requests: RTX 3090 (first {args.hot}) / Radeon VII (next {args.rvii}) / CPU")
    for name, ranking in rankings.items():
        print(name)
        for spec, (counts, _) in zip(args.usage, sources, strict=True):
            gpu, rv, cpu = split(counts, ranking, args.hot, args.rvii)
            total = sum(map(sum, counts))
            print(f"  {spec}: {gpu:6.1%} / {rv:6.1%} / {cpu:6.1%}  ({total} requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
