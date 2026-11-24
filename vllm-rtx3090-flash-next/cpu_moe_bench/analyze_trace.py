"""Analyze a decode trace of the CPU expert kernel (QWEN38_DECODE_TRACE).

Shows where a decode step's time goes, how many tokens the CPU's experts get, and how many of a layer's CPU experts
were CPU experts of the same layer in the previous step (what the expert prefetch, QWEN38_CPU_EXPERTS_PREFETCH, can
hit).

    python3 analyze_trace.py TRACE [--rankings ../rvii_experts/rankings-measured.json] [--hot 24] [--rvii 135]
    python3 analyze_trace.py TRACE --skip-out skip.bin   # also write the skip mask of bench.cpp's replay mode

Only uses the Python standard library. See README.md.
"""

import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
import struct
import sys

REC: struct.Struct = struct.Struct("<qqqii40i")  # t_req, t_cpu_end, t_done (ns), layer, T, ids[40]
LAYERS: int = 48
EXPERTS: int = 512
HERE: Path = Path(__file__).resolve().parent
GPU: int = 0
RVII: int = 1
CPU: int = 2

Record = tuple[int, ...]


def placement(rankings: Path, hot: int, rvii: int) -> list[list[int]]:
    """Per layer and expert: GPU (the first `hot` of the ranking), RVII (the next `rvii`) or CPU."""
    with rankings.open() as fh:
        rank = json.load(fh)
    where = []
    for layer in range(LAYERS):
        w = [CPU] * EXPERTS
        for i, e in enumerate(rank[str(layer)]):
            w[e] = GPU if i < hot else RVII if i < hot + rvii else CPU
        where.append(w)
    return where


def read_steps(path: Path) -> list[list[Record]]:
    """The complete decode steps of a trace (layers 0..47 in order)."""
    data = path.read_bytes()
    steps: list[list[Record]] = []
    cur: list[Record] = []
    for off in range(0, len(data) - REC.size + 1, REC.size):
        r = REC.unpack_from(data, off)
        if r[3] == 0 and cur:
            steps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        steps.append(cur)
    return [s for s in steps if [r[3] for r in s] == list(range(LAYERS))]


def percentiles(values: list[int]) -> list[int]:
    """The 10th, 50th, 90th and 99th percentile in microseconds."""
    v = sorted(values)
    return [round(v[int(q * (len(v) - 1))] / 1e3) for q in (0.1, 0.5, 0.9, 0.99)] if v else []


def main(argv: list[str] | None = None) -> int:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--rankings", type=Path, default=HERE.parent / "rvii_experts" / "rankings-measured.json")
    parser.add_argument("--hot", type=int, default=24, help="experts per layer on the RTX 3090 (QWEN38_HOT_ONLY)")
    parser.add_argument("--rvii", type=int, default=135, help="experts per layer on the Radeon VII (0: none)")
    parser.add_argument("--skip-out", type=Path, help="write uint8 [48][512], 1 = not a CPU expert")
    args = parser.parse_args(argv)

    where = placement(args.rankings, args.hot, args.rvii)
    if args.skip_out:
        args.skip_out.write_bytes(bytes(int(where[ly][e] != CPU) for ly in range(LAYERS) for e in range(EXPERTS)))
    steps = read_steps(args.trace)
    if len(steps) < 2:
        print("fewer than 2 complete decode steps in the trace")
        return 1
    n = len(steps)
    cpu_jobs = rv_jobs = 0
    tokens = Counter()
    compute, rv_wait, gaps, tails = [], [], [], []
    hits = total = 0
    top = {k: [0, 0] for k in (4, 8, 16)}
    prev: list[Counter] | None = None
    for si, step in enumerate(steps):
        sets = []
        for t_req, t_cpu, t_done, layer, t, *ids in step:
            routed = [e for e in ids[: t * 10] if e >= 0]
            cnt = Counter(e for e in routed if where[layer][e] == CPU)
            cpu_jobs += len(cnt)
            rv_jobs += len({e for e in routed if where[layer][e] == RVII})
            tokens.update(cnt.values())
            sets.append(cnt)
            compute.append(t_done - t_req)
            rv_wait.append(t_done - t_cpu)
        gaps.extend(b[0] - a[2] for a, b in pairwise(step))
        if si + 1 < n and steps[si + 1][0][0] - step[-1][2] < 1_000_000_000:
            tails.append(steps[si + 1][0][0] - step[-1][2])
        if prev is not None:
            for layer in range(LAYERS):
                cur, before = sets[layer], prev[layer]
                total += len(cur)
                hits += sum(1 for e in cur if e in before)
                order = [e for e, _ in before.most_common()]
                for k, acc in top.items():
                    chosen = set(order[:k])
                    acc[0] += sum(1 for e in cur if e in chosen)
                    acc[1] += len(chosen)
        prev = sets

    ms = 1e6 * n
    print(f"{n} decode steps (T: {dict(Counter(s[0][4] for s in steps))})")
    print(f"per step: {cpu_jobs / n:.0f} CPU experts, {rv_jobs / n:.0f} Radeon VII experts")
    print(f"tokens per CPU expert: {dict(sorted(tokens.items()))}")
    print(f"per step: CPU layers {sum(compute) / ms:.2f} ms (of which waiting for the Radeon VII "
          f"{sum(rv_wait) / ms:.2f}), GPU between the layers {sum(gaps) / ms:.2f} ms, "
          f"step tail {sum(tails) / max(1, len(tails)) / 1e6:.2f} ms")
    print(f"percentiles (10/50/90/99, us): CPU layer {percentiles(compute)}, gap {percentiles(gaps)}")
    print(f"{hits / max(1, total):.1%} of the CPU experts were CPU experts of the same layer in the previous step")
    pairs = (n - 1) * LAYERS
    for k, (h, p) in top.items():
        print(f"  the previous step's top {k} per layer: {h / pairs:.2f} hits of {p / pairs:.2f} "
              f"({h / max(1, p):.0%}), {h / max(1, total):.1%} of the CPU experts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
