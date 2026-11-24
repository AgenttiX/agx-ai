"""Evaluate predicting a layer's CPU experts from the previous layer's MoE input ("pre-gating"), for the prefetch.

Reads a dump of the decode requests (``QWEN38_DECODE_XDUMP``: per request {layer, T, ids[40], x[4][2560] FP32}) and
the routers (``mlp.gate.weight``, 512 x 2560 BF16) from the checkpoint, and compares, for the same prefetch budget
per layer (``--budget``, the idle pool threads: 7), which of a layer's CPU experts would have been prefetched:

- ``previous step``: the CPU experts of the same layer in the previous decode step, most tokens first (the current
  ``QWEN38_CPU_EXPERTS_PREFETCH``),
- ``pre-gating``: the CPU experts that the layer's router chooses for the previous layer's MoE input, ranked by the
  number of tokens that would choose them, then by their summed router probability,
- ``both``: half of the budget each (the pre-gating ones first), without duplicates.

Also checks that each layer's own router reproduces its routing (the dump's x is the routers' input).

    python3 pregate_eval.py DUMP --model ~/.cache/vllm-models/Qwen3.8-Flash-Next-W4A16-FP8PLE-dense8-view

Needs numpy.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct
import sys

from analyze_trace import CPU, EXPERTS, HERE, LAYERS, placement
import numpy as np

H: int = 2560
TOPK: int = 10
HEAD: struct.Struct = struct.Struct("<42i")  # layer, T, ids[40]
REC_BYTES: int = HEAD.size + 4 * H * 4


CONTAINER_HUB: str = "/root/.cache/huggingface/hub"


def host_path(path: Path, hub: Path) -> Path:
    """path, with a symbolic link into the containers' Hugging Face cache (the checkpoint views) mapped to hub."""
    if path.is_symlink():
        target = str(path.readlink())
        if target.startswith(CONTAINER_HUB):
            return hub / target[len(CONTAINER_HUB):].lstrip("/")
    return path


def load_routers(model: Path, hub: Path) -> np.ndarray:
    """The routers of all layers as FP32 [48, 512, 2560], read from the safetensors files without safetensors."""
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    out = np.zeros((LAYERS, EXPERTS, H), dtype=np.float32)
    for layer in range(LAYERS):
        name = f"model.language_model.layers.{layer}.mlp.gate.weight"
        path = host_path(model / index[name], hub).resolve()
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(n))[name]
            if meta["dtype"] != "BF16" or meta["shape"] != [EXPERTS, H]:
                raise ValueError(f"unexpected router {name}: {meta}")
            a, b = meta["data_offsets"]
            fh.seek(8 + n + a)
            raw = np.frombuffer(fh.read(b - a), dtype=np.uint16)
        out[layer] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(EXPERTS, H)
    return out


def read_steps(path: Path) -> list[list[tuple[int, list[int], np.ndarray]]]:
    """The complete decode steps (layers 0..47 in order) of the dump: per layer (T, ids, x [T, H])."""
    data = path.read_bytes()
    steps, cur = [], []
    for off in range(0, len(data) - REC_BYTES + 1, REC_BYTES):
        head = HEAD.unpack_from(data, off)
        layer, t = head[0], head[1]
        x = np.frombuffer(data, dtype=np.float32, count=t * H, offset=off + HEAD.size).reshape(t, H)
        if layer == 0 and cur:
            steps.append(cur)
            cur = []
        cur.append((layer, t, list(head[2:2 + t * TOPK]), x))
    if cur:
        steps.append(cur)
    return [[(t, ids, x) for _, t, ids, x in s] for s in steps if [r[0] for r in s] == list(range(LAYERS))]


def ranked_cpu_experts(router: np.ndarray, x: np.ndarray, where: list[int]) -> list[int]:
    """The CPU experts that router chooses (top 10 per token) for x, most tokens first, then by probability."""
    logits = x @ router.T  # [T, 512]
    top = np.argsort(-logits, axis=1)[:, :TOPK]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    score: dict[int, list[float]] = defaultdict(lambda: [0, 0.0])
    for t in range(x.shape[0]):
        for e in top[t]:
            if where[e] == CPU:
                score[int(e)][0] += 1
                score[int(e)][1] += float(p[t, e])
    return sorted(score, key=lambda e: (-score[e][0], -score[e][1]))


def main(argv: list[str] | None = None) -> int:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="the checkpoint directory (with the index)")
    parser.add_argument("--hub", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub",
                        help=f"the Hugging Face cache that the view's links to {CONTAINER_HUB} mean")
    parser.add_argument("--rankings", type=Path, default=HERE.parent / "rvii_experts" / "rankings-measured.json")
    parser.add_argument("--hot", type=int, default=24)
    parser.add_argument("--rvii", type=int, default=135)
    parser.add_argument("--budget", type=int, nargs="+", default=[7, 14], help="experts prefetched per layer")
    args = parser.parse_args(argv)

    where = placement(args.rankings, args.hot, args.rvii)
    routers = load_routers(args.model, args.hub)
    steps = read_steps(args.dump)
    if len(steps) < 2:
        print("fewer than 2 complete decode steps in the dump")
        return 1
    print(f"{len(steps)} decode steps")

    same = total_routes = 0
    cpu_total = 0
    hits = {(m, b): 0 for m in ("previous step", "pre-gating", "both") for b in args.budget}
    prev: list[list[int]] | None = None
    for step in steps:
        cur_lists = []
        for layer, (t, ids, x) in enumerate(step):
            own = np.argsort(-(x @ routers[layer].T), axis=1)[:, :TOPK]
            for i in range(t):
                same += len(set(own[i].tolist()) & set(ids[i * TOPK:(i + 1) * TOPK]))
                total_routes += TOPK
            cnt: dict[int, int] = defaultdict(int)
            for e in ids:
                if e >= 0 and where[layer][e] == CPU:
                    cnt[e] += 1
            actual = set(cnt)
            cur_lists.append(sorted(cnt, key=lambda e: -cnt[e]))
            if layer == 0 or prev is None:  # layer 0 follows the previous step's last layer; step 1 has no previous
                continue
            cpu_total += len(actual)
            pre = ranked_cpu_experts(routers[layer], step[layer - 1][2], where[layer])
            old = prev[layer]
            for b in args.budget:
                both = pre[:b - b // 2] + [e for e in old if e not in pre[:b - b // 2]][:b // 2]
                hits[("previous step", b)] += len(actual & set(old[:b]))
                hits[("pre-gating", b)] += len(actual & set(pre[:b]))
                hits[("both", b)] += len(actual & set(both))
        prev = cur_lists

    pairs = (len(steps) - 1) * (LAYERS - 1)
    print(f"each layer's own router reproduces {same / total_routes:.1%} of its routes")
    print(f"CPU experts per layer (layers 1-47): {cpu_total / pairs:.1f}")
    for (m, b), h in hits.items():
        print(f"  {m:13s} budget {b:2d}: {h / pairs:.2f} of the layer's CPU experts prefetched per layer "
              f"({h / max(1, cpu_total):.1%} of them, {h / pairs / b:.0%} of the prefetched ones used)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
