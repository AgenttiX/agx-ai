#!/usr/bin/env python3
"""Test and benchmark client of the Radeon VII worker's prefill path: plays the RTX 3090's side of rvii_protocol.h.

Sends prefill chunks with routing drawn from measured expert usage (``usage/general.bin``: 10 distinct experts per
token, sampled per layer in proportion to their decode counts) and random BF16 x, checks y of some tokens against a
NumPy (FP64) reference from the checkpoint's tensors, and measures the time from writing the request's seq to seeing
the response's seq, for the given numbers of the worker's experts per layer in the mask (the least used of its
experts first, as rvii_prefill.py chooses them).
Run it where the worker's /dev/shm is visible (e.g. a container with --ipc host) with numpy and the checkpoint.
"""

import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rvii
import prepare

# rvii_protocol.h
PF_SHM: str = "/dev/shm/qwen38-rvii-pf"
PF_MAGIC: int = 0x4650494956523351
MAXT: int = 8192
TOPK: int = 10
HIDDEN: int = 2560
N_LAYERS: int = 48
N_EXPERTS: int = 512
REQ: int = 64
RESP: int = 128
MASK: int = 4096
IDS: int = 32768
W: int = IDS + 4 * MAXT * TOPK
X: int = W + 4 * MAXT * TOPK
Y: int = X + 2 * MAXT * HIDDEN


class PfClient:
    """The RTX 3090's side of the prefill protocol."""

    def __init__(self) -> None:
        self.main: bench_rvii.Client = bench_rvii.Client()  # waits until the worker is ready; has the map
        fd = os.open(PF_SHM, os.O_RDWR)
        self.mm: mmap.mmap = mmap.mmap(fd, 0)
        os.close(fd)
        if struct.unpack_from("<Q", self.mm, 0)[0] != PF_MAGIC:
            sys.exit("bench: the prefill segment has no magic")
        self.mask: np.ndarray = np.frombuffer(self.mm, np.uint8, N_LAYERS * N_EXPERTS, MASK).reshape(N_LAYERS, -1)
        self.ids: np.ndarray = np.frombuffer(self.mm, np.int32, MAXT * TOPK, IDS)
        self.w: np.ndarray = np.frombuffer(self.mm, np.float32, MAXT * TOPK, W)
        self.x: np.ndarray = np.frombuffer(self.mm, np.uint16, MAXT * HIDDEN, X)
        self.y: np.ndarray = np.frombuffer(self.mm, np.uint16, MAXT * HIDDEN, Y)
        self.seq: int = max(struct.unpack_from("<q", self.mm, REQ)[0], struct.unpack_from("<q", self.mm, RESP)[0])

    def set_mask(self, per_layer: int) -> None:
        """The worker's per_layer least used experts of each layer (its highest slots: they are in rank order)."""
        m = np.zeros((N_LAYERS, N_EXPERTS), np.uint8)
        for layer in range(N_LAYERS):
            mine = np.nonzero(self.main.map[layer] >= 0)[0]
            mine = mine[np.argsort(self.main.map[layer][mine])]
            if per_layer > 0:
                m[layer, mine[-per_layer:]] = 1
        self.mask[:] = m

    def request(self, layer: int, ids: np.ndarray, w: np.ndarray, x_bf16: np.ndarray) -> tuple[np.ndarray, float]:
        """Send a chunk: ids int32 [T, 10], w float32 [T, 10], x uint16 (BF16) [T, H]. Returns (y [T, H], seconds)."""
        n_tok = ids.shape[0]
        self.ids[: n_tok * TOPK] = ids.reshape(-1)
        self.w[: n_tok * TOPK] = w.reshape(-1)
        self.x[: n_tok * HIDDEN] = x_bf16.reshape(-1)
        struct.pack_into("<ii", self.mm, REQ + 8, n_tok, layer)
        self.seq += 1
        seq = self.seq
        t0 = time.perf_counter()
        struct.pack_into("<q", self.mm, REQ, seq)
        while struct.unpack_from("<q", self.mm, RESP)[0] != seq:
            time.sleep(0.0001)
        dt = time.perf_counter() - t0
        return from_bf16(self.y[: n_tok * HIDDEN]).reshape(n_tok, HIDDEN), dt


def routing(rng: np.random.Generator, prob: np.ndarray, n_tok: int) -> tuple[np.ndarray, np.ndarray]:
    """10 distinct experts per token drawn in proportion to prob (Gumbel top-k), with softmax-like weights."""
    keys = np.log(prob + 1e-12)[None, :] + rng.gumbel(size=(n_tok, N_EXPERTS))
    ids = np.argsort(-keys, axis=1)[:, :TOPK].astype(np.int32)
    w = rng.dirichlet(np.ones(TOPK), n_tok).astype(np.float32)
    return ids, w


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Float32 to BF16 bits (round to nearest even)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(b: np.ndarray) -> np.ndarray:
    """BF16 bits to float32."""
    return (b.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    """Checks and benchmarks the prefill path."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--usage", type=Path, default=Path(__file__).resolve().parent / "usage" / "general.bin")
    p.add_argument("--check", type=int, default=3, help="chunks checked against the reference (8 tokens each)")
    p.add_argument("--tokens", type=int, default=8192, help="tokens per chunk")
    p.add_argument("--per-layer", default="135,96,64,32", help="masked experts per layer to measure")
    p.add_argument("--n", type=int, default=12, help="chunks (layers) per measurement")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    c = PfClient()
    counts = np.fromfile(a.usage, np.uint32).reshape(N_LAYERS, N_EXPERTS).astype(np.float64)
    prob = counts / counts.sum(axis=1, keepdims=True)
    n_tok = a.tokens
    x_bf16 = to_bf16(rng.standard_normal((n_tok, HIDDEN)).astype(np.float32))
    x = from_bf16(x_bf16).astype(np.float64)

    if a.check:
        model_dir = a.model_dir or prepare.find_model_dir()
        with (model_dir / "model.safetensors.index.json").open() as f:
            wm = json.load(f)["weight_map"]
        c.set_mask(int(a.per_layer.split(",")[0]))
        worst = 0.0
        for _ in range(a.check):
            layer = int(rng.integers(N_LAYERS))
            ids, w = routing(rng, prob[layer], n_tok)
            y, _ = c.request(layer, ids, w, x_bf16)
            for t in rng.choice(n_tok, 8, replace=False):
                ref = np.zeros(HIDDEN)
                for e, wt in zip(ids[t], w[t], strict=True):
                    if not c.mask[layer, e]:
                        continue
                    wg, wu, wd = (bench_rvii.dequant(model_dir, wm, layer, int(e), m) for m in ("gate", "up", "down"))
                    g, u = x[t] @ wg, x[t] @ wu
                    ref += np.float64(wt) * ((g / (1 + np.exp(-g)) * u) @ wd)
                err = np.abs(y[t] - ref).max() / max(np.abs(ref).max(), 1e-30)
                worst = max(worst, err)
        print(f"max relative error (max |y - ref| / max |ref|) over {8 * a.check} tokens: {worst:.2e}")
        if worst > 1e-2:
            sys.exit(f"bench: the error {worst:.2e} is too large")

    print(f"\n| experts per layer | pairs per chunk | ms per chunk of {n_tok} tokens (p50) | max |\n|---|---|---|---|")
    for per_layer in map(int, a.per_layer.split(",")):
        c.set_mask(per_layer)
        times, pairs = [], []
        for i in range(a.n + 2):
            layer = int(rng.integers(N_LAYERS))
            ids, w = routing(rng, prob[layer], n_tok)
            _, dt = c.request(layer, ids, w, x_bf16)
            if i >= 2:
                times.append(dt)
                pairs.append(int(c.mask[layer][ids].sum()))
        print(f"| {per_layer} | {np.mean(pairs):.0f} | {np.median(times) * 1e3:.1f} | {max(times) * 1e3:.1f} |",
              flush=True)


if __name__ == "__main__":
    main()
