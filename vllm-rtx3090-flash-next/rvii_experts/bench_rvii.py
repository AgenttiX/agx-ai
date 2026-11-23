#!/usr/bin/env python3
"""Test and benchmark client of the Radeon VII expert worker: plays the CPU side of rvii_protocol.h.

Attaches to the worker's shared memory, sends requests with real expert ids from its map and random x, checks y
against a NumPy (FP64) reference from the checkpoint's tensors, and measures the latency from writing the request's
seq to seeing the response's seq, for T in {1, 4} and J in {1, 4, 8, 12, 16}.
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
import prepare  # the checkpoint reader

# rvii_protocol.h
SHM: str = "/dev/shm/qwen38-rvii"
MAGIC: int = 0x4949565238335651
N_LAYERS: int = 48
N_EXPERTS: int = 512
HIDDEN: int = 2560
MAXT: int = 4
MAP_OFF: int = 4096
REQ: int = 65536
RESP: int = 131072
REQ_JOBS: int = 64
REQ_X: int = 2048
RESP_Y: int = 64

Jobs = list[tuple[int, list[int], list[float]]]


class Client:
    """The CPU side of the protocol."""

    def __init__(self) -> None:
        fd = os.open(SHM, os.O_RDWR)
        self.mm: mmap.mmap = mmap.mmap(fd, 0)
        os.close(fd)
        t0 = time.time()
        while True:
            magic, _ver, ready = struct.unpack_from("<QII", self.mm, 0)
            if magic == MAGIC and ready == 1:
                break
            if time.time() - t0 > 600:
                sys.exit("bench: the worker is not ready")
            time.sleep(0.5)
        self.n_experts: int = struct.unpack_from("<I", self.mm, 48)[0]
        shape = (N_LAYERS, N_EXPERTS)
        self.map: np.ndarray = np.frombuffer(self.mm, np.int16, N_LAYERS * N_EXPERTS, MAP_OFF).reshape(shape).copy()
        self.y: np.ndarray = np.frombuffer(self.mm, np.float32, MAXT * HIDDEN, RESP + RESP_Y)
        self.seq: int = max(struct.unpack_from("<q", self.mm, REQ)[0], struct.unpack_from("<q", self.mm, RESP)[0])

    def request(self, layer: int, jobs: Jobs, x: np.ndarray) -> tuple[np.ndarray, float]:
        """jobs: [(e, [tok...], [wt...])]; x: float32 [T, H]. Returns (y [T, H], latency in s)."""
        n_tok = x.shape[0]
        for j, (e, tok, wt) in enumerate(jobs):
            tok4 = list(tok) + [0] * (MAXT - len(tok))
            wt4 = list(wt) + [0.0] * (MAXT - len(wt))
            struct.pack_into("<ii4i4f", self.mm, REQ + REQ_JOBS + 40 * j, e, len(tok), *tok4, *wt4)
        self.mm[REQ + REQ_X : REQ + REQ_X + x.nbytes] = x.tobytes()
        struct.pack_into("<iii", self.mm, REQ + 8, n_tok, layer, len(jobs))
        self.seq += 1
        seq = self.seq
        t0 = time.perf_counter()
        struct.pack_into("<q", self.mm, REQ, seq)
        while struct.unpack_from("<q", self.mm, RESP)[0] != seq:
            pass
        dt = time.perf_counter() - t0
        return self.y[: n_tok * HIDDEN].reshape(n_tok, HIDDEN).copy(), dt


def dequant(model_dir: Path, wm: dict[str, str], layer: int, e: int, m: str) -> np.ndarray:
    """The FP64 weights [K, N] of a projection."""
    q = prepare.tensor(model_dir, wm, prepare.PREFIX.format(layer, e, m, "qweight"))
    s = prepare.tensor(model_dir, wm, prepare.PREFIX.format(layer, e, m, "scales")).astype(np.float64)
    nib = (q[:, None, :] >> (4 * np.arange(8)[None, :, None])) & 0xF  # [K/8, 8, N]
    w = nib.reshape(-1, q.shape[1]).astype(np.float64) - 8.0
    return w * np.repeat(s, 128, axis=0)


def random_jobs(rng: np.random.Generator, experts: np.ndarray, n_tok: int, n_jobs: int) -> Jobs:
    """n_jobs distinct experts, each with 1..n_tok distinct tokens."""
    jobs = []
    for e in rng.choice(experts, n_jobs, replace=False):
        n = int(rng.integers(1, n_tok + 1))
        tok = sorted(rng.choice(n_tok, n, replace=False).tolist())
        jobs.append((int(e), tok, rng.uniform(0.02, 0.3, n).astype(np.float32).tolist()))
    return jobs


def check(c: Client, rng: np.random.Generator, model_dir: Path, per_layer: dict[int, np.ndarray], n: int) -> None:
    """Compares y with the FP64 reference."""
    with (model_dir / "model.safetensors.index.json").open() as f:
        wm = json.load(f)["weight_map"]
    cache: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
    worst = 0.0
    for n_tok in (1, 4):
        errs = []
        for _ in range(n):
            layer = int(rng.integers(N_LAYERS))
            jobs = random_jobs(rng, per_layer[layer][:12], n_tok, int(rng.integers(1, 9)))
            x = rng.standard_normal((n_tok, HIDDEN)).astype(np.float32)
            y, _ = c.request(layer, jobs, x)
            ref = np.zeros((n_tok, HIDDEN))
            for e, tok, wt in jobs:
                if (layer, e) not in cache:
                    cache[layer, e] = tuple(dequant(model_dir, wm, layer, e, m) for m in ("gate", "up", "down"))
                wg, wu, wd = cache[layer, e]
                for t, w in zip(tok, wt, strict=True):
                    xv = x[t].astype(np.float64)
                    g, u = xv @ wg, xv @ wu
                    ref[t] += np.float64(np.float32(w)) * ((g / (1 + np.exp(-g)) * u) @ wd)
            covered = {t for _, tok, _ in jobs for t in tok}
            if not np.all(y[[t for t in range(n_tok) if t not in covered]] == 0):
                sys.exit("bench: a token row without jobs is not zero")
            errs.append(np.abs(y - ref).max() / max(np.abs(ref).max(), 1e-30))
        worst = max(worst, *errs)
        print(f"T={n_tok}: max relative error (max |y - ref| / max |ref|) over {n} requests: {max(errs):.2e}")
    if worst > 1e-4:
        sys.exit(f"bench: the error {worst:.2e} is too large")


def main() -> None:
    """Checks and benchmarks the worker."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--check", type=int, default=40, help="requests checked against the reference per T")
    p.add_argument("--n", type=int, default=1000, help="requests per latency measurement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", default="1,4", help="token counts of the latency measurement")
    p.add_argument("--J", default="1,4,8,12,16", help="job counts of the latency measurement")
    p.add_argument("--gap-us", type=float, default=0, help="pause between requests (as in decoding: ~1000 us)")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    c = Client()
    print(f"worker: {c.n_experts} experts")
    per_layer = {layer: np.nonzero(c.map[layer] >= 0)[0] for layer in range(N_LAYERS)}
    if a.check:
        check(c, rng, a.model_dir or prepare.find_model_dir(), per_layer, a.check)

    print("\n| T | J | p50 µs | p90 µs | p99 µs | weights GB/s (p50) |\n|---|---|---|---|---|---|")
    for n_tok in map(int, a.T.split(",")):
        for n_jobs in map(int, a.J.split(",")):
            reqs = []
            for _ in range(64):
                layer = int(rng.integers(N_LAYERS))
                x = rng.standard_normal((n_tok, HIDDEN)).astype(np.float32)
                reqs.append((layer, random_jobs(rng, per_layer[layer], n_tok, n_jobs), x))
            lat = []
            for i in range(a.n + 20):
                if a.gap_us:
                    t_end = time.perf_counter() + a.gap_us * 1e-6
                    while time.perf_counter() < t_end:
                        pass
                _, dt = c.request(*reqs[i % len(reqs)])
                if i >= 20:
                    lat.append(dt)
            p50, p90, p99 = (np.percentile(lat, q) * 1e6 for q in (50, 90, 99))
            gbs = n_jobs * prepare.EXPERT_BYTES / (p50 * 1e-6) / 1e9
            print(f"| {n_tok} | {n_jobs} | {p50:.0f} | {p90:.0f} | {p99:.0f} | {gbs:.0f} |", flush=True)


if __name__ == "__main__":
    main()
