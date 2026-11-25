#!/usr/bin/env python3
"""Packs the experts of the Radeon VII expert worker (rvii_worker.hip) from the checkpoint into a cache directory.

The worker keeps the experts of ranks [HOT, HOT + N) of each layer (rankings.json: the runtime's static expert
ranking, most used first; the RTX 3090 keeps the first HOT). Output, in OUT/hot{HOT}-n{N}-{the first 8 hex digits of
the SHA-256 of the rankings file}/ (a changed ranking gets a new cache):
  experts.bin  the experts in the layout of rvii_worker.hip (EXPERT_BYTES each, in slot order)
  slots.bin    int32 (layer, expert) per slot
  meta.json    written last: the cache is complete when it exists
Only numpy and the standard library are needed: the safetensors shards are read through mmap (the page cache).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

N_LAYERS: int = 48
HIDDEN: int = 2560
INTER: int = 640
# The sections of an expert (see rvii_worker.hip): gate/up weights and scales, down weights and scales
GU_W: int = 10 * 20 * 8 * 64 * 16
GU_S: int = 10 * 20 * 64 * 4
DN_W: int = 40 * 5 * 4 * 64 * 16
DN_S: int = 40 * 5 * 64 * 2
EXPERT_BYTES: int = GU_W + GU_S + DN_W + DN_S
PREFIX: str = "model.language_model.layers.{}.mlp.experts.{}.{}_proj.{}"
# AutoRound/GPTQ (v1 format) stores the zero point 8 as 7; 8 itself would also mean 8.
ZEROS_OK: tuple[int, int] = (0x77777777, 0x88888888 - (1 << 32))
DTYPES: dict[str, type] = {"I32": np.int32, "F16": np.float16}

_shards: dict[str, tuple[mmap.mmap, dict, int]] = {}


def tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    """A tensor of the checkpoint as a numpy view of the shard's mmap."""
    fn = weight_map[name]
    if fn not in _shards:
        with (model_dir / fn).open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(mm[:8], "little")
        _shards[fn] = (mm, json.loads(mm[8 : 8 + n]), 8 + n)
    mm, header, base = _shards[fn]
    t = header[name]
    b, e = t["data_offsets"]
    dtype = np.dtype(DTYPES[t["dtype"]])
    return np.frombuffer(mm, dtype, (e - b) // dtype.itemsize, base + b).reshape(t["shape"])


def pack_expert(model_dir: Path, weight_map: dict[str, str], layer: int, e: int) -> bytes:
    """One expert in the layout of rvii_worker.hip (see its header comment)."""
    t = {
        (m, s): tensor(model_dir, weight_map, PREFIX.format(layer, e, m, s))
        for m in ("gate", "up", "down")
        for s in ("qweight", "scales", "qzeros")
    }
    for m in ("gate", "up", "down"):
        z = t[m, "qzeros"]
        if not np.all((z == ZEROS_OK[0]) | (z == ZEROS_OK[1])):
            raise ValueError(f"layer {layer} expert {e} {m}_proj: qzeros are not the zero point 8: {z.ravel()[:4]}")
    if t["gate", "qweight"].shape != (HIDDEN // 8, INTER) or t["down", "qweight"].shape != (INTER // 8, HIDDEN):
        raise ValueError(f"layer {layer} expert {e}: unexpected shapes")
    # gate/up: [r = 16 g + 2 s + h][c = 64 cb + lane][m] -> [cb][g][s][lane][h][m]
    q = np.stack([t["gate", "qweight"], t["up", "qweight"]], -1).reshape(20, 8, 2, 10, 64, 2)
    gu_w = q.transpose(3, 0, 1, 4, 2, 5)
    gu_s = np.stack([t["gate", "scales"], t["up", "scales"]], -1).reshape(20, 10, 64, 2).transpose(1, 0, 2, 3)
    # down: [r = 16 g + 4 s + h][c = 64 cb + lane] -> [cb][g][s][lane][h]
    dn_w = t["down", "qweight"].reshape(5, 4, 4, 40, 64).transpose(3, 0, 1, 4, 2)
    dn_s = t["down", "scales"].reshape(5, 40, 64).transpose(1, 0, 2)
    out = b"".join(np.ascontiguousarray(a).tobytes() for a in (gu_w, gu_s, dn_w, dn_s))
    if len(out) != EXPERT_BYTES:
        raise ValueError(f"layer {layer} expert {e}: {len(out)} bytes")
    return out


def pack_layer(args: tuple[Path, dict[str, str], Path, int, list[int], int]) -> int:
    """Packs a layer's experts into their slots of experts.bin."""
    model_dir, weight_map, path, layer, experts, slot0 = args
    fd = os.open(path, os.O_WRONLY)
    try:
        for i, e in enumerate(experts):
            os.pwrite(fd, pack_expert(model_dir, weight_map, layer, e), (slot0 + i) * EXPERT_BYTES)
    finally:
        os.close(fd)
    return layer


def find_model_dir() -> Path:
    """The checkpoint's snapshot in the HF cache."""
    roots = [os.environ.get("HF_HUB_CACHE", ""), "/root/.cache/huggingface/hub", "~/.cache/huggingface/hub"]
    for r in roots:
        hits = sorted(Path(r).expanduser().glob("models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/*/"))
        if r and hits:
            return hits[-1]
    sys.exit("prepare: the checkpoint albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE is not in the HF cache")


def main() -> None:
    """Packs the experts (if the cache does not exist yet)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=os.environ.get("RVII_MODEL_DIR") or None)
    rankings = os.environ.get("RVII_RANKINGS", Path(__file__).parent / "rankings.json")
    p.add_argument("--rankings", type=Path, default=rankings)
    p.add_argument("--hot", type=int, default=int(os.environ.get("RVII_HOT", "24")))
    p.add_argument("--per-layer", type=int, required=True, help="N: experts per layer")
    p.add_argument("--out", type=Path, default=os.environ.get("RVII_CACHE", "/root/.cache/vllm-models/flash-next-rvii"))
    p.add_argument("--dir", type=Path, help="write the cache into this directory (replacing it) instead of under --out")
    p.add_argument("--jobs", type=int, default=8, help="processes")
    a = p.parse_args()
    model_dir: Path = a.model_dir or find_model_dir()
    if a.dir:
        out = a.dir
        if out.exists():
            shutil.rmtree(out)
    else:
        digest = hashlib.sha256(a.rankings.read_bytes()).hexdigest()[:8]
        out = a.out / f"hot{a.hot}-n{a.per_layer}-{digest}"
        if (out / "meta.json").exists():
            print(f"prepare: {out} exists")
            return
    with a.rankings.open() as f:
        ranks = {int(k): v for k, v in json.load(f).items()}
    with (model_dir / "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]
    sel = {layer: ranks[layer][a.hot : a.hot + a.per_layer] for layer in range(N_LAYERS)}
    slots = [(layer, e) for layer in range(N_LAYERS) for e in sel[layer]]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experts.bin"
    with path.open("wb") as f:
        f.truncate(len(slots) * EXPERT_BYTES)
    t0 = time.time()
    tasks, slot0 = [], 0
    for layer in range(N_LAYERS):
        tasks.append((model_dir, weight_map, path, layer, sel[layer], slot0))
        slot0 += len(sel[layer])
    with ProcessPoolExecutor(a.jobs) as ex:
        for n, layer in enumerate(ex.map(pack_layer, tasks), 1):
            print(f"prepare: layer {layer} ({n}/{N_LAYERS}), {time.time() - t0:.0f} s", flush=True)
    np.array(slots, np.int32).tofile(out / "slots.bin")
    meta = {
        "hot": a.hot,
        "per_layer": a.per_layer,
        "n_experts": len(slots),
        "expert_bytes": EXPERT_BYTES,
        "model_dir": str(model_dir),
        "layout": 1,
    }
    (out / "meta.json.tmp").write_text(json.dumps(meta))
    (out / "meta.json.tmp").rename(out / "meta.json")
    gib = len(slots) * EXPERT_BYTES / 2**30
    print(f"prepare: {len(slots)} experts ({gib:.2f} GiB) in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
