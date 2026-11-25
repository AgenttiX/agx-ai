#!/usr/bin/env python3
"""Automatic re-ranking of the Radeon VII worker's experts from their measured use (RVII_AUTO_RERANK=1).

Runs the worker (rvii_worker) as a child process on a cache of its own (``{RVII_CACHE}/hot{HOT}-n{N}-auto``, packed
from RVII_RANKINGS at the first start), and:

- accumulates the runtime's decode usage counts (``QWEN38_EXPERT_USAGE``: uint32 [48][512] in /dev/shm, cumulative
  since the runtime's start) with an exponential decay (RVII_RERANK_HALF_LIFE_H, default 72 hours), kept in the cache
  directory across restarts;
- every RVII_RERANK_INTERVAL_S (default 600 s), and at once when the RTX 3090's hot experts change (the runtime
  writes them to /dev/shm/qwen38-hot.bin at its start), ranks the experts of each layer by their recent use mixed
  with the general collection (usage/general.bin, weight RVII_RERANK_PRIOR, default 0.2), and if the N best experts
  that the RTX 3090 does not keep would serve at least RVII_RERANK_MIN_GAIN (default 0.02, i.e. 2 % of all decode
  requests) more than the worker's current ones, packs them into a new cache (~15-30 s, in the background) and, once
  the runtime has been idle for RVII_RERANK_IDLE_S (default 120 s), restarts the worker on it (~10-20 s, during which
  the CPU computes the worker's experts);
- writes the ranking (most used first) to ``rankings.json`` in the cache directory, which the runtime uses at its next
  start for its own hot experts if QWEN38_RANKINGS_AUTO points to it.

At least RVII_RERANK_MIN_REQUESTS (default 2e7, ~10000 decode steps) counted requests are needed before a re-ranking.
Only numpy and the standard library are used. See README.md (Automatic re-ranking).
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
L: int = 48
E: int = 512
MAIN_SHM: str = "/dev/shm/qwen38-rvii"
PF_SHM: str = "/dev/shm/qwen38-rvii-pf"
MAIN_REQ: int = 65536  # the request seqs (rvii_protocol.h)
PF_REQ: int = 64


def env_float(name: str, default: float) -> float:
    """A float from the environment."""
    v = os.environ.get(name)
    return float(v) if v else default


def log(msg: str) -> None:
    """A line in the container's log."""
    print(f"rerank: {msg}", file=sys.stderr, flush=True)


def load_ranking(path: Path) -> list[list[int]]:
    """A ranking file as [layer][rank] of all 512 experts (unlisted ones follow by index)."""
    data = json.loads(path.read_text())
    out = []
    for layer in range(L):
        seen: set[int] = set()
        ids = []
        for v in data.get(str(layer), []):
            e = int(v)
            if 0 <= e < E and e not in seen:
                ids.append(e)
                seen.add(e)
        out.append(ids + [e for e in range(E) if e not in seen])
    return out


def save_ranking(path: Path, ranking: list[list[int]]) -> None:
    """A ranking file in the runtime's format (written atomically)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("{\n" + ",\n".join(f'"{layer}": {json.dumps(r)}' for layer, r in enumerate(ranking)) + "\n}\n")
    tmp.replace(path)


def read_seq(path: str, offset: int) -> int | None:
    """A request seq of the worker's shared memory (None if there is none)."""
    try:
        with Path(path).open("rb") as f:
            f.seek(offset)
            return struct.unpack("<q", f.read(8))[0]
    except (OSError, struct.error):
        return None


class Usage:
    """The decayed decode usage counts, from the cumulative counts of the runtime's usage file."""

    def __init__(self, state: Path, half_life_s: float):
        self.state: Path = state
        self.half_life_s: float = half_life_s
        self.acc: np.ndarray = np.zeros((L, E))
        self.prev: np.ndarray | None = None
        self.t: float = time.time()
        if state.exists():
            with np.load(state) as z:
                self.acc = z["acc"].astype(np.float64)
                self.prev = z["prev"].astype(np.uint32) if z["prev"].size else None
                self.t = float(z["t"])

    def update(self, path: Path) -> None:
        """Decay the counts to now and add the runtime's new requests."""
        now = time.time()
        self.acc *= 0.5 ** (max(0.0, now - self.t) / self.half_life_s)
        self.t = now
        try:
            cur = np.fromfile(path, np.uint32)
        except OSError:
            return
        if cur.size != L * E:
            return
        cur = cur.reshape(L, E)
        # A runtime restart starts the counts from zero again.
        delta = cur if self.prev is None or (cur < self.prev).any() else cur - self.prev
        self.acc += delta
        self.prev = cur

    def save(self) -> None:
        """Keep the state across restarts."""
        tmp = self.state.with_name(self.state.name + ".tmp.npz")
        np.savez(tmp, acc=self.acc, prev=self.prev if self.prev is not None else np.zeros(0, np.uint32), t=self.t)
        tmp.replace(self.state)


class Manager:
    """Runs the worker and re-ranks its experts."""

    def __init__(self, a: argparse.Namespace):
        self.a: argparse.Namespace = a
        self.auto: Path = a.cache / f"hot{a.hot}-n{a.per_layer}-auto"
        self.new: Path = self.auto.with_name(self.auto.name + ".new")
        self.base: list[list[int]] = load_ranking(a.rankings)
        prior = np.fromfile(HERE / "usage" / "general.bin", np.uint32).reshape(L, E).astype(np.float64)
        self.prior: np.ndarray = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1)
        self.usage_file: Path = Path(os.environ.get("RVII_RERANK_USAGE", "/dev/shm/qwen38-usage.bin"))
        self.hot_file: Path = Path(os.environ.get("RVII_RERANK_HOT", "/dev/shm/qwen38-hot.bin"))
        self.interval: float = env_float("RVII_RERANK_INTERVAL_S", 600)
        self.min_gain: float = env_float("RVII_RERANK_MIN_GAIN", 0.02)
        self.idle_s: float = env_float("RVII_RERANK_IDLE_S", 120)
        self.min_req: float = env_float("RVII_RERANK_MIN_REQUESTS", 2e7)
        self.prior_w: float = env_float("RVII_RERANK_PRIOR", 0.2)
        self.worker: subprocess.Popen | None = None
        self.stop: bool = False
        self.seqs: tuple[int | None, int | None] = (None, None)
        self.last_active: float = time.time()
        self.pending: dict | None = None  # a packed new cache waiting for an idle runtime

    # ---- the worker
    def start_worker(self) -> None:
        """Start the worker on the current layout."""
        self.worker = subprocess.Popen(["rvii_worker", str(self.auto)])

    def stop_worker(self) -> None:
        """Stop the worker (it finishes the requests in progress)."""
        if self.worker and self.worker.poll() is None:
            self.worker.send_signal(signal.SIGTERM)
            try:
                self.worker.wait(30)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()

    # ---- the layout
    def pack(self, layout: list[list[int]], out: Path) -> bool:
        """Pack ranks [HOT, HOT + N) of the layout ranking into out (prepare.py), at low priority."""
        ranking_file = out.with_name(out.name + "-layout.json")
        save_ranking(ranking_file, layout)
        cmd = ["nice", "-n", "19", sys.executable, str(HERE / "prepare.py"), "--rankings", str(ranking_file),
               "--hot", str(self.a.hot), "--per-layer", str(self.a.per_layer), "--dir", str(out), "--jobs", "4"]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, check=False)
        if r.returncode != 0 or not (out / "meta.json").exists():
            log(f"packing failed ({r.returncode})")
            return False
        ranking_file.replace(out / "layout.json")
        return True

    def current_sets(self) -> list[set[int]]:
        """The worker's experts per layer."""
        slots = np.fromfile(self.auto / "slots.bin", np.int32).reshape(-1, 2)
        sets: list[set[int]] = [set() for _ in range(L)]
        for layer, e in slots:
            sets[int(layer)].add(int(e))
        return sets

    def hot_sets(self) -> list[set[int]]:
        """The RTX 3090's hot experts: as it published them, else the first HOT of the base ranking."""
        try:
            m = np.fromfile(self.hot_file, np.uint8)
            if m.size == L * E:
                m = m.reshape(L, E)
                return [set(np.nonzero(m[layer])[0].tolist()) for layer in range(L)]
        except OSError:
            pass
        return [set(self.base[layer][: self.a.hot]) for layer in range(L)]

    def evaluate(self, usage: Usage, hot: list[set[int]]) -> dict | None:
        """The new layout if it gains enough, else None."""
        acc = usage.acc
        total = acc.sum()
        if total < self.min_req:
            return None
        frac = acc / np.maximum(acc.sum(axis=1, keepdims=True), 1e-9)
        score = self.prior_w * self.prior + (1 - self.prior_w) * frac
        cur = self.current_sets()
        ranking, layout, new_share, cur_share, changed = [], [], 0.0, 0.0, 0
        for layer in range(L):
            order = {e: i for i, e in enumerate(self.base[layer])}
            ranked = sorted(range(E), key=lambda e: (-score[layer, e], order[e]))
            ranking.append(ranked)
            hot_l = sorted(hot[layer], key=lambda e: (-score[layer, e], order[e]))
            rest = [e for e in ranked if e not in hot[layer]]
            new_set = rest[: self.a.per_layer]
            layout.append(hot_l + new_set + rest[self.a.per_layer :])
            new_share += acc[layer, new_set].sum()
            cur_share += sum(acc[layer, e] for e in cur[layer] if e not in hot[layer])
            changed += len(set(new_set) - cur[layer])
        gain = (new_share - cur_share) / total
        save_ranking(self.auto / "rankings.json", ranking)
        log(f"{total:.3g} recent requests: the worker serves {cur_share / total:.1%}, a new ranking would serve "
            f"{new_share / total:.1%} ({changed} experts changed)")
        if gain < self.min_gain:
            return None
        return {"layout": layout, "gain": gain, "changed": changed, "hot": hot}

    # ---- the loop
    def activity(self) -> None:
        """Note when the runtime last sent a request."""
        seqs = (read_seq(MAIN_SHM, MAIN_REQ), read_seq(PF_SHM, PF_REQ))
        if seqs != self.seqs:
            self.seqs = seqs
            self.last_active = time.time()

    def run(self) -> int:
        """The main loop; returns the exit code."""
        a = self.a
        if not (self.auto / "meta.json").exists():
            log(f"packing the first layout from {a.rankings} into {self.auto}")
            if not self.pack(self.base, self.auto):
                return 1
        for d in (self.new,):
            if d.exists():
                shutil.rmtree(d)
        usage = Usage(self.auto / "usage-state.npz", env_float("RVII_RERANK_HALF_LIFE_H", 72) * 3600)
        layout_hot = self.hot_sets()
        self.start_worker()
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, lambda *_: setattr(self, "stop", True))
        t_eval = time.time()
        t_save = time.time()
        while not self.stop:
            time.sleep(5)
            if self.worker and self.worker.poll() is not None:
                log(f"the worker exited ({self.worker.returncode})")
                usage.save()
                return self.worker.returncode or 1
            self.activity()
            now = time.time()
            if now - t_save > 600:
                usage.update(self.usage_file)
                usage.save()
                t_save = now
            hot = self.hot_sets()
            if self.pending is None and (now - t_eval > self.interval or hot != layout_hot):
                t_eval = now
                usage.update(self.usage_file)
                plan = self.evaluate(usage, hot)
                if plan is None and hot != layout_hot:
                    layout_hot = hot  # the worker's experts are fine for the new hot set
                if plan is not None:
                    log(f"packing a new layout (+{plan['gain']:.1%} of the requests, {plan['changed']} experts)")
                    if self.pack(plan["layout"], self.new):
                        self.pending = plan
            if self.pending is not None and now - self.last_active > self.idle_s:
                log("idle: restarting the worker with the new layout")
                self.stop_worker()
                old = self.auto.with_name(self.auto.name + ".old")
                for f in ("usage-state.npz", "rankings.json"):
                    if (self.auto / f).exists():
                        shutil.copy2(self.auto / f, self.new / f)
                self.auto.rename(old)
                self.new.rename(self.auto)
                shutil.rmtree(old)
                layout_hot = self.pending["hot"]
                self.pending = None
                self.start_worker()
        usage.update(self.usage_file)
        usage.save()
        self.stop_worker()
        with contextlib.suppress(OSError):
            if self.new.exists():
                shutil.rmtree(self.new)
        return 0


def main() -> int:
    """Command line interface."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hot", type=int, required=True, help="experts per layer on the RTX 3090")
    p.add_argument("--per-layer", type=int, required=True, help="experts per layer on the Radeon VII")
    p.add_argument("--cache", type=Path, required=True, help="RVII_CACHE")
    p.add_argument("--rankings", type=Path, required=True, help="the base ranking (the first layout and ties)")
    return Manager(p.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
