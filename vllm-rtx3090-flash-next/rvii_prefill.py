# SPDX-License-Identifier: Apache-2.0
"""agx-ai: the Radeon VII worker computes some of its experts in prefill chunks (QWEN38_RVII_PREFILL).

In a prefill chunk of more than QWEN38_CPU_PREFILL_MAX tokens, stream_v2 streams every cold expert of every layer
from RAM to the GPU, and the PCIe copy is the bottleneck. With QWEN38_RVII_PREFILL=N > 0, the Radeon VII expert
worker (rvii_experts/, the prefill part of rvii_protocol.h) computes the N least used of its experts of each layer
instead, and they are not streamed. Per MoE layer, the GPU copies x and the routing into the worker's shared memory
(DMA on a side stream, while it computes its hot experts and streams the others), and before the layer's reduction it
waits for the worker's partial sum (on the GPU) and adds it to its FP32 accumulator.

The worker is used for a chunk if it is ready and alive at the chunk's first layer. If it does not answer within
QWEN38_RVII_PREFILL_TIMEOUT_MS (default 10000), its part of that layer is missing (an error is logged) and it is not
used for prefill again until it restarts.
"""
# ruff: noqa: PLC0415, PLW0603, PTH123  (runs inside the runtime: its relative imports, module state, plain open)
import logging
import mmap
import os
import struct
import time

import numpy as np
import torch  # pyrefly: ignore[missing-import]
from vllm.logger import init_logger  # pyrefly: ignore[missing-import]

logger: logging.Logger = init_logger(__name__)

L: int = 48
E: int = 512
H: int = 2560
TOPK: int = 10
MAXT: int = 8192
MAIN: str = "/dev/shm/qwen38-rvii"
PF: str = "/dev/shm/qwen38-rvii-pf"
MAGIC: int = 0x4949565238335651
PF_MAGIC: int = 0x4650494956523351
MAP: int = 4096
REQ: int = 64
RESP: int = 128
MASK: int = 4096
IDS: int = 32768
W: int = IDS + 4 * MAXT * TOPK
X: int = W + 4 * MAXT * TOPK
Y: int = X + 2 * MAXT * H
SIZE: int = Y + 2 * MAXT * H
N: int = int(os.environ.get("QWEN38_RVII_PREFILL", "0"))
TIMEOUT_NS: int = int(float(os.environ.get("QWEN38_RVII_PREFILL_TIMEOUT_MS", "10000")) * 1e6)


class _Worker:
    """The worker's two segments, mapped while its process (worker_pid) lives."""

    def __init__(self, pid: int):
        from .cpu_experts import extension  # pyrefly: ignore[missing-import]
        self.ext = extension()
        self.pid = pid
        fd = os.open(MAIN, os.O_RDONLY)
        try:
            main = mmap.mmap(fd, MAP + 2 * L * E, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        wmap = np.frombuffer(main, np.int16, L * E, MAP).reshape(L, E).copy()
        main.close()
        fd = os.open(PF, os.O_RDWR)
        try:
            self.mm = mmap.mmap(fd, SIZE)
        finally:
            os.close(fd)
        if struct.unpack_from("<Q", self.mm, 0)[0] != PF_MAGIC:
            self.mm.close()
            raise RuntimeError("the prefill segment has no magic")
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), SIZE, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister of the prefill segment failed ({err})")
        self.addr = self.bytes.data_ptr()
        self.x = self.bytes[X:Y].view(torch.bfloat16).view(MAXT, H)
        self.ids = self.bytes[IDS:W].view(torch.int32).view(MAXT, TOPK)
        self.w = self.bytes[W:X].view(torch.float32).view(MAXT, TOPK)
        self.seq = max(struct.unpack_from("<q", self.mm, REQ)[0], struct.unpack_from("<q", self.mm, RESP)[0])
        self.stream = torch.cuda.Stream()
        self.ok = torch.ones(1, dtype=torch.int32, device="cuda")
        self.timeouts = torch.zeros(1, dtype=torch.int32).pin_memory()
        # Per layer, the N least used of the worker's experts (its slots are in rank order within a layer), except
        # those that this GPU keeps (after an automatic re-ranking, the worker can hold some until it re-ranks again).
        from .expert_store import hot_sets  # pyrefly: ignore[missing-import]
        hot = hot_sets()
        self.sets: list[set[int]] = []
        mask = np.zeros((L, E), np.uint8)
        for layer in range(L):
            mine = np.nonzero(wmap[layer] >= 0)[0]
            mine = mine[np.argsort(wmap[layer][mine])]
            mine = np.array([e for e in mine if int(e) not in set(hot.get(layer, ()))], dtype=np.int64)
            mine = mine[len(mine) - min(N, len(mine)):]
            mask[layer, mine] = 1
            self.sets.append({int(e) for e in mine})
        np.frombuffer(self.mm, np.uint8, L * E, MASK)[:] = mask.reshape(-1)
        logger.info("Radeon VII prefill: the worker computes %d of its experts per layer in prefill chunks",
                    len(self.sets[0]))

    def close(self) -> None:
        """Unmap the segments (the GPU is idle)."""
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaHostUnregister(self.addr)
        del self.x, self.ids, self.w, self.bytes
        self.mm.close()


_worker: _Worker | None = None
_beat: tuple[int, float] = (0, 0.0)  # the last heartbeat seen and when
_active: bool = False                # whether the current chunk uses the worker
_failed_pid: int = 0                 # a worker that did not answer: not used again
_timeouts: int = 0


def _alive() -> int:
    """The worker's pid if it is ready and its heartbeat moves, else 0."""
    global _beat
    try:
        with open(MAIN, "rb") as f:
            head = f.read(56)
    except OSError:
        return 0
    if len(head) < 56:
        return 0
    magic, _ver, ready = struct.unpack_from("<QII", head, 0)
    pid, beat = struct.unpack_from("<qQ", head, 32)
    if magic != MAGIC or ready != 1:
        return 0
    now = time.monotonic()
    if beat != _beat[0]:
        _beat = (beat, now)
    elif now - _beat[1] > 0.5:  # the worker beats every 20 ms
        return 0
    return pid


def plan() -> list[set[int]] | None:
    """The experts that the worker computes in this chunk, per layer, or None if it is not used.

    Called at a prefill chunk's first layer (the previous chunk is done); stream_v2 does not stream these experts.
    """
    global _worker, _active, _failed_pid, _timeouts
    _active = False
    if N <= 0:
        return None
    if _worker is not None and int(_worker.timeouts[0]) != _timeouts:
        _timeouts = int(_worker.timeouts[0])
        _failed_pid = _worker.pid
        logger.error("Radeon VII prefill: the worker did not answer within %.0f ms (its experts of a layer are "
                     "missing from that prefill chunk); not using it until it restarts", TIMEOUT_NS / 1e6)
    pid = _alive()
    if pid in (0, _failed_pid):
        return None
    if _worker is None or _worker.pid != pid:
        if _worker is not None:
            _worker.close()
            _worker = None
        try:
            torch.cuda.synchronize()
            _worker = _Worker(pid)
            _timeouts = 0
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Radeon VII prefill: cannot attach to the worker: %s", e)
            return None
    _active = True
    return _worker.sets


def active() -> bool:
    """Whether the current prefill chunk uses the worker."""
    return _active


def submit(layer: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
    """Send a layer's chunk to the worker (after the current stream's work so far)."""
    w = _worker
    assert w is not None
    T = x.shape[0]
    if T > MAXT:
        raise RuntimeError(f"Radeon VII prefill: {T} tokens exceed {MAXT}")
    w.seq += 1
    ev = torch.cuda.Event()
    ev.record()
    with torch.cuda.stream(w.stream):
        w.stream.wait_event(ev)
        w.x[:T].copy_(x, non_blocking=True)
        w.ids[:T].copy_(topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32), non_blocking=True)
        w.w[:T].copy_(topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float(), non_blocking=True)
        w.ext.rvii_pf_publish(w.addr + REQ, w.seq, T, layer)
    for t in (x, topk_ids, topk_weights):
        t.record_stream(w.stream)


def wait_add(acc: torch.Tensor) -> None:
    """Add the worker's partial sum of the last submitted layer to acc (FP32 [T, H]), on the current stream."""
    w = _worker
    assert w is not None
    w.ext.rvii_pf_wait_add(acc, w.addr + RESP, w.addr + Y, w.seq, TIMEOUT_NS, w.ok, w.timeouts.data_ptr())
