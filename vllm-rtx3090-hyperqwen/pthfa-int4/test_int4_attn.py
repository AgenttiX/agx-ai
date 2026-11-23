"""pth-flash-attn on vLLM's int4_per_token_head KV cache: correctness and kernel timings.

Correctness against an FP64 reference and vLLM's Triton int4 kernel; timings against the int8 pth path and the
Triton int4 path.

The cache is written by vLLM's own int4 writer (reshape_and_cache_int4: RHT, nibble packing, scale with the zero
point in its low mantissa bits) into the layout vLLM allocates (inline [K | ks | V | vs] cells), and read through
PthFlashAttnImpl (the CUSTOM backend: q rotated with the RHT, the kernel, the output rotated back), so the test
covers the same code as the server. The FP64 reference dequantises the cache ((u - zp) * scale), computes the
attention in the RHT basis and rotates back, so it is the exact attention over what the cache holds.

Run in the HyperQwen image with the patched pthfa on PYTHONPATH (see README.md), under the GPU lock:
  python test_int4_attn.py [--quick] [--no-timing] [--no-correctness] [--no-triton]
"""

# The test drives PthFlashAttnImpl and vLLM's int4 helpers through their private members on purpose.
# ruff: noqa: SLF001

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import functools
import math
import statistics
import sys
from typing import Any

from pth_flash_attn.planner import plan_launch, workspace_token_splits  # pyrefly: ignore[missing-import]
import pth_flash_attn.vllm_backend as vb  # pyrefly: ignore[missing-import]
from pth_flash_attn.vllm_backend import PthFlashAttnImpl, PthLaunchSlice  # pyrefly: ignore[missing-import]
import torch  # pyrefly: ignore[missing-import]
from vllm.v1.attention.ops.int4_per_token_head import (  # pyrefly: ignore[missing-import]
    _get_hadamard_matrix,
    _get_rht_signs,
    reshape_and_cache_int4,
    unified_attention_int4,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # pyrefly: ignore[missing-import]
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # pyrefly: ignore[missing-import]

Tensor = Any  # torch.Tensor; the container-only import gives the type checker nothing to check against
DEV = "cuda"
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
RESULTS: list[bool] = []


@dataclass
class Setup:
    """A layer's Impl, its cache and the inputs of one launch."""

    impl: Any
    kv: Tensor  # the packed cache, logical [NB, Hkv, BS, C]
    caches: tuple[Tensor, Tensor, Tensor, Tensor]  # (k, v, ks, vs) views the kernel reads
    full: tuple[Tensor, Tensor]  # padded K/V views the writer gets
    bt: Tensor
    sl: Tensor
    qsl: Tensor
    q: Tensor


def report(ok: bool, what: str) -> None:
    """Print one result line and record it."""
    print(f"  {'OK  ' if ok else 'FAIL'}  {what}", flush=True)
    RESULTS.append(ok)


def impl_for(hq: int, hkv: int, d: int, dtype: str, window: int = 0) -> Any:
    """The CUSTOM backend's Impl of one layer."""
    return PthFlashAttnImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv, alibi_slopes=None,
                            sliding_window=window or None, kv_cache_dtype=dtype)


def make_cache(impl: Any, nb: int, bs: int, hkv: int, d: int, layout: str,
               int4: bool = True) -> tuple[Tensor, Any, tuple[Tensor, Tensor]]:
    """Allocate a zeroed cache as vLLM does and return it, the kernel's views and the writer's views."""
    row = d // 2 if int4 else d
    tdtype = torch.uint8 if int4 else torch.int8
    shape = (nb, bs, hkv, 2 * (row + 4)) if layout == "NHD" else (nb, hkv, bs, 2 * (row + 4))
    kv = torch.zeros(shape, dtype=tdtype, device=DEV)
    if layout == "NHD":
        kv = kv.transpose(1, 2)  # logical [NB, Hkv, BS, C], as vLLM hands it to the backend
    impl._k_scale_cache = impl._v_scale_cache = None
    impl._cache_views = None
    caches = impl._caches(kv)  # (k, v, ks, vs) views the kernel reads
    full_k, full_v = impl._pth_key_value_caches(kv)  # [NB, BS, Hkv, row + 4]: what the writer gets
    return kv, caches, (full_k, full_v)


def write(impl: Any, full: tuple[Tensor, Tensor], k: Tensor, v: Tensor, slots: Tensor, int4: bool = True,
          chunk: int = 32768) -> None:
    """Write K/V into the cache with vLLM's own writer, in chunks."""
    for i in range(0, k.shape[0], chunk):
        if int4:
            reshape_and_cache_int4(k[i:i + chunk], v[i:i + chunk], full[0], full[1], slots[i:i + chunk],
                                   k_scale_cache=impl._k_scale_cache, v_scale_cache=impl._v_scale_cache)
        else:
            triton_reshape_and_cache_flash_per_token_head_quant(
                k[i:i + chunk], v[i:i + chunk], full[0], full[1], impl._k_scale_cache, impl._v_scale_cache,
                slots[i:i + chunk], kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)


def dequant_rot(packed: Tensor, scales: Tensor) -> Tensor:
    """[N, D/2] uint8 nibbles + [N] raw scales -> [N, D] float64 in the RHT basis."""
    lo = (packed & 0xF).double()
    hi = (packed >> 4).double()
    u = torch.stack((lo, hi), dim=-1).flatten(-2)
    bits = scales.view(torch.int32)
    zp = (bits & 0xF).double()
    sc = (bits & ~0xF).view(torch.float32).double()
    return (u - zp[:, None]) * sc[:, None]


def rht_mats(d: int) -> tuple[Tensor, Tensor]:
    """FP64 (M_q, M_o): q' = q @ M_q into the RHT basis, o = o' @ M_o back."""
    h = _get_hadamard_matrix(d, torch.float64, DEV).double()
    s = _get_rht_signs(d, 0, DEV).double()
    return (s[:, None] * h) / d, (h * s[None, :]) / d


def reference(q: Tensor, caches: Any, bt: Tensor, seq_lens: list[int], q_lens: list[int], scale: float,
              window: int = 0, causal: bool = True) -> Tensor:
    """FP64 attention over the dequantised int4 cache (RHT basis), rotated back."""
    k, v, ks, vs = caches
    bs, hkv, d = k.shape[1], k.shape[2], q.shape[2]
    g = q.shape[1] // hkv
    m_q, m_o = rht_mats(d)
    out = torch.empty(q.shape, dtype=torch.float64, device=DEV)
    start = 0
    for r, (seq_len, ql) in enumerate(zip(seq_lens, q_lens, strict=True)):
        pos = seq_len - ql + torch.arange(ql, device=DEV)
        lo = (pos - window + 1).clamp(min=0) if window else torch.zeros_like(pos)
        if causal:
            hi = pos
        else:
            hi = (pos + window - 1).clamp(max=seq_len - 1) if window else torch.full_like(pos, seq_len - 1)
        slots = torch.arange(int(lo.min()), seq_len, device=DEV)
        blocks = bt[r, slots // bs].long()
        offs = slots % bs
        vis = (slots[None, :] >= lo[:, None]) & (slots[None, :] <= hi[:, None])
        for h in range(hkv):
            kd = dequant_rot(k[blocks, offs, h], ks[blocks, offs, h])
            vd = dequant_rot(v[blocks, offs, h], vs[blocks, offs, h])
            qh = q[start:start + ql, h * g:(h + 1) * g].double() @ m_q
            p = (torch.einsum("qgd,kd->gqk", qh, kd) * scale).masked_fill(~vis, -math.inf).softmax(-1)
            out[start:start + ql, h * g:(h + 1) * g] = torch.einsum("gqk,kd->qgd", p, vd) @ m_o
            del kd, vd, p
        start += ql
    return out


class Meta:
    """The slice of PthAttnMetadata that PthFlashAttnImpl.forward reads."""

    num_actual_tokens: int
    decode: Any
    prefill: Any
    causal: bool


class _Workspace:
    """Stands in for vLLM's workspace manager: hands out the partials of one launch."""

    def __init__(self, ws: tuple[Tensor, Tensor] | None) -> None:
        self.ws = ws

    def get_simultaneous(self, *_: object) -> tuple[Tensor, Tensor] | None:
        """Return the partials (ws_o, ws_lse)."""
        return self.ws


def pth_forward(su: Setup, q_lens: list[int], ws_ts: int, window: int = 0, causal: bool = True,
                out: Tensor | None = None) -> tuple[Tensor, Any]:
    """One PthFlashAttnImpl.forward over the whole batch as one decode-like slice; returns (out, plan)."""
    impl, q = su.impl, su.q
    hkv, d = impl.num_kv_heads, impl.head_size
    g = q.shape[1] // hkv
    qmax = max(q_lens)
    plan = plan_launch(num_tokens=q.shape[0], num_reqs=len(q_lens), max_query_len=qmax, G=g, Hkv=hkv, D=d,
                       num_sms=NUM_SMS, is_prefill=qmax > 64 // g, ws_token_splits=ws_ts, window=window)
    part = PthLaunchSlice(plan=plan, token_start=0, token_end=q.shape[0], max_query_len=qmax,
                          query_start_loc=su.qsl, seq_lens=su.sl, block_table=su.bt)
    md = Meta()
    md.num_actual_tokens = q.shape[0]
    md.decode, md.prefill, md.causal = part, None, causal
    ws = None
    if plan.num_splits > 1:
        ws = (torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV),
              torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV))
    impl._no_workspace = torch.empty(0, dtype=torch.float32, device=DEV)
    # forward asks the workspace manager for the partials; give it ours.
    manager = _Workspace(ws)
    vb.current_workspace_manager = lambda: manager
    out = torch.full_like(q, float("nan")) if out is None else out
    impl.forward(None, q, None, None, su.kv, md, out)
    return out, plan


def triton_forward(su: Setup, q_lens: list[int], max_len: int, segm: tuple[Any, ...] | None = None,
                   out: Tensor | None = None) -> Tensor:
    """VLLM's Triton int4 attention on the same cache (it rotates q and the output itself)."""
    impl, q = su.impl, su.q
    out = torch.empty_like(q) if out is None else out
    kw = {}
    if segm is not None:
        kw = {"seq_threshold_3D": segm[0], "num_par_softmax_segments": segm[1], "softmax_segm_output": segm[2],
              "softmax_segm_max": segm[3], "softmax_segm_expsum": segm[4]}
    unified_attention_int4(q, su.full[0], su.full[1], out, cu_seqlens_q=su.qsl, max_seqlen_q=max(q_lens),
                           seqused_k=su.sl, max_seqlen_k=max_len, softmax_scale=impl.scale, window_size=(-1, -1),
                           block_table=su.bt, softcap=0.0, sinks=None, alibi_slopes=None, use_alibi_sqrt=False,
                           qq_bias=None, output_scale=None, mm_prefix_range=None,
                           k_scale_cache=impl._k_scale_cache, v_scale_cache=impl._v_scale_cache, **kw)
    return out


def setup(hq: int, hkv: int, d: int, seq_lens: list[int], q_lens: list[int], layout: str, bs: int,
          window: int = 0, int4: bool = True, qscale: float = 2.0) -> Setup:
    """A cache filled with random K/V of shuffled blocks, and random q."""
    impl = impl_for(hq, hkv, d, "int4_per_token_head" if int4 else "int8_per_token_head", window)
    nbr = [math.ceil(seq_len / bs) for seq_len in seq_lens]
    nb = sum(nbr) + 1
    kv, caches, full = make_cache(impl, nb, bs, hkv, d, layout, int4)
    perm = torch.randperm(nb, device=DEV, dtype=torch.int32)
    bt = torch.zeros(len(seq_lens), max(nbr), dtype=torch.int32, device=DEV)
    used = 0
    for r, (n, seq_len) in enumerate(zip(nbr, seq_lens, strict=True)):
        bt[r, :n] = perm[used:used + n]
        used += n
        pos = torch.arange(seq_len, device=DEV)
        slots = (bt[r, pos // bs].long() * bs + pos % bs)
        # K with a per-channel offset and a few outlier channels (as real keys), V plain normal.
        kk = torch.randn(seq_len, hkv, d, device=DEV) + torch.linspace(-1, 1, d, device=DEV)
        kk[..., :4] *= 6
        vv = torch.randn(seq_len, hkv, d, device=DEV)
        write(impl, full, kk.bfloat16(), vv.bfloat16(), slots, int4)
        del kk, vv
    q = (torch.randn(sum(q_lens), hq, d, device=DEV) * qscale).bfloat16()
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    return Setup(impl, kv, caches, full, bt, sl, qsl, q)


def err_stats(out: Tensor, ref: Tensor) -> tuple[float, float, float]:
    """(max |error|, relative RMS error, max |reference|)."""
    e = (out.double() - ref)
    rel = (e.norm() / ref.norm()).item()
    return e.abs().max().item(), rel, ref.abs().max().item()


def case(hq: int, hkv: int, d: int, seq_lens: list[int], q_lens: list[int], layout: str = "NHD", bs: int = 16,
         ws_ts: int = 1 << 20, window: int = 0, causal: bool = True, triton: bool = True) -> None:
    """One correctness case: pth int4 (and Triton int4) against the FP64 reference."""
    su = setup(hq, hkv, d, seq_lens, q_lens, layout, bs, window)
    out, plan = pth_forward(su, q_lens, ws_ts, window, causal)
    torch.cuda.synchronize()
    ref = reference(su.q, su.caches, su.bt, seq_lens, q_lens, su.impl.scale, window, causal)
    mx, rel, rmax = err_stats(out, ref)
    msg = f"pth max|err| {mx:.2e} rel-RMS {rel:.2e} (|ref|max {rmax:.2f})"
    ok = bool(torch.isfinite(out).all()) and rel < 1e-2 and mx < 0.02 * max(rmax, 1.0)
    if triton and window == 0 and causal:
        tout = triton_forward(su, q_lens, max(seq_lens))
        torch.cuda.synchronize()
        tmx, trel, _ = err_stats(tout, ref)
        dmx = (out.float() - tout.float()).abs().max().item()
        msg += f"; Triton int4 max|err| {tmx:.2e} rel-RMS {trel:.2e}; max|pth - Triton| {dmx:.2e}"
        ok = ok and rel <= 1.5 * trel + 1e-3
    mask = f" W={window} {'causal' if causal else 'non-causal'}" if window or not causal else ""
    report(ok, f"D={d} {hq}/{hkv} q={q_lens} L={seq_lens} {layout} bs={bs}{mask} splits={plan.num_splits}: {msg}")
    del su
    torch.cuda.empty_cache()


def cuda_time(fn: Callable[[], object], reps: int = 20, inner: int = 10) -> float:
    """Median time of one call in us, replayed from a CUDA graph of `inner` calls (as vLLM replays decode steps)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / inner)
    return statistics.median(ts)


def timing(seq_len: int, ql: int, bs: int, triton: bool = True) -> None:
    """Time one layer call of int8 pth, int4 pth (with and without the RHT GEMMs) and Triton int4."""
    hq, hkv, d = 24, 4, 256
    ws_ts = workspace_token_splits(max_num_seqs=4, q_dec_max=8, Hkv=hkv, D=d, num_sms=NUM_SMS)
    res = {}
    for int4 in (False, True):
        su = setup(hq, hkv, d, [seq_len], [ql], "NHD", bs, int4=int4)
        out = torch.empty_like(su.q)
        run_pth = functools.partial(pth_forward, su, [ql], ws_ts, out=out)
        res["int8 pth" if not int4 else "int4 pth"] = cuda_time(run_pth)
        if int4:
            # The kernel alone (q already rotated, output not rotated back).
            rht = su.impl._rht
            su.impl._rht = None
            res["int4 kernel only"] = cuda_time(run_pth)
            su.impl._rht = rht
            if triton:
                nseg = 16
                segm = (128, nseg, torch.empty(ql, hq, nseg, d, dtype=torch.float32, device=DEV),
                        torch.empty(ql, hq, nseg, dtype=torch.float32, device=DEV),
                        torch.empty(ql, hq, nseg, dtype=torch.float32, device=DEV))
                res["int4 Triton"] = cuda_time(functools.partial(triton_forward, su, [ql], seq_len, segm, out))
        del su, out
        torch.cuda.empty_cache()
    print(f"  L={seq_len:>6} q={ql}: " + ", ".join(f"{k} {v:7.1f} us" for k, v in res.items())
          + f"; int4 pth / int8 pth = {res['int4 pth'] / res['int8 pth']:.2f}"
          + (f", Triton / int4 pth = {res['int4 Triton'] / res['int4 pth']:.2f}" if "int4 Triton" in res else ""),
          flush=True)


def graph_case() -> None:
    """A captured decode/verify launch replayed after seq_lens grew on the device equals the eager launch."""
    hq, hkv, d, bs = 24, 4, 256, 816
    ws_ts = workspace_token_splits(max_num_seqs=4, q_dec_max=8, Hkv=hkv, D=d, num_sms=NUM_SMS)
    su = setup(hq, hkv, d, [5000, 9000], [8, 8], "NHD", bs)
    su.sl.copy_(torch.tensor([100, 200], dtype=torch.int32))
    out = torch.empty_like(su.q)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pth_forward(su, [8, 8], ws_ts, out=out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pth_forward(su, [8, 8], ws_ts, out=out)
    su.sl.copy_(torch.tensor([5000, 9000], dtype=torch.int32))
    g.replay()
    torch.cuda.synchronize()
    eager, _ = pth_forward(su, [8, 8], ws_ts)
    ref = reference(su.q, su.caches, su.bt, [5000, 9000], [8, 8], su.impl.scale)
    same = torch.equal(out, eager)
    _, rel, _ = err_stats(out, ref)
    report(same and rel < 1e-2, f"CUDA graph replay after seq_lens grew: bitwise equal to eager {same}, "
                                f"rel-RMS vs reference {rel:.2e}")


def main() -> None:
    """Run the correctness cases and the timings."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-correctness", action="store_true")
    ap.add_argument("--no-triton", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    tri = not a.no_triton
    print(f"== pthfa int4, {torch.cuda.get_device_name(0)} ({NUM_SMS} SMs)", flush=True)
    if not a.no_correctness:
        print("== correctness: target layers (D=256, 24/4 heads)", flush=True)
        for seq_len in ((1024, 32768) if a.quick else (1024, 32768, 131072, 245760)):
            for ql in (1, 8):
                case(24, 4, 256, [seq_len], [ql], bs=816 if seq_len > 2000 else 16, triton=tri)
        case(24, 4, 256, [1, 17, 64, 1000, 4096], [1, 1, 1, 1, 1], triton=tri)
        case(24, 4, 256, [4, 333, 4096], [4, 4, 4], layout="HND", triton=tri)
        case(24, 4, 256, [16, 700, 5000], [16, 16, 16], bs=816, triton=tri)
        case(24, 4, 256, [512, 1500, 37], [512, 300, 37], triton=tri)
        case(24, 4, 256, [6000], [2048], bs=816, triton=tri)
        for ws in (0, 1 << 20):
            case(24, 4, 256, [4, 333, 4096], [4, 4, 4], ws_ts=ws, triton=tri)
        graph_case()
        print("== correctness: DFlash2 drafter layers (D=128, 32/8 heads, window 2048, non-causal)", flush=True)
        for seq_len in (900, 2100, 3000, 10000):
            case(32, 8, 128, [seq_len], [8], bs=864, window=2048, causal=False, triton=False)
        case(32, 8, 128, [3000, 5000], [8, 8], layout="HND", bs=864, window=2048, causal=False, triton=False)
        case(32, 8, 128, [3000], [8], bs=864, window=2048, causal=True, triton=False)
        case(32, 8, 128, [3000], [8], bs=864, window=0, causal=False, triton=False)
        case(32, 8, 128, [3000, 77], [64, 13], bs=16, triton=tri)
    if not a.no_timing:
        print("== time per layer call, CUDA-graph replay (median of 20; int4 pth includes the two RHT matmuls)",
              flush=True)
        for seq_len in ((1024, 32768) if a.quick else (1024, 32768, 102400, 131072, 245760)):
            for ql in (1, 8):
                timing(seq_len, ql, 816, triton=tri)
    print(f"== {sum(RESULTS)}/{len(RESULTS)} OK", flush=True)
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
