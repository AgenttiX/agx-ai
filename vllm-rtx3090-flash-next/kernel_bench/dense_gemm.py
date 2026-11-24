r"""Compare the GEMMs of Qwen3.8-Flash-Next's dense projections on the RTX 3090, in CUDA graphs.

For the int8 (W8A16, group 128) projections of the Gated DeltaNet and attention layers: vLLM's Marlin kernel against
the runtime's int8 skinny Triton kernel (``triton_skinny.skinny_mm_i8``, the int8 lm_head's kernel) with several
configurations, at 4 tokens (decode with MTP) and in prefill (``linear_i8``'s dequantizing path against Marlin).
The error column is relative to the exact product with the int8 weights; Marlin's is not meaningful (this script's
packing for Marlin gives the right timing but not the right values). For the BF16 projections of the
hyper-connections: the BF16 skinny kernel with several configurations.

Run it in the image (it needs ~2 GB of free VRAM; the production server may keep running):

    docker run --rm --gpus all -v "$PWD/kernel_bench":/kb --entrypoint python3 \\
        ghcr.io/agenttix/agx-ai/vllm-flash-next:latest /kb/dense_gemm.py
"""

import argparse
from collections.abc import Callable
from typing import Any

import torch  # pyrefly: ignore[missing-import]
import triton  # pyrefly: ignore[missing-import]
from vllm.model_executor.layers.quantization.utils import marlin_utils  # pyrefly: ignore[missing-import]
from vllm.models.qwen3_8_flash_next.nvidia import triton_skinny as ts  # pyrefly: ignore[missing-import]
from vllm.scalar_type import scalar_types  # pyrefly: ignore[missing-import]

from vllm import _custom_ops as ops  # pyrefly: ignore[missing-module-attribute]

# (N, K, calls per decode step): the GDN in_proj_qkvz and out_proj (36 layers), the attention qkv(+gate) and o_proj
# (12 layers).
INT8_SHAPES: list[tuple[int, int, int]] = [(16384, 2560, 36), (2560, 6144, 36), (13312, 2560, 12), (2560, 6144, 12)]
# The hyper-connections' BF16 projections (2 of each per layer and one more of each at the end).
BF16_SHAPES: list[tuple[int, int, int]] = [(336, 10240, 97), (10240, 320, 97)]
I8_CONFIGS: list[tuple[int, int, int]] = [(32, 1, 3), (32, 2, 3), (64, 1, 3), (64, 1, 4), (16, 2, 4), (16, 4, 4),
                                          (32, 4, 3), (64, 2, 3)]  # BLOCK_N, SPLIT_K, STAGES
BF16_CONFIGS: list[tuple[int, int, int, int]] = [(16, 128, 8, 4), (16, 128, 16, 4), (16, 64, 8, 4), (32, 128, 4, 4),
                                                 (32, 64, 2, 3), (64, 64, 1, 3), (16, 64, 1, 4), (16, 128, 2, 4),
                                                 (32, 128, 1, 4)]  # BLOCK_N, BLOCK_K, SPLIT_K, STAGES


def set_config(n: int, k: int, cfg: tuple[int, ...]) -> None:
    """Make the skinny kernels use cfg for the shape (n, k)."""
    ts._configs[(n, k)] = cfg  # noqa: SLF001 (the runtime caches its choice there)


def graph_time(fn: Callable[[], Any], reps: int = 20, iters: int = 10) -> float:
    """Microseconds per call of fn, replayed in a CUDA graph of reps calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (reps * iters)


def gptq8(w: Any) -> tuple[Any, Any]:
    """Symmetric int8, group 128, as quantize_flash_next_dense.py: (q [K, N] in 0..255, FP16 scales [K/128, N])."""
    n, k = w.shape
    g = w.float().t().reshape(k // 128, 128, n)
    scales = (g.abs().amax(dim=1) * 2 / 255).clamp(min=1e-8)
    q = (torch.round(g / scales[:, None, :]) + 128).clamp(0, 255).to(torch.int32)  # [groups, 128, N]
    return q.reshape(k, n), scales.half()


def marlin_layer(q_kn: Any, scales: Any, n: int, k: int) -> Callable[[Any], Any]:
    """Marlin's GEMM with the uint8b128 GPTQ weight q_kn [K, N] (values 0..255) repacked for it."""
    packed = torch.zeros(k // 4, n, dtype=torch.int32, device=q_kn.device)
    for i in range(4):
        packed |= q_kn[i::4] << (8 * i)
    perm = torch.empty(0, dtype=torch.int, device=q_kn.device)
    mq = ops.gptq_marlin_repack(packed, perm, k, n, 8)
    ms = marlin_utils.marlin_permute_scales(scales, k, n, 128)
    ws = marlin_utils.marlin_make_workspace_new(q_kn.device)
    empty = marlin_utils.marlin_make_empty_g_idx(q_kn.device)

    def apply(x: Any) -> Any:
        return marlin_utils.apply_gptq_marlin_linear(x, mq, ms, None, empty, empty, ws, scalar_types.uint8b128, n, k,
                                                     True)
    return apply


def bench_int8(n: int, k: int, calls: int, prefill: int) -> float:
    """Print Marlin's and the int8 skinny kernel's times for one shape; return Marlin's time per decode step (us)."""
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q_kn, scales = gptq8(w)
    marlin = marlin_layer(q_kn, scales, n, k)
    wi8 = (q_kn - 128).to(torch.int8).t().contiguous()  # [N, K]
    si8 = scales.float().t().contiguous()  # [N, K/128]
    x = torch.randn(4, k, device="cuda").to(torch.bfloat16)
    ref = x.float() @ (wi8.float().view(n, k // 128, 128) * si8[:, :, None]).view(n, k).t()
    mb = n * k / 1e6
    res = {"marlin": (graph_time(lambda: marlin(x)), marlin(x))}
    default = ts.config_for(n, k)
    for bn, split, stages in [(default[0], default[2], default[3]), *I8_CONFIGS]:
        set_config(n, k, (bn, 128, split, stages))
        res[f"i8 {(bn, split, stages)}"] = (graph_time(lambda: ts.skinny_mm_i8(x, wi8, si8)),
                                            ts.skinny_mm_i8(x, wi8, si8))
    set_config(n, k, default)
    print(f"{n} x {k} int8 ({mb:.1f} MB), {calls} per step (BLOCK_N, SPLIT_K, STAGES; the first is the default):")
    for name, (us, out) in res.items():
        err = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"  {name:24s} {us:6.1f} us  {mb / us * 1e3:5.0f} GB/s  err {err:.1e}")
    if prefill:
        xp = torch.randn(prefill, k, device="cuda").to(torch.bfloat16)
        t_m = graph_time(lambda: marlin(xp), reps=3, iters=5)
        t_d = graph_time(lambda: ts.linear_i8(xp, wi8, si8), reps=3, iters=5)
        t_b = graph_time(lambda: torch.nn.functional.linear(xp, w), reps=3, iters=5)
        print(f"  prefill {prefill} tokens: Marlin {t_m:.0f} us, dequantize + BF16 GEMM {t_d:.0f} us, "
              f"BF16 GEMM alone {t_b:.0f} us")
    return res["marlin"][0] * calls


def bench_bf16(n: int, k: int, calls: int) -> None:
    """Print the BF16 skinny kernel's times for one shape with several configurations."""
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(4, k, device="cuda").to(torch.bfloat16)
    mb = n * k * 2 / 1e6
    default = ts.config_for(n, k)
    print(f"{n} x {k} BF16 ({mb:.1f} MB), {calls} per step (BLOCK_N, BLOCK_K, SPLIT_K, STAGES; the first is the "
          "default):")
    for cfg in [default, *BF16_CONFIGS]:
        if cfg[2] > 1 and k < 2 * cfg[1] * cfg[2]:
            continue
        set_config(n, k, cfg)
        us = graph_time(lambda: ts.skinny_mm(x, w))
        print(f"  {cfg!s:24s} {us:6.1f} us  {mb / us * 1e3:5.0f} GB/s  grid {triton.cdiv(n, cfg[0])}x{cfg[2]}")
    set_config(n, k, default)


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefill", type=int, default=2048, help="tokens of the prefill comparison (0: none)")
    args = parser.parse_args()
    torch.manual_seed(0)
    marlin_step = sum(bench_int8(n, k, calls, args.prefill) for n, k, calls in INT8_SHAPES)
    print(f"Marlin per decode step: {marlin_step / 1e3:.2f} ms")
    for n, k, calls in BF16_SHAPES:
        bench_bf16(n, k, calls)


if __name__ == "__main__":
    main()
