#!/usr/bin/env python3
r"""Prefill microbenchmark of vLLM's Triton attention on gfx906, for tuning its launch parameters (see README.md).

``patch`` adds launch-parameter overrides to the installed vLLM, read from the environment and only used for prefill
(more than one query token per sequence): AGX_PF_BM (query rows per block), AGX_PF_TILE (KV tile), AGX_PF_W (warps),
AGX_PF_S (stages) and AGX_PF_WPE (waves per EU), for both the int8 (``triton_unified_attention.py``) and the int4
(``int4_per_token_head.py``) kernel. ``run`` times one 512-token prefill chunk at the end of a context of each given
length, with random caches, for the attention layer types of Gemma 4 26B-A4B or Qwen3.8-27B, and prints the time and
a checksum of the output (the same checksum = the same result). Run each configuration in a new process, in a
throwaway container of the image (``patch`` modifies it):

    docker run --rm --device /dev/dri --device /dev/kfd --group-add video -v "$PWD":/b:ro --entrypoint sh \
        ghcr.io/agenttix/agx-ai/vllm-gfx906:latest -c 'python3 /b/triton_attention_bench.py patch &&
        python3 /b/triton_attention_bench.py run "{\"model\": \"gemma\", \"kv\": \"int8\", \"ctx\": [32768]}" &&
        AGX_PF_BM=32 AGX_PF_TILE=32 python3 /b/triton_attention_bench.py run \
        "{\"model\": \"gemma\", \"kv\": \"int8\", \"ctx\": [32768], \"layers\": [\"global\"]}"'

The configuration (JSON): model (gemma or qwen), kv (int8 or int4), ctx (context lengths), and optionally layers
(names from LAYERS) and block (the KV cache block size: 16 for Gemma, 1552 for Qwen in vLLM; default 16).
"""

import json
from pathlib import Path
import sys
import time
from typing import Any

OPS: Path = Path("/opt/vllm-venv/lib/python3.12/site-packages/vllm/v1/attention/ops")
CHUNK = 512

# Inserted into both launchers: the overrides, only for prefill.
BLOCK_M_OVERRIDE = """    import os as _agx_os
    if max_seqlen_q > 1 and _agx_os.environ.get("AGX_PF_BM"):
        BLOCK_M = int(_agx_os.environ["AGX_PF_BM"])
        BLOCK_Q = BLOCK_M // num_queries_per_kv
"""
TILE_OVERRIDE = """    if max_seqlen_q > 1 and _agx_os.environ.get("AGX_PF_TILE"):
        TILE_SIZE_PREFILL = int(_agx_os.environ["AGX_PF_TILE"])
"""
LAUNCH_OVERRIDES = """{k: int(_agx_os.environ[e]) for k, e in (("num_warps", "AGX_PF_W"), ("num_stages", "AGX_PF_S"),
        ("waves_per_eu", "AGX_PF_WPE")) if max_seqlen_q > 1 and _agx_os.environ.get(e)}"""

# Attention layer types: name, query heads, KV heads, head size, sliding window (0 = none), number of layers
LAYERS: dict[str, list[tuple[str, int, int, int, int, int]]] = {
    "gemma": [("sliding", 16, 8, 256, 1024, 25), ("global", 16, 2, 512, 0, 5)],
    "qwen": [("full", 24, 4, 256, 0, 16)],
}


def insert(text: str, anchor: str, new: str, *, after: bool = False) -> str:
    """Insert new before (or after) the only occurrence of anchor."""
    if text.count(anchor) != 1:
        raise ValueError(f"expected one occurrence of {anchor!r}")
    return text.replace(anchor, anchor + new if after else new + anchor)


def patch() -> None:
    """Add the environment overrides to the installed vLLM."""
    path = OPS / "triton_unified_attention.py"
    s = path.read_text()
    s = insert(s, "    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs\n\n    sliding_window_val",
               BLOCK_M_OVERRIDE)
    s = insert(s, "    if tuned_large_head:\n        TILE_SIZE_PREFILL = 128\n", TILE_OVERRIDE, after=True)
    s = insert(s, "        launch_kwargs.update(num_warps=4, num_stages=1, waves_per_eu=1)\n",
               f"    launch_kwargs.update({LAUNCH_OVERRIDES})\n", after=True)
    path.write_text(s)
    path = OPS / "int4_per_token_head.py"
    s = path.read_text()
    s = insert(s, "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n", BLOCK_M_OVERRIDE, after=True)
    s = insert(s, "        head_size, sliding_window_val, q.element_size(), is_prefill=False\n    )\n", TILE_OVERRIDE,
               after=True)
    s = insert(s, "    _attn_packed[grid](\n", f"    _agx_kw = {LAUNCH_OVERRIDES}\n")
    s = insert(s, "    _attn_packed[grid](\n", "        **_agx_kw,\n", after=True)
    path.write_text(s)


def time_layer(cfg: dict[str, Any], layer: tuple[str, int, int, int, int, int], ctx: int) -> dict[str, float]:
    """Time one prefill chunk of a layer type at a context length."""
    # The script runs in the vLLM image, which has these packages; the development environment does not have them.
    import torch  # noqa: PLC0415  # pyrefly: ignore[missing-import]
    from vllm.v1.attention.ops.int4_per_token_head import (  # noqa: PLC0415  # pyrefly: ignore[missing-import]
        unified_attention_int4,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: PLC0415  # pyrefly: ignore[missing-import]
        unified_attention,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: PLC0415  # pyrefly: ignore[missing-import]

    _, q_heads, kv_heads, head_size, window, n_layers = layer
    block = cfg.get("block", 16)
    n_blocks = (ctx + block - 1) // block + 1
    q = torch.randn(CHUNK, q_heads, head_size, device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    if cfg["kv"] == "int8":
        k_cache = torch.randint(-127, 127, (n_blocks, block, kv_heads, head_size), device="cuda", dtype=torch.int8)
        v_cache = torch.randint(-127, 127, (n_blocks, block, kv_heads, head_size), device="cuda", dtype=torch.int8)
    else:  # two int4 values per byte
        k_cache = torch.randint(0, 255, (n_blocks, block, kv_heads, head_size // 2), device="cuda", dtype=torch.uint8)
        v_cache = torch.randint(0, 255, (n_blocks, block, kv_heads, head_size // 2), device="cuda", dtype=torch.uint8)
    args = {
        "cu_seqlens_q": torch.tensor([0, CHUNK], device="cuda", dtype=torch.int32),
        "max_seqlen_q": CHUNK,
        "seqused_k": torch.tensor([ctx], device="cuda", dtype=torch.int32),
        "max_seqlen_k": ctx,
        "softmax_scale": head_size**-0.5,
        "window_size": (window - 1, 0) if window else (-1, -1),
        "block_table": torch.arange(n_blocks, device="cuda", dtype=torch.int32)[None, :],
        "softcap": 0.0,
        "k_scale_cache": torch.rand(n_blocks, block, kv_heads, device="cuda", dtype=torch.float32) * 0.02,
        "v_scale_cache": torch.rand(n_blocks, block, kv_heads, device="cuda", dtype=torch.float32) * 0.02,
    }

    def call() -> None:
        if cfg["kv"] == "int8":
            unified_attention(q, k_cache, v_cache, out, causal=True, q_descale=None, k_descale=None, v_descale=None,
                              kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, **args)
        else:
            unified_attention_int4(q, k_cache, v_cache, out, sinks=None, alibi_slopes=None, use_alibi_sqrt=False,
                                   qq_bias=None, output_scale=None, mm_prefix_range=None, **args)

    call()  # compile
    torch.cuda.synchronize()
    repeat = 3 if ctx > 40000 else 6
    start = time.perf_counter()
    for _ in range(repeat):
        call()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / repeat * 1000
    return {"ms": round(ms, 2), "ms_all_layers": round(ms * n_layers, 1), "checksum": float(out.float().abs().mean())}


def run(cfg: dict[str, Any]) -> None:
    """Time all selected layer types and context lengths and print the results as JSON."""
    import torch  # noqa: PLC0415  # pyrefly: ignore[missing-import]

    torch.manual_seed(0)
    results = {}
    for layer in LAYERS[cfg["model"]]:
        if cfg.get("layers") and layer[0] not in cfg["layers"]:
            continue
        for ctx in cfg["ctx"]:
            results[f"{layer[0]}@{ctx}"] = time_layer(cfg, layer, ctx)
    print("RESULT " + json.dumps(results))


if __name__ == "__main__":
    if sys.argv[1] == "patch":
        patch()
    else:
        run(json.loads(sys.argv[2]))
