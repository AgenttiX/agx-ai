# pth-flash-attn on vLLM's int4_per_token_head KV cache

An extension of HyperQwen's pth-flash-attn kernel (`pthfa/` of [HyperQwen](https://github.com/syv-ai/HyperQwen) at
commit `3d6c954`, vLLM 0.33.0) that reads vLLM's `int4_per_token_head` KV cache, so that Qwen3.8-27B on the RTX 3090
of agx-z2e holds about twice the context of the int8 cache in the same memory, with the kernel instead of vLLM's slow
Triton int4 attention. Hardware and the rest of the configuration: [`../README.md`](../README.md).

Recommended settings (the image of [`../Dockerfile`](../Dockerfile), see [Files](#files)): `PTH_FA=1`,
`PTH_KV_DTYPE=int4_per_token_head`, `KV_MEM=5950000000` (unchanged), `MAX_LEN=262128`. 262144 is not possible: the
DFlash2 drafter caps the context at 262136 (262144 minus its 8 lookahead positions), and 262128 is the largest
multiple of 16 below that. The KV pool holds 281,001 tokens (int8: 143,870 in the same 5.95 GB), so a full-context
prompt fits with room for vLLM's ~1.05x margin. Measured with these settings plus the 32 GiB CPU KV cache and the
vision tower (on the CPU in the test; `../docker-compose.yml` runs it on the Radeon VII, which takes no VRAM of the
RTX 3090 either): a 259,742-token prompt peaks at 21406 MiB (the server processes), 21416 MiB over the benchmark.

| Model | Context | VRAM (server processes, peak) | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|
| Qwen3.8-27B W4A16, DFlash2 + lookup drafting, **int4** KV cache on pth-flash-attn | **262128** (pool 281,001 tokens in 5.95 GB) | 21406 MiB with a 259,742-token prompt | chat 131 t/s; 146 / 119 / 90 / 71 t/s at 6.4k / 32k / 100k / 200k; edits 280 t/s | 1857 / 1634 / 1233 / 924 t/s at 6.4k / 32k / 100k / 200k; 803 t/s for a 260k prompt | retrieval 121/132 (int8: 123/132) |
| the same with the **int8** KV cache (the configuration of `..`), same image and day | 139264 (pool 143,870 tokens) | 21320 MiB | chat 130 t/s; 122 / 112 / 95 t/s at 6.4k / 32k / 100k; edits 284 t/s | 1858 / 1621 / 1235 t/s | retrieval 123/132 |

Speed is the same within the run-to-run noise of the sampled outputs (the draft acceptance differs between runs by
several points; e.g. 38.6 % vs 28.7 % at 6.4k). The decode attention itself is ~2.2x faster with int4 (see
[Kernel time](#kernel-time)), but DFlash2's 8-token verify, which dominates here, is compute-bound and as fast as
with int8. With the production settings (CPU KV cache 32 GiB, vision tower on the CPU): chat 107 t/s, 121 / 108 /
87 t/s at 6.4k / 32k / 100k, edits 265 t/s, prompt 1691 / 1594 / 1236 t/s.

## Files

- [`pthfa-int4.patch`](pthfa-int4.patch): the patch over HyperQwen's `pthfa/` (int8 path unchanged).
- [`../Dockerfile`](../Dockerfile) applies the patch to the image of this repository, builds and installs it
  (`pthfa/install.sh`, ~2 min, gate test included) and makes the launcher take `PTH_KV_DTYPE` (default
  `int8_per_token_head`, so the image behaves as before without it). [`../docker-compose.int4.yml`](../docker-compose.int4.yml)
  selects the int4 cache and 262128 tokens of context.
- [`serve.sh`](serve.sh): a test server on port 9935 with the settings of `../docker-compose.yml` (`SPEC=dflash2 CTX=long
  PTH_FA=1 KV_MEM=5950000000 EMBED_UVA=1`, 4 sequences), without the CPU KV cache and the vision tower:
  `KV=int4 MAX_LEN=262128 sh serve.sh` (options: `IMAGE`, `KV_OFFLOAD_GB`, `VISION=1`, `MORE_ARGS`). For production:
  `docker compose -f docker-compose.yml -f docker-compose.int4.yml up -d` in `..`.
- [`peak_probe.py`](peak_probe.py): one request with a prompt of a given length (Python source) and the peak VRAM of
  the server processes (the code corpus of `bench_http` ends at ~212k tokens).
- [`test_int4_attn.py`](test_int4_attn.py): correctness against an FP64 reference and vLLM's Triton int4 kernel, and
  kernel timings. In the image (or with the patched package on `PYTHONPATH`):
  `VLLM_INT4_MQ_3D=1 python test_int4_attn.py [--quick] [--no-timing] [--no-correctness]`.

## Design

vLLM's int4 format (`vllm/v1/attention/ops/int4_per_token_head.py`) is kept as it is: its writer, page size
accounting, prefix caching and the inline cell `[K 128 B | k_scale | V 128 B | v_scale]` per (token, head) at head
size 256 (264 B instead of 520 B for int8). The writer applies one randomized Hadamard transform (RHT) to K and V:
`x' = (x * s) @ H`, with the sign vector `s` fixed per head size (seeded, the same for K, V and every layer) and the
unnormalised Hadamard matrix `H`. Each (token, head) is quantised asymmetrically to 4 bits (`(u - zp) * scale`), two
values per byte (even d in the low nibble), with the zero point in the 4 low mantissa bits of the fp32 scale.

- **Python** (`vllm_backend.py`): the CUSTOM backend accepts `int4_per_token_head` at head sizes 128 and 256. Per
  forward it rotates q with one bf16 GEMM (`q @ diag(s) H / D`, so `q' . k' = q . k` and the softmax scale is
  unchanged), runs the kernel in the RHT basis, and rotates the output back (`o' @ H diag(s) / D`), two small GEMMs
  per layer (the matrices' entries are +-2^-k, exact in bf16). The FA2 slab prefill (int8 only) is skipped.
- **Kernel**: the int8 kernels are templates over a stage reader; the int4 kernels are separate instantiations
  (`attn_fwd_i4_kernel`, `attn_fwd_window_i4_kernel`, `attn_dec_i4_kernel`), so the int8 machine code is unchanged.
  The loaders copy the packed cells (rows of D/2 bytes; the warp-specialised register-staged producer copies two
  tokens per warp instruction). The nibbles go straight from shared memory into fp16 MMA fragments with the
  magic-number trick (`0x6400 | u` = 1024 + u, minus 1024 + zp in one `sub.f16x2`), so `u - zp` is exact and the
  zero point costs nothing extra: for K it is subtracted per B column (token), for V per token pair in the
  register. One ldmatrix of K gives 8 nibbles = two k16 steps; q is staged into shared memory with the matching
  d permutation (`pi4`, d and d + 4 adjacent). One LDS.32 of V gives 8 d of a token; the output d mapping changes
  accordingly (64-d groups). Scales are masked (`& ~0xF`) before use.
- **Coverage**: target layers (D = 256, 24/4 heads): the swap-AB decode kernel (up to 32 rows: q <= 5) and the
  16-row kernel (verify of 8 tokens, prefill chunks), whole-cell loader. DFlash2 drafter layers (D = 128, windowed
  non-causal and causal), separate-stream loader. bf16 q only. No register spills (ptxas: 97-221 registers at
  D = 256 decode, 242 at D = 256 WM = 4, 141-217 at D = 128).

## Correctness

`test_int4_attn.py` writes random K/V (with per-channel offsets and outlier channels) through vLLM's own int4 writer
into vLLM's layout and reads it through `PthFlashAttnImpl.forward`. Reference: FP64 attention over the dequantised
cache in the RHT basis, rotated back (the exact attention over what the cache holds). The tolerance is the bf16
output level; the error of vLLM's Triton int4 kernel against the same reference is shown for comparison.

24/24 cases pass (tolerance: relative RMS error < 1 % and at most as large as Triton's error times 1.5). Relative
RMS error of the output, pth int4 / Triton int4, against the reference:

| Case | pth int4 | Triton int4 |
|---|---|---|
| D = 256, q = 1, L = 1k / 32k / 128k / 240k | 4.6e-3 / 6.2e-3 / 5.4e-3 / 6.3e-3 | 6.2e-3 / 1.5e-2 / 1.3e-2 / 1.5e-2 |
| D = 256, q = 8, L = 1k / 32k / 128k / 240k | 3.9e-3 / 3.9e-3 / 4.1e-3 / 5.1e-3 | 6.5e-3 / 1.2e-2 / 1.4e-2 / 1.9e-2 |
| D = 256, batches of decode, 4-/16-token verify, prefill (512, 2048-token chunks), HND and NHD, block 16 / 816, 1 and 21 splits | 2.8e-3 to 4.1e-3 | 4.0e-3 to 7.7e-3 |
| D = 128 drafter layers (32/8 heads, window 2048 non-causal, L 900 to 10k; causal window; non-causal without window; batched prefill) | 3.7e-3 to 4.2e-3 | (Triton: 6.9e-3 for the causal batch) |

Max absolute errors: 4e-3 to 2.2e-2 for outputs up to ~4.9 in magnitude (bf16 output rounding is ~1e-2 there). A
captured CUDA graph replayed after seq_lens grew is bit-identical to the eager launch. The kernel is more accurate
than the Triton int4 path because it subtracts the zero point exactly in fp16 instead of correcting `zp * sum(q)`
after an fp32 dot of large values.

## Kernel time

One attention layer call (24/4 heads, D = 256, block 816), CUDA-graph replay, median; "int4 pth" includes the two
RHT GEMMs, "kernel only" not. Triton int4 is vLLM's `unified_attention_int4` with HyperQwen's 3D (split-KV) path.

| Context | q | int8 pth | int4 pth | int4 kernel only | Triton int4 | int4 / int8 |
|---|---|---|---|---|---|---|
| 1024 | 1 | 13 us | 28 us | 12 us | 42 us | 2.08 |
| 1024 | 8 | 26 us | 40 us | 30 us | 69 us | 1.56 |
| 32768 | 1 | 142 us | 82 us | 66 us | 385 us | 0.58 |
| 32768 | 8 | 185 us | 209 us | 199 us | 839 us | 1.13 |
| 102400 | 1 | 401 us | 183 us | 174 us | 1031 us | 0.46 |
| 102400 | 8 | 582 us | 588 us | 575 us | 2628 us | 1.01 |
| 131072 | 1 | 508 us | 226 us | 222 us | 1311 us | 0.45 |
| 131072 | 8 | 734 us | 748 us | 731 us | 3354 us | 1.02 |
| 245760 | 1 | 943 us | 430 us | 411 us | 2431 us | 0.46 |
| 245760 | 8 | 1375 us | 1393 us | 1367 us | 6338 us | 1.01 |

Decode (q = 1, the swap-AB kernel) is memory-bound and gets the full 2.2x of the halved cache. The 8-token verify
(48 rows: the 16-row kernel with 4 warps) is compute-bound, as with int8, so it stays at the int8 speed. The two RHT
GEMMs cost ~10-15 us per layer, which matters only at short contexts.

## Results

All on 2026-11-23 with the image of this directory over `ghcr.io/syv-ai/hyperqwen:sha-3d6c954`, `serve.sh`
settings (no CPU KV cache, no vision) unless noted; labels in `../../benchmark/results/agx-z2e-kubuntu.jsonl` and
`-quality.jsonl` start with "z2e HyperQwen int4 pth" and "z2e HyperQwen int8 pth (pthfa-int4 image".

```sh
uv run -m agx_ai.benchmark.bench_http --url http://localhost:9935 --env-file llama-cpp/llama-cpp.env --no-embedding \
    --tests chat depth edit --depth 6400 32000 100000 --repeat 1 --label "..."
uv run -m agx_ai.benchmark.quality_http --url http://localhost:9935 --env-file llama-cpp/llama-cpp.env \
    --tests retrieval --depth 8000 32000 100000 --label "..." [--reference "..."]
```

Retrieval (greedy; single lookups / two-step lookups at 8k, 32k, 100k):

| Run | Single | Two-step | Answers identical to int8 | Mean KLD (99 %, first token) | Top-token flips |
|---|---|---|---|---|---|
| int8 | 96/96 | 10 / 11 / 6 of 12 | (reference) | | |
| int8, second start | 96/96 | 10 / 11 / 6 | 132/132 | 0 (bit-identical) | 0 |
| int8, prefill chunk 1024 instead of 2048 (noise floor) | 96/96 | 9 / 11 / 6 | 130/132 | 0.00023 (0.0065, 0.0010) | 2 |
| **int4** | 96/96 | 8 / 11 / 6 | 127/132 | 0.0028 (0.049, 0.0125) | 5 |

int4 changes the answer token distributions about 12 times as much as the noise floor (mean KLD 0.0028 vs 0.00023)
and got 25 instead of 27 of the 36 two-step answers right (the noise floor: 26); all 96 single lookups are correct in
every run. A small but measurable quality cost, which is the price of the 1.9x context.
