# vLLM image for Qwen3.8-Flash-Next with image and video input

`ghcr.io/agenttix/agx-ai/vllm-flash-next`: the single-RTX-3090 runtime of Qwen3.8-Flash-Next by
[DominikBucko/qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090) (its
[v0.1.0 release image](https://github.com/DominikBucko/qwen38-flash-next-3090/releases/tag/v0.1.0), pinned by digest),
with image and video input added. Built by
[`../.github/workflows/deploy-docker-vllm-rtx3090-flash-next.yml`](../.github/workflows/deploy-docker-vllm-rtx3090-flash-next.yml);
locally:

```sh
docker build -t ghcr.io/agenttix/agx-ai/vllm-flash-next:latest .
```

[`docker-compose.yml`](docker-compose.yml) runs it on the RTX 3090 of agx-z2e (`docker compose up -d`, after stopping
[`../vllm-rtx3090`](../vllm-rtx3090) or [`../llama-cpp-rtx3090`](../llama-cpp-rtx3090) on the same GPU and port);
its settings and the measurements behind them are in
[`../vllm-rtx3090/README.md`](../vllm-rtx3090/README.md#qwen38-flash-next). [`../vllm-rtx3070`](../vllm-rtx3070) runs this image
on an 8 GB RTX 3070 too (without MTP, with an int4 lm_head and one hot expert per layer), and also the model with
stock vLLM, with the same `cpu_vision.py` and encoder runner change in [`../vllm/vllm.patch`](../vllm/vllm.patch). The runtime serves the model text-only
(`--language-model-only`), although its model code supports images and videos. This image adds image and video
input and a faster CPU expert kernel:

- [`cpu_vision.py`](cpu_vision.py): the vision tower (Qwen3-VL architecture, 27 layers, 0.84 GiB in BF16) on the CPU,
  with transformers' `Qwen3VLVisionModel` in FP32. The vision tower takes no VRAM, and the text-only speed is
  unchanged. Selected with `QWEN38_VISION_DEVICE=cpu`, which the serve script sets; with `QWEN38_VISION_DEVICE=gpu`,
  vLLM's own vision transformer is used.
- [`vllm-mm.patch`](vllm-mm.patch) (vLLM):
  - the encoder runner leaves the inputs of the vision tower (the pixel values) on the CPU when the model asks for it
    (`mm_encoder_inputs_on_cpu`), so that they never take VRAM; only the embeddings are copied to the GPU, in BF16
    (5 KiB per token). Otherwise, the FP32 pixel values would be copied to the GPU (6 KiB per patch, ~150 MB for a
    1080p video) and back,
  - the Qwen3-VL processor takes a separate pixel budget for videos (`video_max_pixels`, over all frames), as
    `max_pixels` would otherwise limit both the images and each whole video.
- [`flash-next.patch`](flash-next.patch):
  - the model creates the CPU vision tower instead of vLLM's and sets `mm_encoder_inputs_on_cpu`,
  - the serve script enables image and video input (`QWEN38_VISION=1`, the default; `0` = text only as before):
    - images: at most `QWEN38_VISION_MAX_IMAGES` per request (default 4), scaled down to `QWEN38_VISION_MAX_PIXELS`
      (default 1048576 pixels),
    - videos: at most `QWEN38_VISION_MAX_VIDEOS` per request (default 1), sampled at 2 frames per second with vLLM's
      `qwen3_vl` video loader (only the sampled frames are decoded), at most `QWEN38_VISION_VIDEO_MAX_FRAMES` frames
      (default 64, longer videos are sampled more sparsely) and `QWEN38_VISION_VIDEO_MAX_PIXELS` pixels over all
      frames (default 12582912, ~6000 tokens). The model's own default (25 M pixels) would take ~1 min of CPU time
      and ~3 GiB of RAM for a 30-s 1080p video,
    - no encoder profiling: a dummy input of the maximum size would take minutes on the CPU,
  - it sets `gpu-memory-utilization` (`GPU_MEMORY_UTILIZATION`, default 0.8): vLLM checks at startup that this
    share of the GPU is free, and the runtime's default of 0.92 fails next to a desktop. The KV cache size is set
    explicitly anyway,
  - and, for the 8 GB RTX 3070 of [`../vllm-rtx3070`](../vllm-rtx3070/README.md#cpu-experts-2026-10-26) (added on
    2026-10-26): with `QWEN38_LM_HEAD_QUANT=1`, the lm_head is created with the checkpoint's quantization config, so
    that an int4 or int8 lm_head loads, and the serve script requires the MTP drafter only with `MTP_DEPTH` > 0.
    Until 2026-11-23 this was unconditional, which also gave the BF16 lm_head of the checkpoint here a quantized
    method, and the RTX 3090 configuration failed to start with MTP ("reduced draft vocabulary requires an
    unquantized LM head"); the RTX 3070 runs without MTP and was not affected,
  - and (2026-11-24) vLLM filters the top-k/top-p of 2-7 rows of logits with its Triton kernel instead of a full
    sort of the vocabulary: the rejection sampler of MTP 3 does it for 4 rows of 248320 logits in every decode step
    (919 instead of 248 µs; the step's tail on the GPU 4.8 instead of 5.4 ms),
  - and (2026-11-24) with `QWEN38_INT8_SKINNY=1`, the lm_head is converted to int8 after loading (one scale per row
    and 128 inputs, `QWEN38_INT8_SKINNY_MODULES`, default `lm_head`), with an int8 variant of the runtime's skinny
    Triton GEMM for it and for MTP's reduced draft head, which is a view of it. See
    [CPU kernel and prefetch](#cpu-kernel-and-prefetch-2026-11-24).
- [`cpu-moe-opt.patch`](cpu-moe-opt.patch) (2026-11-24, after `rvii-moe.patch`): a faster int8 kernel, a prefetch of
  the next layer's likely experts into the L3 cache (`QWEN38_CPU_EXPERTS_PREFETCH`), the kernel compiled with
  `-mtune=znver2`, and a decode trace for [`cpu_moe_bench`](#offline-benchmark-of-the-cpu-expert-kernel)
  (`QWEN38_DECODE_TRACE`); see [CPU kernel and prefetch](#cpu-kernel-and-prefetch-2026-11-24).
- [`rvii-moe.patch`](rvii-moe.patch) and [`rvii_experts/`](rvii_experts/README.md): cold experts on the Radeon VII
  (2026-11-23), see [Radeon VII experts](#radeon-vii-experts-2026-11-23).
- [`cpu-moe.patch`](cpu-moe.patch) (the runtime's `cpu_moe.cpp`, which computes the cold experts on the CPU during
  decode; compiled at startup), added on 2026-10-26:
  - integer activations, selected at startup (the default stays FP32): `QWEN38_CPU_EXPERTS_INT8=1` quantizes each
    token's activations to int8 with one scale per 32 values (like llama.cpp's Q8_0 activations) and sums the
    products with the int4 weights exactly in int32 with AVX2's `vpmaddubsw`/`vpmaddwd`, instead of converting every
    weight to FP32 for an FMA. A thread computes an expert 2.3 times as fast (in the cache: 25.8 instead of
    11.0 GB/s of weights), so 8 threads reach the memory bandwidth that the FP32 kernel needs 16 or more for, and
    with faster RAM, the kernel has more headroom. The relative error of an expert's output is ~1 % (FP32: 1e-7);
    in the quality test it was not measurable (see
    [`../vllm-rtx3090/README.md`](../vllm-rtx3090/README.md#cpu-expert-kernel-2026-10-26)).
    `QWEN38_CPU_EXPERTS_INT16=1` uses int16 activations with `vpmaddwd` instead: practically exact (6e-5), 1.3 times
    as fast as FP32 per thread,
  - less per-layer overhead: each thread zeroes its own output buffer, and only the threads that computed an expert
    are summed, in parallel (before, one thread zeroed and summed the buffers of all threads in every layer, ~7 ms
    per decode step of 16 threads), and the experts with more tokens are computed first.

The CPU vision tower takes 1.7 GiB of RAM, and a video up to ~1.5 GiB more while it is encoded, so the container
needs a few GiB more than the expert arena (see the `mem_limit` in [`docker-compose.yml`](docker-compose.yml)). transformers logs once at startup that
`video_max_pixels` "is not a valid argument for this processor and will be ignored"; the patch removes it before the
processor calls, and the budget is applied (see the measurements).

## Radeon VII experts (2026-11-23)

Decode is bound by the CPU: per step (4 tokens with MTP 3), the CPU pool computes ~1050 cold experts in ~56 ms from
DDR4, while the RTX 3090 needs ~13 ms. The Radeon VII (16 GB of HBM2 at ~1 TB/s) is otherwise idle in this
configuration, so it now computes a share of the cold experts:

- [`rvii_experts/`](rvii_experts/README.md): a worker in a ROCm container that keeps 135 experts per layer (6480,
  15.3 GiB) in the Radeon VII's VRAM and computes them on request (HIP kernel for gfx906, ~70 µs for 4 tokens and 8
  experts, relative error ~1e-5 against FP64).
- [`rvii-moe.patch`](rvii-moe.patch) (the runtime's `cpu_moe.cpp`, with `QWEN38_RVII=1`): for each decode request of
  a layer, the CPU pool sends the experts that the worker holds to it over `/dev/shm` ([`rvii_experts/rvii_protocol.h`](rvii_experts/rvii_protocol.h),
  both containers with `ipc: host`), computes the others, and adds the worker's partial sum before it answers the RTX
  3090. The worker needs ~0.1 ms per layer and the CPU ~1 ms, so it never delays the answer. If the worker is not
  running, not ready, stops its heartbeat or does not answer within 20 ms (`QWEN38_RVII_TIMEOUT_US`), the CPU computes
  its experts (and re-opens the worker's segment when it is back). `QWEN38_EXPERT_USAGE=<file>` counts the decode
  requests of each expert (uint32 [48][512], written every 256 steps). The decode-step statistics
  (`QWEN38_CPU_EXPERTS_STATS`) show the worker's experts per step and its timeouts.
- **The GPU share off** (`QWEN38_GPU_SHARE_FRAC=0`): the RTX 3090's share of the cold experts (over PCIe, 0.3 before)
  takes the first ~30 % of each layer's cold experts in the order they come, including those of the Radeon VII, which
  then got only 70-100 experts per step.
- **Experts ranked by measured use** ([`rvii_experts/rankings-measured.json`](rvii_experts/rankings-measured.json),
  mounted over the runtime's `static_hot_cache_rankings.json`): the RTX 3090 keeps the first 24 of each layer and the
  Radeon VII the next 135. Counted with `QWEN38_EXPERT_USAGE` on prompts other than the benchmark's chat and edit
  ones (`chat_stream.py`, the raw generation test, 16k and 64k tokens of code, retrieval questions; 4.9 M expert
  requests). By these counts, the runtime's static ranking puts only 11.8 % of the requests on the RTX 3090's 24 and
  40.8 % on the Radeon VII's next 135 (47.3 % for the CPU); the measured one 24.8 % and 49.8 % (25.4 % for the CPU).
  A ranking measured on a first, smaller collection put 60.5 % of the second collection on the Radeon VII, so the
  counts carry over between these workloads. The CPU still computed ~1080 distinct experts per step, as the rarely
  used experts make up most of the distinct ones; the gain was in the code tests (see below).
  [`rvii_experts/rank_experts.py`](rvii_experts/rank_experts.py) builds rankings from usage counts, e.g. weighted
  toward a topic, and [`docker-compose.usage.yml`](docker-compose.usage.yml) collects them (see
  [Choosing the experts](rvii_experts/README.md#choosing-the-experts)).

Measured with `bench_http.py --tests chat depth edit --repeat 2 --depth 6400 32000 --edit-depth 0` (generation in
t/s; experts per step from the decode statistics):

| Configuration | Chat | Depth 6.4k | Depth 32k | Edit | Cold experts per step: CPU / Radeon VII | CPU / GPU ms per step |
|---|---|---|---|---|---|---|
| Before (GPU share 0.3, static ranking, no Radeon VII) | 32.0 | 32.3 | 31.9 | 52.1 | ~1050 / 0 | 56 / 13 |
| Radeon VII, static ranking, GPU share 0.3 | 35.3 | 35.6 | 35.3 | 54.2 | ~970 / 70-100 | 53 / 14 |
| Radeon VII, static ranking, GPU share 0 | 43.4 | 45.9 | 44.8 | 59.0 | ~1100 / ~280 | 49 / 13 |
| Radeon VII, ranking measured for it only (first collection) | 43.4 | 52.0 | 55.8 | 58.9 | ~1070 / ~310 | 49 / 12 |
| **Radeon VII and RTX 3090, measured ranking (current)** | 43.4 | 56.7 | 57.4 | 57.6 | ~1080 / ~290-390 | 46-59 / 12-13 |

- **+36 % in chats, +75-80 % with code context, +11 % in code edits** over the configuration before. Most of it
  comes from the Radeon VII with the GPU share off; the measured ranking added +24-28 % in the code tests (which are
  of the same kind as part of the collection) and nothing in chats (which were in neither collection).
- The RTX 3090 uses the same VRAM (peak 20236 MiB). Prefill does not use the worker (it streams the cold experts
  through the RTX 3090).

The current configuration with the full benchmark (`--tests generation prompt chat depth edit image --repeat 2
--depth 6400 32000 100000 --edit-depth 0 32000 --prompt-tokens 260000`), against the last results of the
configuration before ([`../vllm-rtx3090/README.md`](../vllm-rtx3090/README.md#cpu-expert-kernel-2026-10-26), 2026-10-26):

| | Chat | Gen. 6.4k / 32k / 100k | Edit (short / after 32k) | Prompt 6.4k / 32k / 100k / 260k | Image 1280x960 | Peak VRAM |
|---|---|---|---|---|---|---|
| Before | 32.4 | 33.4 / 32.7 / 32.0 | 52.4 / 51.8 | 1752 / 1994 / 1958 / 2054 | 8.1 s | 20238 MiB |
| **Current** | **42.7** | **55.8 / 55.2 / 52.4** | **59.5 / 59.2** | 1904 / 2347 / 2106 / 2368 | 8.2 s | 20236 MiB |
| Change | +32 % | +67 / +69 / +64 % | +14 % | +9 / +18 / +8 / +15 % | | |

The prompt processing was faster too, probably as the measured ranking keeps more of the experts that the prompts
use on the RTX 3090 (the cold ones are streamed in every prefill chunk).
- **Quality:** the worker computes in FP32 with 24-bit integer activations (relative error ~1e-5, the CPU pool's
  int8 mode ~1e-2). The retrieval questions of the usage collection were answered 42/44 at 8k tokens (as before).
- The worker needs the whole Radeon VII, so [`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii) has to be stopped.
  Without the worker, the CPU computes all cold experts as before (but with the GPU share off).
- **28 instead of 24 hot experts** on the RTX 3090 (the Radeon VII then takes ranks 28-162): chat 46.0, 56.7 / 55.6 t/s
  at 6.4k / 32k, edits 58.6 t/s, i.e. the same within the noise except perhaps chats, for ~1 GiB more VRAM (peak
  21264 MiB). Not used.
- Not done: the Radeon VII's power profile (`high`) might shorten the worker's ~70 µs by ~15 % (it needs root and makes
  no difference to the CPU-bound step); a dynamic selection of the Radeon VII's experts (they are loaded once).

### HyperQwen's optimizations for Flash-Next (2026-11-23)

Checked again against HyperQwen's newest version (`3d6c954`, vLLM 0.33.0, see
[`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md)); none of them would speed up this runtime, whose
decode is bound by the cold experts:

- **pth-flash-attn** and the int8/int4 KV kernels: Flash-Next's attention is Qwen Sparse Attention with its own
  kernels (and the Gated DeltaNet layers), not causal full attention at head size 256, and its KV cache is already
  int8 with 262144 tokens of context.
- **DFlash2 and lookup drafting**: there is no DFlash2 drafter for Flash-Next. Lookup drafting (copying from the
  context) would help only code edits, where MTP 3 already accepts 98 % of its drafts; the step is capped at 4 tokens
  by the CPU expert path (`MAXT`), and longer verify batches would need a larger request slot, new CUDA graphs and
  more CPU work per step (more distinct experts).
- **The FP16 recurrent state** and **the token embedding in pinned RAM**: the runtime keeps one sequence (its
  recurrent state is small), and its token embedding is already in pinned RAM (`QWEN38_EMBED_UVA`).
- **CPU KV offloading**: does not work with this runtime's ring buffer of the sparse attention (see
  [`../vllm-rtx3090/README.md`](../vllm-rtx3090/README.md#other-settings-2026-10-14)).
- **The reduced MTP draft vocabulary and the sampler patches**: the runtime has its own reduced draft vocabulary.

The runtime itself has no newer release than v0.1.0 (2026-09-30).

## CPU kernel and prefetch (2026-11-24)

Where the time of a decode step went with the configuration of 2026-11-23, from a decode trace of the chat, depth and
edit tests (`QWEN38_DECODE_TRACE`, [`cpu_moe_bench/analyze_trace.py`](cpu_moe_bench/analyze_trace.py); 5973 steps of 4
tokens):

| Part of a step | Time | |
|---|---|---|
| CPU experts (48 requests of the RTX 3090) | 32.0 ms | ~13 experts per layer (630 per step; 73 % with 1 token, 18 % 2, 6 % 3, 3 % 4) on 8 threads, ~0.6 ms per layer |
| Waiting for the Radeon VII after the CPU | 0.3 ms | its ~520 experts per step never delay the CPU |
| RTX 3090 between the layers | 13.8 ms | ~0.2 ms per layer: dense layers (int8 `_skinny_dot`, int4 Marlin in the Gated DeltaNet), norms, router, near the memory bandwidth |
| RTX 3090 at the end of the step | 6.8 ms | lm_head 1.45 ms (BF16), 3 MTP drafts ~2 ms (each a 0.38 ms reduced-vocabulary head), sampling 0.9 ms |

So the CPU's RAM was idle ~40 % of a step, and the CPU kernel itself reached ~55 GB/s of the 71 GB/s that the RAM
delivers. Changes ([`cpu-moe-opt.patch`](cpu-moe-opt.patch), [`flash-next.patch`](flash-next.patch)):

- **Prefetch of the next layer's likely experts** (`QWEN38_CPU_EXPERTS_PREFETCH=1`): after answering a layer, each
  idle pool thread reads one expert that the CPU computed for the next layer in the previous step (the ones with the
  most tokens first) into its CCX's L3 cache while the GPU computes, and computes the experts that it read first
  (the others are claimed as before). 37.7 % of a layer's CPU experts were CPU experts of the same layer in the
  previous step; of the 7 prefetched per layer, ~45 % are used. The prefetch has to use loads: on Zen 2, lines that
  `prefetcht0/t2/nta` bring in are not kept in the L3 when the L2 evicts them (a 2.4 MB read afterwards ran at
  ~30 GB/s, as from RAM, and at ~114 GB/s after plain loads), and the first version with prefetch instructions saved
  nothing. A new request stops the prefetch (checked every 4 KB). 2 or 3 per thread did not fit in the ~0.2 ms gap.
- **Loader threads** (`QWEN38_CPU_EXPERTS_LOADER_CPUS`, one per pool thread, on core 2 of each CCX): when a pool
  thread starts an expert, its loader reads the expert's weights in the kernel's order with plain loads, ahead of it,
  into the CCX's L3, so that the pool thread's loads hit the cache; a new expert interrupts the loader. The kernel's
  loop reaches ~61 GB/s with 8 threads, plain loads 71 GB/s with the same access pattern, so this recovers part of
  the difference: -5 % CPU time per step in the replay (63.6 instead of 60.5 GB/s), on the SMT siblings of the pool
  threads (which the runtime keeps idle) only -1 %. The serving processes move to cores 0 and 3 of each CCX
  (`QWEN38_MAIN_CPUS`), 16 cores with SMT.
- **The int8 GEMV:** the products of an activation group (4 packed rows) are summed in int16 (each `vpmaddubsw` lane is
  at most 2 * 15 * 127 = 3810, so 8 fit exactly) and widened once, and 8 instead of 16 columns at a time keep the sums
  of 4 tokens in registers; an empty `asm` barrier keeps GCC from hoisting all 32 products of the 4 rows and spilling
  them. An expert with 4 tokens: 9.0 instead of 7.0 GB/s on one core from the cache (1 token: ~25 as before);
  bit-exact (the integer sums are the same).
- **`-mtune=znver2`** for the kernel (1.5-3 % in the offline benchmark, the same as `-march=znver2`, which would make the
  image require a Zen 2-compatible CPU). GCC 15 and clang 21 were not faster than the image's GCC 13.
- **An int8 lm_head** (`QWEN38_INT8_SKINNY=1`, `flash-next.patch`): the BF16 lm_head (248320 x 2560, 1.27 GB) is
  read in every decode step by the target and, as the reduced draft vocabulary (65812 rows), by each of MTP's 3
  drafts. After loading, it is converted to int8 with one FP32 scale per row and 128 inputs (0.64 GB), and an int8
  variant of the runtime's skinny Triton GEMM reads it (int8 -> BF16 is exact; each group's scale multiplies its
  FP32 dot product). In CUDA graphs: 791 instead of 1483 µs for the lm_head, 211 instead of 399 µs per draft, i.e.
  -1.3 ms per step, and 0.6 GiB less VRAM. MTP's drafter requires an "unquantized" lm_head, which the int8 one
  still is for it (the same method class; the draft head reads the same scales). The other BF16 projections of the
  runtime's skinny GEMM (shared experts, hyper-connections, 437 modules in all) were not faster in int8: they are
  small (3-7 MB) and bound by latency, not by their weight reads. Quality (the retrieval test without MTP, 88
  questions at 8k and 32k tokens, against the BF16 lm_head):

  | Run | Correct | Same answers | Mean KL divergence | 99 % | First token | Top-token flips |
  |---|---|---|---|---|---|---|
  | BF16 lm_head (reference) | 84/88 | | | | | |
  | BF16 lm_head again (noise floor) | 83/88 | 86/88 | 0.00145 | 0.058 | 0.0081 | 2 of 573 |
  | int8, one scale per row | 82/88 | 84/88 | 0.00214 | 0.064 | 0.0126 | 4 of 564 |
  | **int8, one scale per row and 128 inputs (current)** | 80/88 | 84/88 | 0.00164 | 0.049 | 0.0080 | 4 of 567 |

  With one scale per row, the divergence was 1.5 times the noise floor; with group scales it is at the noise floor
  (the two-step lookups, which chain two greedy answers, differ by a few in every run). `QWEN38_INT8_SKINNY=0`
  restores the BF16 lm_head.
- **The sampler** (`flash-next.patch`): vLLM's rejection sampler applies top-k 20 / top-p 0.95 (the model's
  generation config) to the 4 rows of the target's logits with a full sort of the 248320 tokens and a cumulative sum,
  as vLLM uses its Triton top-k/top-p kernel only from 8 rows on: 919 µs per step, with the Triton kernel 248 µs and
  the same tokens kept. The step's tail on the GPU: 4.8 instead of 5.4 ms.

The offline replay of the trace ([`cpu_moe_bench`](#offline-benchmark-of-the-cpu-expert-kernel), 24 layers, 1500 steps,
the traced GPU gaps spun out), CPU time per step of 24 layers: 14.1 ms before, 13.4 ms with the new GEMV, 12.2 ms with
the prefetch too (-14 %). In the server (`--tests chat depth edit --repeat 2 --depth 6400 32000 --edit-depth 0`,
generation in t/s; the decode statistics of the edit test):

| Configuration | Chat | Depth 6.4k | Depth 32k | Edit | CPU / between layers / tail ms per step (edit) |
|---|---|---|---|---|---|
| Before (2026-11-23) | 43.2 | 56.2 | 56.6 | 60.3 | |
| New GEMV, `-mtune=znver2`, prefetch | 48.7 | 59.8 | 59.5 | 63.7 | 43.2-44.9 / 13.3-13.6 / 5.4 |
| The same, again | 47.9 | 59.1 | 57.1 | 63.9 | 43.3-44.4 / 13.1-13.3 / 5.4 |
| And the Triton top-k/top-p | 49.7 | 59.3 | 64.3 | 64.9 | 42.8-44.1 / 13.2-13.3 / 4.8 |
| The same, with the serving processes on cores 0 and 3 of each CCX (control) | 48.2 | 59.5 | 60.2 | 64.8 | 43.6-44.6 / 13.3-13.4 / 4.8 |
| And loader threads on core 2 of each CCX | 50.7 | 61.9 | 63.6 | 66.4 | |
| **And the int8 lm_head (current; from the full benchmark below)** | 52.8 | 65.5 | 65.4 | 68.2 | 41.3-42.3 / 13.5-14.4 / 3.6 |

The trace of the first run with the prefetch: CPU 28.8 instead of 32.0 ms per step (-10 %) for the same ~630 CPU
experts.
The current configuration with the full benchmark (`--tests chat depth edit prompt --repeat 2 --depth 6400 32000
100000 --edit-depth 0 32000 --prompt-tokens 260000`), against the full benchmark of 2026-11-23 (see above):

| | Chat | Gen. 6.4k / 32k / 100k | Edit (short / after 32k) | Prompt 6.4k / 32k / 100k / 260k | Image 1280x960 | Peak VRAM |
|---|---|---|---|---|---|---|
| 2026-11-23 | 42.7 | 55.8 / 55.2 / 52.4 | 59.5 / 59.2 | 1904 / 2347 / 2106 / 2368 | 8.2 s | 20236 MiB |
| 2026-11-24: GEMV, prefetch, sampler | 47.8 | 62.2 / 59.3 / 62.9 | 64.9 / 65.1 | 1939 / 2272 / 2231 / 2366 | | 20236 MiB |
| **2026-11-24: and loader threads, int8 lm_head (current)** | **52.8** | **65.5 / 65.4 / 67.3** | **68.2 / 68.2** | 2010 / 2352 / 2188 / 2366 | 8.2 s | 19598 MiB |
| Change | +24 % | +17 / +18 / +28 % | +15 % | within the noise | the same | -638 MiB |

The chat and depth tests vary with the draft acceptance (36-45 % in these runs); the edit test (98 % acceptance) is the
most stable: +15 %. Per decode step (the edit test's statistics): CPU 41-42 ms (2026-11-23: 46-59 ms in the same
tests), the GPU's tail 3.6 instead of 5.4 ms. Prefill does not use the changed code paths, and the image (vision tower on the serving
processes' CPUs, now 16 cores with SMT instead of 24) took as long as before. The changes of the CPU kernel need no
quality test: its output is the same (6.7e-8 relative difference to the old kernel in the offline benchmark, as
between two runs of the old one, whose threads sum in varying order), the prefetch and the loaders change only when
the weights are read, and the sampler keeps the same tokens; the int8 lm_head was tested (above).

Tried, not better (offline benchmark unless noted):

- **Slices of an expert** as separate work items, for a better load balance (13 experts per layer on 8 threads): with
  the int8 patterns of a layer's activations computed once per layer, a slice costs no more compute (-4 % from the
  cache), but 5 column slices halved the bandwidth from RAM with 8 threads, as each thread reads 512 of every 2560
  bytes of a row and the hardware prefetchers read the neighbouring slices' lines. With plenty of experts per layer
  (no balance problem), the kernel reaches 61-63 GB/s, so the balance costs at most ~10 % anyway.
- **A row-streaming loop order** (each packed row read contiguously, the sums in L1): 10 instead of 17 GB/s with one
  thread from RAM.
- **Longer prefetch distances** (2-3 groups ahead), `prefetcht0` instead of `t1`, prefetches spread over the loop:
  62-63 GB/s vs 62. Plain loads with the same access pattern (800 KB at random 2.5 MB slots) read 71 GB/s, so the
  kernel's loop leaves ~15 %; one thread computes at 17 GB/s from RAM, 25 from the cache, and reads at 28-30.
- **12 or 16 pool threads** (with the new kernel): the same as 8.
- **`torch.compile`** (`--compilation-config '{"mode":3,...}'` instead of the runtime's mode 0): fails at startup
  ("auto_functionalized was not removed"). The GPU's work per layer is mostly weight reads near the bandwidth
  anyway (int8 `_skinny_dot` at ~830 GB/s, Marlin).

Not done:

- **A GPU share of the CPU's experts:** the RTX 3090 waits ~0.5 ms per layer for the CPU, and its PCIe share
  (`QWEN38_GPU_SHARE_FRAC`) could compute some cold experts meanwhile if its selection skipped the Radeon VII's
  experts (it takes the first cold ones in routing order). But it reads them from the same RAM, which the CPU's pool
  now reads at ~63 of 71 GB/s, and without the Radeon VII, shares of 0.15-0.45 made no difference.

### Ideas outside the containers (for review)

These would change the host's configuration, so they are not implemented:

1. **Faster RAM**: DDR4-2666 in 4 channels reads 71 GB/s; the Threadripper 3970X supports DDR4-3200 (and the Infinity
   Fabric clock follows the memory clock up to 1600 MHz). If the modules allow it (BIOS: DOCP/XMP or manual timings),
   +20 % bandwidth would shorten the CPU part (~60 % of a step) nearly proportionally: perhaps +10 % decode. Check the
   stability with a memory test first, then rerun [`cpu_moe_bench/run.sh`](cpu_moe_bench/run.sh) and retune the
   thread count.
2. **L3 cache allocation for the pool cores** (`resctrl`; the CPU has `cat_l3`): the pool uses one core of each 4-core
   CCX, and the serving processes and the desktop on the other three share and evict the CCX's 16 MB of L3 that holds
   the prefetched experts. Reserving most of the L3 ways for the pool cores (e.g. 12 of 16) might keep more of them;
   needs `mount -t resctrl` and a group with the pool CPUs (root).
3. **The CPU frequency policy** `performance` (or the energy-performance preference `performance`) instead of
   `powersave` / `balance_performance`: the pool threads alternate between spinning and short bursts of work every
   ~0.2 ms, and each layer waits for the slowest of 8 threads.
4. **Isolating the pool CPUs** (1, 5, ..., 29) from the scheduler and timer interrupts (`isolcpus=`, `nohz_full=`,
   `irqaffinity=` on the kernel command line): an interrupted pool thread delays the whole layer. The container's
   cpusets would then need the isolated CPUs explicitly (they are already pinned).
5. **Huge pages for the expert arena** at boot (`hugepagesz=1G hugepages=58`): transparent huge pages covered all of
   it now (57.9 GB of `AnonHugePages`), but on 2026-10-14 only 23.6 of 56 GiB after days of uptime, as the RAM was
   fragmented. Reserving them would make it independent of the uptime, but takes the RAM from everything else.
6. **The Radeon VII's power profile** (`high`, see above): ~15 % shorter worker latency, which is not on the critical
   path now (0.3 ms of waiting per step).

## Measurements (2026-10-13, agx-z2e)

RTX 3090 and Threadripper 3970X (32 cores, Zen 2), with the settings of [`docker-compose.yml`](docker-compose.yml)
at the time (20 hot experts, 135168 tokens of context):

- **Vision tower on the GPU** (vLLM's own, the runtime without `--language-model-only`): 0.84 GiB of weights, which
  needed 12 instead of 20 hot experts per layer. The first 1280x960 image then ran out of GPU memory (the encoder's
  activations needed another ~1 GiB). Making room for both (~1.9 GiB) would cost the text path 16 of its 20 hot
  experts per layer, or two thirds of the KV cache.
- **Vision tower on the CPU** (this image, images only): the same VRAM as text-only (14.09 GiB after loading, peak 20804 MiB), and
  the same text speed (chat 27.9 t/s; 29.3 and 27.8 t/s at 6.4k and 32k tokens of context). A 1280x960 image, scaled
  to 1 megapixel (~980 image tokens), takes ~8 s until the first token. It read the text and the shapes, colors and
  positions of a test image correctly.
- The vision tower alone on the CPU, FP32, 1280x960 without scaling (4800 patches): 9.7 s with 16 threads, 6.5 s with
  32 and 9.2 s with 64 (SMT). In BF16, it is 3.5 times as slow (no BF16 instructions on Zen 2). The CPU expert pool is
  idle while the encoder runs, as the runtime serves one request at a time.
- transformers' vision model loads all 333 vision tensors of the checkpoint with no missing or unexpected keys.
- **Videos** (2026-10-13; generated test clips with a red square moving from left to right, and "ALPHA" in the first
  half and "OMEGA" in the second half; both described correctly):

  | Clip | Prompt tokens | First token | RAM while encoding | Peak VRAM |
  |---|---|---|---|---|
  | 20 s, 640x360, 30 fps | 4609 | 28.6 s | +1.18 GiB | 20064 MiB |
  | 30 s, 1920x1080, 30 fps | 5999 | 35.4 s | +1.53 GiB | 20384 MiB |

  The time is mostly the vision tower on the CPU (~1.4 ms per 16x16 patch). With the video limits and the inputs on
  the CPU, the text speed (chat 27.5 t/s; 27.4 and 26.1 t/s at 6.4k and 32k tokens of context), the image time (7.9 s)
  and the peak VRAM (20804 MiB) are unchanged.
- The vision tower's RAM while encoding grows linearly with the patches (it attends within each frame): +0.35 GiB for
  a 1280x960 image (4800 patches), +0.77 GiB for 13200 patches, +1.54 GiB for 26400, +1.64 GiB for 28800.

## Checkpoint with int8 dense layers (2026-10-26)

[`docker-compose.yml`](docker-compose.yml) uses the checkpoint with the Gated DeltaNet and attention projections in
int8 instead of BF16 (round-to-nearest, group size 128, Marlin kernels; -2.6 GiB of VRAM and less to read per decode
step), built in 30 s by [`../vllm-rtx3070/quantize_flash_next_dense.py`](../vllm-rtx3070/quantize_flash_next_dense.py)
as a "view": a directory with one new 2.6 GB shard of the quantized tensors, symbolic links to the original shards
and the other files, and an index and a config for the quantized layers. The runtime's loader honors the index at
the tensor level, so it skips the BF16 originals that are still in the original shards. The shared experts (which
the runtime fuses into the MoE) and the lm_head (created without a quantization config) stay in BF16:

```sh
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro -v ~/.cache/vllm-models:/out \
    -v "$PWD/../vllm-rtx3070":/src:ro --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm:latest \
    /src/quantize_flash_next_dense.py --out /out/Qwen3.8-Flash-Next-W4A16-FP8PLE-dense8-view --bits 8 \
    --lm-head-bits 16 --separate-shard --keep-shared-experts
```

The view's links point to the Hugging Face cache as `/root/.cache/huggingface/hub`, which the compose file mounts
at the same path.

## Offline benchmark of the CPU expert kernel

[`cpu_moe_bench`](cpu_moe_bench) builds the image's `cpu_moe.cpp` into a standalone program (with a stub of the
torch and pybind11 headers) and runs its decode path on 7.7 GB of synthetic int4 experts with decode-like routing
(4 tokens with MTP 3, ~24 cold experts per layer, 48 layers), with the runtime's thread pool. It reproduced the
server's CPU time per decode step within ~7 % (55 vs 59 ms with 16 threads, 66 vs 66 ms with 8), and takes seconds instead of a 5-min server start per variant.
[`cpu_moe_bench/run.sh`](cpu_moe_bench/run.sh) measures the RAM read bandwidth (`bw.cpp`) and then each kernel mode
with 8, 16 and 24 threads, e.g. after a change of the RAM settings (stop the server first):

```sh
cpu_moe_bench/run.sh
```

On 2026-10-26 (DDR4-2666 in 4 channels; RAM read bandwidth 28 GB/s with 1 thread, 71-72 GB/s with 8-16 threads),
ms per decode step (lower is better) and the GB/s of expert weights read:

| Kernel | 8 threads | 16 threads | 24 threads | Relative error |
|---|---|---|---|---|
| FP32, the runtime's kernel | 70.5 (39.7 GB/s) | 56.5 (49.6 GB/s) | 56.0 (50.1 GB/s) | 7e-8 |
| FP32 with `cpu-moe.patch` (less per-layer overhead) | 69.6 (40.3 GB/s) | 55.3 (50.7 GB/s) | 54.4 (51.5 GB/s) | 7e-8 |
| INT16 activations | 55.0 (51.0 GB/s) | 48.8 (57.4 GB/s) | 50.4 (55.6 GB/s) | 6e-5 |
| INT8 activations (groups of 32) | 47.9 (58.5 GB/s) | 46.9 (59.8 GB/s) | 49.2 (57.0 GB/s) | 1.2e-2 |

The same with the experts in the L3 cache (1 thread, 1 token): FP32 11.0, INT16 14.3, INT8 25.8 GB/s, i.e. the
compute limit of one Zen 2 core. Not faster: splitting each expert into 2 or 5 column slices for a better load
balance (10-40 % slower in all modes, also in the server), int8 groups of 64 or 128 instead of 32 (as fast, error
1.4e-2 and 1.6e-2), a prefetch distance of 16 instead of 6 rows (in the server). With faster RAM, the INT8 kernel
should scale furthest: with 8 threads it is at 82 % of the measured bandwidth, and it is far from its compute limit.


Since 2026-11-24 (see [CPU kernel and prefetch](#cpu-kernel-and-prefetch-2026-11-24)), the benchmark can also replay
the server's real routing. Record a decode trace with the server (`QWEN38_DECODE_TRACE=/path/trace.bin` in the
environment and the directory mounted; 184 bytes per layer request, ~55 MB for the chat, depth and edit tests), then:

```sh
python3 cpu_moe_bench/analyze_trace.py trace.bin --skip-out skip.bin   # time per step, tokens per expert, reuse
# the kernel of the image (as in run.sh) and its decode path on the real routing of 24 layers (31 GB of RAM):
g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -mtune=znver2 -pthread -DPF_ROWS=6 -Icpu_moe_bench/stub \
  -DKERNEL='"/path/cpu_moe.cpp"' cpu_moe_bench/bench.cpp -o bench
QWEN38_CPU_EXPERTS_INT8=1 QWEN38_CPU_EXPERTS_PREFETCH=1 BENCH_E=512 BENCH_L=24 BENCH_SKIP=skip.bin \
  BENCH_REPLAY=trace.bin ./bench 1,5,9,13,17,21,25,29 4 7 1500
```

The replay spins out the traced GPU time between the layers, so that the prefetch has the same idle time as in the
server. Per layer, it measured ~12 % less CPU time than the server (24 layers: 14.1 ms; the server's 48: 32.0 ms), as
the server's pool shares the CCXs with the serving processes; the differences between kernel versions carried over.
