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
    [CPU kernel and prefetch](#cpu-kernel-and-prefetch-2026-11-24),
  - and (2026-11-24) the PLE offload's request queue (of one) waits for its thread instead of failing: `put_nowait`
    raised `queue.Full` and killed the engine once when other programs loaded the serving processes' CPUs, and with
    `QWEN38_CPU_EXPERTS_FIFO`, the serve script starts all serving processes off the CPUs of the pool's and the
    loaders' real-time threads (`taskset`). See
    [GPU kernels, the Radeon VII and CPU isolation](#gpu-kernels-the-radeon-vii-and-cpu-isolation-2026-11-24).
- [`cpu-moe-opt.patch`](cpu-moe-opt.patch) (2026-11-24, after `rvii-moe.patch`): a faster int8 kernel, a prefetch of
  the next layer's likely experts into the L3 cache (`QWEN38_CPU_EXPERTS_PREFETCH`), the kernel compiled with
  `-mtune=znver2`, and a decode trace for [`cpu_moe_bench`](#offline-benchmark-of-the-cpu-expert-kernel)
  (`QWEN38_DECODE_TRACE`); see [CPU kernel and prefetch](#cpu-kernel-and-prefetch-2026-11-24). Also the scheduling
  priority of the pool's and the loaders' threads (`QWEN38_CPU_EXPERTS_FIFO` or `QWEN38_CPU_EXPERTS_NICE`), and the
  idle request server polls every 1 ms after 1 s without requests instead of every 50 µs, see
  [GPU kernels, the Radeon VII and CPU isolation](#gpu-kernels-the-radeon-vii-and-cpu-isolation-2026-11-24). And the
  prefetch predicted by the next layer's router (`QWEN38_CPU_EXPERTS_PREGATE`, the routers registered by
  `flash-next.patch`), see [Pre-gating prefetch](#pre-gating-prefetch-2026-11-24).
- [`abliteration.py`](abliteration.py), [`abliteration/`](abliteration) and hooks in `flash-next.patch` (2026-11-24):
  the optional removal of the refusal direction (`QWEN38_ABLITERATION`), see [Abliteration](#abliteration-2026-11-24).
- [`rvii-moe.patch`](rvii-moe.patch) and [`rvii_experts/`](rvii_experts/README.md): cold experts on the Radeon VII
  (2026-11-23), see [Radeon VII experts](#radeon-vii-experts-2026-11-23).
- [`rvii_prefill.py`](rvii_prefill.py) and [`rvii-prefill.patch`](rvii-prefill.patch) (2026-11-25, after
  `cpu-moe-opt.patch`): the Radeon VII computes some of its experts in prefill chunks (`QWEN38_RVII_PREFILL`; the
  bridge kernels and the hooks in `stream_v2.py`), the runtime publishes its hot experts and counts the decode
  requests for the worker's automatic re-ranking, and starts with the worker's ranking (`QWEN38_RANKINGS_AUTO`), see
  [Radeon VII in prefill and automatic re-ranking](#radeon-vii-in-prefill-and-automatic-re-ranking-2026-11-25).
- Tools: [`test_server.sh`](test_server.sh) starts a test server as in the compose file with another image or
  settings, [`kernel_bench/`](kernel_bench/README.md) has microbenchmarks of the RTX 3090's kernels, and
  [`isolate_cpus.py`](isolate_cpus.py) is a prototype (for review, not installed) that reserves the pool's cores
  during requests from the host.
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
4. **Isolating the pool CPUs** from the scheduler and interrupts: done inside the containers with SCHED_FIFO since
   then, and a prototype for reserving the cores from the host only during requests ([`isolate_cpus.py`](isolate_cpus.py)),
   see [Isolating the CPU pool's cores](#isolating-the-cpu-pools-cores).
5. **Huge pages for the expert arena** at boot (`hugepagesz=1G hugepages=58`): transparent huge pages covered all of
   it now (57.9 GB of `AnonHugePages`), but on 2026-10-14 only 23.6 of 56 GiB after days of uptime, as the RAM was
   fragmented. Reserving them would make it independent of the uptime, but takes the RAM from everything else.
6. **The Radeon VII's power profile** (`high`, see above): ~15 % shorter worker latency, which is not on the critical
   path now (0.3 ms of waiting per step). See [Power profiles](#power-profiles) for the commands.

## GPU kernels, the Radeon VII and CPU isolation (2026-11-24)

### The RTX 3090's kernels

A torch profile of the decode steps (CUDA graphs, the configuration before the int8 lm_head): 2370 kernels per step,
18.5 ms of kernel time besides the waits for the CPU. Per Gated DeltaNet layer, the GPU's part before it hands the
layer's cold experts to the CPU (the part on the critical path, ~190 µs):

| Kernel | µs per layer | Reads | |
|---|---|---|---|
| in_proj_qkvz, Marlin W8A16 (16384 x 2560) | 54.6 | 42 MB | 770 GB/s |
| out_proj, Marlin W8A16 (2560 x 6144) | 26.5 | 16 MB | 590 GB/s |
| 4 hyper-connection GEMVs, BF16 skinny Triton (336 x 10240, 10240 x 320) | 42 | 4 x 6.6-6.9 MB | 600-820 GB/s |
| Gated DeltaNet recurrence (`fused_gdn_decode_post_conv_mtp`, in vLLM's `_C`) | 22.6 | | 48 blocks |
| CPU expert submit (40 KB to pinned host memory) | 13-17 | | |
| router, norms and ~20 small kernels | ~30 | | |

Tried with [`kernel_bench/`](kernel_bench/README.md):

- **The int8 skinny Triton kernel** (the int8 lm_head's) instead of Marlin for the W8A16 projections (the checkpoint's
  symmetric int8 with group 128 maps exactly onto its format): with the best configuration per shape, 52.7 instead
  of 55.5 µs, 22.7 instead of 23.2 µs, 43.6 instead of 46.5 µs (attention qkv), i.e. ~0.15 ms per decode step, and
  its dequantizing path for prefill is 30-50 % slower than Marlin (2048 tokens: 2333 instead of 1661 µs for
  in_proj_qkvz). Keeping both would need 1.3 GB more VRAM. Not changed: Marlin is near the practical bandwidth.
- **Other configurations of the BF16 skinny kernel** for the hyper-connections: 10240 x 320 takes 7.3 instead of
  8.0 µs with BLOCK_N 64 (0.07 ms per step), 336 x 10240 none faster. Not changed.
- **The submit kernel** (one block writes x, 4 x 2560 FP32, into pinned host memory for the CPU and publishes the
  request): ~10-11 µs whatever the kernel does (16- or 8-byte loads, all loads first, 2-40 blocks with the last one
  publishing, or a conversion kernel and a copy-engine `cudaMemcpyAsync`): the GPU writes host memory at ~4 GB/s
  here. Only a smaller request (BF16 instead of FP32, a protocol change for the CPU and the Radeon VII) would save
  ~5 µs per layer (0.25 ms per step).
- The Gated DeltaNet recurrence reads and writes its FP32 state (3.1 MB per layer) at ~280 GB/s with 48 blocks on 82
  SMs, so a version with more blocks might save ~0.4 ms per step, but it is compiled into vLLM's `_C` library of
  the base image (no source in the image).

So the RTX 3090's share of a step (~14 ms between the layers, 3.6 ms at the end) is bound by its memory bandwidth and
by small kernels; the CPU's experts (~40 ms) remain the bottleneck.

### `torch.compile`

The runtime runs with `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`: no compilation, but
the whole decode step in one CUDA graph. With mode 3 (vLLM's default), Inductor failed at startup with
"auto_functionalized was not removed": the runtime registers custom ops that write into their inputs (the CPU
experts' wait-and-add into the MoE output, the PLE offload, the hyper-connection and QSA kernels), and Inductor's
functionalization cannot turn them back into in-place calls. What it would take:

- register these ops with their `mutates_args` and either make them functional (return new tensors) or add them to
  vLLM's `splitting_ops`, so that the graph is split around them like around attention, and
- check that the CPU expert bridge's ordering (submit, GPU experts, wait) survives the reordering of the compiled
  graph, which assumes no side effects outside the tensors.

What it could gain: fusing the small elementwise kernels into their neighbours. Per decode step, 1477 kernels take
less than 4 µs (2.9 ms in all), of which the elementwise ones (norms, activations, additions, copies) 1.8 ms; the
runtime already fuses the hyper-connections' and QSA's in Triton. A perfect fusion would save ~1-1.5 ms of a ~55 ms
step (2-3 %), at the cost of minutes of compilation at startup and the risk of the reordering above. Not pursued.

### The Radeon VII

From a decode trace (`QWEN38_DECODE_TRACE`, 13909 steps of chat and code edits, with and without other programs
loading the CPU), the time from the CPU finishing its own experts to the Radeon VII's answer: 6 / 8 / 12 / 16 µs
(10th / 50th / 90th / 99th percentile), i.e. the Radeon VII's ~10 experts per layer (~70-120 µs, see
[`rvii_experts/README.md`](rvii_experts/README.md#measurements)) are done long before the CPU's ~13. Its kernel is
not on the critical path, so making it faster would not change the speed. The delays come from elsewhere: 533 of
668k layers waited more than 1 ms (at most 5 ms), and with the shorter timeout below, the timeouts came only in the
runs where 64 busy threads of other programs loaded the CPU. Changes:

- **The timeout is 2 ms instead of 20 ms** (`QWEN38_RVII_TIMEOUT_US=2000`): if the Radeon VII does not answer within
  2 ms after the CPU has computed its own experts, the CPU computes the Radeon VII's experts for that layer (~0.7 ms)
  instead of waiting up to 20 ms, so a stall (a busy host, a clock change, a hiccup of the worker) costs at most
  ~3 ms per layer. A late answer is ignored by its sequence number, and the worker is used again from the next layer
  on. In 25 min of the final configuration (3 runs with 64 busy threads of other programs, 1 without), it fired 3
  times, all in the first second of the first request. The message is logged for the first 3 and then every 100th
  timeout.
- The idle worker polls every 1 ms after 2 s without requests (before: every 100 µs, ~10000 wake-ups per second).

Tried: the worker with SCHED_FIFO (its polling thread, or all its threads including the HIP runtime's), to protect
it from other programs: with 64 busy threads, the 2-ms timeout then fired 340-680 times in 5 min instead of never
(a spinning real-time thread on the serving processes' CPUs, which the load already crowded, probably delayed the
threads that the worker's answers depend on), and the run with all threads real-time ended with a hung engine.
Not used.

### Isolating the CPU pool's cores

The pool's threads compute the cold experts in ~0.6 ms per layer, and a layer waits for the slowest of 8 threads, so
any other program on their cores delays the whole step. With 64 busy threads of another program (`stress-ng --cpu
64` in the desktop session, i.e. in `user.slice`), unpinned, the speed fell to 40 %. Kept off the pool's and the
loaders' cores (`taskset` on `QWEN38_MAIN_CPUS`, i.e. what isolating these 16 cores would do), it fell only by 3-6 %.
Nice values cannot do this: the containers run in `system.slice`, which has the same CPU weight as `user.slice`, so a
spinning pool thread gets only half of its core next to a busy program however nice it is. The real-time policy
SCHED_FIFO ranks above all normal threads instead, and the threads sleep between requests, so the cores are free
for other programs then: `QWEN38_CPU_EXPERTS_FIFO=10` (`cpu-moe-opt.patch`; with `cap_add: SYS_NICE`). With it, the
serve script starts all serving processes on the CPUs other than the pool's and the loaders' (`taskset`), so that
none of their threads waits behind a real-time thread (the API server and the engine core could run anywhere
before; see [Pre-gating prefetch](#pre-gating-prefetch-2026-11-24) for why not only on `QWEN38_MAIN_CPUS`). The kernel
still gives normal threads 5 % of each CPU (its fair server), so nothing on those CPUs starves.

`--tests chat edit --repeat 2 --edit-depth 0`, generation in tokens/s:

| Configuration | Other programs | Chat | Edit |
|---|---|---|---|
| Normal priority (the configuration of the morning) | none | 52.3 | 67.4 |
| Normal priority | 64 busy threads | 20.0 | 27.0 |
| Normal priority | 64 busy threads, kept off the pool's and the loaders' cores | 49.0 | 65.2 |
| Pool with SCHED_FIFO | none | 53.2 | 68.1 |
| Pool with SCHED_FIFO | 64 busy threads | 46.1 | 63.3 |
| **Pool with SCHED_FIFO, serving processes on their CPUs, Radeon VII timeout 2 ms (current)** | none | 53.0 | 68.1 |
| The same | 64 busy threads (3 runs) | 45.6-46.8 | 63.2-63.7 |

The first run with SCHED_FIFO and the load ended with the engine dead: the PLE offload's request queue holds one
request, and the model thread added the next one with `put_nowait`, which fails when the queue's thread (on the
serving CPUs, which the load now crowded) has not taken the previous one yet. `flash-next.patch` makes it wait
instead. With it, the final configuration ran 4 benchmarks (3 with the load, ~25 min) without errors.

Not done (outside the containers, for review):

- **Reserving the cores during requests from the host**: [`isolate_cpus.py`](isolate_cpus.py) is a prototype of a
  root service that polls vLLM's `vllm:num_requests_running` and, while requests run, moves the desktop's programs
  (`systemctl set-property --runtime user.slice AllowedCPUs=...`) and the movable interrupts (`/proc/irq/*/smp_affinity_list`)
  off the pool's and the loaders' 16 cores (32 hardware threads), and gives them back after 5 s without requests
  (`--dry-run` shows what it would do). The measurement above with the load kept off these cores is what it would
  achieve: SCHED_FIFO already gives most of it (46 / 63 vs 49 / 65 t/s), so it would add a few %. The kernel here
  supports cgroup v2 cpuset partitions (`cpuset.cpus.partition = isolated`), which would also take the cores out of
  the scheduler's load balancing, but the container would then need a threaded cgroup for the pool's threads alone.
  Interrupts are not the problem now: the pool's CPUs get ~20-150 per second (besides the timer), except the request
  server's CPU, whose idle polling caused ~10000 timer interrupts per second (fixed above).
- **`isolcpus=` / `nohz_full=` on the kernel command line** (the kernel has `CONFIG_NO_HZ_FULL`): this would also
  stop the timer tick on the pool's cores, but permanently, so the cores would not be available for other programs
  without requests, which is not wanted here.

### Temperatures

Recorded by `bench_http` since today ([`agx_ai/benchmark/temps.py`](../agx_ai/benchmark/temps.py); the RTX 3090's VRAM
from the [nvidia-vram-hwmon](https://github.com/AgenttiX/nvidia-vram-hwmon) kernel module), min / mean / max °C during
the runs above (~5 min each):

| Device | Sensor | Without other programs (after 3 loaded runs) | With 64 busy threads (the first run) |
|---|---|---|---|
| RTX 3090 | VRAM (GDDR6X) | 92 / 95.7 / 98 | 72 / 89.1 / 94 |
| RTX 3090 | core | 75 / 81.2 / 84 | 62 / 77.7 / 82 |
| Radeon VII | edge | 37 / 38.4 / 40 | 35 / 37.1 / 38 |
| Radeon VII | junction | 41 / 43.7 / 45 | 39 / 42.5 / 44 |
| Radeon VII | memory (HBM2) | 39 / 39.9 / 40 | 37 / 38.6 / 39 |
| CPU (Threadripper 3970X) | Tctl | 58.8 / 63.1 / 72 | 65.3 / 68.4 / 76.6 |

The RTX 3090's GDDR6X is the hottest part: after ~20 minutes of continuous decoding, which reads the VRAM all the
time, it averaged 96 °C and reached 98 °C (the core ~81 °C; the card had logged ~350 s of "SW thermal slowdown"
since the driver loaded, the cause of which nvidia-smi does not say). Micron is quoted to rate GDDR6X for up to
95 °C, and the card throttles its memory at ~110 °C. In the full benchmark (with a 260k-token prompt, whose prefill
runs the GPU at full power for minutes), the VRAM averaged 99.7 °C and reached **108 °C**, i.e. 2 °C below the
throttling point. A lower power limit, a lower memory clock or more airflow would help (see below); in normal use
the model is idle most of the time. The Radeon VII stays cool (at most ~44 °C at the junction): it computes ~10 experts in ~0.1 ms per layer and
idles in between.

### Power profiles

The settings below need root and reset at a reboot; they are host settings, so this repository does not apply them.
To make one permanent, put the command in a systemd service (`Type=oneshot`, `After=nvidia-persistenced.service` for
the RTX 3090) or, for the Radeon VII, a udev rule.

RTX 3090 (`nvidia-smi`; currently: power limit 350 W of 100-350 W, persistence mode on):

```sh
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE,TEMPERATURE   # limits, clocks and the reasons for slowdowns
sudo nvidia-smi -pm 1          # persistence mode (keeps the settings while no program uses the GPU)
sudo nvidia-smi -pl 280        # power limit in W (100-350); back to the default: sudo nvidia-smi -pl 350
sudo nvidia-smi -lgc 210,1695  # limit the core clock range (MHz); undo: sudo nvidia-smi -rgc
sudo nvidia-smi -lmc 405,9501  # limit the memory clock range (MHz; supported: 405, 810, 5001, 9501, 9751);
                               # undo: sudo nvidia-smi -rmc
```

Decode is bound by the CPU and, on the GPU, by memory bandwidth, so a lower power limit or core clock should cost
little decode speed while lowering the core's heat next to the VRAM; prefill (compute bound) would slow down more.
The memory clock is what heats the GDDR6X; one step down (9501 instead of 9751 MHz) costs ~2.5 % of the GPU's
bandwidth, i.e. well under 1 % of decode. To measure a setting:
`uv run -m agx_ai.benchmark.bench_http --url http://localhost:9931 --env-file llama-cpp/llama-cpp.env
--no-embedding --tests chat edit prompt --repeat 2 --edit-depth 0 --label "power limit 280 W"`, which now records
the temperatures too.

Radeon VII (amdgpu's sysfs, `/sys/class/drm/card1/device` = PCI `0000:4f:00.0`; check the card number with
`ls -l /sys/class/drm`):

```sh
D=/sys/class/drm/card1/device
cat $D/power_dpm_force_performance_level   # auto (current)
cat $D/pp_power_profile_mode               # the active profile has a *: COMPUTE (5) while a ROCm program runs
cat $D/pp_dpm_sclk $D/pp_dpm_mclk          # the clock levels, the current one with a *
echo high | sudo tee $D/power_dpm_force_performance_level   # always the highest clocks; back: echo auto
echo manual | sudo tee $D/power_dpm_force_performance_level
echo 5 | sudo tee $D/pp_power_profile_mode                  # COMPUTE, with "manual" (ROCm sets it anyway)
H=$(echo $D/hwmon/hwmon*)
cat $H/power1_cap                           # the power limit in µW: 250000000 (also the maximum)
echo 200000000 | sudo tee $H/power1_cap     # 200 W
```

`high` keeps the Radeon VII at its highest clocks between requests, which made the worker's latency ~15 % shorter
in the offline benchmark ([`rvii_experts/README.md`](rvii_experts/README.md#measurements)), but it is not on the
critical path (see above), and the card then idles at a higher power. Undervolting (`pp_od_clk_voltage`) would need
the kernel parameter `amdgpu.ppfeaturemask=0xffffffff` (OverDrive is off now: `0xfff7bfff`).

## Pre-gating prefetch (2026-11-24)

The prefetch of [CPU kernel and prefetch](#cpu-kernel-and-prefetch-2026-11-24) loads, while the GPU computes between
two layers, the experts that the next layer's CPU computed in the previous decode step (one per idle pool thread, 7
per layer); 42 % of them were used. Instead, the RTX 3090 now predicts the next layer's experts: after handing layer
l's experts to the CPU, while it would otherwise only wait, it applies layer l+1's router to layer l's MoE input (a
4 x 2560 by 2560 x 512 BF16 GEMV and a top-10 per token, a few µs) and writes the 40 expert ids into a pinned "hint"
slot of layer l+1, with the sequence number that layer l+1's request will have. When the CPU has answered layer l,
it prefetches the hinted CPU experts of layer l+1 (not the GPU's or the Radeon VII's), most tokens first, and falls
back to the previous step's if no hint is there (layer 0, after the previous step's last layer).
`QWEN38_CPU_EXPERTS_PREGATE=1` ([`cpu-moe-opt.patch`](cpu-moe-opt.patch): `cpu_experts.py`, `bridge_cuda.cu`,
`cpu_moe.cpp`; the model registers its routers in [`flash-next.patch`](flash-next.patch)). Which experts are loaded
into the cache does not change the result.

Measured first, offline: with `QWEN38_DECODE_XDUMP`, the CPU side dumps each decode request's MoE input and
routing; [`cpu_moe_bench/pregate_eval.py`](cpu_moe_bench/pregate_eval.py) reads the routers from the checkpoint and
compares the choices (1000 decode steps of chat and code edits; each layer's own router reproduces 98.5 % of its
routes from the dumped input, i.e. the dump is right):

| Prefetch of 7 experts per layer chosen by | Of the layer's 11.8 CPU experts, prefetched | Prefetched ones used |
|---|---|---|
| The previous step (before) | 3.53 (30 %) | 50 % |
| **Pre-gating** | **4.91 (42 %)** | **70 %** |
| Half each | 4.45 (38 %) | 64 % |
| 14 per layer: previous step / pre-gating | 4.99 / 7.17 | 36 / 51 % |

In the server (`--tests chat depth edit --repeat 2 --depth 6400 32000 --edit-depth 0`, generation in t/s; CPU time
per step and the prefetched experts that were used, from `QWEN38_CPU_EXPERTS_STATS`):

| Prefetch | Chat | Depth 6.4k | Depth 32k | Edit | CPU ms per step | Used of 336 prefetched per step |
|---|---|---|---|---|---|---|
| **Pre-gating, 1 per thread** | 54.3 | 67.7 | 69.8 | 69.8 | 38.6-39.7 | 202-207 |
| Previous step, 1 per thread (control) | 53.1 | 63.4 | 65.2 | 68.2 | 40.0-41.1 | 138-142 |
| Pre-gating, 2 per thread | 54.3 | 70.2 | 67.3 | 69.9 | 38.1-39.9 | 356-363 of 653 |
| **Pre-gating, 1 per thread (again)** | 55.8 | 64.7 | 70.0 | 69.9 | 38.4-40.0 | 199-207 |

The edits (98 % draft acceptance, the most stable test): +2.4 %; the chat and depth tests vary with the draft
acceptance (35-43 % here). In the server, 61 % of the prefetched experts are used (offline 70 %: there, the Radeon
VII's experts were taken as fixed). Two per thread were not faster, as before: the gap between the layers (~0.2 ms)
fits about one expert per thread. The GPU's extra work happens while it waits for the CPU (between the layers
13.4-13.7 ms per step either way).

The full benchmark of the production configuration (`--tests chat depth edit prompt image --repeat 2 --depth 6400
32000 100000 --edit-depth 0 32000 --prompt-tokens 260000`), against that of the morning:

| | Chat | Gen. 6.4k / 32k / 100k | Edit (short / after 32k) | Prompt 260k | Image 1280x960 | Peak VRAM |
|---|---|---|---|---|---|---|
| 2026-11-24 b: loader threads, int8 lm_head | 52.8 | 65.5 / 65.4 / 67.3 | 68.2 / 68.2 | 2366 | 8.2 s | 19598 MiB |
| **2026-11-24 c: and SCHED_FIFO, Radeon VII timeout 2 ms, pre-gating (current)** | **54.3** | **67.9 / 69.4 / 69.4** | **69.8 / 70.5** | 2376 | 8.1 s | 19600 MiB |
| Change | +3 % | +4 / +6 / +3 % | +2 / +3 % | the same | the same | the same |

The image time is from a separate run on a fresh server: in the full benchmark, the image took 12.3 s, as the serve
script then started all serving processes on `QWEN38_MAIN_CPUS` (16 cores), which took the vision tower's threads
off the SMT siblings of the pool's cores; now only the real-time threads' 16 CPUs are excluded (48 CPUs on all 32
cores). With the load of 64 busy threads, this configuration gave 45.9-46.2 t/s in chats and 64.5-64.9 in edits.

## Abliteration (2026-11-24)

`QWEN38_ABLITERATION=orcarouter` serves the model like
[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored), an
abliterated release of the same base model, without other weights: the release's single refusal direction is
projected out of every write to the residual stream at runtime. This removes the model's safety refusals; unset or
`0` keeps the original behaviour.

It is the implementation of the 2x RTX 3090 runtime
([DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/abliteration.md),
Apache-2.0): [`abliteration.py`](abliteration.py) (copied, with a docstring and lint comments added) and the
direction [`abliteration/orcarouter.json`](abliteration/orcarouter.json) (2560 values, recovered there from the
release's weights; `/opt/qwen38/abliteration` in the image), and the same hooks in `flash-next.patch` for this
runtime's `model.py`, `mtp.py` and `ple_layer.py`. With a unit direction r, the release edits every residual-writing
matrix as W' = W - r (rᵀW); for an output y = W a, that is y - r (r · y), so the runtime projects, in place, the
outputs of the attention and linear-attention blocks, the MoE block, `ple.value_proj` (before the PLE's norm) and
the text token embeddings, in the model and in the MTP head. The MoE projection comes after the CPU's and the Radeon
VII's experts have been added, so it covers them too (projecting the weighted sum of the experts' outputs equals
editing each expert's down projection), and the int4 experts in RAM and in the Radeon VII's VRAM stay as they are.
Also with the token embedding in pinned host memory (`QWEN38_EMBED_UVA`), as the projection acts on the lookup's
result. A bare name resolves to `/opt/qwen38/abliteration/<name>.json`; any other value is a path in the container
to a JSON file with a `refusal_direction` list or a safetensors file with a `refusal_direction` tensor.

The 2x RTX 3090 runtime's check, 8 mild borderline requests (a fake review, an offensive joke, an insult, lock
picking, an excuse for missing work, cheating in an exam, a villain's threat, a role-play without rules) and 2
controls, without thinking, up to 400 tokens, refusals detected by phrase and read through
([`agx_ai/benchmark/refusal_http.py`](../agx_ai/benchmark/refusal_http.py); the runs are in
[`agx-z2e-kubuntu-refusal.jsonl`](../benchmark/results/agx-z2e-kubuntu-refusal.jsonl)):

| `QWEN38_ABLITERATION` | Borderline requests refused | Controls refused | Chat t/s | Edit t/s |
|---|---|---|---|---|
| off | 7 of 8 (all but the villain's threat) | 0 of 2 | 56.3 | 74.0 |
| `orcarouter` | 0 of 8 (all answered) | 0 of 2 (the haiku identical, the capital answer reworded) | 54.9 | 75.1 |

The same 7 of 8 and 0 of 8 as the 2x RTX 3090 runtime measured. The speed is the same within the noise (each
residual write costs a 2560-wide dot product and a rank-1 update per token, in the CUDA graph). Both runs with the
RAM at DDR4-3466 (set in the BIOS on 2026-11-24; before: DDR4-2666, 70 t/s in edits).

## Radeon VII in prefill and automatic re-ranking (2026-11-25)

### The Radeon VII in prefill

In a prefill chunk of more than 384 tokens, the runtime streams every cold expert (488 per layer, 1.2 GB) from RAM to
the RTX 3090 and converts it there (`stream_v2.py`), while the Radeon VII, which holds 135 of them, was idle. The
runtime's profile of a chunk of 8192 tokens (`QWEN38_STREAM_PROFILE=1`, a 32k-token prompt) shows the copy as the
bottleneck: 3.53 s per chunk, of which the copy engine is busy 2.37 s (~24 GB/s), the GPU computes the attention and
the other layers between the MoE layers 1.5 s, the cold GEMMs 0.84 s, the conversion 0.42 s, and waits for the copies
0.65 s.

With `QWEN38_RVII_PREFILL=N` ([`rvii_prefill.py`](rvii_prefill.py), [`rvii-prefill.patch`](rvii-prefill.patch)), the
Radeon VII computes the N least used of its experts of each layer, and the RTX 3090 does not stream them. Per MoE
layer, the RTX 3090 copies x and the routing into the worker's second shared-memory segment (DMA on a side stream,
while it computes its hot experts), streams and computes the other cold experts, and before the layer's reduction
waits on the GPU (a kernel that polls host memory) for the worker's partial sum (BF16) and adds it to its FP32
accumulator. Chunks of up to 384 tokens (computed by the CPU) are unchanged. The worker's side and its kernels are in
[`rvii_experts/README.md`](rvii_experts/README.md#prefill); with all 135 experts and the routing of a real prompt,
it takes ~30 ms per layer of 8192 tokens, of which ~10 ms are the transfers of x and y.

The prompt test at 32k tokens (`--tests prompt --prompt-tokens 32000 --repeat 2`) and the profile of a chunk of 8192
tokens:

| `QWEN38_RVII_PREFILL` | Prompt t/s | s per chunk | Copy busy | Cold GEMMs + conversion | Waiting for the copies / for the Radeon VII |
|---|---|---|---|---|---|
| 0 (before) | 2195 | 3.53 | 2.37 s | 1.26 s | 0.65 / - s |
| 64 (rocBLAS on the Radeon VII, FP32 y) | 2255 | 3.42 | 2.21 s | 1.08 s | 0.44 / 0.27 s |
| 96 | 2441 | 3.16 | 1.99 s | 0.95 s | 0.43 / 0.16 s |
| **112 (current)** | **2470** | **3.13** | 1.90 s | 0.92 s | 0.36 / 0.26 s |
| 135 (all) | 2415 | 3.22 | 1.78 s | 0.88 s | 0.28 / 0.47 s |

With more experts, the RTX 3090 copies and computes less but waits longer for the Radeon VII; 112 balances them
(+12.5 %). The first version, with rocBLAS's FP16 GEMMs and the partial sums in FP32, gained only 3 %: rocBLAS reached
~7 TFLOPS at the experts' sizes, and the FP32 partial sums (84 MB per layer) took PCIe bandwidth from the stream; in
BF16 they are as precise as the RTX 3090's own path, which rounds each expert's output to BF16.

The full benchmark (as before, with abliteration; the RAM at DDR4-3466 since 2026-11-24, so the decode speeds are not
comparable with the last full benchmark at DDR4-2666) and the prompt processing of the images of 2026-11-24 with the
same RAM:

| | Chat | Gen. 6.4k / 32k / 100k | Edit (short / after 32k) | Prompt 6.4k / 32k / 100k / 260k | Image 1280x960 | Peak VRAM |
|---|---|---|---|---|---|---|
| 2026-11-24 c (DDR4-2666) | 54.3 | 67.9 / 69.4 / 69.4 | 69.8 / 70.5 | 1877 / 2251 / 2239 / 2376 | 8.1 s | 19600 MiB |
| Before, DDR4-3466 (the images of 2026-11-24; the prompts at 32k and 260k with `QWEN38_RVII_PREFILL=0`) | 60.4 | 72.4 / - / - | 81.0 / - | 1840 / 2195 / - / 2376 | | |
| **2026-11-25 (current)** | **61.7** | **73.8 / 71.2 / 73.9** | **82.0 / 82.2** | **2105 / 2545 / 2557 / 2639** | 7.5 s | 19984 MiB |

The decode paths did not change: an A/B with chat, depth 6.4k and the short edit gave 60.4 / 72.4 / 81.0 t/s with the
images of 2026-11-24 and 60.7 / 70.8 / 81.5 with the new ones, so the faster decode comes from the RAM (the earlier
checks after the overclock, 56.3 in chats and 74-75 in edits, were lower, perhaps from other load). The prompt
processing gained +11-29 % (the 6.4k prompt of the A/B: 1840 vs 2380 t/s; 260k: 2376 vs 2639; the 32k values of the
table are from different tests, the prompt test and the depth test). The RTX 3090's peak VRAM is ~380 MiB higher
(not investigated; the side stream's copies have their own allocations), the Radeon VII's ~190 MiB (the prefill
buffers, 168 MB); both are within their margins (21.5 of 24 GiB with the desktop, 15.9 of 16 GiB).

Quality: the retrieval test without MTP (88 questions at 8k and 32k tokens, as for the int8 lm_head), against
`QWEN38_RVII_PREFILL=0`:

| Run | Correct | Same answers | Mean KL divergence | 99 % | First token | Top-token flips | Test time |
|---|---|---|---|---|---|---|---|
| `QWEN38_RVII_PREFILL=0` (reference) | 83/88 | | | | | | 265 s |
| **112** | 83/88 | 86/88 | 0.00184 | 0.049 | 0.0109 | 2 of 571 | 214 s |
| 0 again (noise floor) | 82/88 | 85/88 | 0.00271 | 0.077 | 0.0168 | 3 of 567 | 264 s |

The divergence is below the noise floor: the worker's FP16 activations are at least as precise as the RTX 3090's
BF16 ones. (The test time, mostly the prefill of the 8k- and 32k-token prompts, is 19 % shorter.)

### Automatic re-ranking

The experts of the RTX 3090 (24 per layer) and of the Radeon VII (135) were chosen from the usage measured on
2026-11-23 and fixed. Now (`RVII_AUTO_RERANK=1` in the `rvii-experts` service, details in
[`rvii_experts/README.md`](rvii_experts/README.md#automatic-re-ranking)):

- the runtime counts the decode requests of each expert all the time (`QWEN38_EXPERT_USAGE=/dev/shm/qwen38-usage.bin`,
  the counter that [`docker-compose.usage.yml`](docker-compose.usage.yml) enabled for collections; no measurable
  cost) and publishes its hot experts (`/dev/shm/qwen38-hot.bin`);
- [`rvii_experts/rerank.py`](rvii_experts/rerank.py), which now runs the worker, accumulates the counts with a
  72-hour half-life and every 10 min ranks the experts by their recent use (mixed with the general collection at
  20 %). When another set of 135 experts per layer would serve at least 2 % more of the decode requests, it packs
  them in the background and restarts the worker on them once the runtime has been idle for 2 min (2-20 s, while
  the CPU computes the worker's experts);
- the runtime takes its 24 hot experts from the re-ranked ranking at its next start (`QWEN38_RANKINGS_AUTO`, written
  by `rerank.py` into `~/.cache/vllm-models/flash-next-rvii-measured/hot24-n135-auto/rankings.json`; until it
  exists, `rankings-measured.json` as before), and the worker re-ranks for the new hot set.

The first layout is that of `rankings-measured.json`, so nothing changes until the use differs. With the measured mix,
the gain is none; with use far from it (other languages or kinds of text), up to ~10 % in decode was estimated (the
measured ranking gained 24-28 % over the static one in code). `RVII_AUTO_RERANK=0` keeps the fixed placement.

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
