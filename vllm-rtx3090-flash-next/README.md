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
    unquantized LM head"); the RTX 3070 runs without MTP and was not affected.
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

