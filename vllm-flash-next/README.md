# vLLM image for Qwen3.8-Flash-Next with image and video input

`ghcr.io/agenttix/agx-ai/vllm-flash-next`: the single-RTX-3090 runtime of Qwen3.8-Flash-Next by
[DominikBucko/qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090) (its
[v0.1.0 release image](https://github.com/DominikBucko/qwen38-flash-next-3090/releases/tag/v0.1.0), pinned by digest),
with image and video input added. Built by
[`../.github/workflows/deploy-docker-vllm-flash-next.yml`](../.github/workflows/deploy-docker-vllm-flash-next.yml);
locally:

```sh
docker build -t ghcr.io/agenttix/agx-ai/vllm-flash-next:latest .
```

Used by [`../vllm-agx-z2e`](../vllm-agx-z2e/README.md#qwen38-flash-next). The runtime serves the model text-only
(`--language-model-only`), although its model code supports images and videos. This image adds:

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
  - and it sets `gpu-memory-utilization` (`GPU_MEMORY_UTILIZATION`, default 0.8): vLLM checks at startup that this
    share of the GPU is free, and the runtime's default of 0.92 fails next to a desktop. The KV cache size is set
    explicitly anyway.

The CPU vision tower takes 1.7 GiB of RAM, and a video up to ~1.5 GiB more while it is encoded, so the container
needs a few GiB more than the expert arena (see the `mem_limit` in
[`../vllm-agx-z2e/docker-compose.yml`](../vllm-agx-z2e/docker-compose.yml)). transformers logs once at startup that
`video_max_pixels` "is not a valid argument for this processor and will be ignored"; the patch removes it before the
processor calls, and the budget is applied (see the measurements).

## Measurements (2026-10-13, agx-z2e)

RTX 3090 and Threadripper 3970X (32 cores, Zen 2), with the settings of
[`../vllm-agx-z2e/docker-compose.yml`](../vllm-agx-z2e/docker-compose.yml):

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
