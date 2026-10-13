# vLLM image for Qwen3.8-Flash-Next with image input

`ghcr.io/agenttix/agx-ai/vllm-flash-next`: the single-RTX-3090 runtime of Qwen3.8-Flash-Next by
[DominikBucko/qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090) (its
[v0.1.0 release image](https://github.com/DominikBucko/qwen38-flash-next-3090/releases/tag/v0.1.0), pinned by digest),
with image input added. Built by
[`../.github/workflows/deploy-docker-vllm-flash-next.yml`](../.github/workflows/deploy-docker-vllm-flash-next.yml);
locally:

```sh
docker build -t ghcr.io/agenttix/agx-ai/vllm-flash-next:latest .
```

Used by [`../vllm-agx-z2e`](../vllm-agx-z2e/README.md#qwen38-flash-next). The runtime serves the model text-only
(`--language-model-only`), although its model code supports images. This image adds:

- [`cpu_vision.py`](cpu_vision.py): the vision tower (Qwen3-VL architecture, 27 layers, 0.84 GiB in BF16) on the CPU,
  with transformers' `Qwen3VLVisionModel` in FP32. Only the image embeddings are copied to the GPU, so the vision
  tower takes no VRAM, and the text-only speed is unchanged. Selected with `QWEN38_VISION_DEVICE=cpu`, which the serve
  script sets; with `QWEN38_VISION_DEVICE=gpu`, vLLM's own vision transformer is used.
- [`flash-next.patch`](flash-next.patch):
  - the model creates the CPU vision tower instead of vLLM's,
  - the serve script enables image input (`QWEN38_VISION=1`, the default; `0` = text only as before), with at most
    `QWEN38_VISION_MAX_IMAGES` images per request (default 4), images scaled down to
    `QWEN38_VISION_MAX_PIXELS` pixels (default 1048576), no video, and without the encoder profiling (a dummy image
    of the maximum size would take minutes on the CPU),
  - and it sets `gpu-memory-utilization` (`GPU_MEMORY_UTILIZATION`, default 0.8): vLLM checks at startup that this
    share of the GPU is free, and the runtime's default of 0.92 fails next to a desktop. The KV cache size is set
    explicitly anyway.

## Measurements (2026-10-13, agx-z2e)

RTX 3090 and Threadripper 3970X (32 cores, Zen 2), with the settings of
[`../vllm-agx-z2e/docker-compose.yml`](../vllm-agx-z2e/docker-compose.yml):

- **Vision tower on the GPU** (vLLM's own, the runtime without `--language-model-only`): 0.84 GiB of weights, which
  needed 12 instead of 20 hot experts per layer. The first 1280x960 image then ran out of GPU memory (the encoder's
  activations needed another ~1 GiB). Making room for both (~1.9 GiB) would cost the text path 16 of its 20 hot
  experts per layer, or two thirds of the KV cache.
- **Vision tower on the CPU** (this image): the same VRAM as text-only (14.09 GiB after loading, peak 20804 MiB), and
  the same text speed (chat 27.9 t/s; 29.3 and 27.8 t/s at 6.4k and 32k tokens of context). A 1280x960 image, scaled
  to 1 megapixel (~980 image tokens), takes ~8 s until the first token. It read the text and the shapes, colors and
  positions of a test image correctly.
- The vision tower alone on the CPU, FP32, 1280x960 without scaling (4800 patches): 9.7 s with 16 threads, 6.5 s with
  32 and 9.2 s with 64 (SMT). In BF16, it is 3.5 times as slow (no BF16 instructions on Zen 2). The CPU expert pool is
  idle while the encoder runs, as the runtime serves one request at a time.
- transformers' vision model loads all 333 vision tensors of the checkpoint with no missing or unexpected keys.
