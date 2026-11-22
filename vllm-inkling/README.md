# vLLM for Inkling 2 Mini

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, 2026-11-20) with vLLM on
the RTX 3090 (sm_86, 24 GB) of agx-z2e, on the same port as [`../llama-cpp-rtx3090`](../llama-cpp-rtx3090) and
[`../vllm-rtx3090`](../vllm-rtx3090) (9931). The model is a 26B MoE (128 experts, 8 active) whose middle 12 of its 24
layers run three times per token with shared weights; the checkpoint has NVFP4 experts (MXFP8 in the last 4 layers),
everything else BF16, and a DSpark block drafter (`mtp.safetensors`).

vLLM supports the model since [vllm-project/vllm#69028](https://github.com/vllm-project/vllm/pull/69028) (merged
2026-11-21, first in the nightly `647a768` of 2026-11-22), but only for Hopper, Blackwell and DGX Spark: its
quantization config requires SM90. The [`Dockerfile`](Dockerfile) applies [`inkling-sm86.patch`](inkling-sm86.patch)
to that nightly:

- The minimum capability of the inkling2 quantization config is 80 instead of 90.
- Below SM90, the model takes the fallbacks that upstream has for SM12x (consumer Blackwell), which need no Hopper
  features: Triton short convolution and AttnRes instead of the inkling2 CUDA kernels (the wheel builds them for
  9.0a/10.0f/10.3f only, so on sm_86 the ops exist but have no kernel image), no TMA in the k-group linear kernels,
  SM12x's shared-memory limits (~99 KiB per block, as on sm_86), and FlashAttention 2 for the vision tower. The routed
  experts stay on the Hopper path (Humming: NVFP4/MXFP8 weights with BF16 activations), as the SM12x experts use
  native FP4 MMA.
- Below SM90, Triton's programmatic dependent launch is disabled (`gdc_wait` and `gdc_launch_dependents` compile to
  nothing, `launch_pdl` is dropped): ptxas rejects the `griddepcontrol` PTX for sm_86.

It then applies [`inkling-cpu-vision.patch`](inkling-cpu-vision.patch), which runs the vision encoder on the CPU with
`INKLING2_VISION_DEVICE=cpu` ([below](#cpu-vision-encoder-2026-11-22)). Other GPUs take the same paths as upstream. [`../vllm/vllm.patch`](../vllm/vllm.patch) is not applied: none of its
fixes concern this model. Built by
[`../.github/workflows/deploy-docker-vllm-inkling.yml`](../.github/workflows/deploy-docker-vllm-inkling.yml);
locally: `docker build -t ghcr.io/agenttix/agx-ai/vllm-inkling:latest .`. The configuration is selected in
[`docker-compose.yml`](docker-compose.yml). Greedy output starts with the same text as llama.cpp's.

| Configuration | Context | Chat | Code edits | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|
| [`config-inkling-2-mini.yaml`](config-inkling-2-mini.yaml) (default) | 49152 | 94.3 t/s | 89.7 t/s | 4505 t/s for 23k tokens, ~3700 t/s for 43k | 21675 MiB |
| [`config-inkling-2-mini-dspark.yaml`](config-inkling-2-mini-dspark.yaml), DSpark drafter (7 tokens) | 16384 | 116.2 t/s | 295.2 t/s | ~4300 t/s | 21712 MiB |

Both take images, with the vision encoder on the CPU ([CPU vision encoder](#cpu-vision-encoder-2026-11-22)), which
takes no VRAM; on the GPU, it would take 0.6 GiB (~19k tokens of context). vLLM's model takes no videos; llama.cpp
does.

**Compared with llama.cpp on this GPU** ([`../llama-cpp-rtx3090`](../llama-cpp-rtx3090/README.md#inkling-2-mini),
`UD-Q4_K_XL` with Unsloth's DSpark drafter, 118784 tokens of context): vLLM generates 1.8-2.3 times as fast (chat
94-116 vs 51 t/s; edits 295 vs 80 t/s with the drafters) and processes prompts ~9 times as fast (4500 vs 495 t/s),
but fits only 41 % of the context (49152) without its drafter and 14 % with it (16384). Its weights are larger
(18.3 GiB: BF16 embedding, unembedding and attention) and its KV cache is BF16 only below SM89.

## Memory

- The model takes 18.3 GiB, which leaves ~2.9 GB for the KV cache and activations within the ~21.2 GiB of the
  other configurations (the desktop uses ~1.3 GiB of the 24 GiB).
- KV cache: ~32 KiB per token for the 8 full-attention caches (layers 5, 11 and 17 have one per loop pass, so 3 + 3 +
  1 + 1), plus a fixed part for the 40 sliding-window caches (window 1024) and the 24 short-conv states. The fixed
  part grows with the prefill chunk, as each of these caches reserves its window plus a chunk: with
  `max-num-batched-tokens: 2048` (the recipe's default), even 8192 tokens of context needed 4.06 GiB; with 512,
  1.4 GiB. 512 still processes prompts at ~3700-4500 t/s.
- The KV cache is BF16: the model's FP8 KV cache needs SM89+ (E4M3 conversion instructions and Triton's `fp8e4nv`).
- DSpark accepts only 7 drafted tokens (the checkpoint's block size - 1) and takes 0.5 GiB more for its caches.
- Experts in RAM (UVA) would free context, but the model builds its layers without vLLM's offloader, and with the
  drafter, each step verifies 8 tokens, which select most of the experts of a layer: they would cross PCIe in full.

## CPU vision encoder (2026-11-22)

The vision encoder of the model (a ViT with 23 layers of width 1024 and a 2x2 merge projection, 0.6 GiB in BF16) runs
its token embedding, 2D RoPE and fused residual norms as Triton kernels (`ops/vision.py`) and its attention with
FlashAttention, all GPU-only. [`inkling-cpu-vision.patch`](inkling-cpu-vision.patch) runs it on the CPU with
`INKLING2_VISION_DEVICE=cpu` (threads: `INKLING2_VISION_THREADS`, set in [`docker-compose.yml`](docker-compose.yml)):

- The model creates the encoder on the CPU and converts it to FP32 after loading its weights (the Threadripper has no
  fast BF16 matrix units).
- Each Triton op has a PyTorch version that it takes for CPU tensors, with the same BF16 rounding points; the
  attention is PyTorch's scaled dot-product attention per image.
- The encoder moves the pixels to the CPU and its output back to the GPU in BF16.

Check against the GPU encoder (both with the checkpoint's weights, two images of 192 and 30 patches): cosine
similarity of the output embeddings ≥ 0.99993 (mean 0.99997), relative error 0.8 %. A 1280x960 image (794 tokens)
takes 4.0 s until the first token (llama.cpp with its encoder on the CPU: 3.8 s), and a generated test image with a
red square, a blue rectangle and the word "ALPHA" was described correctly. Chat speed and peak VRAM are unchanged
(96.4 t/s, 21684 MiB; with the drafter 117.0 t/s and 296.4 t/s in edits, 21712 MiB).

## Radeon VII

Not feasible: the checkpoint is NVFP4/MXFP8 with kernels for NVIDIA GPUs only (Humming, FlashInfer, Triton with
NVIDIA PTX), the vLLM image for gfx906 ([`../vllm-radeon-vii`](../vllm-radeon-vii)) is v0.30 and has no Inkling
model, and the weights (18.3 GiB) do not fit in its 16 GB. llama.cpp runs the model on that GPU
([`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii/README.md#inkling-2-mini-2026-11-22)).

## Results (2026-11-22)

`bench_http.py`, one request at a time:

| Test | `config-inkling-2-mini.yaml` | `config-inkling-2-mini-dspark.yaml` |
|---|---|---|
| Generation, 256 tokens | 95.4 t/s | 95.3 t/s (acceptance 14 %) |
| Chat, 6 requests | 94.3 t/s (93.8-95.3) | 116.2 t/s (93.5-142.6), acceptance 20 % |
| Edit after 2.8k tokens | 89.7 t/s, prompt 2799 t/s | 295.2 t/s, prompt 4307 t/s, acceptance 80 % |
| Prompt of 23000 tokens | 4505 t/s (24576-token configuration) | |
| Prompt of 43071 tokens | 11.7 s for the request (~3700 t/s) | |
| Peak VRAM of vLLM | 20858 MiB (KV cache 2.0 GB), 21675 MiB (2.85 GB) | 21712 MiB |

The depth test of `bench_http.py` could not run: its prompt builder tokenizes more text than vLLM's `/tokenize`
accepts for this model (131072 tokens).

## Commands

```sh
cd ../benchmark
# A configuration variant on a test server (port 9934; stop the production container first)
CACHE_VOLUME=vllm-inkling_vllm-cache DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    uv run -m agx_ai.benchmark.vllm.test_server ghcr.io/agenttix/agx-ai/vllm-inkling:latest \
    ../vllm-inkling/config-inkling-2-mini.yaml ../llama-cpp/llama-cpp.env 9934
uv run -m agx_ai.benchmark.bench_http --url http://localhost:9934 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
    --tests generation chat edit --repeat 1 --server-config ../vllm-inkling/config-inkling-2-mini.yaml --label "..."
docker rm -f vllm-test
```
