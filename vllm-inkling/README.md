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
- Below SM90, the bf16 projections with at most 16 rows (decoding) take the Triton low-M kernels of SM12x
  (`ll_gemm_sm120.py`) instead of cuBLAS: on the RTX 3090, they read the weights at 790-840 GB/s, cuBLAS at 620-800
  GB/s ([below](#kernels-and-int8-weights-2026-11-22)).

It then applies [`inkling-cpu-vision.patch`](inkling-cpu-vision.patch), which runs the vision encoder on the CPU with
`INKLING2_VISION_DEVICE=cpu` ([below](#cpu-vision-encoder-2026-11-22)), and
[`inkling-int8-cpu-experts.patch`](inkling-int8-cpu-experts.patch), with two options, which the configurations set in
`additional-config` (part of vLLM's compile cache key; else from the environment variables of the same names in upper
case):

- `inkling2_int8_weights`: the BF16 attention projections, LM head and token embedding in int8
  ([below](#kernels-and-int8-weights-2026-11-22)),
- `inkling2_cpu_expert_layers`: the routed experts of these layers in RAM ([below](#experts-in-ram-2026-11-22)).

Other GPUs take the same paths as upstream unless these options are set. [`../vllm/vllm.patch`](../vllm/vllm.patch)
is not applied: none of its fixes concern this model. Built by
[`../.github/workflows/deploy-docker-vllm-inkling.yml`](../.github/workflows/deploy-docker-vllm-inkling.yml);
locally: `docker build -t ghcr.io/agenttix/agx-ai/vllm-inkling:latest .`. The configuration is selected in
[`docker-compose.yml`](docker-compose.yml). Greedy output starts with the same text as llama.cpp's.

| Configuration | Context | Chat | Code edits | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|
| [`config-inkling-2-mini.yaml`](config-inkling-2-mini.yaml) (default) | 98304 | 128.8 t/s, 86.0 t/s after 95k tokens | 120.2 t/s | 4249 t/s for 2.8k tokens, 3125 t/s for 95k | 21726 MiB |
| [`config-inkling-2-mini-dspark.yaml`](config-inkling-2-mini-dspark.yaml), DSpark drafter (7 tokens) | 57344 | 156.6 t/s | 335.8 t/s | 4248 t/s for 2.8k tokens, 3618 t/s for 53k | 21728 MiB |
| [`config-inkling-2-mini-131k.yaml`](config-inkling-2-mini-131k.yaml), experts of 2 layers in RAM | 131072 | 95.8 t/s, 64.5 t/s after 127k tokens | 91.1 t/s | 2722 t/s for 2.8k tokens, 2049 t/s for 127k | 21706 MiB |
| [`config-inkling-2-mini-262k.yaml`](config-inkling-2-mini-262k.yaml), experts of 9 layers in RAM | 262144 | 48.3 t/s, 32.7 t/s after 259k tokens | 47.3 t/s | 1111 t/s for 2.8k tokens, 885 t/s for 259k | 21506 MiB |

All take images, with the vision encoder on the CPU ([CPU vision encoder](#cpu-vision-encoder-2026-11-22)), which
takes no VRAM; on the GPU, it would take 0.6 GiB (~19k tokens of context). vLLM's model takes no videos; llama.cpp
does. Before the int8 weights (2026-11-22, the first version), the default configuration had 49152 tokens at 94.3 t/s
and the drafter one 16384 tokens at 116.2 t/s (edits 295.2 t/s).

**Compared with llama.cpp on this GPU** ([`../llama-cpp-rtx3090`](../llama-cpp-rtx3090/README.md#inkling-2-mini),
`UD-Q4_K_XL` with Unsloth's DSpark drafter, 118784 tokens of context: chat 51 t/s, 31 t/s at 100k, edits 80 t/s,
prompts 495-219 t/s): vLLM generates 2.5 times as fast with 98304 tokens (129 t/s; with its drafter 157 t/s in chats
and 336 t/s in edits with 57344 tokens), 1.9 times as fast with 131072 tokens, and about as fast with 262144 tokens
(48.3 t/s, 33 t/s after 259k tokens), and processes prompts 2-8 times as fast. The weights of the configurations
are the checkpoint's (NVFP4/MXFP8 experts) with int8 attention projections, LM head and embedding (16.1 GiB on the
GPU), and the KV cache is BF16 below SM89.

## Kernels and int8 weights (2026-11-22)

A profile of decoding (PyTorch profiler, `profiler-config` with `/start_profile`) with the first version of the
default configuration: the GPU was busy 97 % of each 10.3 ms step (95.5 t/s), so the kernels limit the speed, not
the CPU. Per step:

| Kernels | Time per step | Reads at |
|---|---|---|
| Routed experts (Humming, NVFP4/MXFP8, 48 layer visits) | 2.45 ms | 680-815 GB/s |
| Attention projections `qkvr` (cuBLAS gemv, 48 visits; loop layers after their first pass project k and v from other inputs: 3 launches) | 2.65 ms | 480-810 GB/s |
| LM head (cuBLAS, 926 MB) | 1.05 ms | 880 GB/s |
| Attention output `wo_ud` (cuBLAS gemm, 48 visits) | 1.03 ms | 655 GB/s |
| ~1000 small kernels (attention, short convolutions, AttnRes, norms, routing) | ~3 ms | |

The routed experts already run near the bandwidth of the GPU (936 GB/s), and the small kernels would need fused
kernels. The bf16 weights outside the experts (attention, LM head) take 45 % of the step:

- The low-M Triton kernels of SM12x read them faster than cuBLAS (`wo_ud` 18.0 vs 22.9 us, the 3 projections of a loop
  layer 47.8 vs 60.3 us in one launch): 102.9 t/s instead of 95.5 (+8 %), the same perplexity (prefill still uses
  cuBLAS). [`inkling-sm86.patch`](inkling-sm86.patch) enables them below SM90.
- In int8 (`inkling2_int8_weights`, `ops/w8_linear.py` of the patch), they take half the memory and half the reads:
  symmetric int8 with one fp32 scale per output row (`amax / 127`), BF16 activations and fp32 accumulation. Batches of
  up to 16 rows run split-K Triton kernels derived from the low-M kernels, which convert the weights to BF16 in
  registers (`qkvr` 25.9 vs 47.2 us, the 3 projections of a loop layer 26.1 vs 47.8 us in one launch, `wo_ud` 11.3 vs
  18.0 us, the LM head 528 vs 1049 us: 765-877 GB/s of int8); larger batches (prompts) dequantize the weights in row
  chunks for cuBLAS. The embedding table is int8 too (dequantized per token). Decoding: 126-129 t/s instead of 95.5
  (+35 %), 1.46 GiB less VRAM (the weights on the GPU: 16.14 instead of 17.6 GiB), which the default configuration
  spends on 98304 instead of 49152 tokens of context.

Perplexity of the model (`prompt_logprobs`, 12837 tokens: the GPL-3 and Apache-2.0 license texts, `bench_http.py` and
`../vllm/README.md`): 2.2851 with the BF16 weights (cuBLAS or the low-M kernels), 2.2896 with int8 (+0.2 %); the
greedy output of a story prompt is the same for its first ~30 tokens. The DSpark drafter shares the int8 LM head and
embedding: 73 % of its drafts are accepted in code edits (BF16: 80 %), and edits are still faster (336 vs 295 t/s).
FP8 weights would be as small, but the RTX 3090 has no FP8 conversion instructions (Triton's `fp8e4nv` needs SM89).

## Experts in RAM (2026-11-22)

With `inkling2_cpu_expert_layers`, the routed experts of the given layers (their Humming weights and block scales)
move to pinned host memory after loading, and the expert kernels read them over PCIe 4.0 x16 through unified virtual
addressing, as vLLM's UVA offloader does (the model builds its layers without vLLM's offloader). A decoding step reads
only the 8 experts of each token: 36 MB per NVFP4 layer (0.53 GiB in RAM), 65 MB per MXFP8 layer (0.98 GiB), so each
NVFP4 layer costs ~1.3 ms per token (26-28 GB/s, the speed of the link; one layer: 126 -> 108 t/s). Layers 6-17 run 3
times per token, so the configurations take layers outside the loop (0-5, 18-23). Prompts read all experts of these
layers once per chunk of 512 tokens (`max-num-batched-tokens`; a larger chunk would take more KV cache).

- 131072 tokens: layers 0 and 1 (1.06 GiB), 96 t/s.
- 262144 tokens: layers 0-5, 18 and 19 (NVFP4) and 20 (MXFP8), 5.2 GiB, 48 t/s. 9.1 GiB of KV cache.

Computing the experts on the CPU (as llama.cpp's `--n-cpu-moe`) would read them from RAM at ~80 GB/s instead of
~26 GB/s over PCIe, but vLLM has no CPU path for NVFP4 experts, and each layer would wait for the CPU in the middle of
the CUDA graph. With the drafter, each step verifies 8 tokens, which select most of the experts of a layer: the
drafter configuration keeps all experts on the GPU.

## Memory

- The weights take 16.14 GiB with `inkling2_int8_weights` (18.3 GiB in BF16), which leaves ~4.4 GB for the KV cache
  and activations within the ~21.2 GiB of the other configurations (the desktop uses ~1.3 GiB of the 24 GiB).
- KV cache: ~31 KiB per token for the 8 full-attention caches (layers 5, 11 and 17 have one per loop pass, so 3 + 3 +
  1 + 1), plus a fixed part of ~1.15 GiB for the 40 sliding-window caches (window 1024) and the 24 short-conv states.
  The fixed part grows with the prefill chunk, as each of these caches reserves its window plus a chunk: with
  `max-num-batched-tokens: 2048` (the recipe's default), even 8192 tokens of context needed 4.06 GiB; with 512,
  1.4 GiB. 512 still processes prompts at ~3000-4300 t/s.
- The KV cache is BF16: the model's FP8 KV cache needs SM89+ (E4M3 conversion instructions and Triton's `fp8e4nv`).
- DSpark accepts only 7 drafted tokens (the checkpoint's block size - 1) and takes 0.7 GiB more for its weights and
  caches.

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

`bench_http.py`, one request at a time, with the image of the patches above; the long prompts are source files of this
repository (the depth test of `bench_http.py` cannot run: its prompt builder tokenizes more text than vLLM's
`/tokenize` accepts, twice `max-model-len`). Generation after a long prompt is measured with the prompt in the prefix
cache.

| Test | default (98304) | DSpark (57344) | 131072, 2 layers in RAM | 262144, 9 layers in RAM |
|---|---|---|---|---|
| Generation, 256 tokens | 128.7 t/s | 119.9 t/s (acceptance 14 %) | 96.1 t/s | 48.2 t/s |
| Chat, 6 requests | 128.8 t/s (128.6-129.2) | 156.6 t/s (111.3-225.7), acceptance 23 % | 95.8 t/s (95.3-96.3) | 48.3 t/s (48.2-48.3) |
| Edit after 2.8k tokens | 120.2 t/s, prompt 4249 t/s | 335.8 t/s, prompt 4248 t/s, acceptance 73 % | 91.1 t/s, prompt 2722 t/s | 47.3 t/s, prompt 1111 t/s |
| Long prompt | 95157 tokens: 30.5 s (3125 t/s) | 52588 tokens: 14.5 s (3618 t/s) | 126506 tokens: 61.7 s (2049 t/s) | 258544 tokens: 292 s (885 t/s) |
| Generation after it | 86.0 t/s | | 64.5 t/s | 32.7 t/s |
| Image 1280x960 (794 tokens), 16 CPU threads | 4.1 s | | | |
| Peak VRAM of vLLM | 21726 MiB | 21728 MiB | 21706 MiB | 21506 MiB |

The first version (BF16 weights, cuBLAS): the default configuration with 49152 tokens generated 95.4 t/s (chat 94.3,
edits 89.7 t/s, a 43071-token prompt in 11.7 s), the drafter one with 16384 tokens 116.2 t/s in chat and 295.2 t/s in
edits (acceptance 80 %); both peaked at ~21.7 GiB.

## Commands

```sh
cd ../benchmark
# A configuration variant on a test server (port 9934; stop the production container first)
CACHE_VOLUME=vllm-inkling_vllm-cache \
    DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e INKLING2_VISION_DEVICE=cpu -e INKLING2_VISION_THREADS=16" \
    uv run -m agx_ai.benchmark.vllm.test_server ghcr.io/agenttix/agx-ai/vllm-inkling:latest \
    ../vllm-inkling/config-inkling-2-mini.yaml ../llama-cpp/llama-cpp.env 9934
uv run -m agx_ai.benchmark.bench_http --url http://localhost:9934 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
    --tests generation chat edit --repeat 1 --server-config ../vllm-inkling/config-inkling-2-mini.yaml --label "..."
docker rm -f vllm-test
# A decoding profile (add to a configuration variant:
# profiler-config: {"profiler": "torch", "torch_profiler_dir": "/prof", "torch_profiler_with_stack": false}
# and mount a directory with DOCKER_ARGS="... -v /tmp/prof:/prof"), then:
curl -X POST -H "Authorization: Bearer $LLAMA_API_KEY" http://localhost:9934/start_profile  # then a request
curl -X POST -H "Authorization: Bearer $LLAMA_API_KEY" http://localhost:9934/stop_profile
```
