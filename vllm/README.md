# vLLM image

`ghcr.io/agenttix/agx-ai/vllm`: the official vLLM image for NVIDIA GPUs with a few fixes, not specific to any
machine, for x86-64 and ARM64 (the patch applies to both variants of the base image, checked on 2026-10-11). Built by [`../.github/workflows/deploy-docker-vllm.yml`](../.github/workflows/deploy-docker-vllm.yml);
locally:

```sh
docker build -t ghcr.io/agenttix/agx-ai/vllm:latest .
```

Used by [`../vllm-big-machine`](../vllm-big-machine) and [`../vllm-agx-z2e`](../vllm-agx-z2e). For gfx906 (Radeon VII), see
[`../vllm-radeon-vii`](../vllm-radeon-vii) instead.

- `vllm/vllm-openai:latest` is v0.31.0 with CUDA 13.0 and PyTorch 2.13 (the `-cu129` tags are the CUDA 12.9
  builds). It is pinned as `v0.31.0` in the [`Dockerfile`](Dockerfile).
- [`vllm.patch`](vllm.patch), applied by the [`Dockerfile`](Dockerfile):
  - Qwen3.5/3.8 passes its quantization config to the token embedding (main model and MTP module), so that
    checkpoints with a quantized embedding load. Without it, e.g. JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB fails with
    `There is no module or parameter named 'embed_tokens.weight_packed' in Qwen3_5Model`. The same fix as in
    [`../vllm-radeon-vii`](../vllm-radeon-vii) and the checkpoint's model card.
  - The native CPU KV offloading is allowed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which vLLM
    rejects for all KV connectors, because RDMA connectors register GPU memory that the VMM allocator could remap.
    The native offloading copies with device pointers, and the KV cache is allocated once and never freed. The
    alternative, `enable-cumem-allocator`, works on CUDA but needs more memory: the single-GPU Gemma 4 26B-A4B of
    [`../vllm-big-machine`](../vllm-big-machine) ran out of it.
  - Triton attention (`triton_unified_attention.py`), added on 2026-10-12 and measured on the RTX 3090 of
    [`../vllm-agx-z2e`](../vllm-agx-z2e) with Qwen3.8-27B, an int8 KV cache and MTP:
    - The split-KV ("3D") kernel is also used for the verification batches of speculative decoding (up to 8 query
      tokens per sequence), not only for single-token decoding. Without it, MTP decoding ran one program per KV head
      and sequence over the whole context and slowed down with it: 58.6 t/s at 6.4k and 23.7 t/s at 32k tokens of
      context, with it 83.5 and 64.0 t/s. The kernel and the reduction already handle several query tokens per
      sequence; the patch only checks that the segment buffers (one row per token) are large enough.
    - On sm_86, prefill with head size 256 uses 64 query rows per block and a KV tile of 16 (upstream: 16 and 32,
      with a tuned variant only for Blackwell): 890 instead of 743 t/s for a 32k prompt, 582 instead of 395 t/s for
      96k (tested: block sizes 16-128, tiles 16-128, 4 or 8 warps).
    - Checked with [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py): the
      long-context retrieval answers (88 questions at 17k and 61k tokens) are identical to the unpatched image.
      The greedy outputs diverge after 35 % of their length on average, as the floating-point sums are reordered;
      changing only the prefill chunk size of the unpatched image (2048 -> 1024) gives a similar divergence (49 %).
