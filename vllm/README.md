# vLLM image

`ghcr.io/agenttix/agx-ai/vllm`: the official vLLM image for NVIDIA GPUs with a few fixes, not specific to any
machine. Built by [`../.github/workflows/deploy-docker-vllm.yml`](../.github/workflows/deploy-docker-vllm.yml);
locally:

```sh
docker build -t ghcr.io/agenttix/agx-ai/vllm:latest .
```

Used by [`../vllm-big-machine`](../vllm-big-machine). For gfx906 (Radeon VII), see
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
