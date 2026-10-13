# vLLM image

`ghcr.io/agenttix/agx-ai/vllm`: the official vLLM image for NVIDIA GPUs with a few fixes, not specific to any
machine, for x86-64 and ARM64 (the patch applies to both variants of the base image, checked on 2026-10-11). Built by [`../.github/workflows/deploy-docker-vllm.yml`](../.github/workflows/deploy-docker-vllm.yml);
locally:

```sh
docker build -t ghcr.io/agenttix/agx-ai/vllm:latest .
```

Used by [`../vllm-big-machine`](../vllm-big-machine), [`../vllm-agx-z2e`](../vllm-agx-z2e) and
[`../vllm-agx-ai`](../vllm-agx-ai). For gfx906 (Radeon VII), see
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
  - UVA weight offloading (`offload-backend: uva` with `cpu-offload-params`), added on 2026-10-13 for
    Qwen3.8-Flash-Next with its experts in RAM on the 8 GB RTX 3070 of [`../vllm-agx-ai`](../vllm-agx-ai):
    - Pinned host memory is allocated with exactly its size (`empty_pinned` and `pin_memory_exact` in
      `utils/torch_utils.py`: an anonymous memory mapping registered with `cudaHostRegister`, unregistered when it
      is freed). `pin_memory=True` uses PyTorch's caching host allocator, which rounds every block up to the next
      power of two: the 47.6 GiB n-gram table of Qwen3.8-Flash-Next took 64 GiB and its 838 MB expert tensors 1 GiB
      each, which exceeded the 128 GiB of RAM. Used by the UVA offloader, by the re-offloading of weights after
      their post-processing (`model_loader/utils.py`) and by the pinned n-gram table of Qwen4Exp.
    - The offloader no longer skips a module whose first parameter is already on the CPU: the decoder layer that
      holds the pinned n-gram table kept its experts (1.25 GiB) on the GPU.
    - Qwen4Exp offloads its token embedding too (`cpu-offload-params: embed_tokens`; it is created outside
      `make_layers`, which offloads only the decoder layers), and passes its quantization config to the lm_head,
      so that a quantized lm_head loads (checkpoints without one stay unquantized).
    - The Marlin MoE post-processing of GPTQ/AutoRound experts no longer transposes the weights to N-first copies
      and back: with UVA, each copy was new GPU memory (1.2 GiB per layer for Qwen3.8-Flash-Next). The w13 and w2
      weights are now transposed and repacked one after the other, and the repack reads the offloaded weights
      directly.
    - The UVA lookup of the pinned n-gram table copies FP8 rows as `uint8`, as Triton has no FP8 e4m3 type before
      sm_89 (`type fp8e4nv not supported in this architecture`).
  - Native CPU KV offloading with an EAGLE/MTP drafter whose layers share the full-attention KV cache group, added on
    2026-10-13 (`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`). With the default
    `prefix_cache_retention_interval` of 0, sliding-window groups store only the window ending at the prompt's replay
    boundary, but the lookup drops the volatile trailing chunk of the drafter's group and therefore needs the window
    ending one chunk earlier. For Gemma 4 with MTP (5 sliding-window groups, and the full-attention group as the
    drafter group), no window was ever complete, so nothing was loaded and every evicted conversation was computed
    again. The sliding-window tail is now also kept for the boundary one chunk earlier (one more 16-token block per
    conversation). On the RTX 3070 of [`../vllm-agx-ai`](../vllm-agx-ai), a 26616-token conversation evicted by two
    98k-token prompts came back in 2.2 s instead of 29 s (381 MB loaded), with the same greedy answer as before.
  - Image and video input for Qwen3.8-Flash-Next (Qwen4Exp) with the vision tower on the CPU, added on 2026-10-13 for
    [`../vllm-agx-ai`](../vllm-agx-ai), ported from [`../vllm-flash-next`](../vllm-flash-next/README.md):
    `models/qwen4_exp/nvidia/cpu_vision.py` (a new file, the same module as there) runs transformers' Qwen3-VL vision
    model in FP32 on the CPU when `QWEN38_VISION_DEVICE=cpu` (threads: `QWEN38_VISION_THREADS`), the model then
    creates it instead of vLLM's vision transformer and sets `mm_encoder_inputs_on_cpu`, and the encoder runner
    (`v1/worker/gpu/mm/encoder_runner.py`) leaves the pixel values of such a model on the CPU, so that they never take
    VRAM. The separate pixel budget for videos of `vllm-mm.patch` is not needed in vLLM 0.31, which takes
    per-modality processor settings (`mm-processor-kwargs: {"max_pixels": ..., "videos_kwargs": {"max_pixels": ...}}`).

