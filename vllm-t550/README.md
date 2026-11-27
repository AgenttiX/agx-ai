# vLLM on the NVIDIA T550 Laptop GPU

vLLM configurations for Gemma 4 E2B and E4B QAT on the NVIDIA T550 Laptop GPU (4 GB, Turing TU117) under Docker
Desktop with WSL 2 on Windows: [`docker-compose.yml`](docker-compose.yml) with
[`config-gemma-4-e2b-qat.yaml`](config-gemma-4-e2b-qat.yaml) (the default) or
[`config-gemma-4-e4b-qat.yaml`](config-gemma-4-e4b-qat.yaml), selected with its volume line. The image is
[`../vllm`](../vllm/README.md)'s, whose `vllm.patch` has the Turing and Gemma 4 changes these need. The benchmark
results of this computer are saved with the hostname `t550`.

The same port as [`../llama-cpp-t550`](../llama-cpp-t550/README.md) (9931), as only one of them fits on the GPU at a
time. **llama.cpp is the better choice on this GPU** for E4B (more than twice the context, twice the prompt
processing speed and faster chat, even at a lower power limit) and for its other models. vLLM's E2B is the fastest
model on this GPU in chat (66.5 t/s), for when the smaller model is good enough.

## Summary (2026-11-27)

vLLM v0.33.0 with [`../vllm/vllm.patch`](../vllm/vllm.patch). tg = generation, pp = prompt processing; "chat" = the
chat test of [`bench_http.py`](../agx_ai/benchmark/README.md), the realistic generation speed with MTP.

| Model | Context | VRAM (peak) | RAM (pinned) | tg: chat / at 16k / at 32k / at 64k | pp: 4096 tokens / at 16k / at 32k / at 64k | Notes |
|---|---|---|---|---|---|---|
| Gemma 4 E2B QAT W4A16, int8 embeddings, MTP 2, 30 W | 131072 | 3129 MiB | 2.6 GiB | 66.5 / 18.4 / 11.1 / 6.5 t/s | 199 / 94 / 56 / 31 t/s | The default. Long contexts are slow (attention) |
| same, 20 W (with `expandable_segments`, 3279 MiB) | 131072 | 3279 MiB | 2.6 GiB | 58.3 / 16.2 / 10.0 / 5.8 t/s | 158 / 80 / 48 / 27 t/s | |
| Gemma 4 E4B QAT W4A16, int8 embeddings, int4 lm_head, MTP 2, 30 W | 30720 | 3859 MiB | 3.3 GiB | 25.8 / 11.5 / 9.0 (28k) / - t/s | 115 / 68 / 50 (28k) / - t/s | **llama.cpp is better** ([`../llama-cpp-t550`](../llama-cpp-t550/README.md), at 20 W: chat 30.6, pp 245 t/s, 69632 tokens) |

Not measured: the quality of E2B against E4B, and E2B with llama.cpp on this GPU.

## Hardware and software

| Component | |
|---|---|
| OS | Windows 11 25H2 (build 26200), Docker Desktop (CLI plugin 4.100.1, engine 29.9.3) with the WSL 2 backend (WSL 3.0.5, kernel 6.18) |
| CPU | Intel Core i7-1260P (Alder Lake, 4 performance + 8 efficiency cores, 16 threads) |
| RAM | 32 GB DDR4-3200 (2 x 16 GB SO-DIMM, dual channel); the WSL 2 VM gets 15.5 GiB of it |
| NPU | none |
| iGPU | Intel Iris Xe Graphics (96 EU, shares the system RAM), drives the display |
| dGPU | NVIDIA T550 Laptop GPU: Turing TU117 (sm_75, no tensor cores), 4 GB GDDR6 (64-bit, 96 GB/s), PCIe link x4 (gen 4), driver 596.52 (CUDA 13.2) |
| dGPU power limit | 20 W in the "balanced" power mode of Windows, 30 W in "best performance" (stated per result) |

## Checkpoint

Google's int4 QAT checkpoints in compressed-tensors format (`google/gemma-4-E2B-it-qat-w4a16-ct`, 7.9 GB, and
`google/gemma-4-E4B-it-qat-w4a16-ct`, 11 GB) have the linear layers in int4 (group size 32), but the token embedding,
the lm_head and the per-layer embedding (PLE) in BF16. [`../vllm/quantize_gemma4_ct_embedding.py`](../vllm/quantize_gemma4_ct_embedding.py)
quantizes them (round-to-nearest) and copies everything else, in the image (a few minutes per model, the PLE in chunks so
that it fits the RAM of the WSL 2 VM; `--ple-bits` is in the image from its next build after 2026-11-27):

```sh
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub -v ~/.cache/vllm-models:/models \
    -e HF_HUB_OFFLINE=1 --entrypoint quantize_gemma4_ct_embedding.py ghcr.io/agenttix/agx-ai/vllm:latest \
    --repo google/gemma-4-E2B-it-qat-w4a16-ct --embed-bits 8 --lm-head-bits 8 --ple-bits 8 \
    --out /models/gemma-4-E2B-it-qat-w4a16-ct-embed8-head8-ple8
docker run ... --repo google/gemma-4-E4B-it-qat-w4a16-ct --embed-bits 8 --lm-head-bits 4 --ple-bits 8 \
    --out /models/gemma-4-E4B-it-qat-w4a16-ct-embed8-head4-ple8
```

| Model | Checkpoint | On the GPU | In pinned RAM (UVA) |
|---|---|---|---|
| E2B | 4.85 GiB | linear 0.97 GiB, lm_head int8 0.38 GiB | embedding 0.38 GiB, PLE 2.22 GiB |
| E4B | 6.63 GiB | linear 2.05 GiB, lm_head int4 0.32 GiB | embedding 0.63 GiB, PLE 2.67 GiB |

The vision and audio encoders (0.9 GiB) stay in the checkpoint but are not loaded (`language-model-only`). The MTP
drafters (`google/gemma-4-E{2,4}B-it-qat-q4_0-unquantized-assistant`, 0.18 GB) are downloaded from Hugging Face.
The PLE must be int8: vLLM creates it on the GPU before it is offloaded, and in BF16 (4.4 GiB for E2B) it does not
fit even for that moment. E4B has an int4 lm_head: an int8 one would take 0.32 GiB of its KV cache (~23k of its
31k tokens); not tried since `expandable_segments` is off (below), which is what made E4B fit at all.

## GPU memory under Windows

CUDA reports only ~3.2 GiB of the 4 GiB as free (3300 MiB for a bare CUDA context, both natively on Windows and in
WSL 2, at 20 W and 30 W), although only ~18 MiB are used by other processes and 151 MiB are reserved by the driver.
This is the budget of the Windows driver (WDDM), not a limit. Measured in WSL 2 (allocate, touch, then read all of
it):

| Allocated by one process | Read bandwidth |
|---|---|
| up to 3712 MiB | 87 GiB/s (in VRAM) |
| 3840 MiB | 70 GiB/s (WDDM starts to page to system RAM) |
| 4352 MiB | 30 GiB/s |
| same, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fails at 3456 MiB: "CUDA driver error: device not ready" |

llama.cpp allocates with `cudaMalloc` and never asks for the free memory, so it uses up to ~3.9 GiB (also in Docker,
re-checked after the Docker Desktop update). vLLM sizes itself from the free memory: `gpu-memory-utilization: 0.9`
refuses to start, so the configurations set `kv-cache-memory-bytes` and `gpu-memory-utilization: 0.78`. And
PyTorch's `expandable_segments`, used by the other vLLM configurations of this repository, must stay off: with it,
E4B failed to load with the error above at ~3480 MiB, with both an int8 and an int4 lm_head; without it, E4B loads
and peaks at 3859 MiB, and E2B peaks at 3129 instead of 3279 MiB. Pinned host memory (the UVA offload) did not change
any of these limits. The Docker Desktop update of 2026-11-26 changed nothing here: the llama.cpp container still
loads E4B with 3899 MiB.

## Optimizations

Measured with E2B at 20 W (bench_http, single requests):

| Change | Generation | Prompt processing (4096) | Chat |
|---|---|---|---|
| Marlin int4 kernel (vLLM's default) | 15.5 t/s | 92 t/s | 14.7 t/s |
| Exllama kernel (`VLLM_DISABLED_KERNELS=MarlinLinearKernel`) | 43.7 t/s | 65 t/s | |
| + exllama row chunks of 48 / 32 / **50** (`VLLM_EXLLAMA_ROW_CHUNK`) | | 114 / 107 / **119** t/s | |
| + MTP 1 / **2** / 3 draft tokens (no MTP: 40.9; 12 chat requests) | | | 49.4 / **55.6** / 56.5 t/s |

- **Marlin** relies on tensor cores (the TU117 has none usable). The exllama (GPTQ) kernel is 2.8 times as fast for
  generation; it dequantizes into a K x N FP16 scratch per layer, which `vllm.patch` limits by splitting wide layers
  into column chunks (`VLLM_EXLLAMA_MAX_SCRATCH_MB=128`; Gemma 4's lm_head would need 1.3 GiB).
- **Prefill:** above 50 rows, exllama dequantizes and calls cuBLAS, whose FP16 GEMM runs at ~0.4 TFLOPS here
  (`volta_h884gemm`, profiled with Nsight Systems). Row chunks of 50 keep exllama's own kernel.
- **MTP 2**: 3 draft tokens are within the noise of 2 (acceptance 40 vs. 50 %), with more variance.
- **Attention** is ~30 % of prefill and dominates decode at long contexts (Triton, with the Turing tiles of
  `vllm.patch`; FlashAttention 2 needs sm_80). Not optimized further.
- **KV cache:** int8 (`int8_per_token_head`). E2B's sliding-window layers need only their window, so 0.52 GB hold
  147k tokens, more than the model's 131072; E4B needs ~14 KB per token.
- The 256-token generation test of `bench_http` reports nonsense with MTP on this server (e.g. 1600 t/s with 1.2 %
  draft acceptance), probably as the prompt ends the answer early: use the chat test.

