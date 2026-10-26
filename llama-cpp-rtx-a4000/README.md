# llama.cpp on 2x RTX A4000

llama.cpp for [PaperQA2](https://github.com/Future-House/paper-qa) RAG on a workstation with 2x RTX A4000 (16 GB each).
One `llama-server` router ([`docker-compose.yml`](docker-compose.yml), [`config.ini`](config.ini) and a preset)
serves an LLM and an embedding model on port 9931. Both models are loaded at the same time.
They are unloaded from VRAM after 10 minutes of inactivity and reloaded on the next request.
The preset is selected in [`docker-compose.yml`](docker-compose.yml):

| Preset | Model | Settings | Runs on |
|---|---|---|---|
| [`preset-qwen3.8-27b-2gpu.ini`](preset-qwen3.8-27b-2gpu.ini) (default) | `Qwen/Qwen3.8-27B`: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_M | Tensor split 1:1, MTP, ctx 98304, 4 slots | Both GPUs |
| | `Qwen/Qwen3-Embedding-4B`: [Qwen/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) Q6_K | Layer split 4:3, ctx 4096, 1 slot | Both GPUs |
| [`preset-gemma-4-26b-a4b-qat-2gpu.ini`](preset-gemma-4-26b-a4b-qat-2gpu.ini) | `google/gemma-4-26b-a4b-qat`: [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) UD-Q4_K_XL | Layer split 4:3, MTP, ctx 262144, 4 slots, ubatch 1024 | Both GPUs |
| | `Qwen/Qwen3-Embedding-4B` Q6_K | as above | Both GPUs |
| [`preset-gemma-4-26b-a4b-qat-1gpu.ini`](preset-gemma-4-26b-a4b-qat-1gpu.ini) | `google/gemma-4-26b-a4b-qat` UD-Q4_K_XL | Experts of 19 of 30 layers on the CPU (16 threads), MTP, ctx 262144, 4 slots | GPU 1 + CPU |
| | `Qwen/Qwen3-Embedding-4B` Q6_K | ctx 4096, 1 slot | GPU 0 |

The same models and configurations on vLLM, and the comparison with these presets for PaperQA2 (vLLM answers
1.6-2.5 times as fast with the same model), are in [`../vllm-rtx-a4000`](../vllm-rtx-a4000/README.md).

## Hardware

- OS: Ubuntu 24.04.5 LTS (kernel 7.0.0-30-generic), NVIDIA driver 610.57.04 (CUDA 13.3)
- CPU: AMD Ryzen Threadripper PRO 3975WX (32 cores, 64 threads)
- RAM: 192 GB registered DDR4 ECC in 8 channels (4 x 32 GB and 4 x 16 GB); the frequency needs root to read
- NPU: none
- GPU: 2 x NVIDIA RTX A4000, PCIe 4.0 x16 each, no NVLink; GPU 0 also runs the desktop session (~110 MiB)
- GPU architecture: Ampere (GA104)
- VRAM: 16 GB GDDR6 per GPU (16376 MiB reported by `nvidia-smi`), 32 GB in total

The machine is shared with other users' jobs, so the container has a memory limit of 48 GiB (see
[`docker-compose.yml`](docker-compose.yml)).

## The image (build b11541, 2026-10-10)

`ghcr.io/ggml-org/llama.cpp:server-cuda13` was updated from build b11046 (2026-09-19) to b11541. The new build needs
three changes in [`docker-compose.yml`](docker-compose.yml), without which the default preset did not start:

- `NVIDIA_DISABLE_REQUIRE=1`: the image requires CUDA >= 13.4, but the driver provides 13.3 (as in
  [`../llama-cpp-rtx3070`](../llama-cpp-rtx3070)).
- `LLAMA_ARG_PORT=9931`: the image sets `LLAMA_ARG_PORT=8080`, which overrides the port of `config.ini`, so the server
  listened on 8080 and the published port 9931 reset every connection.
- `NCCL_P2P_DISABLE=1` and `shm_size: 2g`: the tensor split now reduces with NCCL
  (`ggml_backend_cuda_comm_allreduce_nccl`). Peer-to-peer copies between the GPUs return corrupted data on this
  machine (see [`../vllm-rtx-a4000`](../vllm-rtx-a4000/README.md#peer-to-peer-copies-between-the-gpus-are-broken)),
  so NCCL failed with `unhandled cuda error` in `ncclGroupEnd`; without peer-to-peer copies, it communicates through
  `/dev/shm`, whose default of 64 MiB was too small (`unhandled system error`).

The NCCL buffers take more VRAM than the previous all-reduce, so the default preset no longer fit with
Qwen3-Embedding-8B. It now uses Qwen3-Embedding-4B, which retrieved as well as the 8B model on the PaperQA2 corpus
(the right chunk among the top 10 for 92.2 vs 91.4 % of the questions) at 1.6 times the speed, see
[the embedding test](../vllm-rtx-a4000/README.md#embedding-models). This frees 1.5 GiB per GPU, which also made room for
50 % more context for Qwen3.8-27B (98304 instead of 65536). The model id changed, so PaperQA2 builds a new index
(~12 min).

## VRAM budget of the default preset

Measured with both models loaded (`nvidia-smi`; idle, and peak under a mixed load):

| Context | GPU0 idle | GPU1 idle | Peak (GPU0 / GPU1) |
|---|---|---|---|
| 81920 | 14875 MiB | 15046 MiB | 14999 / 15170 MiB (mixed load: 8 requests and embeddings) |
| 98304 | 15451 MiB | 15622 MiB | 15551 / 15722 MiB (a 96500-token prompt, then 4 x 23000-token prompts with embeddings) |

The total is 16376 MiB per GPU. The embedding model's compute buffer is ~600 MiB per 1024 tokens of ubatch-size,
so ubatch-size 4096 did not fit next to the LLM; 2048 does (an input longer than that is processed in two batches).

## Context size of Qwen3.8-27B

Probed on 2026-10-11 with b11541 and both models loaded, each with a cold start, a prompt of almost the full context
and a mixed load (`benchmark/llama_cpp_bench_http.py --mixed`); 98304 also with an unload/reload cycle through the
router API (`POST /models/unload`):

| Context | Result |
|---|---|
| 81920, 98304 | Stable |
| 106496 | Loads, but dies with the first long prompt (`CUDA error: out of memory` in `cudaGraphInstantiate`) |

The KV cache costs 32 KiB per token on each GPU (+258 MiB per GPU for +8192 tokens), see the model section
in [`preset-qwen3.8-27b-2gpu.ini`](preset-qwen3.8-27b-2gpu.ini). Before (b11046, Qwen3-Embedding-8B, 2026-09-21), 65536 was the maximum.

## Gemma 4 26B-A4B QAT

Gemma 4 26B-A4B is a mixture-of-experts model with 4B active parameters, so it generates 2.5 times as fast as
Qwen3.8-27B, but it gave shallower answers in PaperQA2. Its KV cache is small (20 KiB per token for the 5 global
layers, a sliding window of 1024 tokens for the other 25), so its full context of 262144 tokens fits.

**2 GPUs** ([`preset-gemma-4-26b-a4b-qat-2gpu.ini`](preset-gemma-4-26b-a4b-qat-2gpu.ini)): everything fits on the
GPUs, with the embedding model.

- Tensor split loads (with ubatch-size 1024; with 2048, the MTP drafter's compute buffer did not fit), but the first
  request crashes in the flash attention kernel (`fattn.cu:762: fatal error`), likely because the single 512-wide KV
  head of the global layers cannot be split. So tensor parallelism does not work for this model.
- Layer split works. With 2 GPUs, llama.cpp keeps several copies of the compute buffers for pipeline parallelism:
  with ubatch-size 2048, Gemma asked for 5 GiB on GPU 0, so the preset uses 1024. Split 1:1 ran out of memory on
  GPU 1, which holds the output layer and more of the embedding model, so the split is 4:3.
- Experts on the CPU are not needed, and slower: with `n-cpu-moe = 6` (and ubatch-size 2048), chat 94 instead of
  135 t/s, 4 x 8000-token prompts in 22.8 instead of 16.7 s.
- Peak 15949 / 13128 MiB, also with a prompt of 250000 tokens (1223 t/s).

**1 GPU** ([`preset-gemma-4-26b-a4b-qat-1gpu.ini`](preset-gemma-4-26b-a4b-qat-1gpu.ini)), like
[`../llama-cpp-rtx3070`](../llama-cpp-rtx3070): Gemma on GPU 1 with the experts of 19 layers on the CPU (~8 GB;
the container used 12.4 GiB of RAM; 16 threads), the embedding model on GPU 0. Variants, all with ctx 262144 and 4 slots:

| Experts on the CPU | ubatch-size | KV cache | Result |
|---|---|---|---|
| 4, 12, 16 | 2048 | f16 | Do not load (the KV cache, then the compute buffers of the model, 1.6 GiB, and of the MTP drafter, 1.3 GiB) |
| 19 | 2048 | f16 | chat 59.6 t/s, pp 2363 t/s, 4 x 8000 tokens in 26.9 s, peak 15957 MiB on GPU 0 (**preset**, now on GPU 1) |
| 16 | 1024 | f16 | chat 62.9 t/s, pp 1983 t/s, 4 x 8000 tokens in 27.7 s: 5 % faster chat (18 % in the generation test), 16 % slower prompts |
| 18 | 2048 | f16 | On GPU 1: dies with a long prompt (`CUDA error: out of memory`) |

## Speed

tg = token generation, pp = prompt processing. Measured with
[`benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py) (results in
[`benchmark/results/`](../benchmark/results/)) with b11541; generation in t/s, prompt processing in parentheses:

| Test | Qwen3.8-27B ([`preset-qwen3.8-27b-2gpu.ini`](preset-qwen3.8-27b-2gpu.ini)) | Gemma, 2 GPUs | Gemma, 1 GPU + CPU |
|---|---|---|---|
| Generation, 256 tokens | 65.2 | 124.7 | 57.8 |
| Chat, 6 requests | 52.5 | 134.7 | 60.6 |
| Code context of 6.4k tokens | 50.4 (932) | 77.6 (2982) | 55.5 (1910) |
| Code context of 32k tokens | 47.7 (896) | 72.4 (2794) | 53.3 (1928) |
| Prompt of 16k tokens | (981) | (3306) | (2394) |
| 4 x (8000-token prompt + 256 generated) | 44.7 s | 16.7 s | 25.4 s |
| 8 x (8000-token prompt + 256 generated), with embeddings | 187.7 s | 58.3 s | 55.6 s |
| Embeddings, 16 x 2048 tokens | 3368 | 3380 | 3307 |

PaperQA2 (the three questions of [`../vllm-rtx-a4000`](../vllm-rtx-a4000/README.md#paperqa2)), time per
question: Qwen3.8-27B 89-103 s with thinking at "low" and 47-50 s without; Gemma without thinking 25-29 s on 2 GPUs
and 34-40 s on 1 GPU + CPU. Building the index of the 71 papers with Qwen3-Embedding-4B: 702 s.

### Before: b11046, Qwen3.8-27B + Qwen3-Embedding-8B

Qwen3.8-27B UD-Q4_K_M (tensor split, MTP, ctx 55296 at the time, 4 slots) + Qwen3-Embedding-8B Q6_K:
- Single request: tg 50-67 t/s, acc 50-70 %
- 4 concurrent requests with 11k-token prompts, while embedding 2 x 30k tokens:
  pp 144-295 t/s and tg 8-33 t/s per request, ~76 t/s in total, ~90 s wall time per request
- Embeddings during the above: 2 x 30k tokens in ~50 s (~600 t/s per batch)

### Previous combination: Gemma 4 31B + Qwen3-Embedding-4B

Gemma 4 31B QAT UD-Q4_K_XL (tensor split, MTP, ctx 24576, 4 slots) + Qwen3-Embedding-4B Q6_K:
- Single request: tg 64-75 t/s, acc 65-85 %, pp 890 t/s for a 4.8k-token prompt
- 4 concurrent requests with 2.8k-token prompts: tg 28-39 t/s per request, ~136 t/s in total
- The same while embedding 2 x 30k tokens: pp 111-141 t/s and tg 22-31 t/s per request, ~109 t/s in total
- Embeddings during the above: 2 x 30k tokens in ~29 s (~1000 t/s per batch)

Gemma 4 31B had a larger and faster-accepted MTP draft, but its sliding-window KV cache (~900 MiB per slot)
and the ~80 KiB per token global-attention KV cache limited the context size to 24576 tokens.
