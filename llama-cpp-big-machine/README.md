# llama.cpp on the Big Machine

llama.cpp for [PaperQA2](https://github.com/Future-House/paper-qa) RAG on the Big Machine.
One `llama-server` router ([`docker-compose.yml`](docker-compose.yml), [`config.ini`](config.ini),
[`preset.ini`](preset.ini)) serves an LLM and an embedding model on port 9931.
Both models are loaded at the same time, one half of each on each GPU.
They are unloaded from VRAM after 10 minutes of inactivity and reloaded on the next request.

| Model | Settings | Runs on |
|---|---|---|
| `Qwen/Qwen3.8-27B`: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_M | Tensor split 1:1, MTP, ctx 65536, 4 slots | Both GPUs |
| `Qwen/Qwen3-Embedding-8B`: [Qwen/Qwen3-Embedding-8B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF) Q6_K | Layer split 4:3, ctx 5120, 1 slot | Both GPUs |

## Hardware

- OS: Linux, kernel 7.0.0-30-generic, glibc 2.39 (TODO: distribution and version)
- CPU: AMD Ryzen Threadripper PRO 3975WX (32 cores, 64 threads)
- RAM: TODO (size, type, frequency)
- NPU: none
- GPU: 2 x NVIDIA RTX A4000
- GPU architecture: Ampere (GA104)
- VRAM: 16 GB GDDR6 per GPU (16376 MiB reported by `nvidia-smi`), 32 GB in total

## VRAM budget

Measured with both models loaded (`nvidia-smi`; idle, and peak under a mixed load of 4 requests and embeddings):

| GPU | Qwen3.8-27B | Embedding model | Idle | Peak |
|---|---|---|---|---|
| GPU0 | ~11.1 GiB (10848 MiB at ctx 55296, +258 MiB at 65536) | 4432 MiB | 15829 MiB | 15947 MiB |
| GPU1 | ~11.0 GiB | 4414 MiB | 15692 MiB | 15810 MiB |

The total is 16376 MiB per GPU. The LLM needs ~120 MiB more per GPU under load, and loading a model needs transient
memory beyond its steady-state footprint, so ~400 MiB per GPU must stay free at idle for the models to reload reliably
after the idle unload. The worker processes stay alive after the idle unload and keep ~600 MiB of VRAM each.

## Context size of Qwen3.8-27B

Probed on 2026-09-21 with both models loaded. Each candidate was tested with a cold start, two unload/reload cycles
through the router API and a mixed load (`benchmark/llama_cpp_bench_http.py --mixed`):

| Context | Result |
|---|---|
| 55296, 63488, 65536 | Stable |
| 67584 | Loads and reloads, but the process dies under load (`CUDA error: out of memory` in `cudaGraphInstantiate`) |
| 69632, 71680 | Fails to load (`cudaMalloc failed: out of memory` for a ~136 MiB buffer on GPU 0) |

The KV cache costs 32 KiB per token on each GPU (+258 MiB per GPU for +8192 tokens), see the model section
in [`preset.ini`](preset.ini). The embedding model's slot count does not affect VRAM use;
its context is set by the PaperQA2 chunk size.

## Speed

tg = token generation, pp = prompt processing, acc = accepted MTP draft tokens.
Newer measurements are in [`benchmark/results/`](../benchmark/results/), measured with
[`benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py).

### Qwen3.8-27B + Qwen3-Embedding-8B

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
