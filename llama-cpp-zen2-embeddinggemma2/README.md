# EmbeddingGemma 2 on the CPU of agx-z2e with llama.cpp

Serves the text embeddings of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on the
Threadripper 3970X (Zen 2) of agx-z2e with llama-server, so that the GPUs stay free for the LLMs
([`../llama-cpp-rtx3090`](../llama-cpp-rtx3090), [`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii)).
[`../vllm-zen2-embeddinggemma2`](../vllm-zen2-embeddinggemma2) serves the same model on the same port with vLLM,
which is **2 times faster** on this CPU (see the table); run only one of the two.

| Model | Context | VRAM | RAM | 500-token texts | 100-token texts | Single text | Notes |
|---|---|---|---|---|---|---|---|
| EmbeddingGemma 2 `BF16` ([ggml-org GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)), 8 slots, 32 pinned threads | 8192 | none | 2.3-2.9 GiB (peak RSS, 0.5 GiB of it locked weights) | 2200-2300 t/s | 2100-2200 t/s | 0.26 s (500 tokens), 0.07 s (100 tokens) | vLLM: 4300-4700 t/s and 4900-5000 t/s ([`../vllm-zen2-embeddinggemma2`](../vllm-zen2-embeddinggemma2)) |

## Hardware and software

| | |
|---|---|
| Computer | Desktop computer with an ASUS ROG Zenith II Extreme (TRX40) motherboard |
| OS | Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0-42-generic |
| CPU | AMD Ryzen Threadripper 3970X (Zen 2, 32 cores, 64 threads, AVX2, no AVX-512 or BF16 instructions) |
| RAM | 128 GB DDR4-2666 (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPUs (not used) | NVIDIA GeForce RTX 3090 (Ampere, 24 GB GDDR6X), AMD Radeon VII (Vega 20, GCN 5, 16 GB HBM2), NVIDIA GeForce GTX TITAN (Kepler, 6 GB GDDR5, unused) |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-b12004` (CPU, commit 3c809f4), which loads its `haswell` kernels (AVX2, FMA, F16C) on Zen 2 |

## Usage

```sh
docker compose up -d
```

- llama-server runs as a router with the shared settings of [`../llama-cpp/config.ini`](../llama-cpp/config.ini)
  and the API key of `../llama-cpp/llama-cpp.env` (see [`../llama-cpp`](../llama-cpp/README.md)), on port 9933.
  The model is downloaded to the Hugging Face cache on the first start (0.6 GB) and loads in ~1 s.
- The API is OpenAI-compatible at http://localhost:9933/v1/embeddings, with the model id `google/embeddinggemma-2`.
  The embeddings have 768 dimensions and are normalized (mean pooling, as in the sentence-transformers configuration).
  The router also lists the other models of the Hugging Face cache, but does not load them
  (`LLAMA_ARG_MODELS_AUTOLOAD=false`; requests for them fail with "model is not loaded"). Without that, a request for
  one of them loaded it on the CPU and evicted the embedding model (`models-max = 1`), see below.
- The weights are locked in RAM (`load-mode = mmap+mlock`, `ulimits: memlock: -1`).
- llama.cpp ignores the `dimensions` parameter. For Matryoshka truncation to 512, 256 or 128 dimensions, the client
  has to truncate and renormalize the embeddings (vLLM does it with `dimensions`).
- Inputs of up to 8192 tokens work; longer ones are rejected with HTTP 500 ("input (9003 tokens) is too large to
  process").
- Use the task prefixes of the [model card](https://huggingface.co/google/embeddinggemma-2#1-task-instruction-prefixes),
  e.g. `task: search result | query: {query}` for queries and `title: none | text: {document}` for documents.
- The vision and audio encoders (`mmproj-embeddinggemma-2-*.gguf`) are not loaded, so only text is supported.

## Performance

Measured on 2026-10-30 with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py)
`--tests embedding --embed-texts 64 --embed-tokens 500` (and 100), i.e. one request with 64 texts, with the LLM
servers of the GPUs idle. The results of the production configuration are in
`../benchmark/results/agx-z2e-kubuntu.jsonl` (labels starting with "zen2 llama.cpp"). The sweep used test servers
with the same settings as command-line arguments and 16-64 texts; the first request after a start is ~10 % slower.

| Setting | 500-token texts | 100-token texts | Notes |
|---|---|---|---|
| Q8_0, 8 threads, 4 slots | 940-1000 t/s | 1090-1110 t/s | |
| Q8_0, 16 threads, 4 slots | 1580-1650 t/s | 1800-1820 t/s | |
| Q8_0, 32 threads, 4 slots | 2100-2310 t/s | 2060-2220 t/s | |
| Q8_0, 64 threads (SMT), 4 slots | 770-880 t/s | 450-570 t/s | Single text 0.7-1.2 s |
| Q8_0, 28 threads, 8 slots | 1820-1970 t/s | 2320-2380 t/s | |
| Q8_0, 32 threads, 1 slot | 1870 t/s | 1590-1610 t/s | |
| Q8_0, 32 threads, 8 slots | 1880-2190 t/s | 2330-2490 t/s | |
| Q8_0, 32 threads, 16 slots | 1410-1750 t/s | 2430-2470 t/s | |
| Q8_0, 32 threads, 8 slots, ubatch 2048 | 2120-2250 t/s | 2110-2290 t/s | |
| BF16, 32 threads, 4 slots | 2240-2310 t/s | 2050-2090 t/s | |
| BF16, 32 threads, 8 slots | 2170-2300 t/s | 2220-2320 t/s | |
| BF16, 32 threads, 8 slots, flash attention off | 870-950 t/s | 1740-1750 t/s | |
| **BF16, 32 threads pinned to the physical cores, 8 slots** (this configuration) | **2200-2380 t/s** | **2140-2360 t/s** | |
| The same, 16 texts of 2000 tokens | 1700 t/s | | Single text 1.5 s |
| The same, 2 texts of 8000 tokens | 1420 t/s | | Single text 7.2 s |

- One thread per physical core is best. With 64 threads, the SMT siblings compete for the same cores and the
  threads wait for each other (2.5-4 times slower).
- BF16 was as fast as Q8_0: llama.cpp's CPU kernels convert the BF16 weights cheaply with AVX2 and compute in FP32,
  and the speed is limited by the compute of the attention and of the small matrices (hidden size 512), not by the
  memory bandwidth. The unquantized weights are therefore used.
- Pinning the threads to the cores 0-31 (`cpu-range`, `cpu-strict`) was ~3 % faster.
- Flash attention (auto = on) is important for longer texts.
- The model is not causal, so each input has to fit into one ubatch (`ubatch-size` = 8192 = the context).
  The texts of a request are processed in the same ubatch up to the number of slots.
- The embeddings match sentence-transformers (FP32 on the CPU) with a cosine similarity of 0.99999-1.00000 (six texts
  of 16-1500 tokens, including ones longer than the 512-token sliding window).

## Optimization with the optimize-llama-cpp skill (2026-10-31)

Most steps of [`.claude/skills/optimize-llama-cpp`](../.claude/skills/optimize-llama-cpp/SKILL.md) are for LLMs on
GPUs (MTP, KV cache type, `n-cpu-moe`, vision encoder, VRAM headroom) and do not apply. The threads, slots, ubatch,
flash attention and pinning were measured above. In addition:

| Model file (the final settings, `server-b12004`, 2 rounds in alternating order) | 500-token texts | 100-token texts |
|---|---|---|
| BF16 (published) | 2230-2370 t/s | 2310-2410 t/s |
| F16 (converted from BF16 with `llama-quantize`) | 2180-2320 t/s | 2250-2340 t/s |
| F32 (converted) | 2300-2460 t/s | 2260-2420 t/s |
| Q8_0 (published) | 2110-2270 t/s | 2440-2540 t/s |

- The four types are within 5 % of each other. F32 is ~4 % faster for 500-token texts but would need a locally
  converted file; Q8_0 ~5 % faster for 100-token texts but quantized. The published BF16 stays.
- The image is pinned (`server-b12004`). It contains no Zen 2 kernel variant, but the `haswell` one has the same
  instruction set extensions, so a native build was not tried.
- `load-mode = mmap+mlock` (the catalog's choice for CPU-only models; `mlock = 1` is not a preset option in b12004):
  the 0.5 GiB of weights stay in RAM (`VmLck`), with no change in speed (2270 and 2370 t/s).
- **Router pitfall:** `llama_cpp_bench_http.py --tests embedding` picked the first non-embedding model of the cache
  (Gemma 4 31B) as its "chat model" for the server properties and warmed it up, so the router loaded it on the CPU
  (10 GiB peak RSS) and then evicted it again for the embedding test. The script now uses the embedding model for
  the properties when only the embedding test runs, and the server no longer loads other models on request. The
  embedding results in `../benchmark/results/agx-z2e-kubuntu.jsonl` from before the fix were measured after the
  embedding model had been reloaded, so they are valid, but their `chat_model` is that LLM.
- The final configuration: 2240-2270 t/s (64 x 500 tokens), 2150-2370 t/s (64 x 100 tokens), 1860 t/s (16 x 2000
  tokens), 1510 t/s (2 x 8100 tokens, 6.8 s for one), and a cosine similarity of 0.99999-1.00000 with
  sentence-transformers.

Commands:

```sh
docker compose down
# A test server with llama-server arguments (the scratch script used for the tables; the GGUF is mounted as /g):
docker run -d --name llama-test -p 9933:8080 -v /home/mika/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    --entrypoint /app/llama-server ghcr.io/ggml-org/llama.cpp:server-b12004 \
    -hf ggml-org/embeddinggemma-2-GGUF:BF16 --offline --host 0.0.0.0 --port 8080 --embedding -a embeddinggemma-2 \
    -c 8192 -b 8192 -ub 8192 -t 32 -tb 32 -np 8 --cpu-range 0-31 --cpu-strict 1 --cpu-range-batch 0-31 --cpu-strict-batch 1
cd ../benchmark
./llama_cpp_bench_http.py --url http://localhost:9933 --tests embedding --embedding-model embeddinggemma-2 \
    --embed-texts 64 --embed-tokens 500 --label "..."
docker rm -f llama-test
# F16/F32 from the BF16 GGUF
docker run --rm -v /home/mika/.cache/huggingface/hub:/hub:ro -v "$PWD/gguf:/out" --entrypoint sh \
    ghcr.io/ggml-org/llama.cpp:full-b12004 -c '/app/llama-quantize \
    $(ls /hub/models--ggml-org--embeddinggemma-2-GGUF/snapshots/*/embeddinggemma-2-BF16.gguf) /out/embeddinggemma-2-F32.gguf F32'
```
