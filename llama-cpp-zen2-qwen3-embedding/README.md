# Qwen3-Embedding on the CPU of agx-z2e with llama.cpp

Serves [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (default) or
[Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) on the Threadripper 3970X (Zen 2) of agx-z2e
with llama-server, so that the GPUs stay free for the LLMs. The model is selected with the volume lines in
[`docker-compose.yml`](docker-compose.yml). [`../vllm-zen2-qwen3-embedding`](../vllm-zen2-qwen3-embedding) serves the
same models on the same port with vLLM: it is **1.5 times faster for the 0.6B model, but slower for the 4B model**.
Run only one of them (also not at the same time as the EmbeddingGemma 2 configurations on the same port).

| Preset | Model | Context | VRAM | RAM | 500-token texts | 100-token texts | Notes |
|---|---|---|---|---|---|---|---|
| [`preset-qwen3-embedding-0.6b.ini`](preset-qwen3-embedding-0.6b.ini) (default) | Qwen3-Embedding-0.6B `Q8_0`, 4 slots, 32 pinned threads | 8192 | none | 1.8 GiB (peak RSS, 0.6 GiB locked weights) | 1290 t/s | 1400 t/s | vLLM: 2070 and 2040 t/s ([`../vllm-zen2-qwen3-embedding`](../vllm-zen2-qwen3-embedding)) |
| [`preset-qwen3-embedding-4b.ini`](preset-qwen3-embedding-4b.ini) | Qwen3-Embedding-4B `Q6_K`, 4 slots, 32 pinned threads | 8192 | none | 4.7 GiB (peak RSS, 3.1 GiB locked weights) | 300 t/s | 270 t/s | Faster than vLLM (170 and 270 t/s) |

The 0.6B model embeds 4.3 times as fast as the 4B model. On the PaperQA2 corpus of
[`../vllm-rtx-a4000`](../vllm-rtx-a4000/README.md#embedding-models), the 4B model found the right chunk among the top 10
for 92 % of the questions and the 0.6B model for 89 %; see also the [retrieval test](#retrieval-test) below.

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
  The GGUF is downloaded to the Hugging Face cache on the first start (0.6 GB or 3.3 GB) and loads in a few seconds.
- The API is OpenAI-compatible at http://localhost:9933/v1/embeddings, with the model id `Qwen/Qwen3-Embedding-0.6B`
  or `Qwen/Qwen3-Embedding-4B`. The embeddings have 1024 (0.6B) or 2560 (4B) dimensions and are normalized
  (last-token pooling; llama.cpp appends the `<|endoftext|>` token like the Hugging Face tokenizer).
- llama.cpp ignores the `dimensions` parameter. For Matryoshka truncation, the client has to truncate and renormalize
  the embeddings (vLLM does it with `dimensions`).
- Queries should have an instruction, e.g.
  `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}`, and the
  documents none (see the [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage)).
- Inputs of up to 8192 tokens work (the model supports 32768; raise `ctx-size` if needed).
- The router also lists the other models of the Hugging Face cache, but does not load them
  (`LLAMA_ARG_MODELS_AUTOLOAD=false`, see [`../llama-cpp-zen2-embeddinggemma2`](../llama-cpp-zen2-embeddinggemma2)).
- The weights are locked in RAM (`load-mode = mmap+mlock`, `ulimits: memlock: -1`).

## Performance

Measured on 2026-10-31 with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py)
`--tests embedding --embed-texts 64 --embed-tokens 500` (and 100; 16 texts for the 4B model), i.e. one request with
64 or 16 texts, two runs each, on test servers with the settings as command-line arguments (`--kv-unified`, like the
router). The results of the production configurations are in `../benchmark/results/agx-z2e-kubuntu.jsonl` (labels
starting with "zen2 llama.cpp b12004 Qwen3-Embedding").

### Qwen3-Embedding-0.6B

| Setting (32 threads pinned to the cores 0-31 unless noted) | 500-token texts | 100-token texts |
|---|---|---|
| f16, 8 slots, ubatch 8192 | 1050-1070 t/s | 1200 t/s |
| f16, 4 slots, ubatch 8192 | 1150-1160 t/s | 1180 t/s |
| f16, 16 slots, ubatch 8192 (context 16384) | 910-940 t/s | 1190 t/s |
| f16, 8 slots, ubatch 4096 | 1060-1080 t/s | 1210 t/s |
| f16, 8 slots, ubatch 2048 | 1130-1140 t/s | 1200 t/s |
| f16, 4 slots, ubatch 2048 | 1150-1160 t/s | 1170-1180 t/s |
| f16, 4 slots, ubatch 1024 | 1160 t/s | 1170 t/s |
| f16, 2 slots, ubatch 2048 | 1170-1180 t/s | 1100 t/s |
| f16, 8 slots, 24 threads (cores 0-23) | 840-850 t/s | 960 t/s |
| f16, 8 slots, 32 threads not pinned | 1010-1040 t/s | 1160 t/s |
| f16, 8 slots, flash attention off | 380-410 t/s | 750 t/s |
| **Q8_0, 4 slots, ubatch 2048** (this preset) | **1290 t/s** | **1400 t/s** |
| The same, 16 texts of 2000 tokens | 1100 t/s | |
| The same, 2 texts of 8000 tokens | 600 t/s | |

### Qwen3-Embedding-4B

| Setting (32 pinned threads) | 500-token texts | 100-token texts |
|---|---|---|
| Q8_0, 4 slots, ubatch 2048 | 210 t/s | 233 t/s |
| Q8_0, 2 slots, ubatch 2048 | 229 t/s | 231 t/s |
| Q8_0, 4 slots, ubatch 512 | 231 t/s | 233 t/s |
| **Q6_K, 4 slots, ubatch 2048** (this preset) | **300 t/s** | **272 t/s** |
| The same, 4 texts of 2000 tokens | 274 t/s | |
| The same, 1 text of 8000 tokens | 182 t/s (44 s) | |

- One thread per physical core, pinned (~4 % faster), and flash attention (2.6 times faster for 500-token texts),
  as for EmbeddingGemma 2.
- The model is causal, so long inputs are split into ubatches, and 1024-8192 made no difference.
- The 0.6B model is compute-bound: Q8_0 was 11-19 % faster than f16. The 4B model reads more weights per token, and
  Q6_K was 17-43 % faster than Q8_0.
- Cosine similarity with sentence-transformers (FP32 on the CPU; six texts of 6-1500 tokens): 0.6B f16 1.00000,
  0.6B Q8_0 0.9983-0.9994, 4B Q8_0 0.9994-0.9996, 4B Q6_K 0.9933-0.9978. vLLM (FP32): 1.00000 for both.

## Retrieval test

As the PaperQA2 corpus is not on this computer,
[`../benchmark/embedding_retrieval_eval.py`](../benchmark/embedding_retrieval_eval.py) was run on a corpus of this
repository's Markdown files and ten packages of the Python 3.14 standard library (`asyncio`, `email`, `json`, `http`,
`concurrent`, `logging`, `unittest`, `importlib`, `xml/etree`, `sqlite3`), in chunks of 2500 characters (1013 chunks,
2.4M characters). Qwen3.8-27B ([`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen)) wrote one question for each of
300 random chunks (291 questions; the prompt was generalized from physics papers to any document for this). Each model
then ranked all chunks for each question, without an instruction, as PaperQA2 does. The results are in
`../benchmark/results/agx-z2e-kubuntu-embedding.jsonl`.

| Model | R@1 | R@5 | R@10 | MRR | Time for the 1013 chunks |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B f16 | 0.680 | 0.897 | 0.935 | 0.774 | 525 s |
| Qwen3-Embedding-0.6B Q8_0 (this preset) | 0.680 | 0.890 | 0.935 | 0.772 | 446 s |
| Qwen3-Embedding-4B Q6_K | 0.667 | 0.880 | 0.931 | 0.760 | 1976 s |

- Q8_0 retrieves as well as f16 for the 0.6B model (two questions of 291 dropped out of the top 5).
- On this corpus, the 4B model is no better than the 0.6B model (within a few questions), at 4.4 times the time. On
  the PaperQA2 corpus of physics papers, it was 3 points better at R@10, so the choice depends on the documents.

## Commands

```sh
docker compose down
# A test server (the scratch script used for the tables)
docker run -d --name llama-test -p 9933:8080 -v /home/mika/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    --entrypoint /app/llama-server ghcr.io/ggml-org/llama.cpp:server-b12004 \
    -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --offline --host 0.0.0.0 --port 8080 --embedding --pooling last \
    --kv-unified -a embed -c 8192 -b 2048 -ub 2048 -t 32 -np 4 \
    --cpu-range 0-31 --cpu-strict 1 --cpu-range-batch 0-31 --cpu-strict-batch 1
cd ../benchmark
./llama_cpp_bench_http.py --url http://localhost:9933 --tests embedding --embedding-model embed \
    --embed-texts 64 --embed-tokens 500 --label "..."
# The retrieval test (chunks.jsonl as above; the questions once, with the LLM on port 9931)
uv run ./embedding_retrieval_eval.py questions --chunks chunks.jsonl --out questions.jsonl --n 300 \
    --url http://localhost:9931 --env-file ../llama-cpp/llama-cpp.env --concurrency 8
uv run ./embedding_retrieval_eval.py eval --chunks chunks.jsonl --questions questions.jsonl \
    --url http://localhost:9933 --model embed --label "..."
docker rm -f llama-test
```

Note for zsh: `"$M:Q8_0"` is expanded with the `:Q` modifier (to `...GGUF8_0`); write `"${M}:Q8_0"`.
