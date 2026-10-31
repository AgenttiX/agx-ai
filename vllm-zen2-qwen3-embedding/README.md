# Qwen3-Embedding on the CPU of agx-z2e with vLLM

Serves [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (default) or
[Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) on the Threadripper 3970X (Zen 2) of agx-z2e
with the CPU build of vLLM, so that the GPUs stay free for the LLMs. The model is selected with the volume lines in
[`docker-compose.yml`](docker-compose.yml). [`../llama-cpp-zen2-qwen3-embedding`](../llama-cpp-zen2-qwen3-embedding)
serves the same models on the same port with llama.cpp, which is **slower for the 0.6B model but faster for the 4B
model** (Q6_K). Run only one of them (also not at the same time as the EmbeddingGemma 2 configurations on the same
port).

| Config | Model | Context | VRAM | RAM | 500-token texts | 100-token texts | Notes |
|---|---|---|---|---|---|---|---|
| [`config-qwen3-embedding-0.6b.yaml`](config-qwen3-embedding-0.6b.yaml) (default) | Qwen3-Embedding-0.6B (BF16 checkpoint in FP32), up to 64 sequences and 16384 tokens per step | 8192 | none | 12.7 GiB (6 GiB of it KV cache) | 2070 t/s | 2040 t/s | 1.5 times faster than llama.cpp (1290 and 1400 t/s) |
| [`config-qwen3-embedding-4b.yaml`](config-qwen3-embedding-4b.yaml) | Qwen3-Embedding-4B (BF16 checkpoint in FP32) | 8192 | none | 28.4 GiB | 170 t/s | 270 t/s | llama.cpp Q6_K is faster (300 and 270 t/s) and uses 4.7 GiB |

The embeddings match sentence-transformers exactly (cosine similarity 1.00000 for both models). Matryoshka truncation
works with the `dimensions` parameter of the API (`hf-overrides: {"is_matryoshka": true}`; vLLM renormalizes).
On the PaperQA2 corpus of [`../vllm-rtx-a4000`](../vllm-rtx-a4000/README.md#embedding-models), the 4B model found the
right chunk among the top 10 for 92 % of the questions and the 0.6B model for 89 %; see also the
[retrieval test](../llama-cpp-zen2-qwen3-embedding/README.md#retrieval-test) of the llama.cpp configuration.

## Hardware and software

| | |
|---|---|
| Computer | Desktop computer with an ASUS ROG Zenith II Extreme (TRX40) motherboard |
| OS | Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0-42-generic |
| CPU | AMD Ryzen Threadripper 3970X (Zen 2, 32 cores, 64 threads, AVX2, no AVX-512 or BF16 instructions) |
| RAM | 128 GB DDR4-2666 (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPUs (not used) | NVIDIA GeForce RTX 3090 (Ampere, 24 GB GDDR6X), AMD Radeon VII (Vega 20, GCN 5, 16 GB HBM2), NVIDIA GeForce GTX TITAN (Kepler, 6 GB GDDR5, unused) |
| vLLM | `vllm/vllm-openai-cpu:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1` (0.33.0rc1.dev320), unchanged, the same as [`../vllm-zen2-embeddinggemma2`](../vllm-zen2-embeddinggemma2) |

## Usage

```sh
docker compose up -d
```

- The selected `config-*.yaml` is given to `vllm serve --config`, with the API key of `../llama-cpp/llama-cpp.env`
  (see [`../llama-cpp`](../llama-cpp/README.md)), on port 9933. The model has to be in the Hugging Face cache
  (`HF_HUB_OFFLINE=1`). The cache directory belongs to root, so download it with a container:

  ```sh
  docker run --rm -v /home/mika/.cache/huggingface/hub:/root/.cache/huggingface/hub --entrypoint hf \
      vllm/vllm-openai-cpu:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1 download Qwen/Qwen3-Embedding-0.6B
  ```

- The API is OpenAI-compatible at http://localhost:9933/v1/embeddings, with the model id `Qwen/Qwen3-Embedding-0.6B`
  or `Qwen/Qwen3-Embedding-4B`. The embeddings have 1024 or 2560 dimensions and are normalized (last-token pooling).
- Queries should have an instruction, e.g.
  `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}`, and the
  documents none (see the [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage)).
- Inputs of up to 8192 tokens work (the model supports 32768); longer ones are rejected with HTTP 400.

## Performance

Measured on 2026-10-31 with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py)
`--tests embedding --embed-texts 64 --embed-tokens 500` (and 100; 16 texts for the 4B model), two runs each, with
[`../benchmark/vllm_test_server.sh`](../benchmark/vllm_test_server.sh) (`GPUS=none`) and the LLM servers of the GPUs
idle. The results of the production configurations are in `../benchmark/results/agx-z2e-kubuntu.jsonl` (labels
starting with "zen2 vLLM ... Qwen3-Embedding").

**The prefix cache:** vLLM enables it for causal pooling models like Qwen3-Embedding (not for EmbeddingGemma 2), and
the benchmark repeats the same texts, so the second run of the 4B model was served from the cache (175 -> 1030 t/s).
The benchmark now sends a random `cache_salt` with each embedding request to vLLM, like with its other requests. The
runs marked "no salt" below were measured before this fix; for the 0.6B model, the 32k tokens of a 64 x 500 request
did not fit into the cache (18,688 tokens), and the results agree with the salted ones.

### Qwen3-Embedding-0.6B (FP32, `VLLM_CPU_KVCACHE_SPACE=4`)

| Image, setting | 500-token texts | 100-token texts |
|---|---|---|
| nightly, 64 sequences, 8192 batched tokens (no salt) | 1980-1990 t/s | 2030-2080 t/s |
| v0.32.0, the same (no salt) | 1990 t/s | 1980-1990 t/s (the first request took 49 s) |
| nightly, 16 sequences, 8192 batched tokens (no salt) | 1890-1900 t/s | 1930-1940 t/s |
| nightly, 64 sequences, 2048 batched tokens (no salt) | 1950 t/s | 1860-1960 t/s |
| nightly, 64 sequences, 4096 batched tokens (no salt) | 2010-2030 t/s | 1970-2040 t/s |
| nightly, 64 sequences, 16384 batched tokens (no salt) | 2110-2120 t/s | 2060-2080 t/s |
| nightly, 128 sequences, 16384 batched tokens (no salt) | 2100-2120 t/s | 2060-2080 t/s |
| nightly, 64 sequences, 8192 batched tokens | 1940-1950 t/s | 2010-2050 t/s |
| **nightly, 64 sequences, 16384 batched tokens** (this configuration) | **2070-2080 t/s** | **2000-2040 t/s** |
| The same, `enable-prefix-caching: false` | 2080-2090 t/s | 2010-2070 t/s |
| Production (`VLLM_CPU_KVCACHE_SPACE=6`): 64 x 500, 64 x 100 | 2070 t/s | 2040 t/s |
| Production: 16 texts of 2000 tokens, 2 texts of 8000 tokens | 1680 t/s, 860 t/s (9.3 s for one) | |

### Qwen3-Embedding-4B (FP32, `VLLM_CPU_KVCACHE_SPACE=6`)

| Setting | 500-token texts | 100-token texts |
|---|---|---|
| 64 sequences, 16384 batched tokens | 170-210 t/s | 236-267 t/s |
| The same, `enable-prefix-caching: false` | 196-219 t/s | 237-267 t/s |
| Production: 16 x 500, 16 x 100 | 172 t/s | 269 t/s |
| Production: 4 texts of 2000 tokens, 1 text of 8000 tokens | 178 t/s, 180 t/s (44 s) | |

- 16384 batched tokens were 6 % faster than 8192 for 500-token texts; the KV cache has to hold them (224 KiB per
  token in FP32 for the 0.6B model, 288 KiB for the 4B model), hence `VLLM_CPU_KVCACHE_SPACE=6`.
- The prefix cache costs nothing and is left on (it helps when many queries share an instruction).
- For the 4B model, llama.cpp with the Q6_K GGUF is up to 1.7 times faster for 500-token texts and as fast for
  100-token texts, and even its f16 GGUF (200 t/s) is faster than vLLM in FP32. The model is compute-bound on this CPU
  (16 llama.cpp threads were 40 % slower than 32), and llama.cpp's Q6_K kernels are its fastest. vLLM has no BF16 or int8 path that would help on Zen 2 (see
  [`../vllm-zen2-embeddinggemma2`](../vllm-zen2-embeddinggemma2/README.md)).
- The settings of the EmbeddingGemma 2 configuration (FP32, automatic thread binding, `torch.compile`) were kept;
  they are measured in [`../vllm-zen2-embeddinggemma2/README.md`](../vllm-zen2-embeddinggemma2/README.md).

## Commands

```sh
docker compose down
cd ../benchmark
GPUS=none DOCKER_ARGS="-e VLLM_CPU_KVCACHE_SPACE=6" ./vllm_test_server.sh \
    vllm/vllm-openai-cpu:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1 /tmp/config-variant.yaml ../llama-cpp/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp/llama-cpp.env --tests embedding \
    --embed-texts 64 --embed-tokens 500 --server-config /tmp/config-variant.yaml --label "..."
docker rm -f vllm-test
```
