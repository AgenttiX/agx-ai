# EmbeddingGemma 2 on the CPU of agx-z2e with vLLM

Serves the text embeddings of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on the
Threadripper 3970X (Zen 2) of agx-z2e with the CPU build of vLLM, so that the GPUs stay free for the LLMs.
[`../llama-cpp-zen2-embeddinggemma2`](../llama-cpp-zen2-embeddinggemma2) serves the same model on the same port with
llama.cpp, which is half as fast on this CPU; run only one of the two.

| Model | Context | VRAM | RAM | 500-token texts | 100-token texts | Single text | Notes |
|---|---|---|---|---|---|---|---|
| EmbeddingGemma 2 (Google's BF16 checkpoint run in FP32), up to 64 sequences per step, 31 pinned threads | 8192 | none | 6.0-6.3 GiB | 4300-4700 t/s | 4900-5000 t/s | 0.18 s (500 tokens), 0.07 s (100 tokens) | 2 times faster than llama.cpp (2200-2300 t/s); needs a vLLM newer than v0.32.0 (see below) |

## Hardware and software

| | |
|---|---|
| Computer | Desktop computer with an ASUS ROG Zenith II Extreme (TRX40) motherboard |
| OS | Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0-42-generic |
| CPU | AMD Ryzen Threadripper 3970X (Zen 2, 32 cores, 64 threads, AVX2, no AVX-512 or BF16 instructions) |
| RAM | 128 GB DDR4-2666 (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPUs (not used) | NVIDIA GeForce RTX 3090 (Ampere, 24 GB GDDR6X), AMD Radeon VII (Vega 20, GCN 5, 16 GB HBM2), NVIDIA GeForce GTX TITAN (Kepler, 6 GB GDDR5, unused) |
| vLLM | `vllm/vllm-openai-cpu:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1` (0.33.0rc1.dev320), unchanged |

## Usage

```sh
docker compose up -d
```

- [`config.yaml`](config.yaml) is given to `vllm serve --config`, with the API key of `../llama-cpp/llama-cpp.env`
  (see [`../llama-cpp`](../llama-cpp/README.md)), on port 9933. The model has to be in the Hugging Face cache
  (`HF_HUB_OFFLINE=1`). The cache directory belongs to root, so download it with a container:

  ```sh
  docker run --rm -v /home/mika/.cache/huggingface/hub:/root/.cache/huggingface/hub --entrypoint hf \
      vllm/vllm-openai-cpu:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1 download google/embeddinggemma-2
  ```

  The server starts in ~50 s.
- The API is OpenAI-compatible at http://localhost:9933/v1/embeddings, with the model id `google/embeddinggemma-2`.
  The embeddings have 768 dimensions and are normalized (mean pooling, as in the sentence-transformers configuration).
- Matryoshka truncation: `"dimensions": 512`, `256` or `128` in the request (`hf-overrides` in
  [`config.yaml`](config.yaml)); vLLM renormalizes the truncated embeddings. Other values are rejected.
- Inputs of up to 8192 tokens work; longer ones are rejected with HTTP 400.
- Use the task prefixes of the [model card](https://huggingface.co/google/embeddinggemma-2#1-task-instruction-prefixes),
  e.g. `task: search result | query: {query}` for queries and `title: none | text: {document}` for documents.
- The vision and audio encoders are disabled (`limit-mm-per-prompt`), so only text is supported.

## v0.32.0 gives wrong embeddings on the CPU

vLLM v0.32.0 (`vllm/vllm-openai-cpu:v0.32.0`), the first release that supports EmbeddingGemma 2, ran fine and fast
(3400-3500 t/s for 500-token texts), but its embeddings had a cosine similarity of only 0.77-0.91 with those of
sentence-transformers. Comparing the hidden states of each layer with transformers showed that the token and
per-layer embeddings were identical, but the output of the first attention layer was not, and it matched
transformers exactly when that was given a causal mask. The model declares its attention as bidirectional
(`AttentionType.ENCODER_ONLY`), but the metadata builder of the CPU attention backend (`CPU_ATTN`) only turned off the
causal mask for cross-attention. The BERT-like models are not affected, as they use the `EncoderOnlyAttention`
wrapper, which sets `causal = False` itself. `attention-backend: CPU_ATTN` and `enforce-eager: true` gave the same
results. vLLM's model code also logs that it forces `TRITON_ATTN`, but on the CPU the result was the same.

This was fixed upstream by [vllm-project/vllm#63738](https://github.com/vllm-project/vllm/pull/63738) on 2026-10-27,
so this configuration uses an unchanged nightly image of vLLM's main branch from 2026-10-30 instead of patching
v0.32.0. With it, the embeddings match sentence-transformers with a cosine similarity of 1.00000 (six texts of
16-1500 tokens, including ones longer than the 512-token sliding window), and it is also 25-45 % faster. Switch to
v0.33.0 once it is released.
The two runs labeled "zen2 vLLM 0.32.0" in `../benchmark/results/agx-z2e-kubuntu.jsonl` are of the broken version.

## Performance

Measured on 2026-10-30 with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py)
`--tests embedding --embed-texts 64 --embed-tokens 500` (and 100), i.e. one request with 64 texts, with
[`../benchmark/vllm_test_server.sh`](../benchmark/vllm_test_server.sh) (`GPUS=none`) and the LLM servers of the GPUs
idle. The results of the production configuration are in `../benchmark/results/agx-z2e-kubuntu.jsonl` (labels
starting with "zen2 vLLM").

| Image, setting | 500-token texts | 100-token texts | Notes |
|---|---|---|---|
| v0.32.0, FP32, 8 sequences, 8192 batched tokens | 3490-3550 t/s | 2390-2430 t/s | Wrong embeddings (see above) |
| v0.32.0, FP32, 16 sequences, 16384 batched tokens | 3500-3560 t/s | 2910-2960 t/s | |
| v0.32.0, FP32, 64 sequences, 4096 batched tokens | 3530-3540 t/s | 3310-3320 t/s | |
| v0.32.0, FP32, 64 sequences, 8192 batched tokens | 3490-3560 t/s | 3330-3420 t/s | |
| v0.32.0, FP32, 64 sequences, 16384 batched tokens | 3470-3510 t/s | 3350-3410 t/s | |
| v0.32.0, FP32, 64 sequences, 32768 batched tokens | 3110-3250 t/s | 3390-3440 t/s | |
| v0.32.0, FP32, 8 sequences, 32 threads on the cores 0-31 | 3340-3430 t/s | 2490-2590 t/s | |
| v0.32.0, BF16 | 680-690 t/s | 680-690 t/s | |
| v0.32.0, FP16, 64 sequences | 550-570 t/s | 680 t/s | |
| nightly, FP32, 16 sequences, 16384 batched tokens | 4430-4440 t/s | 4310-4390 t/s | |
| nightly, FP32, 64 sequences, 16384 batched tokens | 4480-4510 t/s | 4660-4900 t/s | |
| nightly, FP32, 64 sequences, 8192 batched tokens, 32 threads on the cores 0-31 | 4260-4350 t/s | 4570-4770 t/s | |
| **nightly, FP32, 64 sequences, 8192 batched tokens** (this configuration) | **4300-4680 t/s** | **4880-5030 t/s** | |
| The same, 16 texts of 2000 tokens | 3430 t/s | | Single text 0.59 s |
| The same, 2 texts of 8000 tokens | 1570 t/s | | Single text 5.1 s |

- FP32: the Zen 2 CPU has no BF16 instructions (AVX-512 BF16 or AMX), and PyTorch's BF16 and FP16 matrix
  multiplications were 5-6 times slower. The BF16 checkpoint converts to FP32 exactly, so this loses nothing.
  (In llama.cpp, the GGUF type only sets how the weights are stored, and its BF16 weights are computed in FP32.)
- The default thread binding (`VLLM_CPU_OMP_THREADS_BIND=auto`) pins one OpenMP thread to each physical core except
  one, which it leaves for the engine and the API server (cores 32-62, the SMT siblings of 0-30, which are the same
  physical cores). Binding 32 threads to the cores 0-31 was not faster.
- More sequences per step help short texts: 64 sequences were 40 % faster than 8 for 100-token texts. The number of
  batched tokens (4096-16384) made no difference beyond the noise.
- vLLM batches all texts of a request (up to 64 sequences and 8192 tokens) into one forward pass, while llama.cpp
  processes at most one text per slot (8) in a ubatch, which likely explains part of the difference.
