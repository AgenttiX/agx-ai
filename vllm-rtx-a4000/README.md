# vLLM on 2x RTX A4000

vLLM for [PaperQA2](https://github.com/Future-House/paper-qa) RAG on a workstation with 2x RTX A4000 (16 GB each),
as an alternative to [`../llama-cpp-rtx-a4000`](../llama-cpp-rtx-a4000). The LLM is served on port 9932 and the
embedding model on port 9934, with the API key of the llama.cpp server, so both can run at the same time on different
GPUs. [`../../phd-research/phd_code/tools/paperqa_ask.py`](../../phd-research/phd_code/tools/paperqa_ask.py) uses
them with `PAPERQA_BACKEND=vllm`.

**Summary.** For PaperQA2, the time per question is dominated by the generated tokens (the evidence summaries), not
by prompt processing. With the same model, vLLM answers 2.2-2.5 times as fast as llama.cpp and builds the index 2-7
times as fast, as it batches the concurrent requests, processes prompts faster and, for Qwen3.8-27B, runs
[HyperQwen](#hyperqwen-and-the-int8-kv-cache-2026-10-26) (see [Comparison with llama.cpp](#comparison-with-llama-cpp)).
The largest effect is the model and its thinking, though: Gemma 4 26B-A4B without thinking answers a question in
~10 s, Qwen3.8-27B in ~25 s without thinking and ~43 s with reasoning effort "low", with the most complete answers.
[Recommendation](#recommendation): configuration 3 (Qwen3.8-27B with HyperQwen and Qwen3-Embedding-0.6B on both GPUs),
the default.

## Configurations

The configuration is selected in [`.env`](.env) (`COMPOSE_PROFILES` and the config files), so that a plain
`docker compose up -d`, which `paperqa_ask.py` also runs when the server is down, starts it. Context = `max-model-len`.
The profile `llm` starts stock vLLM with `LLM_CONFIG`, the profile `hyperqwen` HyperQwen for Qwen3.8-27B (configured in
[`docker-compose.yml`](docker-compose.yml)).

| # | GPUs | LLM (port 9932) | Embedding model (port 9934) | Context | LLM chat | PaperQA2 per question |
|---|---|---|---|---|---|---|
| 1 | 0+1 | Qwen3.8-27B, TP 2, MTP | - (with 8: CPU) | 229376 | 55 t/s | 56-61 s (1 + 8) |
| 2 | 0+1 | Gemma 4 26B-A4B QAT, TP 2, MTP | - (with 8: CPU) | 262144 | 198 t/s | 8-11 s without thinking (2 + 8) |
| 3 | 0+1 | Qwen3.8-27B, HyperQwen, TP 2, DFlash2 | Qwen3-Embedding-0.6B, TP 2 | 262144 | 98 t/s | 40-46 s, 24-26 s without thinking |
| 3a | 0+1 | Qwen3.8-27B, TP 2, MTP | Qwen3-Embedding-0.6B, TP 2 | 196608 | 53 t/s | 61-70 s, 31-34 s without thinking |
| 4 | 0+1 | Gemma 4 26B-A4B QAT, TP 2, MTP | Qwen3-Embedding-4B (FP8), TP 2 | 114688 | 195 t/s | 10-11 s without thinking |
| 5 | 1 | Qwen3.8-27B (int4 embedding) | - (with 7: GPU 0) | 40960 | 27 t/s | 130-150 s, 53-55 s without thinking (5 + 7) |
| 6 | 1 | Gemma 4 26B-A4B QAT (int4 embedding) | - (with 7: GPU 0) | 30720 | 125 t/s | 13-15 s without thinking (6 + 7) |
| 7 | 0 | - | Qwen3-Embedding-4B | 8192 | - | 7000 t/s of embeddings |
| 8 | CPU | - | Qwen3-Embedding-0.6B (16 threads) | 8192 | - | 1100 t/s of embeddings |

PaperQA2 times: the three test questions below, with the default `PAPERQA_THINKING=low` unless noted. Gemma with
thinking took 68-116 s per question and failed one of the three (see [PaperQA2](#paperqa2)). Configuration 3a is the
stock-vLLM alternative to 3 (block 3a in `.env`), and HyperQwen also
runs without the embedding server (`COMPOSE_PROFILES=hyperqwen`), faster than 1.

The YAML files for `vllm serve --config` are `config-*.yaml`; each one explains its settings. The compose file sets
the environment shared by all of them (`NCCL_P2P_DISABLE`, `PYTORCH_CUDA_ALLOC_CONF`, the caches) and memory limits,
and the settings of HyperQwen.

## Hardware

| | |
|---|---|
| OS | Ubuntu 24.04.5 LTS (kernel 7.0.0-30-generic), NVIDIA driver 610.57.04 (CUDA 13.3) |
| CPU | AMD Ryzen Threadripper PRO 3975WX (32 cores, 64 threads, Zen 2: AVX2, no AVX-512 or BF16) |
| Computer | Lenovo ThinkStation P620 (30E0003EMT) |
| RAM | 192 GB DDR4-3200 registered ECC in 8 channels, one DIMM per channel: 4 x 32 GB Micron 36ASF4G72PZ-3G2R1 and 4 x 16 GB Micron 18ASF2G72PDZ-3G2E1, all dual-rank, running at 3200 MT/s (`dmidecode`) |
| NPU | none |
| GPU | 2 x NVIDIA RTX A4000, PCIe 4.0 x16 each, no NVLink; GPU 0 also runs the desktop session (Xorg, ~110 MiB) |
| GPU architecture | Ampere (GA104, compute capability 8.6) |
| VRAM | 16 GB GDDR6 with ECC per GPU (16376 MiB, of which CUDA sees 15.6 GiB) |

The machine is shared: other users' simulations run on ~10 cores, so the configurations use at most 16 threads and
the containers have memory limits (40 GiB for the LLM, 24 GiB and 16 GiB for the embedding models).

## Setup

```sh
# Image: vLLM v0.31.0 with ../vllm/vllm.patch (also built by ../.github/workflows/deploy-docker-vllm.yml)
docker build -t ghcr.io/agenttix/agx-ai/vllm:latest ../vllm
# Checkpoints into the Hugging Face cache
uvx --from 'huggingface_hub[hf_xet]' hf download cyankiwi/Qwen3.8-27B-AWQ-INT4
uvx --from 'huggingface_hub[hf_xet]' hf download JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
uvx --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen3-Embedding-4B
uvx --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen3-Embedding-0.6B
# Gemma: the bit-exact QAT checkpoints (~10 min each on the CPU, ~30 GB of RAM), see ../vllm-radeon-vii/README.md
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub -v ~/.cache/vllm-models:/out \
    --entrypoint repack_gemma4_qat.py ghcr.io/agenttix/agx-ai/vllm:latest --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub -v ~/.cache/vllm-models:/out \
    --entrypoint repack_gemma4_qat.py ghcr.io/agenttix/agx-ai/vllm:latest \
    --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 --embed-bits 16 --lm-head-bits 8
# HyperQwen: its checkpoints into ~/.cache/vllm-models/hyperqwen (~22 GB of downloads, ~6 min)
docker compose run --rm hyperqwen-prepare
# Select the configuration in .env, stop the llama.cpp server if it uses the same GPUs, and start
docker compose -f ../llama-cpp-rtx-a4000/docker-compose.yml down
docker compose up -d
```

The first start compiles kernels for 2-4 min; later starts take 1-2.5 min (the caches are in the `vllm-cache`
volume), with HyperQwen 6-7 min (its caches are in the `hyperqwen-cache` volume). To switch configurations, stop all
profiles first: `docker compose --profile llm --profile hyperqwen --profile embedding --profile embedding-cpu down`.
For a one-off configuration, the variables of `.env` can also be given on the command line, e.g.
`COMPOSE_PROFILES=llm LLM_CONFIG=config-gemma-4-26b-a4b-qat-1gpu.yaml LLM_GPUS=1 docker compose up -d`.

## Images

- The GPU configurations use `ghcr.io/agenttix/agx-ai/vllm:latest`: `vllm/vllm-openai:v0.31.0` (CUDA 13.0) with the
  fixes of [`../vllm/vllm.patch`](../vllm/vllm.patch) and the pth-flash-attn kernel, see
  [`../vllm/README.md`](../vllm/README.md).
- HyperQwen uses its own image of its commit `ce9615c` (2026-10-22, vLLM 0.31.0), pinned by digest, as
  [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md).
- The CPU embedding server uses `vllm/vllm-openai-cpu:v0.31.0` unchanged.

## Peer-to-peer copies between the GPUs are broken

`nvidia-smi topo -p2p r` reports peer-to-peer reads as OK and PyTorch's `can_device_access_peer(0, 1)` returns
`True`, but a GPU-to-GPU copy of 64 MiB arrives with wrong data. Tensor parallelism therefore hung:

- NCCL hung in its initialization (both workers at 100 % CPU for 10+ min after `vLLM is using nccl==2.30.7`).
  `NCCL_P2P_DISABLE=1` makes NCCL communicate through host memory (in `docker-compose.yml`).
- vLLM's custom all-reduce also uses peer-to-peer writes: with it, the engine hung after the warmup, both GPUs at
  100 % utilization at idle power. `disable-custom-all-reduce: true` in the TP configurations.

This is typical of the AMD IOMMU translating the peer-to-peer DMA. The kernel command line has neither `iommu=pt` nor
`amd_iommu=off`; with `iommu=pt` (needs root and a reboot), peer-to-peer copies, NCCL P2P and the custom all-reduce
would likely work and make tensor parallelism faster. Without them, Qwen3.8-27B generates 38 t/s with TP 2 (without
MTP), while the weights alone would allow ~45 t/s.

## Checkpoints

| Model | Checkpoint | On the GPU |
|---|---|---|
| Qwen3.8-27B, HyperQwen (3) | HyperQwen's re-quantization of [`dbirks/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) (the "fast" variant: lm_head and MTP module in int4 GPTQ, embedding in int8 in pinned RAM), and the [DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) in W4A16 | 7.3 GiB per GPU with the drafter |
| Qwen3.8-27B, 2 GPUs | [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4): AWQ int4, group size 32, asymmetric; Gated DeltaNet layers, embedding, lm_head and MTP head in BF16 | 9.3 GiB per GPU |
| Qwen3.8-27B, 1 GPU | [`JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB`](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB) (`mtp-int8` revision): GPTQ int4, group size 128, embedding and lm_head int4 | 13.1 GiB |
| Gemma 4 26B-A4B QAT, 2 GPUs | Built from Google's Q4_0 GGUF (bit-exact int4, group size 32) with `--embed-bits 16 --lm-head-bits 8` | 8.8 GiB per GPU with the MTP drafter |
| Gemma 4 26B-A4B QAT, 1 GPU | The same with the embedding and the lm_head in int4 | 14.0 GiB |
| Gemma 4 MTP drafter | [`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant) (BF16), shares the KV cache of the main model | (included above) |

All run with the Marlin kernels (`MarlinLinearKernel`, and the `MARLIN` WNA16 MoE backend for Gemma's experts,
which handles their intermediate size of 704). Gemma runs in FP16 like the scales of the Q4_0 GGUF, which BF16 would
round; the repacking script [`../vllm/repack_gemma4_qat.py`](../vllm/repack_gemma4_qat.py) is shared with
[`../vllm-radeon-vii`](../vllm-radeon-vii/README.md#checkpoints-2026-10-09).
NeoChen1024/gemma-4-26B-A4B-it-qat-W4A16 is the same conversion but with the embedding and the vision tower in BF16
and Marlin padding (19.2 GB), too large here. EmbeddingGemma 2 is not supported by vLLM 0.31.

## Memory

- **2 GPUs, LLM alone** (1, 2): `gpu-memory-utilization: 0.95`. Qwen gets 4.46 GiB of KV cache per GPU (int8,
  256682 tokens, for 229376 of context), Gemma 5.0 GiB (374k tokens, as only its 5 global layers have a full KV cache).
- **2 GPUs, LLM and embedding model** (3, 3a, 4): both are split with tensor parallelism, as the LLM gets the same KV
  cache on both GPUs. Both KV caches are set explicitly (`kv-cache-memory-bytes`, HyperQwen's `KV_MEM`). vLLM still
  checks at startup that `gpu-memory-utilization` of the GPU memory is free, and the two servers start at the same
  time, so the embedding server has `0.1` and the LLM a value below what the embedding server leaves free. Next to the
  4B embedding model (3.5 GiB per GPU with its runtime, FP8), the stock Qwen with a BF16 KV cache had only ~1 GiB of KV
  cache per GPU (~26k tokens), so configurations 3 and 3a use the 0.6B model (1.5 GiB per GPU). With HyperQwen's
  smaller weights (7.3 instead of 9.5 GiB per GPU), configuration 3 has 5.6 GB of KV cache per GPU (int8, 299648
  tokens) for the full 262144 tokens of context; 3a has 3.9 GB (int8, 209818 tokens) for 196608 tokens. Gemma's KV
  cache is compact, so configuration 4 uses the 4B model and gets 114688 tokens (121k in the KV cache). With 2.1 GB of
  KV cache (131072 tokens), the embedding server ran out of memory under load after both had started at the same time.
- **1 GPU** (5, 6): on GPU 1, which has no desktop session. As on the Radeon VII, the KV cache is set explicitly
  (profiling leaves only 0.16 GiB for Gemma), int8 per token and head, with Triton attention, smaller prefill chunks
  and fewer CUDA graphs. FlashAttention 2, vLLM's default on Ampere, supports no quantized KV cache.
- Every configuration was stress-tested with a prompt of almost the full context and concurrent requests, while
  embedding batches ran on the same GPUs (`llama_cpp_bench_http.py --mixed`), configurations 3 and 4 after starting
  both servers at the same time with `docker compose up -d`. The peaks were 15600-15960 MiB of the 16376 MiB
  (configuration 3: 15723 MiB, with a 255k-token prompt 15697 MiB).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is needed for the single-GPU Gemma to load at all: without it,
  the Marlin repacking of the experts after loading ran out of memory.

## Speed

Measured with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py) (the results are in
[`../benchmark/results/big-machine.jsonl`](../benchmark/results/big-machine.jsonl)), with the previous llama.cpp
production preset (build b11046, Qwen3.8-27B UD-Q4_K_M with MTP, tensor split, and Qwen3-Embedding-8B) for
comparison; the current llama.cpp presets are compared in [Comparison with llama.cpp](#comparison-with-llama-cpp):

| Test | Qwen TP 2 (1) | Qwen 1 GPU (5) | Gemma TP 2 (2) | Gemma 1 GPU (6) | llama.cpp Qwen 2 GPUs (b11046) |
|---|---|---|---|---|---|
| Generation, 256 tokens | 63.2 | 27.5 | 253.6 | 129.7 | 61.3 |
| Chat, 6 requests | 55.4 | 27.1 | 198.3 | 125.5 | 55.5 |
| Code context of 6.4k tokens | 47.4 (1405) | 25.9 (813) | 140.4 (4764) | 101.2 (3345) | 52.3 (956) |
| Code context of 32k tokens | 32.6 (1282) | 22.1 (490) | 77.5 (3175) | 83.8 (2619) at 16k | 49.2 (929) |
| Prompt of 16k tokens | (1401) | (628 at 20k) | (4184) | (2175 at 29k) | (1003) |
| 4 x (8000-token prompt + 256 generated) | 28.0 s | - | 9.0 s | - | 43.1 s |
| 8 x (8000-token prompt + 256 generated) | 55.7 s | - | 16.8 s | - | 95.3 s |

Generation in t/s, prompt processing in parentheses. vLLM processes prompts 1.4 times as fast as llama.cpp with the
same model and finishes concurrent requests ~1.6 times as fast, but its attention slowed down more with the context
length (Qwen at 32k tokens: 32.6 vs 49.2 t/s). These Qwen measurements are from before the int8 KV cache and
HyperQwen; for the current ones, see [below](#hyperqwen-and-the-int8-kv-cache-2026-10-26).

**MTP.** Qwen's own MTP head with 2 drafted tokens: chat 37.5 -> 55.4 t/s, 6.4k context 36.6 -> 47.4 t/s, at the
cost of 0.6 GiB per GPU (KV cache 142k -> 105k tokens at `gpu-memory-utilization: 0.92`). Gemma's drafter: 198 t/s
in chat. On one GPU, MTP does not fit next to a useful context (as on the Radeon VII).

**Prefill chunk.** `max-num-batched-tokens: 8192` instead of the default 2048: no faster prompt processing
(1401 vs 1444 t/s) and 0.5 GiB less KV cache, so the default is kept for 2 GPUs.

**Embeddings.** Qwen3-Embedding-4B on one GPU: 7000 t/s (16 x 1024 tokens); 0.6B on both GPUs: 34000 t/s; 8B in
FP8 on one GPU: 3400 t/s. On the CPU, the 0.6B model embeds 1100 t/s with 16 threads, 940 t/s with 24 and 1120 t/s
with 32, i.e. more threads do not help next to the other users' jobs. llama.cpp (one slot, as in its preset):
4B Q6_K ~3200 t/s, 8B Q6_K ~2000 t/s.

## HyperQwen and the int8 KV cache (2026-10-26)

Configurations 3 and 1 (Qwen3.8-27B on both GPUs) with the optimizations of the RTX 3090 configurations of agx-z2e:

- From [`../vllm-agx-z2e`](../vllm-agx-z2e/README.md#qwen38-27b): the int8 KV cache read by the pth-flash-attn
  kernel of [`../vllm`](../vllm/README.md) (`attention-backend: CUSTOM`, also for the MTP drafter), and the recurrent
  state in FP16. Now in the stock-vLLM configurations (1, 3a).
- From [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md): HyperQwen's image, checkpoint and DFlash2
  drafter (configuration 3). HyperQwen supports tensor parallelism; its authors measured it on 2x RTX 3080 without
  peer-to-peer copies, like here. It runs with `--tensor-parallel-size 2 --disable-custom-all-reduce` and
  `NCCL_P2P_DISABLE=1`, and `EMBED_UVA=1` keeps the int8 token embedding in pinned RAM.
- From [`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next/README.md): nothing. Its runtime, CPU expert kernel
  and CPU vision tower are for Qwen3.8-Flash-Next with its experts in RAM, not for a dense model on two GPUs.

All next to the 0.6B embedding server (TP 2), with the benchmark of [Commands](#commands) (`--depth 6400 32000 80000`,
plus the `edit` test of [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md#hyperqwens-modes): a
2.2k-token file output again with one identifier renamed). The results are in
[`../benchmark/results/big-machine.jsonl`](../benchmark/results/big-machine.jsonl):

| Test | Stock vLLM, BF16 KV (3 before) | Stock vLLM, int8 KV + pth-flash-attn (3a) | HyperQwen `SPEC=mtp` | **HyperQwen `SPEC=dflash2` (3)** |
|---|---|---|---|---|
| Context | 94208 | 196608 | 131072 (tested) | **262144** |
| Generation, 256 tokens | 63.5 | 62.5 | 92.8 | **90.4** |
| Chat, 6 requests | 54.6 | 52.5 | 91.2 | **98.0** |
| Code context of 6.4k tokens | 49.0 (1327) | 56.8 (1305) | 93.6 (1388) | **91.7 (1389)** |
| Code context of 32k tokens | 31.2 (1203) | 54.3 (1178) | 91.8 (1251) | **89.5 (1259)** |
| Code context of 80k tokens | 20.8 (1034) | 53.3 (998) | 85.2 (1053) | **85.8 (1058)** |
| Code context of 200k tokens | - | - | - | **86.3 (754)** |
| Edit (2.2k tokens copied with a change) | 72.0 | 77.0 | 134.6 | **243.5** |
| Prompt of 16k tokens | (1328) | (1295) | (1353) | **(1366)** |
| 4 x (8000-token prompt + 256 generated) | 29.4 s | 29.0 s | 25.9 s | **25.0 s** |
| 8 x (8000-token prompt + 256 generated), with embeddings | 116.2 s | 115.3 s | 99.5 s | **99.8 s** |
| PaperQA2, thinking "low" (q1 / q2 / q3) | (60-64 s) | 61 / 61 / 70 s | 44 / 46 / 48 s | **40 / 41 / 46 s** |
| PaperQA2, no thinking | (29-32 s) | 32 / 31 / 34 s | 26 / 27 / 29 s | **24 / 25 / 26 s** |
| Peak VRAM per GPU (with the embedding server) | 15873 MiB | 15907 MiB | 13275 MiB (4 GB KV cache) | **15723 MiB** |

Generation in t/s, prompt processing in parentheses. The PaperQA2 times are with the 75 papers of 2026-10-26 (index
built in 126 s); the ones in parentheses are from the [PaperQA2](#paperqa2) section (71 papers). The HyperQwen modes
ran with a 4 GB KV cache (KV_MEM) except the final column, whose larger cache made no difference to the speed in
the other tests.

- **HyperQwen with DFlash2 is the fastest** in everything except the 4 concurrent requests, where its MTP mode is as
  fast: PaperQA2 answers 1.5 times as fast as with stock vLLM (and 2.2 times as fast as llama.cpp), chats 1.8 times,
  long contexts 2.9-4.1 times, and code edits 3.4 times, with the full context of the model next to the embedding
  server. HyperQwen's checkpoint is a different int4 quantization (AutoRound, int4 lm_head), but its answers covered
  the same points with the same excerpts as stock vLLM's (compared for question 2 in both modes). The draft
  acceptance is lower than with MTP (~30 % of 7 drafted tokens per step vs ~55 % of 2-3), but more tokens are
  accepted per step.
- **The int8 KV cache with pth-flash-attn** (3a) does not change the speed at short contexts, but keeps the
  generation at ~54 t/s up to 80k tokens of context (BF16 with FlashAttention: 31 t/s at 32k, 21 t/s at 80k), and
  the KV cache holds twice as many tokens.
- **The prompt processing** is about the same in all variants (1300-1400 t/s at 16k), so the concurrency tests,
  which are mostly prompt processing, improve the least. HyperQwen's `INT8_ACT` (W4A8 prefill, +20-30 % prompt
  processing on the RTX 3090) was not tried, as it costs +4 % perplexity.
- **No CPU KV offloading in configuration 3**: the GPUs hold 299648 tokens, so a conversation evicted from them has
  had more than 8 GiB of newer KV blocks (~0.1 MB per token with the recurrent-state snapshots) written to the CPU
  cache after it, and is gone from there too. Its second worker also could not pin the region at TP 2
  (`cudaHostRegister failed for rank=1`), so it would stay pageable. The stock configurations keep their 8 GiB.
- HyperQwen's launcher warns that `KV_MEM` is above its default; that limit is for the whole model on one 24 GB
  GPU, while `KV_MEM` is per GPU here. The stress test passed (15697 MiB with a 255k-token prompt, 4 and 8 concurrent
  30k-token prompts with embeddings at the same time).
- HyperQwen takes 6-7 min to start (stock vLLM: 3-5 min). Its MTP mode leaves more context in the same memory
  (8 instead of 4 sequences, 206289 instead of 195715 tokens with 4 GB), but is not needed for PaperQA2's 4
  concurrent requests.

## Embedding models

Which embedding model finds the right excerpts on the actual corpus? [`../benchmark/embedding_retrieval_eval.py`](../benchmark/embedding_retrieval_eval.py)
gave 300 random chunks of the PaperQA2 index (1668 chunks of 5000 characters, ~2M tokens, 71 papers) to Gemma 4,
which wrote one specific question per chunk (269 questions; 31 chunks such as reference lists were skipped). Each
embedding model then ranked all 1668 chunks for each question, without an instruction prefix, as PaperQA2 does.
Recall@k: how often the chunk that the question was written from is among the top k. PaperQA2 gives the top
`evidence_k = 10` chunks to the LLM. The results are in
[`../benchmark/results/big-machine-embedding.jsonl`](../benchmark/results/big-machine-embedding.jsonl).

| Model | Server | R@1 | R@5 | R@10 | MRR | Dimensions | Time for the 1668 chunks |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-4B BF16 | vLLM, 1 GPU | 0.651 | 0.877 | **0.922** | 0.757 | 2560 | 308 s |
| Qwen3-Embedding-4B FP8 | vLLM, 1 GPU | 0.651 | 0.874 | **0.922** | 0.756 | 2560 | 361 s |
| Qwen3-Embedding-4B Q6_K | llama.cpp, 1 GPU | 0.654 | 0.874 | **0.922** | 0.756 | 2560 | 636 s |
| Qwen3-Embedding-8B FP8 | vLLM, 1 GPU | 0.651 | 0.881 | 0.918 | 0.755 | 4096 | 680 s |
| Qwen3-Embedding-8B BF16 | vLLM, TP 2 | 0.643 | 0.874 | 0.918 | 0.754 | 4096 | 418 s |
| Qwen3-Embedding-8B Q6_K (llama.cpp production) | llama.cpp, 1 GPU | 0.639 | 0.874 | 0.914 | 0.748 | 4096 | 1025 s |
| Qwen3-Embedding-0.6B BF16 | vLLM, 1 GPU | 0.569 | 0.844 | 0.892 | 0.687 | 1024 | 51 s |
| EmbeddingGemma 2 BF16 | llama.cpp, 1 GPU | 0.320 | 0.554 | 0.673 | 0.430 | 768 | 265 s |

- The 4B model retrieves as well as the 8B model on this corpus, at about half the time and memory. FP8 and Q6_K
  lose nothing measurable.
- The 0.6B model misses the right chunk among the top 10 for 3 more questions in 100, but embeds 6 times as fast.
- EmbeddingGemma 2 (September 2026, 768 dimensions) is trained with task prefixes (`task: search result | query: `,
  `title: none | text: `), which PaperQA2 does not add; without them, it is far behind. vLLM 0.31 does not support it.
- The questions are generated from single chunks, and some could be answered by several chunks or papers, so the
  absolute recall is a lower bound; the comparison between the models is what matters.

## PaperQA2

[`paperqa_ask.py`](../../phd-research/phd_code/tools/paperqa_ask.py) with `PAPERQA_BACKEND=vllm`, a separate index per
embedding model (`PAPERQA_INDEX_DIR`), and three questions:

1. What sets the peak frequency of the sound-wave GW spectrum?
2. How does the sound speed affect the GW spectrum from sound waves?
3. What are the main sources of uncertainty in predicting the gravitational wave spectrum from a first-order phase
   transition?

Time per question in s (the `ask` call; +3 s for starting Python), and the number of excerpts used for the answer
(contexts, at most 10):

| Configuration | Thinking | q1 | q2 | q3 | Generated tokens per question |
|---|---|---|---|---|---|
| 1 + 8: Qwen TP 2, 0.6B on the CPU | low | 56 (10) | 59 (10) | 61 (10) | 5600-6300 |
| 3: Qwen TP 2 + 0.6B TP 2 | low | 60 (10) | 62 (10) | 64 (8) | 5600-6100 |
| 3: Qwen TP 2 + 0.6B TP 2 | off | 29 (10) | 29 (10) | 32 (7) | 1700-2000 |
| 5 + 7: Qwen 1 GPU, 4B on GPU 0 | low | 130 (10) | 146 (10) | 150 (10) | 5800-7300 |
| 5 + 7: Qwen 1 GPU, 4B on GPU 0 | off | 53 (10) | 55 (8) | 54 (7) | 1600-2000 |
| 2 + 8: Gemma TP 2, 0.6B on the CPU | on | 84 (9) | 89 (6) | 105 (2) | 20000-24000 |
| 2 + 8: Gemma TP 2, 0.6B on the CPU | off | 8 (10) | 11 (10) | 9 (4) | 1200-2000 |
| 4: Gemma TP 2 + 4B TP 2 | on | 94 (9) | 68 (7) | 113 (3) | 15000-24000 |
| 4: Gemma TP 2 + 4B TP 2 | off | 10 (10) | 11 (10) | 10 (5) | 1500-2000 |
| 6 + 7: Gemma 1 GPU, 4B on GPU 0 | off | 13 (10) | 14 (6) | 15 (5) | 1600-1700 |

Building the index of the 71 papers (2M tokens): 97 s with the 0.6B model on both GPUs (3), 309 s with the 4B model
next to Gemma (4), 39 min with the 0.6B model on the CPU (8).

Findings:

- **The generated tokens dominate.** A question needs ~16-22k prompt tokens (the evidence summaries of 10-20
  excerpts and the answer), which take only a few seconds, but Qwen at reasoning effort "low" generates ~600 tokens per
  summary. So the generation speed under 4 concurrent requests and the amount of thinking decide.
- **Gemma 4 thinks much more.** It has no effort levels, so `PAPERQA_THINKING=low` means full thinking: up to 3000
  tokens per summary, ~22k per question, and fewer excerpts were judged relevant. PaperQA2 sends temperature 0, and in
  the first run one response looped until the 262k context was full (the speculative decoding acceptance at 100 %
  for minutes) and the agent timed out after 500 s. `paperqa_ask.py` therefore caps the generated tokens per request
  (`PAPERQA_MAX_TOKENS`, default 8192). Even then, the third question failed once, as the answer step got no content.
  Without thinking (`PAPERQA_THINKING=0`), Gemma is reliable and 6-9 times as fast.
- **Prefix caching and CPU KV offloading do not help PaperQA2.** Asking question 1 again took as long (56 s): 1600 of
  72k prompt tokens hit the prefix cache, none the CPU cache, as the summaries are generated anew and the KV cache
  on the GPUs (75-374k tokens) is far larger than one question needs. For Qwen, the Gated DeltaNet states are also
  only cached at 784-token blocks. The offloading is kept for long chats in the 2-GPU configurations (8 GiB of pinned
  RAM), but not on 1 GPU, where its GPU buffers made Qwen run out of memory.
- The answer quality is compared in [Comparison with llama.cpp](#comparison-with-llama-cpp).

## Comparison with llama.cpp

The llama.cpp presets of [`../llama-cpp-rtx-a4000`](../llama-cpp-rtx-a4000) on the same machine (build b11541,
see its README), with the same benchmark and the same PaperQA2 questions:

| | llama.cpp | vLLM |
|---|---|---|
| **Qwen3.8-27B, 2 GPUs** | UD-Q4_K_M, tensor split, MTP, + Qwen3-Embedding-4B Q6_K | HyperQwen (W4A16), TP 2, DFlash2, + Qwen3-Embedding-0.6B (3) |
| Context | 98304 | 262144 |
| Chat / 32k context | 52.5 / 47.7 t/s | 98.0 / 89.5 t/s |
| Prompt processing (16k) | 981 t/s | 1366 t/s |
| 4 x 8000-token prompts | 44.7 s | 25.0 s |
| PaperQA2, thinking "low" | 89-103 s per question | 40-46 s |
| PaperQA2, no thinking | 47-50 s | 24-26 s |
| **Gemma 4 26B-A4B QAT, 2 GPUs** | UD-Q4_K_XL, layer split, MTP, + Qwen3-Embedding-4B Q6_K | int4 (Google's Q4_0), TP 2, MTP, + Qwen3-Embedding-4B FP8 (4) |
| Context | 262144 | 114688 (262144 without the embedding model) |
| Chat / 32k context | 134.7 / 72.4 t/s | 198.3 / 77.5 t/s |
| Prompt processing (16k) | 3306 t/s | 4184 t/s |
| 4 x 8000-token prompts | 16.7 s | 9.0 s |
| PaperQA2, no thinking | 25-29 s | 10-11 s |
| **Gemma 4 26B-A4B QAT, 1 GPU** | 19 layers of experts on the CPU, + embedding on GPU 0 | all on GPU 1 (6), + Qwen3-Embedding-4B on GPU 0 (7) |
| Context | 262144 | 30720 |
| Chat / 32k context | 60.6 / 53.3 t/s | 125.5 / 83.8 t/s (at 16k) |
| 4 x 8000-token prompts | 25.4 s | 13.2 s (4 x 6000) |
| PaperQA2, no thinking | 34-40 s | 13-15 s |
| **Building the PaperQA2 index** | 702 s (4B, 1 slot) | 97 s (0.6B, 3), 309 s (4B FP8, 4) |
| **Start** | ~15 s; the models are unloaded after 10 min of inactivity and reloaded in seconds | 1.5-2.5 min; the GPU memory stays allocated |

- **Speed.** With the same model, vLLM answers PaperQA2 questions 2.2 (Qwen with HyperQwen; 1.6 with stock vLLM) to
  2.5 (Gemma) times as fast and builds the index 2-7 times as fast. It processes prompts faster, batches the
  concurrent requests better, and its embedding server is twice as fast as llama.cpp's single slot. llama.cpp fits
  more context with Gemma on one GPU (262144 with the experts on the CPU vs 30720). Before the int8 KV cache and
  HyperQwen, llama.cpp generated faster with Qwen at long contexts (32k: 47.7 vs 32.6 t/s); now vLLM does (89.5 t/s).
- **Quality.** The answers of the same model on the two backends are equivalent; the differences between the runs
  are within the variation of the excerpts that are selected. Between the models, Qwen3.8-27B with thinking at
  "low" gave the most complete and precise answers (e.g. for question 2: the shift of the peak to $k_* \sim 2 c_s k_p$,
  the low-frequency tilt $k^{3-2\nu}$, the change of the fluid solutions and the $f_2 \propto 1/c_s$ break). Without
  thinking, Qwen covered nearly the same points. Gemma without thinking gave well-structured but shallower answers,
  missing e.g. the peak shift in question 2 and keeping fewer excerpts in question 3 (2-5 instead of 7-10).
- **Mixed.** llama.cpp for the LLM and vLLM for the embeddings works (`PAPERQA_EMBED_BACKEND=vllm`; tested with
  llama.cpp's single-GPU Gemma and configuration 7: 33-38 s per question, the same as with llama.cpp's own embedding
  model). The question time is dominated by the LLM, so a mixed setup only speeds up building the index. vLLM is
  faster in both roles, so mixing is useful only if the LLM has to run on llama.cpp (see below).
- **Tensor split for Gemma in llama.cpp** loads (with ubatch-size 1024), but crashes on the first request
  (`fattn.cu:762: fatal error`); layer split works. Experts on the CPU with 2 GPUs (`n-cpu-moe = 6`) are slower
  (chat 94 vs 135 t/s) and not needed, as the full context fits on the GPUs.

## Recommendation

For PaperQA2 on this machine: **vLLM, configuration 3** (Qwen3.8-27B with HyperQwen and Qwen3-Embedding-0.6B, both
with tensor parallelism on both GPUs, the default in [`.env`](.env)), with `PAPERQA_BACKEND=vllm`:

- ~43 s per question at reasoning effort "low", or ~25 s without thinking (`PAPERQA_THINKING=0`) with answers of
  nearly the same quality. Stock vLLM (3a) needs 1.5 times as long, llama.cpp 2.2 times as long with the same model.
- The index of all 75 papers is rebuilt in ~2 min.
- 262144 tokens of context for other clients, at ~86-98 t/s from short chats to 200k tokens of context.

If speed matters more than depth, **configuration 4** with `PAPERQA_THINKING=0`: Gemma 4 answers in ~10 s, with
shallower answers. Gemma's thinking is not worth it in PaperQA2: it made the questions 6-9 times as slow, looped once
and failed one question.

A mixed setup brings no benefit for PaperQA2. llama.cpp remains the better choice when the GPUs should be free most
of the time (its router unloads idle models after 10 minutes and reloads them in seconds, while vLLM keeps the GPU
memory allocated), for long single conversations (Gemma with 262144 tokens of context on one GPU, faster generation
at long contexts), or for switching between many models on one port.

## Commands

```sh
cd ../benchmark
# Serving speed of the LLM, with the embedding server (--mixed: embedding batches during the concurrent requests)
./llama_cpp_bench_http.py --url http://localhost:9932 --embedding-url http://localhost:9934 \
    --env-file ../llama-cpp/llama-cpp.env --tests generation prompt concurrency embedding chat depth edit \
    --prompt-tokens 16000 --concurrency 1 4 8 --concurrent-prompt-tokens 8000 --depth 6400 32000 80000 --repeat 1 \
    --chat-n-predict 256 --mixed --server-config ../vllm-rtx-a4000/config-qwen3.8-27b-tp2.yaml --label "..."
# Stress test of the memory: a prompt of almost the full context, then concurrent long prompts with embeddings
./llama_cpp_bench_http.py --url http://localhost:9932 --embedding-url http://localhost:9934 \
    --env-file ../llama-cpp/llama-cpp.env --tests prompt concurrency --prompt-tokens 255000 \
    --concurrency 4 8 --concurrent-prompt-tokens 30000 --mixed --repeat 1 --timeout 3600 \
    --server-config ../vllm-rtx-a4000/docker-compose.yml --label "..."
# PaperQA2 (from ../../phd-research; time each question)
PAPERQA_THINKING=low uv run python -m phd_code.tools.paperqa_ask ask "What sets the peak frequency of the sound-wave GW spectrum?"
# A configuration variant on a test server (port 9933); stop the production containers first
CACHE_VOLUME=vllm-rtx-a4000_vllm-cache GPUS=0,1 DOCKER_ARGS="--memory 40g -e NCCL_P2P_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" ./vllm_test_server.sh \
    ghcr.io/agenttix/agx-ai/vllm:latest /tmp/config-variant.yaml ../llama-cpp/llama-cpp.env
# The CPU image (no GPUs, see docker-compose.yml)
GPUS=none NAME=vllm-cpu DOCKER_ARGS="--cpus 16 -e VLLM_CPU_OMP_THREADS_BIND=nobind -e OMP_NUM_THREADS=16" \
    ./vllm_test_server.sh vllm/vllm-openai-cpu:v0.31.0 ../vllm-rtx-a4000/config-embedding-cpu.yaml \
    ../llama-cpp/llama-cpp.env 9935
# Embedding retrieval test: export the chunks of a PaperQA2 index, write the questions once, then test each model
cd ../../phd-research && .venv/bin/python -c 'import glob, json, pickle, zlib, pathlib
for f in sorted(glob.glob("literature/index/pqa_index_*/docs/*")):
    d = pathlib.Path(f).read_bytes()
    try: d = zlib.decompress(d)
    except zlib.error: pass
    for t in pickle.loads(d).texts: print(json.dumps({"name": t.name, "text": t.text}))' > /tmp/chunks.jsonl
cd ../agx-ai/benchmark
uv run ./embedding_retrieval_eval.py questions --chunks /tmp/chunks.jsonl --out /tmp/questions.jsonl --n 300 \
    --url http://localhost:9932 --env-file ../llama-cpp/llama-cpp.env --concurrency 8
uv run ./embedding_retrieval_eval.py eval --chunks /tmp/chunks.jsonl --questions /tmp/questions.jsonl \
    --url http://localhost:9934 --env-file ../llama-cpp/llama-cpp.env --label "..."
```
