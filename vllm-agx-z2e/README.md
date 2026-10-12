# vLLM on agx-z2e (RTX 3090)

vLLM on the RTX 3090 of agx-z2e, as an alternative to [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e) on the same GPU
and port (9931). Only one of the two fits on the GPU at a time. [`docker-compose.yml`](docker-compose.yml) has two
services:

- `vllm` (the default, `docker compose up -d`): Qwen3.8-27B, Gemma 4 26B-A4B or Gemma 4 31B entirely on the GPU, with
  the image of [`../vllm`](../vllm) (`vllm/vllm-openai:v0.31.0`, CUDA 13.0, with the fixes of
  [`../vllm/vllm.patch`](../vllm/vllm.patch)). The model is selected by commenting the config volume lines in or out.
- `vllm-flash-next` (`docker compose stop vllm && docker compose --profile flash-next up -d vllm-flash-next`):
  Qwen3.8-Flash-Next with most of its experts in RAM, with a third-party vLLM runtime (see
  [Qwen3.8-Flash-Next](#qwen38-flash-next)).

All use the API key of the llama.cpp servers. Like the llama.cpp presets, each configuration keeps vLLM at ≤ 21 GiB of
VRAM (peaks 20804-20988 MiB with full-context prompts). The desktop used 2.7 GiB during these measurements, which
left 0.8-1.0 GiB free:

| Configuration | Model | KV cache | Context | Chat | Generation at 6.4k / 32k / 100k tokens | Prompt processing |
|---|---|---|---|---|---|---|
| [`config-qwen3.8-27b.yaml`](config-qwen3.8-27b.yaml) (default) | Qwen3.8-27B GPTQ int4, MTP 2 | int8 | 147456 | 77-86 t/s | 83 / 68 / 45 t/s | 1160 t/s at 6.4k, 450 t/s at 145k |
| [`config-gemma-4-26b-a4b-qat.yaml`](config-gemma-4-26b-a4b-qat.yaml) | Gemma 4 26B-A4B QAT (Google's Q4_0, bit-exact), MTP 2 | int8 | 163840 | 255 t/s | 196 / 130 / 84 t/s, 70 t/s at 150k | 5030 t/s at 6.4k, 1020 t/s at 153k |
| [`config-gemma-4-31b-qat.yaml`](config-gemma-4-31b-qat.yaml) | Gemma 4 31B QAT (Google's w4a16), no MTP | int4 | 40960 | 36 t/s | 32 / 24 (at 36.5k) t/s | 775 t/s at 6.4k, 535 t/s at 39.5k |
| `vllm-flash-next` | Qwen3.8-Flash-Next W4A16 + FP8 n-gram table, MTP 3, experts in RAM | int8 | 135168 | 27.5 t/s | 28 / 27 (at 36.5k) / 29 t/s | 1650 t/s at 6.4k, 2060 t/s at 133k |

**Compared with llama.cpp on this GPU** ([`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e/README.md#results-of-the-current-presets)):

- **Qwen3.8-27B: vLLM is better.** It fits 44 % more context (147456 vs 102400) and generates 50-75 % faster at
  every context length (chat 77-86 vs 49 t/s, 68 t/s at 32k vs 42 t/s at 36.5k, 45 vs 26 t/s at 100k). llama.cpp processes long
  prompts faster (100k tokens: 776 vs 553 t/s).
- **Gemma 4 26B-A4B: it depends on the context length.** vLLM is faster in chats and short contexts (255 vs 216 t/s,
  6.4k: 196 vs 183 t/s), llama.cpp at long ones (32-36k: 169 vs 130 t/s, 100k: 121 vs 84 t/s) and in prompt processing
  of long prompts (2 times as fast). The context is about the same (163840 vs 165888).
- **Gemma 4 31B: llama.cpp is much better.** It fits 1.5 times the context (61440 vs 40960) with MTP and generates
  twice as fast (77 vs 36 t/s in chat). The dense 31B model leaves vLLM no room for the MTP drafter next to a useful
  context (see [Gemma 4 31B](#gemma-4-31b)).
- **Qwen3.8-Flash-Next: vLLM for long prompts, llama.cpp for the longest contexts and images.** Both generate
  ~28 t/s in chats, bound by the memory bandwidth of the CPU. vLLM stays at 27-29 t/s up to 100k tokens of context
  (llama.cpp: 20 t/s at 100k) and processes prompts 2.4-2.7 times as fast (1900-2060 vs 690-780 t/s), with MTP.
  llama.cpp fits the full 262144 tokens of context (vLLM: 135168), runs the vision encoder (vLLM: text only), starts in
  1.5 min instead of 7, and uses the stock image. See
  [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e/README.md#qwen38-flash-next).

The rest of this file has the measurements behind the settings. They were made on 2026-10-12 with the tools in
[`../benchmark`](../benchmark/README.md), on test servers ([`../benchmark/vllm_test_server.sh`](../benchmark/vllm_test_server.sh),
port 9933) with the production containers stopped. The results are in
[`../benchmark/results/agx-z2e-kubuntu.jsonl`](../benchmark/results/agx-z2e-kubuntu.jsonl) (labels starting with
"z2e vLLM"). VRAM figures are those of the vLLM processes (`nvidia-smi --query-compute-apps`), i.e. without the desktop.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-38-generic), KDE Plasma desktop (Wayland) on the RTX 3090 |
| CPU | AMD Ryzen Threadripper 3970X (Zen 2, 32 cores, 64 threads, 8 CCXs of 4 cores; AVX2, no AVX-512) |
| RAM | 128 GB DDR4-2666 in 4 channels (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPU | NVIDIA GeForce RTX 3090, which also drives the desktop (1.9-2.7 GiB of VRAM); driver 610.43.02 (CUDA 13.3), power limit 350 W (default), PCIe 4.0 x16 |
| GPU architecture | Ampere (GA102, sm_86) |
| VRAM | 24 GB GDDR6X (24576 MiB) |
| Other GPUs | AMD Radeon VII ([`../vllm-radeon-vii`](../vllm-radeon-vii)), NVIDIA GeForce GTX TITAN (Kepler, unused) |

## Setup

```sh
# Image: ../vllm (built by ../.github/workflows/deploy-docker-vllm.yml)
docker pull ghcr.io/agenttix/agx-ai/vllm:latest
# Qwen3.8-27B into the Hugging Face cache
uvx --from 'huggingface_hub[hf_xet]' hf download JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8
# Gemma 4 26B-A4B: the bit-exact QAT checkpoint (~10 min on the CPU, ~30 GB of RAM; see ../vllm-radeon-vii/README.md),
# then the embedding in int8
docker run --rm --memory 40g -v "$PWD/../vllm-radeon-vii":/src:ro -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    -v ~/.cache/vllm-models:/out --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm:latest \
    /src/repack_gemma4_qat.py --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 --embed-bits 16 --lm-head-bits 8
docker run --rm --memory 60g -v "$PWD/..":/src:ro -v ~/.cache/vllm-models:/out --entrypoint python3 \
    ghcr.io/agenttix/agx-ai/vllm:latest /src/vllm-agx-z2e/quantize_gemma4_ct_embedding.py \
    --repo /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 --embed-bits 8 --lm-head-bits 8 \
    --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed8-head8
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
# Gemma 4 31B: Google's compressed-tensors checkpoint (23 GB), then the embedding and the lm_head in int4 (~5 min)
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-31B-it-qat-w4a16-ct
docker run --rm --memory 60g -v "$PWD/..":/src:ro -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    -v ~/.cache/vllm-models:/out --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm:latest \
    /src/vllm-agx-z2e/quantize_gemma4_ct_embedding.py --lm-head-bits 4 --out /out/gemma-4-31B-it-qat-w4a16-ct-embed4-head4
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-31B-it-qat-q4_0-unquantized-assistant  # MTP variant
# Qwen3.8-Flash-Next (129 GB) and its runtime
uvx --from 'huggingface_hub[hf_xet]' hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
    --revision ef554143369a706525336f6b42a09094835dc077
docker pull ghcr.io/dominikbucko/qwen38-flash-next-3090@sha256:7f176605b59c462af21b1b62fd854f9f3cca96e3d11a295581297483d45490a3
# Stop llama.cpp on this GPU and start vLLM
docker compose -f ../llama-cpp-agx-z2e/docker-compose.yml stop
docker compose up -d
```

The first start compiles kernels for a few minutes; later starts take 1.5-2 min (the caches are in the `vllm-cache`
volume). The Flash-Next runtime takes ~7 min to start, as it loads 125 GB and copies 56 GiB of experts into its RAM
arena.

## The Triton attention fix (in ../vllm/vllm.patch)

On Ampere, vLLM supports quantized KV caches only with its Triton attention: FlashAttention 2 supports none, the
Triton backend's FP8 cache needs sm_89+, and FlashInfer's FP8 cache works but allows no full CUDA graphs with
speculative decoding. With Triton attention and MTP, the generation slowed down sharply with the context length
(Qwen3.8-27B: 58.6 t/s at 6.4k and 23.7 t/s at 32k tokens of context), as the kernel splits the KV cache over several
programs ("3D" kernel) only for single-token decoding, not for the 3-token verification batches of MTP. The
batches then ran with one program per KV head (4 for Qwen) over the whole context. The patch also uses the split
kernel for these batches, and larger query blocks for the prefill on sm_86 (see [`../vllm/README.md`](../vllm/README.md)):

| Test (t/s, prompt processing in parentheses) | Qwen3.8-27B unpatched | patched | Gemma 4 26B-A4B unpatched | patched |
|---|---|---|---|---|
| Generation at 6.4k tokens of context | 58.6 (1088) | 83.5 (1102) | 112.1 (4935) | 207.9 (5133) |
| Generation at 32k | 23.7 (725) | 64.0 (714) | 43.3 (2939) | 127.2 (3012) |
| Generation at 100k | | | 16.7 (1398) | 84.1 (1399) |
| Prompt of 32k / 96k tokens | 743 / 395 | 890 / 582 | 3044 / - | 3130 / - |

The long-context retrieval answers of [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py)
are the same with and without the patch (Qwen: 88/88 identical at 17k and 61k tokens; Gemma: 43/44 identical, same
score). Before the patch, FlashInfer with an FP8 cache was the fastest option at long contexts (Qwen: 55 t/s at 32k),
but 37 % slower in chats (51.5 vs 81 t/s) because of the piecewise CUDA graphs.

Until the image of [`../vllm`](../vllm) is rebuilt by the workflow, the measurements were made with a local build of it.

## Qwen3.8-27B

[`JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB`](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB)
(`mtp-int8` revision): GPTQ int4 (group size 128), int4 embedding and lm_head, MTP head in int8; 13.5 GiB on the GPU
with the MTP head, without the vision tower. The other int4 checkpoints keep the embedding and the lm_head in BF16
(2.4 GiB each), which would cost ~100k tokens of context here.

| Variant (all with MTP 2) | Context | Chat | Gen. 6.4k | Gen. 32k | Prompt 32k / 96-145k | Peak VRAM |
|---|---|---|---|---|---|---|
| Triton, int8 KV, unpatched | 98304 | 81.0 | 58.6 | 23.7 | - / 399 (96k) | 19196 MiB |
| FlashInfer, FP8 KV, unpatched | 98304 | 51.5 | 54.0 | 55.0 | - / 807 (96k) | 19886 MiB |
| FlashAttention 2, BF16 KV, unpatched | 49152 | 85.0 | 60.7 | 30.1 | - | 19116 MiB |
| **Triton, int8 KV, patched (current)** | 147456 | 76.5-86.3 | 83.4 | 68.3-68.6 | 934 / 450 (145k) | 20976 MiB |
| The same with MTP 3 | 147456 | 88.0 | 83.4 | 71.1 | 919 / - | 21036 MiB |
| The same with 4096-token prefill chunks | 139264 | 81.9 | 84.9 | 64.0 | 926 / - | 20938 MiB |

- The context is the maximum for the 21 GiB: the KV cache of 6.4 GB holds 158480 tokens (~40 KiB per token including the
  recurrent states), and vLLM needs 1.05-1.07 times the context. 163840 tokens (6.9 GB) ran out of memory on the
  first request with the desktop using 2.7 GiB.
- MTP 3 was 2-4 % faster than MTP 2 (within the run-to-run variation of 76-86 t/s in chat) and leaves less context.
- At 100k tokens of context: 44.6 t/s (553 t/s prompt processing); llama.cpp: 25.9 t/s (776 t/s).
- **CPU KV offloading** (16 GiB of pinned RAM, `kv-offloading-backend: native`) takes no GPU memory and does not slow
  down generation. A 25k-token conversation that was evicted from the GPU by a 140k-token prompt came back in 4.0 s
  instead of 25.8 s.

## Gemma 4 26B-A4B

The bit-exact checkpoint of [`../vllm-radeon-vii`](../vllm-radeon-vii/README.md#gemma-4-26b-a4b-qat) (all linear
weights from Google's Q4_0 GGUF), with Google's QAT MTP drafter. The 24 GB leave room for a better embedding than
on the 16 GB GPUs:

| Embedding / lm_head | On the GPU with the drafter | KV cache | Context | Chat | Gen. 32k | Gen. 100k |
|---|---|---|---|---|---|---|
| fp16 / int8 | 16.09 GiB | 2.8 GB, 108455 tokens | 106496 | 263.5 | 134.0 | 83.8 |
| **int8 / int8 (current)** | 15.41 GiB | 4.2 GB, 173323 tokens | 163840 | 255.0 | 130.2 | 84.0 |

The int8 embedding (round-to-nearest, group size 128) of
[`quantize_gemma4_ct_embedding.py`](quantize_gemma4_ct_embedding.py) is practically lossless and gives ~55k more
tokens of context. With the int4 embedding and lm_head of the 16 GB GPUs, the context could be ~12k tokens larger.
At 150k tokens of context: 69.9 t/s (1004 t/s prompt processing); llama.cpp at 163k: 103.5 t/s (2008 t/s).
With the CPU KV offloading of the configuration (16 GiB, as for Qwen), the peak with a 16k-token context was
20938 MiB (31B: 20948 MiB).

## Gemma 4 31B

[`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) has the linear
weights in int4 (group size 32) but the embedding and the lm_head in BF16 (2.6 GiB each), 20.6 GiB on the GPU.
[`quantize_gemma4_ct_embedding.py`](quantize_gemma4_ct_embedding.py) quantizes both to int4 (with an int8 lm_head, the
model ran out of memory while loading): 16.9 GiB on the GPU. That leaves ~3 GB for everything else, and vLLM needs
much more KV cache for this model than llama.cpp: it reserves the sliding window and a prefill chunk for each of the 50
sliding-window layers, and pads the KV cache of the 10 global layers (4 KV heads of 512) to the page size of the
sliding-window layers (16 KV heads of 256).

| Variant | KV cache | Context | Chat | Gen. 6.4k | Gen. 16k | Gen. 36.5k | Peak VRAM |
|---|---|---|---|---|---|---|---|
| int8 KV, no MTP | 2.3 GB | 12176 at most | (not benchmarked) | | | | |
| **int4 KV, no MTP (current)** | 2.9 GB | 40960 | 35.6 | 32.1 | 27.6 | 23.8 | 20988 MiB |
| int4 KV, MTP 2 | 1.3 GB | 8192 | 74.8 | 45.8 | | | 20782 MiB |
| llama.cpp: q8_0 KV, MTP | | 61440 | 76.9 | 72.7 | 68.1 | 55.8 | |

- `hf-overrides` limits `max_position_embeddings` to the context: the rotary embedding caches are otherwise
  allocated for 262144 positions (0.375 GiB).
- The int4 KV cache retrieves as well as int8: at 9k tokens of context, 43/44 correct with both; at 21k (int4 only),
  42/44.
- With MTP, the drafter (0.9 GB) leaves room for only 8192 tokens of context; this variant is commented out in the
  configuration.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a 125B MoE (512 experts, 10 active, 6B active parameters) with a 51B-parameter n-gram embedding
table and a hybrid attention (Gated DeltaNet and Qwen Sparse Attention). Stock vLLM 0.31 supports it, but only with
the whole model on GPUs: its CPU offloading (`cpu-offload-gb`) copies the offloaded weights to the GPU for every step,
and `VLLM_PLE_CPU_OFFLOAD` keeps the n-gram table pinned in RAM in BF16 (102 GB) or FP8 (51 GB), which does not fit
next to the 60+ GB of int4 experts. The vLLM CPU backend would need the whole model in RAM too (and AVX-512 for its
fast paths).

[`DominikBucko/qwen38-flash-next-3090`](https://github.com/DominikBucko/qwen38-flash-next-3090) is a vLLM build for
exactly this case (one RTX 3090 and 64 GB of RAM): the GPU keeps the dense weights and the most used ("hot") experts,
an AVX2 thread pool computes the other ("cold") experts from RAM during decode while the GPU computes a share of them
over PCIe, and prefill streams all experts through the GPU. The n-gram table (FP8) is read in place from the
checkpoint. It uses [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)
(Intel's AutoRound W4A16 experts and an FP8 n-gram table, 125 GB). With 68 GiB for the container, all 23808 cold
experts fit in its RAM arena (56 GiB). The compose file uses the author's
[v0.1.0 release image](https://github.com/DominikBucko/qwen38-flash-next-3090/releases/tag/v0.1.0), pinned by
digest; the later commits of the repository (up to `10b24c2`, 2026-10-11) change only documentation. None of the
fixes of [`../vllm/vllm.patch`](../vllm/vllm.patch) would help this model: its sparse attention has its own split-K
kernels, it does not use the Qwen3.5/3.8 model code, and its decode is bound by the CPU. The settings are in the
`environment` of the service in [`docker-compose.yml`](docker-compose.yml).

| Settings | Chat | Gen. 6.4k | Gen. 32k | CPU / GPU time per decode step | Peak VRAM |
|---|---|---|---|---|---|
| 16 hot experts per layer, GPU share 0.22 (default) | 26.9 | 27.6 | 26.8 | | 20330 MiB |
| **20 hot experts, GPU share 0.3 (current)** | 27.9 | 28.2 | 28.0 | 55 / 16 ms | 20724 MiB |
| 20 hot experts, GPU share 0.5 (capped at 8 experts per layer) | 29.0 | 27.9 | 27.0 | 55 / 16 ms | 20724 MiB |
| 20 hot experts, GPU share 0.5 with up to 16 experts per layer, 28 CPU threads | 24.3 | 23.7 | 24.7 | 48 / 33 ms | 20724 MiB |

- **Decode is bound by the CPU's memory bandwidth.** The runtime's defaults (32 hot experts, 23.4 GB) do not fit in
  21 GB; each hot expert per layer costs 116 MiB of VRAM. Per decode step (4 tokens with MTP 3), the CPU computes
  ~800 cold experts (1.9 GB of weights) in ~55 ms, ~35 GB/s from the 4 channels of DDR4-2666, while the GPU needs only
  ~16 ms. The runtime's author measured 43-51 t/s on a Threadripper PRO with 8 memory channels. More GPU share
  (PCIe reads) or more CPU threads (which then compete with the serving processes) made it slower.
- MTP with 3 drafted tokens accepts ~2 tokens per step with the server's sampling (temperature 1.0); MTP 2 does not
  start, as its prefix-cache block size (3184 tokens) conflicts with the runtime's check of the retention interval
  (a multiple of 3200).
- **Prefill** streams all experts through the GPU in 8192-token chunks: 1650-2060 t/s, without slowing down with the
  context (Qwen Sparse Attention).
- The runtime serves one request at a time (others queue), up to 135168 tokens, text only. It takes ~7 min to start:
  it loads the 125 GB checkpoint and copies the cold experts into a RAM arena with transparent huge pages (only 23.6 of
  56 GiB got huge pages, as the RAM was fragmented after days of uptime).
- The serve script of the image has no option for `gpu-memory-utilization`, whose default (0.92) fails vLLM's startup
  check next to the desktop, so the compose entrypoint adds `--gpu-memory-utilization 0.8` with `sed`.

Results with the current settings (`--tests generation prompt chat depth --prompt-tokens 133000`):

| Test | vLLM | llama.cpp ([preset](../llama-cpp-agx-z2e/preset-qwen3.8-flash-next.ini)) |
|---|---|---|
| Context | 135168 | 262144 |
| Generation, 256 tokens | 32.2 t/s (MTP acceptance 45 %) | 27.6 t/s |
| Chat, 6 requests | 27.5 t/s (23.4-34.0) | 27.6 t/s |
| Depth 6.4k / 16k / 36.5k / 100k | 28.1 / 26.8 / 27.4 / 28.7 t/s | 26.6 / 26.3 / 24.9 / 20.4 t/s |
| Prompt processing at the same depths | 1652 / 1957 / 1941 / 1893 t/s | 604 / 684 / 687 / 689 t/s |
| Full-context prompt | 2058 t/s (133000 tokens) | 778 t/s (260000 tokens) |
| Peak VRAM | 20804 MiB | 20618 MiB |
| RAM | 68 GiB container limit (56 GiB of experts) | ~66 GB |

## Commands

```sh
cd ../benchmark
# Serving speed with a full-context prompt, chat and depth (the configuration of docker-compose.yml)
./llama_cpp_bench_http.py --url http://localhost:9931 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 145000 --depth 6400 32000 100000 --repeat 1 \
    --server-config ../vllm-agx-z2e/config-qwen3.8-27b.yaml --label "..."
# A configuration variant on a test server (port 9933; stop the production container first)
docker compose -f ../vllm-agx-z2e/docker-compose.yml stop
CACHE_VOLUME=vllm-agx-z2e_vllm-cache GPUS=0 DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    ./vllm_test_server.sh ghcr.io/agenttix/agx-ai/vllm:latest /tmp/config-variant.yaml ../llama-cpp-radeon-vii/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 1 --depth 6400 32000 --server-config /tmp/config-variant.yaml --label "..."
# Long-context retrieval (output quality of a KV cache type or a patch)
./llama_cpp_quality_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \
    --tests retrieval --depth 16000 60000 --logprobs 0 --label "..."
docker rm -f vllm-test && docker compose -f ../vllm-agx-z2e/docker-compose.yml start
```
