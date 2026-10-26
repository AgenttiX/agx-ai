# vLLM on agx-z2e (RTX 3090)

vLLM on the RTX 3090 of agx-z2e, as an alternative to [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e) on the same GPU
and port (9931). Only one of them fits on the GPU at a time. There are two Docker Compose configurations:

- [`docker-compose.yml`](docker-compose.yml) (`docker compose up -d`): Qwen3.8-27B, Gemma 4 26B-A4B or Gemma 4 31B
  entirely on the GPU, with the image of [`../vllm`](../vllm) (`vllm/vllm-openai:v0.31.0`, CUDA 13.0, with the fixes
  of [`../vllm/vllm.patch`](../vllm/vllm.patch)). The model is selected by commenting the config volume lines in or
  out.
- [`../vllm-rtx3090-flash-next/docker-compose.yml`](../vllm-rtx3090-flash-next/docker-compose.yml) (`docker compose
  stop` here, then `docker compose up -d` there): Qwen3.8-Flash-Next with most of its experts in RAM, with a
  third-party vLLM runtime and its vision tower on the CPU for images and videos (see
  [Qwen3.8-Flash-Next](#qwen38-flash-next)).

For Qwen3.8-27B, [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md) (the HyperQwen image, with the
DFlash2 drafter) is faster for one to three users or agents on the same GPU and port: chat 136 vs 83 t/s, 128 / 110 /
80 t/s at 6.4k / 32k / 100k tokens of context, and 290 t/s when the model copies code, with 139264 tokens of context.
The configuration here stays the stock-vLLM alternative, and is faster for 4 concurrent agents.

All use the API key of the llama.cpp servers. Like the llama.cpp presets, each configuration keeps vLLM at ≤ 21 GiB of
VRAM (peaks 20804-20988 MiB with full-context prompts). The desktop used 2.7 GiB during these measurements, which
left 0.8-1.0 GiB free:

| Configuration | Model | KV cache | Context | Chat | Generation at 6.4k / 32k / 100k tokens | Prompt processing |
|---|---|---|---|---|---|---|
| [`config-qwen3.8-27b.yaml`](config-qwen3.8-27b.yaml) (default) | Qwen3.8-27B GPTQ int4, MTP 2, pth-flash-attn | int8 | 155648 | 81-86 t/s | 83 / 79 / 66 t/s, 60 t/s at 153k | 1210 t/s at 6.4k, 690 t/s at 153k |
| [`config-gemma-4-26b-a4b-qat.yaml`](config-gemma-4-26b-a4b-qat.yaml) | Gemma 4 26B-A4B QAT (Google's Q4_0, bit-exact), MTP 2 | int8 | 163840 | 255 t/s | 196 / 130 / 84 t/s, 70 t/s at 150k | 5030 t/s at 6.4k, 1020 t/s at 153k |
| [`config-gemma-4-31b-qat.yaml`](config-gemma-4-31b-qat.yaml) | Gemma 4 31B QAT (Google's w4a16), no MTP | int4 | 40960 | 36 t/s | 32 / 24 (at 36.5k) t/s | 775 t/s at 6.4k, 535 t/s at 39.5k |
| [`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next) | Qwen3.8-Flash-Next W4A16 (int8 dense layers) + FP8 n-gram table, MTP 3, experts in RAM | int8 | 262144 | 32.4 t/s | 33 / 33 / 32 t/s, 52 t/s in code edits | 1750 t/s at 6.4k, 2050 t/s at 260k |

**Compared with llama.cpp on this GPU** ([`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e/README.md#results-of-the-current-presets)):

- **Qwen3.8-27B: vLLM is better.** It fits 52 % more context (155648 vs 102400) and generates 65-155 % faster at
  every context length (chat 81-86 vs 49 t/s, 79 t/s at 32k vs 42 t/s at 36.5k, 66 vs 26 t/s at 100k), and processes
  long prompts about as fast (100k tokens: 812 vs 776 t/s). With
  [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md), even faster.
- **Gemma 4 26B-A4B: it depends on the context length.** vLLM is faster in chats and short contexts (255 vs 216 t/s,
  6.4k: 196 vs 183 t/s), llama.cpp at long ones (32-36k: 169 vs 130 t/s, 100k: 121 vs 84 t/s) and in prompt processing
  of long prompts (2 times as fast). The context is about the same (163840 vs 165888).
- **Gemma 4 31B: llama.cpp is much better.** It fits 1.5 times the context (61440 vs 40960) with MTP and generates
  twice as fast (77 vs 36 t/s in chat). The dense 31B model leaves vLLM no room for the MTP drafter next to a useful
  context (see [Gemma 4 31B](#gemma-4-31b)).
- **Qwen3.8-Flash-Next: vLLM is better, except for images.** Both fit the full 262144 tokens of context and generate
  ~27-32 t/s in chats, bound by the memory bandwidth of the CPU (vLLM: 52 t/s when copying code with MTP). vLLM stays at ~28 t/s up to 100k tokens of context
  (llama.cpp: 20 t/s at 100k), processes prompts 2.4-2.8 times as fast (1650-2090 vs 600-780 t/s) and supports
  videos (~35 s for 30 s of 1080p). llama.cpp processes images faster (3.2 s vs ~8 s for 1280x960, as vLLM's vision
  tower runs on the CPU), starts in 1.5 min instead of 7, and uses the stock image. See
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
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub -v ~/.cache/vllm-models:/out \
    --entrypoint repack_gemma4_qat.py ghcr.io/agenttix/agx-ai/vllm:latest \
    --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 --embed-bits 16 --lm-head-bits 8
docker run --rm --memory 60g -v ~/.cache/vllm-models:/out --entrypoint quantize_gemma4_ct_embedding.py \
    ghcr.io/agenttix/agx-ai/vllm:latest --repo /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 \
    --embed-bits 8 --lm-head-bits 8 --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed8-head8
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
# Gemma 4 31B: Google's compressed-tensors checkpoint (23 GB), then the embedding and the lm_head in int4 (~5 min)
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-31B-it-qat-w4a16-ct
docker run --rm --memory 60g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub -v ~/.cache/vllm-models:/out \
    --entrypoint quantize_gemma4_ct_embedding.py ghcr.io/agenttix/agx-ai/vllm:latest \
    --lm-head-bits 4 --out /out/gemma-4-31B-it-qat-w4a16-ct-embed4-head4
uvx --from 'huggingface_hub[hf_xet]' hf download google/gemma-4-31B-it-qat-q4_0-unquantized-assistant  # MTP variant
# Qwen3.8-Flash-Next (129 GB) and its runtime
uvx --from 'huggingface_hub[hf_xet]' hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
    --revision ef554143369a706525336f6b42a09094835dc077
docker pull ghcr.io/agenttix/agx-ai/vllm-flash-next:latest
# Stop llama.cpp on this GPU and start vLLM
docker compose -f ../llama-cpp-agx-z2e/docker-compose.yml stop
docker compose up -d
# Or Qwen3.8-Flash-Next instead
docker compose stop && docker compose -f ../vllm-rtx3090-flash-next/docker-compose.yml up -d
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
| Triton, int8 KV, patched (until 2026-10-26) | 147456 | 76.5-86.3 | 83.4 | 68.3-68.6 | 934 / 450 (145k) | 20976 MiB |
| The same with MTP 3 | 147456 | 88.0 | 83.4 | 71.1 | 919 / - | 21036 MiB |
| The same with 4096-token prefill chunks | 139264 | 81.9 | 84.9 | 64.0 | 926 / - | 20938 MiB |
| **pth-flash-attn, int8 KV, FP16 recurrent state (current, 2026-10-26)** | 155648 | 82.8 | 83.4 | 79.4 | 1054 / 693 (153k) | 21036 MiB |

On 2026-10-26, the attention moved to the pth-flash-attn kernel of [`../vllm`](../vllm/README.md) (`attention-backend:
CUSTOM`, also for the MTP drafter), and the recurrent state to FP16, both from
[HyperQwen](../vllm-rtx3090-hyperqwen/README.md#combining-the-optimizations). Measured that day with the same tests
(the Triton row again for comparison: chat 86.3, 79.6 / 66.7 / 40.5 t/s at 6.4k / 32k / 100k): 83.4 / 79.4 / 66.3 t/s,
prompt processing 1209 / 1054 / 812 instead of 1143 / 867 / 544 t/s. The FP16 state gives 6.5 % more tokens in the
same KV cache (168848 instead of 158480), with no measurable effect on the speed. The notes below are from 2026-10-12
with the Triton attention.

Checked on 2026-10-26 with the current settings:

- **MTP 3 instead of 2** (with CUDA graphs for up to 16 tokens): only edits are faster (125.6 / 119.3 instead of
  112.2 / 102.9 t/s, short / after 32k tokens), everything else slower: chat 76.7 instead of 82.8 t/s, 77.9 / 72.6 /
  61.3 instead of 83.4 / 79.4 / 66.3 t/s at 6.4k / 32k / 100k tokens of context, prompt processing 1038 / 909 / 699
  instead of 1209 / 1054 / 812 t/s. MTP 2 stays.
- **The CPU KV offloading** of 16 GiB is large enough: a 25030-token conversation that two 100k-token prompts had
  evicted from the GPU came back in 1.7 s instead of 25.0 s, with the same greedy answer
  ([`../benchmark/vllm_prefix_return.py`](../benchmark/vllm_prefix_return.py)). This image stores ~36 KB per token
  (8.1 GB for the 225k tokens of the test), much less than HyperQwen's ~100 KB with its recurrent-state snapshots
  (see [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md#cpu-kv-offloading)), so 16 GiB holds ~470k
  tokens.

- The context was the maximum for the 21 GiB: the KV cache of 6.4 GB holds 158480 tokens (~40 KiB per token including the
  recurrent states), and vLLM needs 1.05-1.07 times the context. 163840 tokens (6.9 GB) ran out of memory on the
  first request with the desktop using 2.7 GiB.
- MTP 3 was 2-4 % faster than MTP 2 (within the run-to-run variation of 76-86 t/s in chat) and leaves less context.
- At 100k tokens of context: 44.6 t/s (553 t/s prompt processing); llama.cpp: 25.9 t/s (776 t/s).
- **CPU KV offloading** (16 GiB of pinned RAM, `kv-offloading-backend: native`) takes no GPU memory and does not slow
  down generation. A 25k-token conversation that was evicted from the GPU by a 140k-token prompt came back in 4.0 s
  instead of 25.8 s.

## Gemma 4 26B-A4B

The bit-exact checkpoint of [`../vllm-radeon-vii`](../vllm-radeon-vii/README.md#checkpoints-2026-10-09) (all linear
weights from Google's Q4_0 GGUF), with Google's QAT MTP drafter. The 24 GB leave room for a better embedding than
on the 16 GB GPUs:

| Embedding / lm_head | On the GPU with the drafter | KV cache | Context | Chat | Gen. 32k | Gen. 100k |
|---|---|---|---|---|---|---|
| fp16 / int8 | 16.09 GiB | 2.8 GB, 108455 tokens | 106496 | 263.5 | 134.0 | 83.8 |
| **int8 / int8 (current)** | 15.41 GiB | 4.2 GB, 173323 tokens | 163840 | 255.0 | 130.2 | 84.0 |

The int8 embedding (round-to-nearest, group size 128) of
[`../vllm/quantize_gemma4_ct_embedding.py`](../vllm/quantize_gemma4_ct_embedding.py) is practically lossless and gives
~55k more tokens of context. With the int4 embedding and lm_head of the 16 GB GPUs, the context could be ~12k tokens
larger. At 150k tokens of context: 69.9 t/s (1004 t/s prompt processing); llama.cpp at 163k: 103.5 t/s (2008 t/s). With
the CPU KV offloading of the configuration (16 GiB, as for Qwen), the peak with a 16k-token context was 20938 MiB (31B:
20948 MiB).

## Gemma 4 31B

[`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) has the linear weights
in int4 (group size 32) but the embedding and the lm_head in BF16 (2.6 GiB each), 20.6 GiB on the GPU.
[`../vllm/quantize_gemma4_ct_embedding.py`](../vllm/quantize_gemma4_ct_embedding.py) quantizes both to int4 (with an
int8 lm_head, the model ran out of memory while loading): 16.9 GiB on the GPU. That leaves ~3 GB for everything else,
and vLLM needs much more KV cache for this model than llama.cpp: it reserves the sliding window and a prefill chunk for
each of the 50 sliding-window layers, and pads the KV cache of the 10 global layers (4 KV heads of 512) to the page size
of the sliding-window layers (16 KV heads of 256).

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
(Intel's AutoRound W4A16 experts and an FP8 n-gram table, 125 GB). With 68 GiB for the container (71 GiB since images and videos run on the CPU), all 23808 cold
experts fit in its RAM arena (56 GiB). The compose file uses [`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next/README.md):
the author's [v0.1.0 release image](https://github.com/DominikBucko/qwen38-flash-next-3090/releases/tag/v0.1.0)
(the later commits of the repository, up to `10b24c2` of 2026-10-11, change only documentation) with image and video
input, the vision tower running on the CPU so that it takes no VRAM and does not slow down text. A 1280x960 image takes
~8 s until the first token, a 30-s video ~35 s (~6000 tokens). With the vision tower on the GPU, the first image ran
out of GPU memory. None of the fixes of
[`../vllm/vllm.patch`](../vllm/vllm.patch) would help this model: its sparse attention has its own split-K kernels, it
does not use the Qwen3.5/3.8 model code, and its decode is bound by the CPU. The settings are in the `environment` of
the service in [`../vllm-rtx3090-flash-next/docker-compose.yml`](../vllm-rtx3090-flash-next/docker-compose.yml).

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
- The runtime serves one request at a time (others queue). It takes ~7 min to start:
  it loads the 125 GB checkpoint and copies the cold experts into a RAM arena with transparent huge pages (only 23.6 of
  56 GiB got huge pages, as the RAM was fragmented after days of uptime).
- The runtime's serve script had no option for `gpu-memory-utilization`, whose default (0.92) fails vLLM's startup
  check next to the desktop; [`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next/README.md) adds it (0.8).

### Context size (2026-10-13)

With the runtime's defaults, vLLM fitted 135168 tokens of context and llama.cpp the full 262144, although vLLM's int8
KV cache needs less per token. What takes the 21 GiB (llama.cpp's weights estimated from its per-tensor types):

| | vLLM (runtime defaults, 20 hot experts) | llama.cpp ([preset](../llama-cpp-agx-z2e/preset-qwen3.8-flash-next.ini)) |
|---|---|---|
| Dense weights (Gated DeltaNet, attention, shared experts, lm_head, hyper-connections, routers) | 7.97 GiB in BF16 | ~3.7 GiB in 4-6-bit types (Q4_K-Q6_K, IQ4) |
| Token embedding | 1.18 GiB in BF16 on the GPU | IQ4_XS in RAM |
| Routed experts on the GPU | 20 per layer, 2.27 GiB | 6 of 48 layers, ~6.5 GiB |
| MTP module | 1.65 GiB, and its attention layer makes the KV cache ~1.5 times as large per token | none (not in llama.cpp yet) |
| KV cache | 3.05 GB for 141504 tokens (~21.5 KB per token with MTP, ~14.5 KB without) | 7.2 GiB for 262144 tokens (~29 KiB per token, q8_0) |
| Vision encoder | 0 (on the CPU) | ~1 GiB |
| Activations and buffers | ~1.8 GiB above the idle level during prefill (8192-token chunks, expert streaming) | ~2.4 GiB (ubatch 2048) |

The main difference is the precision of the dense weights: the checkpoint has everything but the routed experts and
the n-gram table in BF16 (9.15 GiB, ~5.4 GiB more than llama.cpp's quantization), and the MTP module. Quantizing the
dense weights to int8 like [`../vllm-rtx3070`](../vllm-rtx3070/README.md#qwen38-flash-next) would save ~3.3 GiB, but it
needs a re-quantized 65 GB checkpoint and int8 support in the runtime's custom kernels. Not needed here, as the two
cheaper steps were enough:

- **The token embedding in pinned RAM** (`QWEN38_EMBED_UVA=1`, an option of the runtime): the GPU reads the rows of
  the prompt's tokens over PCIe. -1.18 GiB of VRAM, no measurable effect on the speed.
- **Fewer hot experts**: decode is bound by the CPU's memory bandwidth, so 14 instead of 20 hot experts per layer
  (-0.68 GiB) cost no measurable speed either.

| Variant (all with the token embedding in RAM) | Context | KV cache | Chat | Gen. 6.4k / 32k / 100k | Prompt 260k | Peak VRAM |
|---|---|---|---|---|---|---|
| Before: embedding on the GPU, MTP 3, 20 hot experts | 135168 | 3.05 GB | 27.5 | 28.1 / 27.4 (36.5k) / 28.7 | 2058 t/s (133k) | 20804 MiB |
| MTP 3, 20 hot experts | 135168 | 3.05 GB | 27.1 | 28.0 / 29.1 / - | - | 19684 MiB |
| No MTP, 20 hot experts | 135168 | 3.05 GB (209941 tokens) | 25.5 | 25.5 / 25.4 / - | - | 17360 MiB |
| No MTP, 40 hot experts | 262144 | 4.05 GB | out of GPU memory on the first request (idle 19758 MiB) | | | |
| No MTP, 36 hot experts | 262144 | 3.85 GB (278812 tokens) | 25.8 | 26.3 / 25.8 / 26.4 | 2177 t/s | 20858 MiB |
| MTP 3, 7 hot experts | 262144 | 5.95 GB (333406 tokens) | 26.0 | 26.3 / 26.5 / 28.2 | 2046 t/s | 20816 MiB |
| **MTP 3, 14 hot experts (current)** | 262144 | 5.15 GB (287594 tokens) | 26.5 | 26.6 / 28.3 / 28.4 | 2087 t/s | 20876 MiB |

- Without MTP, decode is ~8 % slower at the same context; its KV cache is a third smaller, so more experts fit on the
  GPU, but that does not make up for it.
- The peak is ~1.8 GiB above the idle level after loading (prefill buffers), so the idle level has to stay below
  ~19.1 GiB.
- The runtime needs a KV cache of ~1.05 times the context. With MTP, the cache costs ~15 KB per additional token
  (5.95 GB held 333406 tokens), less than the ~21.5 KB per token of the smaller cache suggested.

### CPU pool threads (2026-10-26)

With `QWEN38_CPU_EXPERTS_STATS=64`, the runtime logs the decode step times. With the settings of 2026-10-14, a step
(4 tokens with MTP 3) took 63 ms of CPU time for ~920 cold experts (~37 GB/s from the RAM), 20 ms of GPU time and a
6.5 ms tail, one after the other: the GPU waits for the CPU's result in every layer. Each layer has only ~19 cold
experts, one work item each, so the 24 threads of the pool are not all busy, but the pool is bound by the memory
bandwidth anyway: splitting each expert into 5 slices of 128 columns (`QWEN38_CPU_EXPERTS_SLICES`) made it slower.
Fewer threads leave more cores to the serving processes that drive the GPU:

| CPU pool (`QWEN38_CPU_EXPERTS_CPUS`) | Chat | Gen. 6.4k | Edit (copy) | CPU / GPU / tail per step |
|---|---|---|---|---|
| 24 threads, 3 per 4-core CCX (the runtime's default) | 25.6 | 27.4 | 39.9 | 63 / 20 / 6.5 ms |
| 48 threads, the same cores with SMT | 21.6 | 22.7 | 37.8 | 77 / 18 / 6.2 ms |
| 24 threads with 5 slices per expert | 25.2 | 25.2 | 39.7 | 71 / 16 / 5.6 ms |
| **16 threads, 2 per CCX (the best with FP32)** | 28.0 | 30.0 | 46.0 | 59 / 16 / 5.6 ms |
| 16 threads, GPU share 0.45 | 27.2 | 27.6 | 46.2 | 59 / 17 / 5.6 ms |
| 16 threads, prefetch distance 16 instead of 6 rows (`PF_ROWS` of the kernel) | 27.7 | 27.7 | 45.7 | 62 / 16 / 5.6 ms |
| 8 threads, 1 per CCX | 25.4 | 26.5 | 42.5 | 66 / 16 / 5.6 ms |

One run of each test (the chat test varies by ~±10 %, the depth test by ~±5 %; the edit test, a file copied with
one change at a draft acceptance of 98 %, by ~1 %). With 16 threads and `--repeat 2`: chat 28.7 t/s,
27.0 / 27.6 / 26.2 t/s at 6.4k / 32k / 100k tokens of context, 45.9 / 45.6 t/s in edits (short / after 32k tokens),
prompt processing 1690 / 1986 / 1984 t/s, peak 20896 MiB. So +15 % in code edits and +12 % in chats; at long contexts,
the difference is within the noise.

Not applicable here (checked on 2026-10-26):

- **HyperQwen's optimizations** (see [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md)) are for the
  Qwen3.5/3.8 model code with full attention: the pth-flash-attn kernel needs causal attention with head size 256
  (Flash-Next has Qwen Sparse Attention with its own kernels), DFlash2 and its lookup drafting need a DFlash2 drafter,
  which does not exist for Flash-Next, and HyperQwen's image has no Flash-Next support. The FP16 recurrent state would
  only save VRAM (one sequence), and more hot experts on the GPU made little difference (see above).
- **Stock vLLM** ([`../vllm`](../vllm)'s image, as in [`../vllm-rtx3070`](../vllm-rtx3070/README.md#qwen38-flash-next))
  reads the cold experts over PCIe (~25 GB/s, ~2.4 GB per decode step with MTP 3) instead of computing them on the
  CPU, which reads ~41 GB/s. In this runtime, a larger GPU share (more PCIe reads) did not help either, as both read the
  same RAM.

### CPU expert kernel (2026-10-26)

The RAM of this machine reads 71-72 GB/s with 8-16 threads (28 GB/s with one), but the CPU pool reached only ~41 GB/s:
one thread computed an expert at only ~8 GB/s (11 GB/s from the cache), as the runtime's AVX2 kernel converts each
int4 weight to FP32 for an FMA, and one thread zeroed and summed the output buffers of all threads in every layer.
[`../vllm-rtx3090-flash-next/cpu-moe.patch`](../vllm-rtx3090-flash-next/README.md) adds int8 and int16 activations
(summed exactly in int32 with `vpmaddubsw`/`vpmaddwd`) and removes that overhead. Developed with the offline
benchmark of [`../vllm-rtx3090-flash-next/cpu_moe_bench`](../vllm-rtx3090-flash-next/README.md#offline-benchmark-of-the-cpu-expert-kernel),
which has the kernel measurements; in the server (one run each, MTP 3):

| CPU experts | Chat | Gen. 6.4k | Edit (copy) | CPU / GPU / tail per step |
|---|---|---|---|---|
| FP32, 16 threads (above) | 28.0 | 30.0 | 46.0 | 59 / 16 / 5.6 ms |
| INT16 activations, 16 threads | 29.7 | 27.5 | 48.3 | 55 / 18 / 5.7 ms |
| INT8 activations, 8 threads | 29.1 | 30.5 | 50.1 | 53 / 17 / 5.6 ms |
| INT8, 8 threads, 28 instead of 14 hot experts (163840 tokens of context) | 31.8 | 30.0 | 51.0 | 50 / 17 / 5.6 ms |
| INT8, 8 threads, int8 dense layers | 30.1 | 31.1 | 51.7 | 53 / 14 / 5.5 ms |
| INT8, 8 threads, int8 dense layers, 36 hot experts | 33.1 | 34.9 | 53.2 | 48 / 15 / 5.6 ms |
| INT8, 8 threads, int8 dense layers, 32 hot experts (peak 21876 MiB: too much) | 34.2 | 32.8 | 52.7 | 50 / 15 / 5.6 ms |
| **INT8, 8 threads, int8 dense layers, 24 hot experts (current)** | 31.2 | 32.9 | 52.3 | 51 / 14 / 5.6 ms |

- **Integer activations**: INT8 with 8 threads is the fastest, and leaves the most cores to the serving processes.
  Its relative error per expert is ~1 % (FP32: 1e-7; INT16: 6e-5). The quality test (retrieval at 8k and 32k tokens
  of context without MTP, 88 questions, with the token probabilities of all answer tokens) shows no difference: INT16,
  which is practically exact, measures the noise floor of the GPU's non-deterministic sums.

  | Run (no MTP) | Correct | Same answers as FP32 | Mean KL divergence | Top-token flips |
  |---|---|---|---|---|
  | FP32 CPU experts (reference) | 83/88 | | | |
  | INT16 activations | 82/88 | 86/88 | 0.0027 | 2 of 575 |
  | INT8 activations | 82/88 | 85/88 | 0.0026 | 3 of 570 |
  | INT8 activations, int8 dense layers, 24 hot experts (current) | 83/88 | 85/88 | 0.0018 | 3 of 570 |

- **More hot experts** pay off little on their own: 28 instead of 14 per layer cut the cold experts per step by
  6 % (946 -> 887) and made edits 2 % faster. The int8 dense layers (Gated DeltaNet and attention projections, see
  [`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next/README.md#checkpoint-with-int8-dense-layers-2026-10-26))
  free 2.6 GiB and also shorten the GPU part of a step (17 -> 14 ms), as there is less to read; with the freed VRAM,
  24 hot experts fit (each costs ~200 MiB of peak VRAM with the prefill buffers).
- The current configuration with `--repeat 2`: chat 32.4 t/s, 33.4 / 32.7 / 32.0 t/s at 6.4k / 32k / 100k tokens of
  context, 52.4 / 51.8 t/s in edits (short / after 32k tokens), prompt processing 1752 / 1994 / 1958 t/s and 2054 t/s
  for 260000 tokens, peak 20238 MiB. Compared with the settings of 2026-10-14 measured the same day (24 FP32
  threads): chat +27 %, generation +22-31 %, edits +31 %.

Tested afterwards, not better (2026-10-26, one run each, the current configuration otherwise):

- **GPU share** 0.15 / 0.3 (current) / 0.45: edits 52.2 / 52.3 / 52.5 t/s, CPU 52 / 51 / 50 ms and GPU 14 / 14 /
  17 ms per step: no difference.
- **28 hot experts**: edits 49.8 t/s (24: 52.3), as the GPU part grows (16 instead of 14 ms per step) by more than
  the CPU part shrinks; peak 21286 MiB.
- **An int8 lm_head** (1.2 GB of BF16 read in every step) does not start: the MTP drafter computes its reduced draft
  vocabulary with the target's lm_head, which the runtime requires to be unquantized (`reduced draft vocabulary
  requires an unquantized LM head`). It would also need the model to pass its quantization config to the lm_head.
- **More tokens per step for edits** (a step verifies at most 4 tokens, and copying code accepts 98 % of the 3 MTP
  drafts): not attempted. Batches of more than 4 tokens leave the runtime's CUDA-graph decode path for the CPU
  experts and fall back to its prefill paths (all cold experts streamed through the GPU, or the CPU without CUDA
  graphs). It would need a larger request slot in the CPU kernel, the GPU bridge kernel and the GPU share, CUDA
  graphs for the larger batches, and a drafter that copies from the context in addition to MTP, which vLLM does not
  combine (one method at a time; its `ngram` and `suffix` methods alone would lose MTP's ~2 accepted tokens per step
  in normal text).

Ideas not tried yet:

1. **The CPU frequency policy** `performance` instead of `powersave`/`balance_performance` (a system-wide setting,
   so not changed): the serving processes, which gained most from the freed cores, may gain from it too.
2. **Faster RAM**: the INT8 kernel reads ~59 GB/s with 8 threads, 82 % of the measured bandwidth, and is far from
   its compute limit, so it should gain from faster RAM almost proportionally (measure with
   [`../vllm-rtx3090-flash-next/cpu_moe_bench/run.sh`](../vllm-rtx3090-flash-next/cpu_moe_bench/run.sh) first, then
   retune the thread count). The rest of a step (GPU ~14 ms, tail ~5.6 ms of ~70 ms) does not change.

### Other settings (2026-10-14)

- **CPU KV offloading does not work with this runtime.** vLLM's native offloading (`kv-offloading-size`) requires
  every KV cache group's block size to be a multiple of the prefix-cache hash size (3200 tokens), but the runtime keeps
  the raw keys of the sparse attention's open compression group in a ring buffer of 8 rows (`CircularBufferSpec`:
  the compression ratio 4 plus the 3 drafted tokens), and the connector fails at startup (`tokens_per_block=8 not
  divisible by tokens_per_hash=3200`). The ring needs no offloading (vLLM's GPU prefix cache already excludes it and
  gives each request a fresh one), but skipping it would mean patching the per-group bookkeeping of the offloading
  connector's scheduler and worker. Stock vLLM's implementation of the model, which has no such group, offloads
  fine (see [`../vllm-rtx3070`](../vllm-rtx3070/README.md#qwen38-flash-next)). The startup check also rejects KV
  connectors with `expandable_segments`, as in [`../vllm/vllm.patch`](../vllm/vllm.patch). Without offloading, the
  GPU prefix cache of ~287k tokens still serves a returning conversation as long as other prompts have not evicted it.
- **MTP 1** (with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`, as the runtime only accepts multiples of 3200 tokens,
  which fit the block size of MTP 3 only): slower, chat 24.0 t/s and 23.1 / 26.2 t/s at 6.4k / 32k tokens of context,
  vs 26.5 and 26.6 / 28.3 t/s with MTP 3. The CPU part of a decode step falls from ~55 to ~36 ms, but the fixed parts
  (~17 ms on the GPU, ~4 ms at the end of the step) stay, and only ~1.6 instead of ~2.05 tokens are accepted per step.
  MTP 1 would have room for 310789 instead of 287594 tokens in the same KV cache.
- **`VLLM_QSA_EXACT_TOPK=1`** is no quality setting: it replaces the top-k kernel of the sparse attention's block
  selection with `torch.topk` and deterministic ordering, which selects the same blocks apart from ties.
- **The CPU's frequency policy** was `powersave` with the energy-performance preference `balance_performance`
  (amd-pstate). As decode is bound by the CPU's memory accesses, `performance` might be a few % faster; not tested, as
  it is a system-wide setting. Transparent huge pages (`madvise`) work: 54.7 of the 56.4 GiB of the expert arena got
  huge pages.

Results with the settings of 2026-10-14 (24 CPU pool threads; `--tests generation prompt chat depth --prompt-tokens
260000`), with llama.cpp for comparison. For the current settings, see [CPU expert kernel](#cpu-expert-kernel-2026-10-26):

| Test | vLLM | llama.cpp ([preset](../llama-cpp-agx-z2e/preset-qwen3.8-flash-next.ini)) |
|---|---|---|
| Context | 262144 | 262144 |
| Generation, 256 tokens | 29.6 t/s (MTP acceptance 44 %) | 27.6 t/s |
| Chat, 6 requests | 26.5 t/s (22.3-35.7) | 27.6 t/s |
| Depth 6.4k / 32k (llama.cpp: 36.5k) / 100k | 26.6 / 28.3 / 28.4 t/s | 26.6 / 24.9 / 20.4 t/s |
| Prompt processing at the same depths | 1649 / 1982 / 1979 t/s | 604 / 687 / 689 t/s |
| Full-context prompt (260000 tokens) | 2087 t/s | 778 t/s |
| Image 1280x960, prompt processing | 8.1 s (vision tower on the CPU, image scaled to 1 megapixel) | 3.2 s |
| Video 30 s 1080p | 35.4 s until the first token (5999 tokens) | not supported |
| Peak VRAM | 20876 MiB | 20618 MiB |
| RAM | 71 GiB container limit (56.4 GiB of experts, 1.7 GiB vision tower, 1.2 GiB token embedding, up to ~1.5 GiB more for a video) | ~66 GB |

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
