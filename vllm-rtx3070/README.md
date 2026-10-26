# vLLM on RTX 3070

vLLM on the RTX 3070 (8 GB) of the `agx-ai` LXC container, as an alternative to
[`../llama-cpp-rtx3070`](../llama-cpp-rtx3070) on the same GPU and port (9931). Only one of the two fits on the GPU at a
time. [`docker-compose.yml`](docker-compose.yml) uses the image of [`../vllm`](../vllm) (`vllm/vllm-openai:v0.31.0`,
CUDA 13.0, with the fixes of [`../vllm/vllm.patch`](../vllm/vllm.patch)) and the API key of the llama.cpp server.
The model is selected by commenting the config volume lines in or out.

| Configuration | Model | Context | Chat | Generation at 6.4k / 36.5k tokens | Prompt processing |
|---|---|---|---|---|---|
| [`config-gemma-4-26b-a4b-qat.yaml`](config-gemma-4-26b-a4b-qat.yaml) (default) | Gemma 4 26B-A4B QAT (Google's Q4_0, bit-exact), experts in RAM, MTP 2 | 98304 | 32 t/s | 30 / 27 t/s | 1100 t/s at 6.4k, 870 t/s at 36.5k, 580 t/s at 96k |
| [`config-qwen3.8-flash-next.yaml`](config-qwen3.8-flash-next.yaml) | Qwen3.8-Flash-Next W4A16 with int8 dense layers and an int4 lm_head, experts and n-gram table in RAM, FP8 KV cache, no MTP, image and video input (vision tower on the CPU) | 57344 | 15.4 t/s | 15.4 / 15.4 (at 32k) t/s | 377 t/s at 6.4k, 404 t/s at 32k, 455 t/s at 56k |

**Compared with llama.cpp on this GPU** ([`../llama-cpp-rtx3070`](../llama-cpp-rtx3070/README.md)): **llama.cpp is
clearly better for Gemma 4 26B-A4B.** It generates 1.7 times as fast (chat ~55 vs 32 t/s, 45-49 vs 27 t/s at 36-70k
tokens), processes long prompts twice as fast (~1220 t/s at 119k vs 580 t/s at 96k) and fits 25 % more context
(122880 vs 98304). Neither fits the model in 8 GB, but they keep the experts in RAM differently: llama.cpp computes
them on the 16-core EPYC, while vLLM has no CPU kernels for them, so the GPU reads the selected experts from RAM over
PCIe for every token.

**Qwen3.8-Flash-Next: vLLM is faster, llama.cpp had more context.** vLLM generates 13 % faster (15.4 vs 13.6 t/s,
15.4 vs 12.1 t/s at 32-36.5k tokens) and processes prompts ~15-20 % faster, as the GPU reads the experts over PCIe
faster than the CPU decodes llama.cpp's IQ quantization types, and it also takes videos. llama.cpp fitted 81920 tokens
of context instead of 57344, started in 1.5 min instead of 13 and needed neither the re-quantized checkpoint nor the
patched image; its GGUF was removed on 2026-10-13 to free disk space, see [Qwen3.8-Flash-Next](#qwen38-flash-next).

The rest of this file has the measurements behind the settings. They were made on 2026-10-12 and 2026-10-13 with the
tools in [`../benchmark`](../benchmark/README.md), on test servers
([`../benchmark/vllm_test_server.sh`](../benchmark/vllm_test_server.sh), port 9933) with the production containers
stopped. The results are in [`../benchmark/results/agx-ai.jsonl`](../benchmark/results/agx-ai.jsonl) (labels starting
with "agx-ai vLLM"). VRAM figures are those of the vLLM processes (`nvidia-smi --query-compute-apps`); the GPU has
no display, and 7839 MiB of its 8192 MiB are usable.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS in an LXC container on Proxmox VE (kernel 7.0.14-14-pve), host `agx-h12` (Supermicro H12SSL-i) |
| CPU | AMD EPYC 7302 (Zen 2, 16 cores, 32 threads; AVX2, no AVX-512), of which the container gets 24 threads (see [`../llama-cpp-rtx3070`](../llama-cpp-rtx3070/README.md)) |
| RAM | 256 GB DDR4-2400 ECC RDIMM: 8 × 32 GB SK Hynix HMA84GR7AFR4N-UH (dual-rank), one per channel in all 8 channels (theoretical 153.6 GB/s); 144 GiB available to the container (128 GiB until 2026-10-13). ~60 GB/s copy bandwidth in the container (PyTorch, 16 threads) |
| NPU | none |
| GPU | NVIDIA GeForce RTX 3070, headless; driver 610.57.04 (CUDA 13.3), core clock locked to 1600 MHz by [`../llama-cpp-rtx3070/gpu-power-limit`](../llama-cpp-rtx3070/gpu-power-limit), PCIe 4.0 x16 |
| GPU architecture | Ampere (GA104, sm_86) |
| VRAM | 8 GB GDDR6 (8192 MiB) |

**The 1600 MHz clock lock (or the 170 W power limit) is required**, as for llama.cpp: without it, long prompts can
crash the GPU (Xid 79), see [`../llama-cpp-rtx3070`](../llama-cpp-rtx3070/README.md#xid-79-crashes). During these
tests, the clock lock was active (1590 MHz at ~125 W under full load) and the power limit was the default 240 W,
which is now also the default of the script.

## Setup

```sh
# Image: ../vllm (built by ../.github/workflows/deploy-docker-vllm.yml)
docker pull ghcr.io/agenttix/agx-ai/vllm:latest
# Gemma 4 26B-A4B: the bit-exact QAT checkpoint with an fp16 embedding (~10 min on the CPU, ~30 GB of RAM; downloads
# Google's Q4_0 GGUF, 15 GB, which can be deleted afterwards)
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub -v ~/.cache/vllm-models:/out \
    --entrypoint repack_gemma4_qat.py ghcr.io/agenttix/agx-ai/vllm:latest \
    --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 --embed-bits 16 --lm-head-bits 8
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub python:3-slim sh -c \
    "pip install -q 'huggingface_hub[hf_xet]' && hf download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
# Qwen3.8-Flash-Next (125 GB, must stay in the cache), then its dense layers in int8 and its lm_head in int4
# (~10 min, 65 GB)
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub python:3-slim sh -c \
    "pip install -q 'huggingface_hub[hf_xet]' && hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
    --revision ef554143369a706525336f6b42a09094835dc077 --exclude 'docs/*' --exclude 'runtime/*'"
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro -v ~/.cache/vllm-models:/out \
    -v "$PWD":/src:ro --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm:latest /src/quantize_flash_next_dense.py \
    --out /out/Qwen3.8-Flash-Next-W4A16-FP8PLE-dense-int8-head4 --bits 8 --lm-head-bits 4
# Stop llama.cpp on this GPU and start vLLM
docker compose -f ../llama-cpp-rtx3070/docker-compose.yml stop
docker compose up -d
```

The Qwen3.8-Flash-Next configuration needs the fixes of 2026-10-13 in [`../vllm/vllm.patch`](../vllm/vllm.patch)
(UVA weight offloading), which are in the image once the workflow has rebuilt it after the push. Until then, build
it locally with `docker build -t ghcr.io/agenttix/agx-ai/vllm:latest ../vllm` (done on agx-ai on 2026-10-13). The
Gemma configuration works with and without them.

The first start compiles kernels for a few minutes; later starts of Gemma take ~3 min (the caches are in the `vllm-cache`
volume), most of it for loading the 15 GiB checkpoint and pinning 12 GiB of it in RAM.

## Gemma 4 26B-A4B

The bit-exact checkpoint of [`../vllm-radeon-vii`](../vllm-radeon-vii/README.md#checkpoints-2026-10-09) (all linear
weights from Google's Q4_0 GGUF), with Google's QAT MTP drafter, as in [`../vllm-rtx3090`](../vllm-rtx3090). It does
not fit in 8 GB: the experts alone take 11.2 GiB.

**Experts in RAM.** vLLM 0.31 has no CPU kernels for the experts of a GPU model, and no expert offloading in the
upstream code (only RFCs and forks, e.g. [vllm-project/vllm#38256](https://github.com/vllm-project/vllm/issues/38256)).
It does have generic weight offloading. With `offload-backend: uva` and `cpu-offload-params: [experts, embed_tokens]`,
the selected parameters are moved to pinned RAM and the GPU reads them directly over PCIe (unified virtual
addressing, zero-copy), without copying whole layers. The fused MoE kernel reads only the 8 experts per layer that
the router selects, ~765 MiB per token, so generation is bound by PCIe 4.0 x16 at ~20 GB/s. The embedding is only
looked up, so it stays in fp16 in RAM (on the 16 GB GPUs, the embedding has to be quantized to int4 to fit). On
the GPU remain the attention, the shared MLP, the routers, the int8 lm_head and the MTP drafter (4.12 GiB according
to vLLM). The `prefetch` backend would copy whole layers to the GPU instead, i.e. all 128 experts of a layer for
every token.

**Prompt processing** reads all experts of every layer over PCIe for every 2048-token chunk, so it is 2-2.5 times
slower than in llama.cpp, which also moves all expert weights over PCIe for every batch of a prompt (see
[its README](../llama-cpp-rtx3070/README.md#xid-79-crashes)), but with bulk copies instead of the zero-copy reads of
the MoE kernel.

| Variant (all with an int8 KV cache, one sequence) | Context | Chat | Gen. 6.4k | Gen. 36.5k | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|---|
| No MTP | 32768 | 26.6 | 25.2 | | 1135 t/s at 6.4k | 6118 MiB |
| MTP 2 | 40960 | 31.6 | 30.2 | | 1102 t/s at 6.4k | 6442 MiB |
| MTP 2 | 81920 | 31.6 | | 27.3 | 868 t/s at 36.5k, 649 t/s at 79k | 7502 MiB |
| MTP 2, 4096-token prefill chunks | 40960 | | | 26.7 | 934 t/s at 36.5k, 923 t/s at 39k | 7320 MiB |
| MTP 2, KV cache 2.85 GB | 98304 | 32.0 | | | 584 t/s at 96k | 7722 MiB |
| MTP 3, KV cache 2.7 GB, CPU KV offloading | 98304 | 31.7 | 29.9 | | 1073 t/s at 6.4k | 7622 MiB |

- **MTP** with 2 drafted tokens: +19 % (acceptance 65-70 %). The 3 verified tokens select more distinct experts than
  one token, so each step reads more from RAM; 3 drafted tokens were not faster (acceptance 57-61 %).
- **Context.** The KV cache takes ~27 KiB per token with int8 (including the sliding-window layers), and vLLM needs
  ~1.05 times the context. With 2.85 GB (108630 tokens), the peak was 7722 MiB with a 96000-token prompt, too close
  to the 7839 MiB, so the configuration uses 2.7 GB (102909 tokens) for the same 98304 tokens of context.
- **4096-token prefill chunks** were only 8 % faster at 36.5k tokens, but the sliding-window layers then need a
  larger KV cache per sequence, which halved the context (40352 tokens at most with 1.68 GiB).
- **Experts on the GPU.** Each layer of experts kept on the GPU (lowering `cpu-offload-gb`) would cost 408 MiB, i.e.
  ~15k tokens of context, and remove only 1/30 of the PCIe reads of a token. Not tested.
- **CPU KV offloading** (`kv-offloading-size: 16`, `kv-offloading-backend: native`) keeps prefix cache blocks evicted
  from the GPU in 16 GiB of RAM. Test: a 26616-token conversation, then two 98k-token prompts that evict it from the
  GPU, then the conversation again. With MTP, it was computed again (29 s) until a fix on 2026-10-13: the connector
  stored the blocks, but never loaded any. Debug logging in the offloading scheduler showed why: Gemma 4 has five
  sliding-window KV cache groups and one full-attention group, which also holds the MTP drafter's layers. With the
  default `prefix_cache_retention_interval` of 0, the sliding-window groups store only the window ending at the end
  of the prompt (blocks 1599-1662), but the lookup drops the last chunk of the drafter's group as volatile, so a hit
  ends one chunk earlier and needs blocks 1598-1661. No window was complete, so the lookup always returned 0 (without
  MTP, it came back in 0.3 s). [`../vllm/vllm.patch`](../vllm/vllm.patch) now also keeps the sliding-window tail for
  that boundary: the conversation came back in 2.2 s (381 MB loaded), with the same greedy answer as when it was
  first computed. [`../vllm-rtx3090/config-gemma-4-26b-a4b-qat.yaml`](../vllm-rtx3090/config-gemma-4-26b-a4b-qat.yaml)
  uses the same combination and gets the fix with the image.

Results of the configuration (`--tests generation prompt chat depth --prompt-tokens 96000 --depth 6400 36500`, with
the image including the fixes of 2026-10-13):

| Test | Result |
|---|---|
| Generation, 256 tokens | 36.7 t/s (MTP acceptance 98 % with the repetitive text of this test) |
| Chat, 6 requests | 32.0 t/s (28.4-35.4, acceptance 70 %) |
| Depth 6.4k / 36.5k | 30.0 / 27.6 t/s (prompt 1102 / 868 t/s) |
| Prompt of 96000 tokens | 584 t/s |
| Peak VRAM | 7582 MiB |

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a 125B MoE (512 experts, 10 active per token, 48 layers, 12 of them with Qwen Sparse Attention
and 36 with Gated DeltaNet) with a 51B-parameter n-gram embedding table. On agx-z2e it runs with the single-RTX-3090
runtime of DominikBucko/qwen38-flash-next-3090 (see [`../vllm-rtx3090`](../vllm-rtx3090/README.md#qwen38-flash-next)),
which needs 9.8 GB of VRAM for its BF16 dense weights alone and has no option for smaller GPUs. Stock vLLM 0.31 runs
the model with the experts read from RAM over UVA as for Gemma, but only with six fixes and a re-quantized
checkpoint (and two more fixes for images and videos and for the FP8 KV cache). The vLLM CPU backend is no alternative: it would need the whole model in RAM, the model has no CPU
implementation in vLLM, and the fast paths need AVX-512.

**Checkpoint.** [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)
(the checkpoint of the agx-z2e runtime) is the only one that stock vLLM loads with an FP8 n-gram table (47.6 GiB
instead of 95 GiB in BF16): the routed experts in Intel's AutoRound int4 (58.4 GiB), everything else in BF16 (9.2 GiB).
The other int4 checkpoints keep the n-gram table in BF16, or need patched vLLM builds for their FP8/INT6 layers.
[`quantize_flash_next_dense.py`](quantize_flash_next_dense.py) quantizes the BF16 Gated DeltaNet, attention and
shared expert projections to int8 and the lm_head to int4 with round-to-nearest (group size 128, the GPTQ format of
the experts), which takes the weights on the GPU from 8.0 GiB (without the embedding) to 4.83 GiB (5.03 GiB with an
int8 lm_head, as until 2026-10-13). The relative error of the int8 weights is 0.8 % (int4: 14 %); for the lm_head,
int4 cost less quality than the FP8 KV cache, see [Context size](#context-size). The output directory gets new shards for the quantized tensors (65 GB) and
symbolic links to the n-gram table and the other files of the source, which must stay in the Hugging Face cache. Since
2026-10-26, `--separate-shard` writes only the quantized tensors into one new shard (2.6 GB) and links all source
shards, for loaders that honor the index at the tensor level (not stock vLLM 0.31, which reads every tensor of a listed
shard), and `--keep-shared-experts` keeps the shared experts in BF16: used by
[`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next/README.md#checkpoint-with-int8-dense-layers-2026-10-26).

**The fixes**, in [`../vllm/vllm.patch`](../vllm/vllm.patch) (details in [`../vllm/README.md`](../vllm/README.md)),
found one at a time:

| Problem | Symptom | Fix |
|---|---|---|
| PyTorch's pinned memory rounds every block up to the next power of two | the 47.6 GiB n-gram table took 64 GiB, the 838 MB expert tensors 1 GiB each: the RAM ran out (the engine was killed silently at 129.5 of 131 GB) | exactly sized pinned memory (`cudaHostRegister` of an anonymous mapping); now 113 GB of RAM in use, 14-17 GB free |
| The offloader skipped a module whose first parameter is already on the CPU | layer 1, which holds the pinned n-gram table, kept its experts (1.25 GiB) on the GPU | look for the device among all parameters |
| The embedding is created outside `make_layers` | `cpu-offload-params: embed_tokens` had no effect (1.2 GiB on the GPU) | Qwen4Exp offloads it |
| The lm_head was created without the quantization config | the int8 lm_head did not load | pass it (checkpoints without a quantized lm_head are unaffected) |
| The Marlin post-processing of GPTQ experts transposed them to GPU copies and back | 2.8 GB of temporary GPU memory per layer, out of memory | transpose views only, repack w13 and w2 one after the other, reading the offloaded weights directly |
| Triton has no FP8 e4m3 type before sm_89 | `type fp8e4nv not supported in this architecture` in the n-gram lookup | copy the FP8 rows as uint8 |

Two settings are needed as well: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (the first request ran out of GPU memory
with the default of 512 MiB, which the memory profiling does not account for) and `TORCHINDUCTOR_COMPILE_THREADS=1`
(no RAM for parallel compile workers), both in [`docker-compose.yml`](docker-compose.yml).

**Memory.** ~107 GiB of the container's RAM are pinned (experts, n-gram table and embedding), and 4.83 GiB of
weights are on the GPU (5.03 GiB with the int8 lm_head). `free -m` after loading (the pinned weights and the offloading buffer appear as
"shared" and "buff/cache", but cannot be freed):

| Configuration | Container RAM | Available RAM |
|---|---|---|
| Text only, no CPU KV offloading | 128 GiB | 17181 MiB |
| Text only, 8 GiB of CPU KV offloading | 128 GiB | 8731 MiB |
| Vision tower on the CPU, no CPU KV offloading | 128 GiB | 13903 MiB, ~1.5 GiB less while a 30-s 1080p video is encoded |
| **Vision tower on the CPU, 8 GiB of CPU KV offloading (current)** | 144 GiB (since 2026-10-13) | 22070 MiB, 20733 MiB at the lowest while a 30-s 1080p video is encoded |

With 128 GiB, both together would have left only ~5.5 GiB, so the container got 16 GiB more. With the final
configuration of 2026-10-14 (FP8 KV cache, int4 lm_head, 57344 tokens), 21678 MiB are available after loading.
The KV cache on the GPU is described in [Context size](#context-size); MTP does not fit. Loading takes 13-23 min
(~10-20 min for reading 125 GB and pinning the weights, as the page cache has no room).

| Variant (int8 lm_head, BF16 KV cache) | Context | Chat | Gen. 6.4k / 16k | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|
| Eager mode, KV cache 0.8 GB | 12288 | 7.4 | 7.7 / - | 383 t/s at 6.4k | 7104 MiB |
| CUDA graphs for decoding, KV cache 0.8 GB | 12288 | 15.3 | 15.2 / - | 396 t/s at 6.4k | 7062 MiB |
| CUDA graphs, KV cache 1.1 GB (until 2026-10-13) | 20480 | 15.3 | 15.2 / 15.2 | 396 / 421 t/s at 6.4k / 16k, 478 t/s at 19.5k | 7322 MiB |

- **Generation** reads ~1.2 GB of experts per token over PCIe (10 of 512 experts in each of the 48 layers) and does
  not slow down with the context (Qwen Sparse Attention).
- **CPU KV offloading** (`kv-offloading-size: 8`; with 128 GiB of RAM, 16 GiB would not have fit next to the pinned
  weights) works for this hybrid model: an 8961-token conversation that two 18k-token prompts had evicted from the GPU came back in 5.7 s
  (358 MB loaded, the same time as a hit in the GPU cache) instead of 22.6 s. The stored blocks take ~34 KB per token
  (attention KV and Gated DeltaNet states), so the 8 GiB hold ~250k tokens. With the vision tower and 144 GiB of RAM
  (2026-10-13), a 9378-token conversation came back in 6.0 s instead of 22.8 s, and the images, videos and chat speed
  (15.3 t/s) were unchanged.
- **Images and videos** run through the vision tower on the CPU (`QWEN38_VISION_DEVICE=cpu`, 16 threads; see
  [`../vllm/vllm.patch`](../vllm/vllm.patch)), ported from the agx-z2e runtime image of
  [`../vllm-rtx3090-flash-next`](../vllm-rtx3090-flash-next/README.md): transformers' Qwen3-VL vision model in FP32, outside vLLM's
  memory accounting, with the pixel values kept on the CPU and only the BF16 embeddings copied to the GPU. The vision
  tower takes no VRAM (peak 7360 MiB instead of 7322 MiB), and the text speed is unchanged (chat 15.4 t/s, 15.3 t/s at
  16k tokens). The sizes are limited as on agx-z2e: images to 1 megapixel (`max_pixels`), videos to 64 frames sampled
  at 2 per second (vLLM's `qwen3_vl` video loader decodes only these) and 12.6 M pixels over all frames
  (`videos_kwargs.max_pixels`; vLLM 0.31 takes per-modality settings, so the video budget patch of agx-z2e is not
  needed). Test media as on agx-z2e (`temperature 0`), all described correctly (the text, the shapes with their colors
  and positions; the red square moving from left to right and "ALPHA" changing to "OMEGA"):

  | Input | Prompt tokens | First token | RAM while encoding |
  |---|---|---|---|
  | Image 1280x960, scaled to 1 megapixel | 1004 | 14.3-14.9 s | -0.2 GiB |
  | Image 1280x960 without scaling (`images_kwargs.max_pixels` did not apply) | 1232 | 17.1 s | -0.3 GiB |
  | Video, 20 s, 640x360, 30 fps | 4608 | 52-59 s | -1.0 GiB |
  | Video, 30 s, 1920x1080, 30 fps | 5998 | 66-74 s | -1.5 GiB |

  The time is mostly the vision tower: ~3 ms per 16x16 patch with the 16 cores of the EPYC (agx-z2e: ~1.4 ms with 32
  cores). The prompt token counts are the same as on agx-z2e.
- **Quality.** In the long-context retrieval test of
  [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py) (44 questions in a 17360-token
  context), the int8 checkpoint answered 38/44 (single lookups 32/32, two-step 6/12) and llama.cpp's IQ3_S 37/44
  (32/32, 5/12), so the round-to-nearest int8 layers do not cost measurable quality.

### Context size

Measured on 2026-10-13 and -14, following the steps that gave agx-z2e the full context (its token embedding in RAM
and fewer experts on the GPU, see [`../vllm-rtx3090`](../vllm-rtx3090/README.md#context-size-2026-10-13)). Here, the
embedding and all experts were already in RAM, so the room had to come from the KV cache itself and the dense weights.

**Why the KV cache was so large.** It took ~45 KiB per token in BF16, although the attention of the 12 QSA layers
needs only 24 KiB. vLLM gives all KV cache groups of this hybrid model the same page size, and the attention block
is made as large as needed for one Gated DeltaNet state per page: "Setting attention block size to 1568 tokens to
ensure that attention page size is >= mamba page size" (groups `[1568, 1568, 1568, 1568, 4, 1568]`). With prefix
caching (`mamba-cache-mode: align`), the Gated DeltaNet groups take a page per block too. The memory per token is
therefore set by the state size divided by the tokens per page.

**FP8 KV cache.** An FP8 cache halves the attention page per token, so a page holds 3136 tokens, which roughly halves
the memory per token of all groups: 1.1 GB held 42598 tokens instead of 24041. This model's sparse attention kernel
reads the FP8 cache through Triton's e4m3 type, which needs sm_89; [`../vllm/vllm.patch`](../vllm/vllm.patch) now
reads it as uint8 below sm_89 and converts it with bit operations (exact for all finite values). The first version
of the conversion made prompt processing twice as slow (228 instead of 421 t/s at 16k tokens), as the kernel's
one-warp prefill tiles spilled; a cheaper conversion (shifting the bits into FP16 positions and multiplying by 256)
takes 15.9 instead of 14.6 ms per layer and chunk.

**int4 lm_head** (0.6 -> 0.3 GiB, ~12k tokens of context with the FP8 cache), built with
`quantize_flash_next_dense.py --lm-head-bits 4`.

**Quality** in the retrieval test of [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py)
(44 questions in a 17360-token context, with the probabilities of the top 5 tokens at every answer position), compared
with FP8 KV and an int8 lm_head:

| Variant | Score | Identical answers | Mean KL divergence | Top-token flips |
|---|---|---|---|---|
| FP8 KV, int8 lm_head (reference) | 39/44 | | | |
| The same with 1024-token prefill chunks (noise floor, lossless in principle) | 39/44 | 44/44 | 0.0020 | 0 |
| BF16 KV, int8 lm_head | 38/44 | 43/44 | 0.0056 | 1 |
| FP8 KV, int4 lm_head | 39/44 | 42/44 | 0.0042 | 2 |

Both changes cost very little: the FP8 cache's divergence from BF16 is ~3 times the noise floor, and the int4 lm_head's
is lower than that.

| Variant (all with CUDA graphs, vision tower on the CPU) | KV cache | Context | Chat | Gen. 6.4k / 32k | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|---|
| BF16 KV, int8 lm_head (until 2026-10-13) | 1.1 GB, 24041 tokens | 20480 | 15.3 | 15.2 / - (15.2 at 16k) | 421 t/s at 16k | 7322 MiB |
| FP8 KV (first conversion), int8 lm_head | 1.1 GB, 42598 tokens | 32768 | 15.3 | - / - (15.2 at 16k) | 228 t/s at 16k, 240 t/s at 31k | 7594 MiB (too tight) |
| FP8 KV, int8 lm_head | 1.0 GB, 42130 tokens | 36864 | 15.3 | 15.2 / - (15.2 at 16k) | 377 / 399 t/s at 6.4k / 16k, 456 t/s at 35.5k | 7362 MiB |
| FP8 KV, int4 lm_head | 1.3 GB, 60948 tokens | 49152 | 15.4 | - / - (15.4 at 16k) | 399 t/s at 16k, 456 t/s at 48k | 7320 MiB |
| **FP8 KV, int4 lm_head (current)** | 1.3 GB, 63488 tokens | 57344 | 15.4 | 15.4 / 15.4 | 377 / 404 t/s at 6.4k / 32k, 455 t/s at 56k | 7342 MiB |
| The same with 4096-token prefill chunks | 1.0 GB | 40960 | out of GPU memory on the first long prompt (7571 MiB after loading) | | | |

- **Prefix cache granularity.** The prefix cache reuses whole blocks, which are now 3136 tokens instead of 1568, so
  the end of a conversation since the last block boundary is processed again on every turn: re-sending a 9378-token
  conversation took 9.7 s with FP8 (4.7-6.0 s with BF16), i.e. up to ~7 s, ~3.5 s on average at ~420 t/s.
  For long conversations, the larger context is worth more.
- **CPU KV offloading** works with the FP8 cache: after two 38.5k-token prompts had evicted a 9378-token conversation,
  it came back in 9.8 s (246 MB loaded) instead of 24.1 s. The stored blocks take ~18 KB per token, so the 8 GiB hold
  roughly 450k tokens.
- **Images and videos** were unchanged (1004 prompt tokens and 15.6 s for the image, 58.2 and 78.4 s for the videos).
- **Not done:**
  - Prefill chunks of 4096 tokens would stream the experts half as often, but run out of GPU memory (see above).
  - MTP: the module does not fit, and with 10 of 512 experts per token, verifying 3-4 tokens would read nearly 3-4
    times the experts per step over PCIe for ~2 accepted tokens. llama.cpp's MTP was slower here for the same reason.
  - int4 for the Gated DeltaNet projections would free ~1 GiB (~40k tokens), but round-to-nearest int4 has a 14 %
    relative error on these weights, and a better (calibrated) quantization would need a GPTQ run.
  - More experts on the GPU: the offloading works per layer (1.22 GiB each).
  - Without prefix caching, the Gated DeltaNet groups would need no pages per block, but every turn would process
    the whole conversation again.

**Compared with llama.cpp** ([`../llama-cpp-rtx3070`](../llama-cpp-rtx3070/README.md#qwen38-flash-next)): vLLM generates
13 % faster (15.4 vs 13.6 t/s in chat, 15.4 vs 12.1 t/s at 32-36.5k tokens) and processes prompts ~15 % faster at
6.4k tokens (377 vs 328 t/s), as the GPU reads the experts over PCIe faster than the 16 cores decode llama.cpp's IQ
types. Both run their vision encoder on the CPU: a 1280x960 image takes ~15 s until the first token with vLLM
(scaled to 1 megapixel, 1004 tokens) and ~18 s with llama.cpp (1261 tokens), and vLLM also takes videos. llama.cpp fit
more context (81920 vs 57344), started in 1.5 min instead of 13-23, and needed no re-quantized checkpoint or patched
image. Its GGUF was removed on 2026-10-13 to free disk space.

## Commands

```sh
cd ../benchmark
# Serving speed (the configuration of docker-compose.yml)
./llama_cpp_bench_http.py --url http://localhost:9931 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 96000 --depth 6400 36500 --repeat 1 \
    --server-config ../vllm-rtx3070/config-gemma-4-26b-a4b-qat.yaml --label "..."
# A configuration variant on a test server (port 9933; stop the production container first)
docker compose -f ../vllm-rtx3070/docker-compose.yml stop
CACHE_VOLUME=vllm-rtx3070_vllm-cache DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCHINDUCTOR_COMPILE_THREADS=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128" ./vllm_test_server.sh ghcr.io/agenttix/agx-ai/vllm:latest /tmp/config-variant.yaml
./llama_cpp_bench_http.py --url http://localhost:9933 --no-embedding --tests chat depth --repeat 1 --depth 6400 \
    --server-config /tmp/config-variant.yaml --label "..."
docker rm -f vllm-test && docker compose -f ../vllm-rtx3070/docker-compose.yml start
```
