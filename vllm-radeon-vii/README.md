# vLLM on the Radeon VII

vLLM for the Radeon VII (gfx906, 16 GB HBM2) on agx-z2e, as an alternative to
[`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii) on the same GPU and port (9932). Only one of the two fits on the
GPU at a time. The image is [`aiinfos/vllm-gfx906-mobydick`](https://github.com/ai-infos/vllm-gfx906-mobydick)
(vLLM v0.30.0 with ROCm 7.14 and custom gfx906 kernels) with a few fixes ([`Dockerfile`](Dockerfile),
[`vllm-gfx906.patch`](vllm-gfx906.patch)), built by
[`../.github/workflows/deploy-docker-vllm-radeon-vii.yml`](../.github/workflows/deploy-docker-vllm-radeon-vii.yml).
The model is selected in [`docker-compose.yml`](docker-compose.yml). There are two configurations, both with the
language model entirely on the GPU, an int8 KV cache, prefix caching and 8 GiB of CPU KV offloading (2026-10-14), and
images and videos with the vision tower on the CPU, which takes no VRAM and no context
([images and videos](#images-and-videos-2026-10-29)):

| Configuration | Model | Context | VRAM (peak) | RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|---|---|
| [`config-gemma-4-26b-a4b-qat.yaml`](config-gemma-4-26b-a4b-qat.yaml) (default) | Gemma 4 26B-A4B QAT, int4 bit-exact with Google's Q4_0 (built locally, see below) | 47104 | 16280 MiB of 16368 | 11.2 GiB (8 GiB of it the KV offloading) | 110 t/s, chat 106 t/s; 69/58/45 t/s at 6.4k/16k/32k tokens of context | 804/532/342 t/s at 6.4k/16k/32k, 263 t/s for 45.6k | [`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii) is better for this model, also for images and videos; an image ~9 s, a video 6 frames |
| [`config-qwen3.8-27b.yaml`](config-qwen3.8-27b.yaml) | Qwen3.8-27B GPTQ int4 with int4 embedding and lm_head | 33792 | 16285 MiB of 16368 | 11.0 GiB (8 GiB of it the KV offloading) | 44 t/s, chat 43 t/s; 37/30/25 t/s at 6.4k/16k/28k | 240/179/136 t/s at 6.4k/16k/28k, 125 t/s for 32.3k | faster than llama.cpp up to ~28k tokens of context, which fits more context; an image ~30 s (llama.cpp: ~14 s), a video ~2000 tokens |

**Compared with the llama.cpp presets on this GPU:**

- **Gemma: llama.cpp is better.** It fits 1.9 times the context (88064), generates 1.5-1.7 times as fast at 6-36k
  tokens of context (101/91/78 t/s) and processes prompts 1.6-3 times as fast. Short chats are equally fast.
- **Qwen: vLLM is faster for short and medium contexts, llama.cpp for long ones.** vLLM generates 45 % faster in chat
  (43.2 vs 29.9 t/s, without speculative decoding vs llama.cpp's MTP), 31 % faster at 6.4k tokens of context, 10 %
  faster at 16k and as fast at 28-30k (24.7 vs 24.3 t/s). llama.cpp fits 1.4 times the context (47104) and processes
  long prompts up to 1.7 times as fast.

The reasons:

1. **Memory.** Qwen's weights take as much as llama.cpp's GGUF (13.25 vs 13.27 GiB), Gemma's 0.8 GiB more (14.04 GiB:
   the embedding and the lm_head are untied, and the MoE kernel needs 0.36 GiB of zero points). vLLM's runtime
   (HIP kernels, activations and scratch buffers, CUDA graphs, and the recurrent state or the sliding-window cache for
   a whole prefill chunk) takes another 1.0-1.35 GiB, so only 0.9-1.3 GiB are left for the KV cache. The int8 cache
   needs ~20 KiB per token for Gemma (like llama.cpp's q8_0) and ~33 KiB for Qwen (llama.cpp: 44 KiB).
2. **Attention.** The fast custom gfx906 flash attention of the image supports only head sizes up to 256 and only an
   fp16 KV cache. Gemma 4 has global layers with head size 512, so vLLM uses Triton attention for all of its layers,
   and an fp16 KV cache for Qwen fits only ~11k tokens. Triton attention on gfx906 (no matrix cores) slows down much
   more with the context length than llama.cpp's flash attention, in prompt processing above all: at 32k tokens of
   context, Gemma's 5 global layers take ~2 s per 512-token chunk.
3. **Speculative decoding** (MTP) does not fit next to a useful context: Qwen's MTP head is another 15-28 % faster
   (51-56 t/s in chat) but leaves 4.6-9k tokens of context, and Gemma's separate MTP drafter (0.8 GB) does not fit at
   all.

Returning to a long conversation is faster with vLLM, as prefix cache blocks evicted from the GPU are kept in RAM
(Gemma: 0.2 s instead of 27 s for a 15000-token prompt).

The benchmark tools are in [`../benchmark`](../benchmark) (`llama_cpp_bench_http.py` supports vLLM); the commands
are at the end.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-38-generic) |
| CPU | AMD Ryzen Threadripper 3970X (32 cores, 64 threads) |
| RAM | 128 GB DDR4-2666 (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPU | AMD Radeon VII, used only for this server (the RTX 3090 in the same machine runs [`../llama-cpp-rtx3090`](../llama-cpp-rtx3090)) |
| GPU architecture | GCN 5.1 (Vega 20, gfx906), 60 CUs, PCIe 3.0 x16 |
| VRAM | 16 GB HBM2 (16368 MiB) |

## Setup

```sh
# Gemma: build the checkpoint (downloads Google's 14.4 GB Q4_0 GGUF and ~2.5 GB of the unquantized QAT checkpoint;
# ~10 min on the CPU and ~30 GB of RAM). The script is a command in the NVIDIA image of ../vllm; this image runs it
# from ../vllm instead.
docker run --rm -v "$PWD/../vllm":/src:ro -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    -v ~/.cache/vllm-models:/out --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm-gfx906:latest \
    /src/repack_gemma4_qat.py --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct
# A checkpoint built before 2026-10-29 lacks the vision tower's linear weights: add them (1.1 GB, vision.safetensors)
docker run --rm -v "$PWD/../vllm":/src:ro -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    -v ~/.cache/vllm-models:/out --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm-gfx906:latest \
    /src/repack_gemma4_qat.py --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct --vision-only
# Qwen: download the checkpoint (14.8 GB) into the Hugging Face cache
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub --entrypoint hf \
    ghcr.io/agenttix/agx-ai/vllm-gfx906:latest download JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8
docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml stop
docker compose up -d
```

The first start compiles kernels for ~6 min; later starts take ~1.5-3 min (the compile caches are in the
`vllm-cache` volume). The model ids, the port and the API key (`LLAMA_API_KEY` of
[`../llama-cpp/llama-cpp.env`](../llama-cpp/README.md)) are the same as for llama.cpp, so clients such as
LiteLLM work with either server, as long as the prompts fit in vLLM's smaller context.

## Checkpoints (2026-10-09)

vLLM v0.30.0 no longer loads GGUF files, so the GGUFs of the llama.cpp presets cannot be used. The checkpoints are
in the same Hugging Face cache (`~/.cache/huggingface/hub`) where possible, and in `~/.cache/vllm-models` (mounted as
`/models`) when built locally.

### Gemma 4 26B-A4B QAT

Google's quantization-aware training (QAT) targets ggml's Q4_0: int4 with an fp16 scale per block of 32 weights,
symmetric. That is exactly compressed-tensors' int4 `pack-quantized` format with group size 32, which vLLM runs with
its exllama and gfx906 W4A16 MoE kernels, so Google's own Q4_0 GGUF can be converted without any rounding.

The int4 checkpoints of this model on Hugging Face are not on Google's grid. For
[`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4)
(int4, group size 32, made with an MSE observer), compared block by block with Google's GGUF (64 rows of two
attention matrices):

| | `self_attn.q_proj` of layer 0 | `self_attn.o_proj` of layer 5 |
|---|---|---|
| int4 values equal | 61 % | 71 % |
| scales equal | 0.2 % | 0.5 % |
| mean relative difference of the dequantized weights | 12.7 % | 5.5 % |

Its experts correlate with Google's at only 0.98-0.99. Its dense MLP is unquantized fp16 (the QAT weights), and
ggml's Q4_0 rounding (`d = max / -8`, `q = trunc(x / d + 8.5)`) applied to it reproduces Google's GGUF bit for bit, so
the GGUF is the exact QAT target. Besides, that checkpoint is 17.2 GB, with the dense MLP, the embedding and the vision
tower in fp16, and does not fit.

[`../vllm/repack_gemma4_qat.py`](../vllm/repack_gemma4_qat.py) therefore builds the checkpoint from Google's two QAT
repositories: every linear weight of the language model (attention, dense MLP, experts) from the Q4_0 GGUF, and the
norms, routers, embedding and vision tower from the unquantized QAT checkpoint (downloaded with HTTP range requests,
~2.5 GB of 52 GB). The embedding and the lm_head are untied, because vLLM cannot use a quantized embedding as the
lm_head, and both are quantized to int4 with group size 32 (round-to-nearest; the GGUF has them in Q6_K). The vision
tower stays fp16 and is not loaded (`language-model-only`).

| Variant | Size of the language model | Result |
|---|---|---|
| Embedding fp16, lm_head int8 | 15.0 GiB | runs out of memory while loading (15.62 GiB allocated) |
| **Embedding and lm_head int4, group size 32 (current)** | 13.7 GiB | 14.04 GiB on the GPU after loading |

The extra 0.36 GiB on the GPU are zero points, which the gfx906 W4A16 MoE kernel needs even for symmetric weights.

### Qwen3.8-27B

[`JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB`](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB):
GPTQ int4 with group size 128 (calibrated on ultrachat), and int4 embedding and lm_head (round-to-nearest), 13.25 GiB on
the GPU without the vision tower. Its model card reports a mean KL divergence of 0.04 against BF16. The usual int4
checkpoints of this model are 19-21 GB, as they keep the embedding and the lm_head (2.4 GiB each) in BF16; none of them
fit. The `mtp-int8` revision has the MTP head in int8 (0.4 GiB instead of 0.8).

## Optimizations from the other configurations (2026-10-14)

The configurations of [`../vllm-rtx3090`](../vllm-rtx3090), [`../vllm-rtx3070`](../vllm-rtx3070) and
[`../vllm`](../vllm) (vLLM 0.31 on NVIDIA) found several improvements since the measurements below. Tested here:

| Change | Result | Used |
|---|---|---|
| `hf-overrides` with `max_position_embeddings` = the context (from Gemma 4 31B on the RTX 3090) | The rotary embedding caches are no longer allocated for 262144 positions: 0.33 GiB more for the KV cache with Gemma (two caches, head sizes 256 and 512), 0.11 GiB with Qwen | yes: Gemma 34816 -> 47104 tokens of context, Qwen 30720 -> 33792, at the same peak VRAM |
| More split-KV segments in decoding (`VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS`, default 16) | See below; the image's own comment suggests 64-128 for gfx906 | yes, 128 |
| Split-KV kernel for the verification batches of speculative decoding (the Triton fix of [`../vllm/vllm.patch`](../vllm/vllm.patch)) | Already in this image (vLLM 0.30 with the gfx906 fork's changes) for the int8 cache | - |
| Larger query blocks in prefill (the sm_86 tuning of [`../vllm/vllm.patch`](../vllm/vllm.patch)), tuned for gfx906 | See below: faster in isolation, but not in the server | no |
| `int4_per_token_head` KV cache (from Gemma 4 31B on the RTX 3090) | Gemma fits 81920 tokens of context, but its kernels are 2-3 times as slow on gfx906 | no |
| Gemma's MTP drafter, Gemma's int8 embedding | Do not fit in 16 GB | no |

**Split-KV segments.** With one sequence, the decode kernel runs one program per KV head and segment: with the
default of 16 segments, 32 programs for Gemma's global layers (2 KV heads) and 64 for Qwen (4 KV heads), on 60 CUs.
More segments fill the GPU at long contexts; the extra buffers are a few MB (+20-40 MiB at the peak). Generation
in t/s:

| Segments | Gemma chat | Gemma 6.4k / 16k / 32k | Qwen chat | Qwen 6.4k / 16k / 28k |
|---|---|---|---|---|
| 16 (before) | 109.6 | 60.7 / 43.1 / 30.9 | 43.5 | 34.6 / 25.6 / 19.4 |
| 64 | 108.3 | 72.8 / 56.6 / 40.8 | 43.3 | 37.3 / 29.5 / 23.9 |
| **128 (current)** | 105.3-105.8 | 68.9-69.3 / 57.4-57.6 / 45.0-45.1 | 43.3 | 37.0 / 30.2 / 24.8 |

128 segments are 4 % slower than 16 in Gemma's short chats and 13-46 % faster from 6.4k tokens of context on.

**Prefill tuning.** A microbenchmark of the Triton attention kernels with the configurations' shapes
([`triton_attention_bench.py`](triton_attention_bench.py): one 512-token chunk at the end of the context, random
caches) found faster launch parameters for gfx906: 32 query rows per block
and a KV tile of 32 (the image uses 16 and 16). In isolation, Gemma's attention took 15 % less time per chunk at 32k
tokens of context (global layers 417 -> 361 ms, sliding-window layers 6.3 -> 4.4 ms), and Qwen's 30-50 % less with
its block size of 1552. In the server, Gemma processed a 32k prompt more slowly (308-312 instead of 342 t/s, every
variant of the two parameters), and one of the three requests of a depth test generated no tokens; Qwen was 7-19 %
faster (213 instead of 179 t/s at 16k), but needed 70 MiB more VRAM at the peak. The change was dropped. A larger
query block does not help on gfx906 as it does on sm_86: without matrix cores, the kernel is limited by its
arithmetic (~1.3 TFLOPS for Gemma's global layers), not by reading the KV cache.

**int4 KV cache** (`int4_per_token_head`, with a Hadamard rotation): ~12 KB per token instead of ~27 KB for Gemma, so
81920 tokens of context fit in about the same memory. But its kernel has no gfx906 tuning: Gemma generated 37 instead
of 106 t/s in chat and 14 instead of 45 t/s at 32k tokens of context, processed prompts at half the speed (176 instead
of 342 t/s at 32k), and a prompt of 80400 tokens did not finish in 15 minutes. Its split-KV kernel also does not
handle the verification batches of speculative decoding.

## Fixes in the image

[`vllm-gfx906.patch`](vllm-gfx906.patch), applied by the [`Dockerfile`](Dockerfile):

- **The exllama GEMM wastes up to 2.4 GiB.** `ops.gptq_gemm` allocates a K x N fp16 scratch buffer for dequantizing
  on every call, even when it does not use it (fewer than 50 rows). For the lm_head of a large vocabulary that is
  2.4 GiB (Qwen, 248320 x 5120) or 1.4 GiB (Gemma) of peak memory, which vLLM's profiling then subtracts from the KV
  cache: Qwen failed to start (`OOM ... trying to allocate 2543845376 bytes`). Measured in isolation, the scratch
  buffer is allocated for 1-64 rows alike, while the lm_head takes 0.83 ms for one row, i.e. the kernel does not
  dequantize. The patch splits matrices whose scratch buffer would exceed 512 MiB (the lm_heads) into column chunks.
  Each chunk costs one more kernel launch, ~18 us per forward pass on gfx906 (Qwen's lm_head in 39 chunks of 64 MiB:
  1.56 instead of 0.83 ms), so the chunks are as large as the largest scratch buffer that another layer needs anyway
  (Qwen: 340 MiB for the MLP, i.e. 7 chunks; Gemma: the minimum of 64 MiB, 23 chunks). A first version that chunked
  every matrix above 64 MiB, including Qwen's MLP projections in all 64 layers, cost ~15 ms per token: Qwen generated
  26.1 instead of 43.5 t/s in chat. The image's Triton W4A16 kernel, which has no scratch buffer, is no alternative:
  it is 10 times slower on gfx906 (10.6 vs 0.84 ms for Qwen's lm_head, 0.87 vs 0.07 ms for an MLP projection).
- **Qwen's MTP head with a quantized embedding.** The MTP module did not pass the quantization config to its
  embedding, like the patch of the checkpoint's model card (the main model already had the fix).
- **Triton attention crashed** with `'CommonAttentionMetadata' object has no attribute '_seq_lens_cpu'` (a field that
  this build of vLLM removed). The patch passes `None`, so that the Triton kernels are used instead of the optional
  torch SDPA paths.
- **CPU KV offloading with `expandable_segments`.** vLLM rejects the combination for all KV connectors, because RDMA
  connectors register GPU memory that the VMM allocator could remap. The native CPU offloading only copies with device
  pointers, and the KV cache is allocated once and never freed, so the patch exempts it. The alternative,
  `enable-cumem-allocator`, makes the engine exit without an error message after loading the weights.
- **Videos of checkpoints without a video preprocessor config** (Gemma 4) failed with `unhashable type: 'dict'`: vLLM
  looks up the video processor of the model type in transformers, whose mapping is `{backend: class name}` since
  transformers 5.19. The patch takes the class name.

[`cpu-vision.patch`](cpu-vision.patch) and the modules it imports run the vision towers on the CPU, see
[images and videos](#images-and-videos-2026-10-29).

## Images and videos (2026-10-29)

Both checkpoints have their vision towers (Gemma 1.1 GiB, Qwen 0.86 GiB in 16 bits), but on the GPU they and their
activations would cost ~1-2 GiB of the 0.9-1.3 GiB of the KV cache, so the configurations were text only
(`language-model-only`). Now the towers run on the CPU, in FP32 with 32 threads, as in
[`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md#images-and-videos):

- [`cpu_vision_gemma4.py`](cpu_vision_gemma4.py) builds transformers' Gemma 4 vision model (the same module that
  vLLM uses on the GPU) on the CPU and encodes images and video frames with it; vLLM's embedder projects its output
  on the GPU (1152 values per token).
- [`cpu_vision_qwen.py`](cpu_vision_qwen.py) is the Qwen3-VL tower of HyperQwen's configuration, with the output in
  the model's FP16.
- [`cpu-vision.patch`](cpu-vision.patch) creates them in `gemma4_mm.py` and `qwen3_5.py` (with
  `GEMMA4_VISION_DEVICE=cpu` and `QWEN38_VISION_DEVICE=cpu`, set in [`docker-compose.yml`](docker-compose.yml)), and
  makes vLLM's V1 model runner leave the pixel values of such a model on the CPU.
- [`cpu_onednn.py`](cpu_onednn.py): the PyTorch of the image has no MKL or other BLAS library, so its CPU matrix
  multiplications fall back to a single-threaded loop (~7 GFLOP/s). A Gemma image took 925 s. The module computes the
  linear layers with oneDNN, which the build has (~100 times as fast: 0.035 instead of 3.3 s for a 2520 x 1152 x 4304
  product), and the attention one head at a time with oneDNN instead of SDPA (1.9 times as fast). A Gemma image then
  took 12.5 s, with the same output (largest difference 0.002 of values up to 945).
- The configurations allow up to 64 images per request (as agent sessions collect screenshots) and 1 video, and skip
  vLLM's encoder profiling, which would take minutes on the CPU.

**The Gemma checkpoint lacked the vision tower's linear weights.** The pattern of
[`../vllm/repack_gemma4_qat.py`](../vllm/repack_gemma4_qat.py) for the linear weights that come from the GGUF
(`.layers.N.self_attn.q_proj.` etc.) also matched those of the vision tower, which therefore were neither fetched
nor written, and with `language-model-only` no one noticed. vLLM loaded the tower with uninitialized weights: Gemma
answered that it saw no image. The script now matches only the language model, and `--vision-only` adds the 189
missing tensors to an existing checkpoint as `vision.safetensors` (see [setup](#setup)). The other configurations
with these checkpoints are text only.

**Gemma's videos have at most 6 frames.** vLLM processes Gemma 4's images and videos with bidirectional attention,
so each one has to fit in a single prefill chunk (`max-num-batched-tokens`, 512 here; larger chunks need more VRAM),
and a frame takes 78 tokens. The limit is `"num_frames": 6` in `limit-mm-per-prompt` and in `media-io-kwargs` (468
tokens). Images take 280 tokens. Qwen's images and videos are chunked like text: images are scaled down to
1 megapixel (~1000 tokens), videos sampled at 2 frames per second, at most 64 frames and 4.2 megapixels over all
frames (~2000 tokens). With 12.6 megapixels (~4600 tokens, the limit of HyperQwen's configuration), the VRAM peak
rose to 16340 MiB, 55 MiB over that of the longest text prompt; with 4.2 megapixels it is 16295 MiB.

Measured with generated test media (greedy, without thinking; all answers correct unless noted):

| Model | Input | Prompt tokens | Time of the request | VRAM peak |
|---|---|---|---|---|
| Gemma | 1280x960 image with text and three shapes | 297 | 8.0 s (an image again: 0.1 s, from the caches) | 16230 MiB |
| Gemma | 16 1280x960 images, each with its number | 4320 | 106 s | 16230 MiB |
| Gemma | 46k tokens of code and the image | 46286 | 179 s | 16230 MiB |
| Gemma | 20-s 640x360 video (a red square moving to the right, "ALPHA" and then "OMEGA") | 480 | 13 s | 16230 MiB |
| Gemma | the same in 1080p, 30 s | 480 | 11 s | 16230 MiB |
| Qwen | the image | 1001 | 31.8 s | 16258 MiB |
| Qwen | 16 images | 15614 | 447 s | |
| Qwen | the 640x360 video, 4.2 megapixels | 2022 | 24 s | 16295 MiB |
| Qwen | the 1080p video, 4.2 megapixels | 2272 | 28.5 s | 16295 MiB |
| Qwen | 26k tokens of code and the 1080p video, 12.6 megapixels | 31989 | 300 s | 16342 MiB |

Gemma's answers about the videos had the text right, but with 6 frames the square's movement partly wrong. The
peaks of the text-only stress tests are 16280 (Gemma) and 16285 MiB (Qwen), so neither context had to be reduced.
Qwen's tower is ~4 times as slow per image as in HyperQwen's image (7.5 s for the same image), which has MKL;
llama.cpp's encoder on the CPU takes ~14 s for it ([`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii/README.md)).

## Memory (2026-10-09, before the changes of 2026-10-14)

The KV cache gets what remains of the 15.98 GiB that PyTorch sees after the weights, vLLM's runtime and the
activations. Findings, with Qwen unless noted:

| Setting | Effect |
|---|---|
| CUDA graphs: the default 51 batch sizes up to 512 (Qwen3.5-0.8B) / sizes 1-8 / only `FULL_DECODE_ONLY` with size 1-2 | 5.89 / 0.83 / 0.11-0.26 GiB |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 840 MiB less VRAM after startup with the same KV cache (16129 -> 15289 MiB), the fragmentation of PyTorch's allocator |
| `gpu-memory-utilization` (0.97-0.99) with profiling | 0.5-0.6 GiB of KV cache; profiling counts ~1.1 GiB of non-PyTorch memory and 0.6-0.9 GiB of activations |
| `kv-cache-memory-bytes` set explicitly, stress-tested with full-context prompts | 1.30 GiB of KV cache for Qwen, 0.88 GiB for Gemma (which has 0.8 GiB more weights) |
| `max-num-batched-tokens` 512 instead of 2048 | smaller activation peak during prompt processing |
| `max-num-seqs` 1 | one recurrent state (Qwen, 0.32 GiB) and one sliding-window cache (Gemma, ~0.4 GiB) |
| `language-model-only` | no vision tower (Qwen 0.9 GiB, Gemma 1.1 GiB) |
| `cpu-offload-params: [embed_tokens]` | no effect: the UVA offloader only covers the decoder layers |

## KV cache and attention backend (2026-10-09)

Qwen3.8-27B has 16 full-attention layers with a KV cache (4 KV heads of 256) and 48 Gated DeltaNet layers with a
fixed-size recurrent state. An fp16 cache needs 64 KiB per token, int8 or fp8 ~33 KiB, plus 0.32 GiB for the
recurrent state. The custom gfx906 flash attention (`CUSTOM`) only supports an fp16 cache; quantized caches need
`TRITON_ATTN`. Qwen without MTP, each at the largest context that fits, generation in t/s, prompt processing in
parentheses:

| KV cache, backend | Context | Chat | Depth 6.4k | Depth 16k | Depth 28-30k | Prompt of ctx - 1.5k tokens | Peak VRAM |
|---|---|---|---|---|---|---|---|
| fp16, `CUSTOM` | 11264 | 40.1 | 37.4 (261) | - | - | (256) | 16114 MiB |
| **int8 per token and head, `TRITON_ATTN` (current)** | 30720 | 43.5 | 34.6 (238) | 25.6 (178) | 19.4 (135) | (133) | 16286 MiB |
| llama.cpp `UD-IQ4_XS`, q8_0, MTP (for comparison) | 47104 | 29.9 | 28.1 (249) | 27.3 (240) | 24.3 (225) | (211) | 16298 MiB |

The fp16 cache with the custom kernel is 8-10 % faster at 6.4k tokens of context but fits only a third of the
context (it also allocates gather buffers that grow with the context). fp8 (e4m3) was as fast as int8 per token and
head (measured before the chunking fix), and int8 is the format closest to llama.cpp's q8_0, whose quality was within
the noise for this model (see [`../llama-cpp-radeon-vii/README.md`](../llama-cpp-radeon-vii/README.md)); vLLM's int8
cache was not tested for quality. Triton attention loses much more speed with the context length than llama.cpp's
flash attention: from 6.4k to 28k tokens of context, generation drops by 44 % (llama.cpp: 14 % to 30k).

Gemma 4 26B-A4B has 25 sliding-window layers (head size 256, window 1024) and 5 global ones (head size 512). vLLM
forces `TRITON_ATTN` for all layers of a model with mixed head sizes. Patching that out (in a test image), so that the
sliding-window layers use `CUSTOM`, needs an fp16 cache for them: with all layers fp16, 8k tokens of context, chat was
73 t/s instead of 111 and the depth test at 6.4k the same (60 t/s), so the patch was dropped. Mixing an fp16 cache for
the sliding-window layers (`kv-cache-dtype-skip-layers: [sliding_window]`) with int8 for the global ones made vLLM
need 4 times the memory per token. All layers therefore use Triton with the int8 cache, ~20 KiB per token plus ~0.4 GiB
for the sliding windows.

## Context size (2026-10-09, before the changes of 2026-10-14)

Each configuration was started with `kv-cache-memory-bytes` and `max-model-len` set to the largest context that the
KV cache holds, and then stress-tested with the full benchmark below, including a prompt of ctx-size - 1500 tokens;
peak VRAM from `amd-smi`:

| Model | `kv-cache-memory-bytes` | Context | VRAM after startup | Peak VRAM |
|---|---|---|---|---|
| Gemma | 600000000 | 16384 | 15841 MiB | 15923 MiB |
| **Gemma (current)** | 950000000 | 34816 | 16181 MiB | 16263 MiB (16282 MiB in the production container) |
| **Qwen (current)** | 1400000000 | 30720 | 16145 MiB | 16286 MiB |
| Qwen, fp16 cache | 1150000000 | 11264 | 15933 MiB | 16114 MiB |

Both configurations have ~80 MiB of headroom at the peak, like the llama.cpp presets (70 MiB); if vLLM ever runs out
of memory, lower `kv-cache-memory-bytes` and `max-model-len` together (~2.9k tokens per 0.1 GB for Qwen, ~5k for
Gemma). Before the chunking fix, which leaves Qwen's 340 MiB MLP scratch buffer unchunked, Qwen fit 40960 tokens.

## Speed (2026-10-09, before the changes of 2026-10-14)

The current configurations, measured with the full command below (Gemma in the production container, Qwen on a test
server with the same image, configuration and environment), with the llama.cpp presets for comparison (from
[`../llama-cpp-radeon-vii/README.md`](../llama-cpp-radeon-vii/README.md), with MTP). Generation in t/s, prompt
processing in parentheses:

| Test | Gemma (34816) | llama.cpp Gemma (88064) | Qwen (30720) | llama.cpp Qwen (47104) |
|---|---|---|---|---|
| Generation, 256 tokens (repeated text, temperature 0) | 113.0 | 124.2 | 44.3 | 34.7 |
| Prompt of ctx - 1500 tokens | (333) | (761 at 86564) | (133) | (211 at 45604) |
| Chat, 6 requests | 109.0 | 113.7 | 43.5 | 29.9 |
| Code context of 6.4k tokens | 60.5 (802) | 101.1 (1260) | 34.6 (238) | 28.1 (249) |
| Code context of 16k tokens | 42.9 (531) | 90.5 (1180) | 25.6 (178) | 27.3 (240) |
| Code context of 28-36.5k tokens | 30.8 (342) at 32k | 77.9 (1008) at 36.5k | 19.4 (135) at 28k | 24.3 (225) at 30k |

Without speculative decoding, vLLM generates as fast as (Gemma) or faster than (Qwen) llama.cpp with MTP in short
chats. vLLM's dense int4 GEMMs (exllama) are faster than llama.cpp's for Qwen, but its attention slows down much more
with the context length, and its prompt processing is slower.

## MTP for Qwen (2026-10-09)

Speculative decoding with Qwen's own MTP head (`mtp-int8` revision, 0.4 GiB), int8 cache, `TRITON_ATTN`. The MTP
head, its attention layer and the recurrent states of the drafted positions leave much less room for the context:

| Drafted tokens | Context | Generation, 256 tokens (acceptance) | Chat (acceptance) | Depth 6.4k (acceptance) | Peak VRAM |
|---|---|---|---|---|---|
| none (current) | 30720 | 44.3 | 43.5 | 34.6 (238) | 16286 MiB |
| 1 | 9216 | 56.6 (84 %) | 51.2 (72 %) | 39.9 (232) (72 %) | 16273 MiB |
| 2 | 4608 (4704 fit) | 72.1 (80 %) | 55.8 (54 %) | - | 16195 MiB |

MTP is another 15-28 % faster, but 4.6-9k tokens of context are too little for agents. The settings for one drafted
token are at the end of [`config-qwen3.8-27b.yaml`](config-qwen3.8-27b.yaml).

For Gemma, MTP needs Google's separate drafter
([`gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant),
0.8 GB in fp16), which does not fit next to the main model; it was not tried.

## Prefix caching and CPU KV offloading (2026-10-09)

Automatic prefix caching is on: a request that starts with the same tokens as an earlier one reuses its KV cache. With
one sequence and ~31-35k tokens of KV cache, only one or two long conversations fit on the GPU, so
`kv-offloading-size: 8` keeps the blocks evicted from the GPU in 8 GiB of pinned RAM (vLLM's native CPU offloading).
Test: three different 15000-token code contexts, then the same three again (with the GPU cache alone, each evicts the
next):

| Model | First time | Again, GPU cache only | Again, with CPU offloading |
|---|---|---|---|
| Gemma | 27.4 s | 27.4 s | **0.2 s** |
| Qwen (before the chunking fix) | 85 s | (not measured; all misses as with Gemma) | **8.9 s** |

For Qwen, the recurrent states of the Gated DeltaNet layers are only checkpointed at some positions, so part of the
prompt is processed again. The V2 model runner crashes with offloading (`Memory Fault Error ... kernel:
_gather_block_tables_kernel`), so the V1 runner is used (`VLLM_USE_V2_MODEL_RUNNER=0`); it is as fast (Gemma: chat 109.1
vs 111.5 t/s, the depth tests within 1 %). The offloading takes no GPU memory and no CPU time while idle.

## Other findings

- `llama_cpp_bench_http.py` and `llama_cpp_quality_http.py` now support vLLM, and `vllm_test_server.sh` starts test
  servers (see [`../benchmark/README.md`](../benchmark/README.md)). vLLM's speeds are timed on the client from
  streamed responses.
- The first start takes ~6 min, mostly compiling Triton kernels (210 s of "profiling/warmup") and `torch.compile`
  (90-110 s). With the caches in the `vllm-cache` volume, later starts take 1.5-3 min.
- The weights load in 13-15 s from the page cache.

## Commands

```sh
cd ../benchmark
# Serving speed: generation, a prompt of ctx-size - 1500 tokens, chat, depth
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 45600 --depth 6400 16000 32000 --repeat 1 --chat-n-predict 256 \
    --server-config ../vllm-radeon-vii/config-gemma-4-26b-a4b-qat.yaml --label "..."
# The same for Qwen (config-qwen3.8-27b.yaml)
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 32300 --depth 6400 16000 28000 --repeat 1 --chat-n-predict 256 \
    --server-config ../vllm-radeon-vii/config-qwen3.8-27b.yaml --label "..."
# A configuration variant on a test server (stop the production container first)
docker compose -f ../vllm-radeon-vii/docker-compose.yml stop
GPUS=amd DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_USE_V2_MODEL_RUNNER=0 \
    -e VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS=128" \
    ./vllm_test_server.sh ghcr.io/agenttix/agx-ai/vllm-gfx906:latest /tmp/config-variant.yaml \
    ../llama-cpp/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 1 --depth 6400 16000 --server-config /tmp/config-variant.yaml --label "..."
docker rm -f vllm-test && docker compose -f ../vllm-radeon-vii/docker-compose.yml start
```
