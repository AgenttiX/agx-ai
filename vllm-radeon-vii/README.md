# vLLM on the Radeon VII

vLLM for the Radeon VII (gfx906, 16 GB HBM2) on agx-z2e, as an alternative to
[`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii) on the same GPU and port (9932). Only one of the two fits on the
GPU at a time. The image is [`aiinfos/vllm-gfx906-mobydick`](https://github.com/ai-infos/vllm-gfx906-mobydick)
(vLLM v0.30.0 with ROCm 7.14 and custom gfx906 kernels) with a few fixes ([`Dockerfile`](Dockerfile),
[`vllm-gfx906.patch`](vllm-gfx906.patch)), built by
[`../.github/workflows/deploy-docker-vllm-radeon-vii.yml`](../.github/workflows/deploy-docker-vllm-radeon-vii.yml).
The model is selected in [`docker-compose.yml`](docker-compose.yml). There are two configurations, both entirely on
the GPU, text only, with an int8 KV cache, prefix caching and 8 GiB of CPU KV offloading:

| Configuration | Model | Context | Chat generation | Prompt processing |
|---|---|---|---|---|
| [`config-gemma-4-26b-a4b-qat.yaml`](config-gemma-4-26b-a4b-qat.yaml) (default) | Gemma 4 26B-A4B QAT, int4 bit-exact with Google's Q4_0 (built locally, see below) | 34816 | ~110 t/s, 61/43/31 t/s at 6.4k/16k/32k tokens of context | ~800 t/s at 6.4k, ~340 t/s at 32k |
| [`config-qwen3.8-27b.yaml`](config-qwen3.8-27b.yaml) | Qwen3.8-27B GPTQ int4 with int4 embedding and lm_head | 30720 | ~44 t/s, 35/26/19 t/s at 6.4k/16k/28k tokens of context | ~240 t/s at 6.4k, ~135 t/s at 28k |

**Compared with the llama.cpp presets on this GPU:**

- **Gemma: llama.cpp is better.** It fits 2.5 times the context (88064), generates 1.7-2.5 times as fast at 6-36k
  tokens of context (101/91/78 t/s) and processes prompts 1.6-3 times as fast. Short chats are equally fast.
- **Qwen: vLLM is faster for short contexts, llama.cpp for long ones.** vLLM generates 45 % faster in chat (43.5 vs
  29.9 t/s, without speculative decoding vs llama.cpp's MTP) and 23 % faster at 6.4k tokens of context, about as fast
  at 16k, and 20 % slower at 28-30k. llama.cpp fits 1.5 times the context (47104) and processes long prompts up to
  1.7 times as fast.

The reasons:

1. **Memory.** Qwen's weights take as much as llama.cpp's GGUF (13.25 vs 13.27 GiB), Gemma's 0.8 GiB more (14.04 GiB:
   the embedding and the lm_head are untied, and the MoE kernel needs 0.36 GiB of zero points). vLLM's runtime
   (HIP kernels, activations and scratch buffers, CUDA graphs, and the recurrent state or the sliding-window cache for
   a whole prefill chunk) takes another 1.0-1.35 GiB, so only 0.9-1.3 GiB are left for the KV cache. The int8 cache
   needs ~20 KiB per token for Gemma (like llama.cpp's q8_0) and ~33 KiB for Qwen (llama.cpp: 44 KiB).
2. **Attention.** The fast custom gfx906 flash attention of the image supports only head sizes up to 256 and only an
   fp16 KV cache. Gemma 4 has global layers with head size 512, so vLLM uses Triton attention for all of its layers,
   and an fp16 KV cache for Qwen fits only ~11k tokens. Triton attention on gfx906 slows down much more with the
   context length than llama.cpp's flash attention.
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
| GPU | AMD Radeon VII, used only for this server (the RTX 3090 in the same machine runs [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e)) |
| GPU architecture | GCN 5.1 (Vega 20, gfx906), 60 CUs, PCIe 3.0 x16 |
| VRAM | 16 GB HBM2 (16368 MiB) |

## Setup

```sh
# Gemma: build the checkpoint (downloads Google's 14.4 GB Q4_0 GGUF and ~2.5 GB of the unquantized QAT checkpoint;
# ~10 min on the CPU and ~30 GB of RAM)
docker run --rm -v "$PWD":/src:ro -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    -v ~/.cache/vllm-models:/out --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm-gfx906:latest \
    /src/repack_gemma4_qat.py --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct
# Qwen: download the checkpoint (14.8 GB) into the Hugging Face cache
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub --entrypoint hf \
    ghcr.io/agenttix/agx-ai/vllm-gfx906:latest download JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8
docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml stop
docker compose up -d
```

The first start compiles kernels for ~6 min; later starts take ~1.5-3 min (the compile caches are in the
`vllm-cache` volume). The model ids, the port and the API key (`LLAMA_API_KEY` of
[`../llama-cpp-radeon-vii/llama-cpp.env`](../llama-cpp-radeon-vii)) are the same as for llama.cpp, so clients such as
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

[`repack_gemma4_qat.py`](repack_gemma4_qat.py) therefore builds the checkpoint from Google's two QAT repositories:
every linear weight of the language model (attention, dense MLP, experts) from the Q4_0 GGUF, and the norms,
routers, embedding and vision tower from the unquantized QAT checkpoint (downloaded with HTTP range requests, ~2.5 GB
of 52 GB). The embedding and the lm_head are untied, because vLLM cannot use a quantized embedding as the lm_head,
and both are quantized to int4 with group size 32 (round-to-nearest; the GGUF has them in Q6_K). The vision tower stays
fp16 and is not loaded (`language-model-only`).

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

## Memory (2026-10-09)

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

## Context size (2026-10-09)

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

## Speed (2026-10-09)

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
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 33300 --depth 6400 16000 32000 --repeat 1 --chat-n-predict 256 \
    --server-config ../vllm-radeon-vii/config-gemma-4-26b-a4b-qat.yaml --label "..."
# The same for Qwen (config-qwen3.8-27b.yaml)
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 29200 --depth 6400 16000 28000 --repeat 1 --chat-n-predict 256 \
    --server-config ../vllm-radeon-vii/config-qwen3.8-27b.yaml --label "..."
# A configuration variant on a test server (stop the production container first)
docker compose -f ../vllm-radeon-vii/docker-compose.yml stop
DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_USE_V2_MODEL_RUNNER=0" \
    ./vllm_test_server.sh ghcr.io/agenttix/agx-ai/vllm-gfx906:latest /tmp/config-variant.yaml \
    ../llama-cpp-radeon-vii/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 1 --depth 6400 16000 --server-config /tmp/config-variant.yaml --label "..."
docker rm -f vllm-test && docker compose -f ../vllm-radeon-vii/docker-compose.yml start
```
