# vLLM optimizations in this repository

A catalog of the vLLM settings, patches and runtimes that were tried in the configurations of this repository,
what they did, and when they were rejected. The measurements and their full context are in the README files of the
configurations; this file collects them by topic, so that a new configuration can start from what is known.
The Claude Code skill [`optimize-vllm`](../.claude/skills/optimize-vllm/SKILL.md) applies them. For llama.cpp, see
[`../llama-cpp/OPTIMIZATIONS.md`](../llama-cpp/OPTIMIZATIONS.md).

Abbreviations: tg = generation, pp = prompt processing, TTFT = time to the first token, TP = tensor parallelism.
Configurations: [vllm](.) (the generic image and [`vllm.patch`](vllm.patch)), [radeon-vii](../vllm-radeon-vii),
[rtx-a4000](../vllm-rtx-a4000), [rtx3070](../vllm-rtx3070), [rtx3090](../vllm-rtx3090),
[flash-next](../vllm-rtx3090-flash-next), [hyperqwen](../vllm-rtx3090-hyperqwen), [frame](../vllm-frame).

## Order of priority

What has given the largest gains, roughly in order:

1. **Size the KV cache explicitly** (`kv-cache-memory-bytes`) instead of relying on the profiling of
   `gpu-memory-utilization`, and free the VRAM that vLLM allocates by default for things the configuration does not
   need (CUDA graph sizes, RoPE caches for 262144 positions, the vision tower, BF16 embeddings, extra sequences).
2. **An int8 KV cache with a fast attention kernel**: `kv-cache-dtype: int8_per_token_head` (2x the context of
   BF16) with `attention-backend: CUSTOM` (pth-flash-attn, Ampere) or `TRITON_ATTN` with the split-KV fix of
   `vllm.patch`. This made tg flat over the context instead of falling to a third at 100k tokens.
3. **Speculative decoding**: MTP with 2 drafted tokens, or DFlash2 (hyperqwen) where available.
4. **Full CUDA graphs for decoding** with only the capture sizes that are used.
5. **CPU KV offloading** for conversations that return after other prompts (agents): TTFT from 25 s to 2 s.

## Memory: `kv-cache-memory-bytes` and `gpu-memory-utilization`

- **Profiling wastes memory:** with `gpu-memory-utilization` alone, ~0.8 GiB stayed unused on the Radeon VII and
  ~1 GiB on the RTX 3070, and the 1-GPU A4000 Gemma got only 0.16 GiB of KV cache. Every single-GPU configuration
  sets `kv-cache-memory-bytes` (e.g. 6.4 GB on the RTX 3090 for Qwen3.8-27B at 155648 tokens).
- **vLLM still checks at startup** that `gpu-memory-utilization` x the total memory is free. Set it below what the
  desktop or the other servers leave (0.8 on the RTX 3090 with a desktop; hyperqwen `GPU_UTIL=0.85`).
- **Two servers on the same GPUs** (rtx-a4000 LLM + embedding server): an explicit KV size for both, a small
  `gpu-memory-utilization` for the embedding server (0.05; 0.1 failed to start), and the LLM's set below what is
  left. Stress-test after a joint `docker compose up -d`.
- **`max-model-len`**: raise it to what the startup log says the KV cache holds ("GPU KV cache size"), divided by
  ~1.05-1.1: the scheduler needs extra blocks for the recurrent states of hybrid models.
- **Probe:** a prompt of `max-model-len - 1500` tokens, concurrent long prompts, embeddings and media if they are
  served, and the peak VRAM per process. Many configurations ran out of memory on the *first request*, not at
  startup. On an OOM, lower `kv-cache-memory-bytes` and `max-model-len` together (Qwen3.8-27B int8: ~2.9k tokens
  per 0.1 GB; Gemma 4 26B-A4B ~5k).
- **Headroom:** radeon-vii ~80 MiB of 16368; rtx3090 ≤21 GiB next to the desktop; rtx3070 ~7.6 of 7.84 GiB
  (7722 MiB was too close); rtx-a4000 peaks 15.6-15.96 of 16.38 GiB. Another program opening the GPU (a video
  player) crashed hyperqwen at ~1 GiB of headroom.

## Freeing VRAM

- **`hf-overrides: {"text_config": {"max_position_embeddings": <max-model-len>}}`**: the RoPE caches are otherwise
  allocated for the trained maximum (262144): +0.1-0.33 GiB (radeon-vii Gemma 34816 -> 47104 tokens). Use whenever
  the context is below the trained maximum.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: -840 MiB after startup on radeon-vii, and needed for the
  Marlin MoE repack on the 1-GPU A4000. It conflicts with KV connectors (CPU offloading) unless `vllm.patch` is used.
- **CUDA graphs** (see below): the default 51 capture sizes took 5.89 GiB on gfx906; sizes 1-8 0.83 GiB.
- **`max-num-seqs`** as small as the client needs (1 for one user, 4 for agents or PaperQA2): each sequence costs a
  sliding-window cache (Gemma ~0.2-0.4 GiB) or a recurrent state (Qwen3.8 ~0.32 GiB).
- **`max-num-batched-tokens`** 512-2048: the activation peak, and for Gemma 4 one prefill chunk per sliding-window
  layer.
- **`mamba-ssm-cache-dtype: float16`** for the Gated DeltaNet states of Qwen3.8: +6.5 % tokens, no speed change.
- **The vision tower**: `language-model-only: true` (Qwen 0.9 GiB ≈ 25k tokens, Gemma 1.1 GiB ≈ 45k), or better, the
  CPU vision towers below.
- **Smaller embeddings and lm_head:** quantized (int4/int8, see Quantization) or in pinned RAM
  (`cpu-offload-params: [embed_tokens]` with UVA; hyperqwen `EMBED_UVA=1`, flash-next `QWEN38_EMBED_UVA=1`: -1.18 GiB
  without a measurable speed cost). The usual int4 checkpoints of Qwen3.8-27B keep 2.4 GiB of BF16 embedding and
  lm_head, ~100k tokens of context on the RTX 3090.

## KV cache dtype and attention backend

- **`kv-cache-dtype: int8_per_token_head`** (the format closest to llama.cpp's q8_0) is the default choice: ~2x the
  tokens of BF16. FlashAttention 2 (the Ampere default) supports no quantized cache, so it needs
  `attention-backend: CUSTOM` or `TRITON_ATTN`.
- **`attention-backend: CUSTOM`** (pth-flash-attn from hyperqwen, built into the generic image for sm_80/86/89 on
  x86-64): int8 KV, causal head size 256 or sliding-window/non-causal 128 (Qwen3.5/3.8); other shapes are refused at
  startup. The MTP drafter also needs `"attention_backend": "CUSTOM"` in `speculative-config`. RTX 3090 Qwen3.8-27B:
  tg at 100k 41 -> 66-69 t/s, pp at 100k 544 -> ~800 t/s. Works with CPU KV offloading.
- **Triton split-KV fix** (`vllm.patch`): the 3D split-KV kernel for speculative-decoding verification batches
  (≤8 query tokens). Qwen MTP at 32k 23.7 -> 64 t/s; Gemma at 100k 16.7 -> 84 t/s; identical answers.
- **Triton prefill tuning for sm_86** (head size 256): pp at 96k 395 -> 582 t/s. The same tuning on gfx906 was
  faster in a microbenchmark but slower in the server and lost a request: verify kernel wins in the server.
- **gfx906:** `VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS=128` (default 16) fills the 60 CUs with few KV heads:
  Gemma at 32k 30.9 -> 45 t/s, short chats -4 %. The gfx906 CUSTOM flash attention (fp16 KV only) gave a third of the
  context and was rejected. `VLLM_USE_V2_MODEL_RUNNER=0` with CPU KV offloading (memory fault otherwise).
- **FP8 KV:** FlashInfer FP8 allows only piecewise CUDA graphs with MTP (chat 51 vs 81 t/s), rejected. Triton FP8
  needs sm_89+; `vllm.patch` reads e4m3 as uint8 for the Flash-Next QSA kernel on Ampere. For hybrid models FP8
  halves the attention page, so it nearly halves the memory per token in all groups (1.1 GB: 42598 vs 24041 tokens)
  but doubles the prefix-cache granularity. Quality: KL 0.0056 vs a noise floor of 0.0020.
- **`int4_per_token_head`** (Hadamard rotation): 3.4x the context of int8 for Gemma 4 31B on the RTX 3090 with
  equal retrieval at 9k (42/44 vs 43/44 at 21k), so it is used there; on gfx906 it was 3x slower and rejected. Its
  split-KV kernel does not handle verification batches.
- `kv-cache-dtype-skip-layers: [sliding_window]` (fp16 sliding window + int8 global) needed 4x the memory per token.
- **Gemma 4 31B:** vLLM reserves a sliding window plus a prefill chunk for each of its 50 sliding-window layers and
  pads the global layers to their page size, so its KV cache is much larger than llama.cpp's (hence
  `max-num-batched-tokens: 512`).

## Quantization formats and checkpoints

- **Kernels:** W4A16 runs on Marlin on Ampere (`MARLIN` MoE backend for Gemma's intermediate size 704) and on exllama
  on gfx906 (the gfx906 Triton W4A16 kernel is 10x slower).
- **Gemma 4 QAT:** [`repack_gemma4_qat.py`](repack_gemma4_qat.py) repacks Google's QAT Q4_0 GGUF bit-exactly into
  compressed-tensors int4 (group size 32) with `dtype: float16`; third-party AWQ checkpoints are off Google's grid.
  The embedding and lm_head must be untied (`--embed-bits`, `--lm-head-bits`;
  [`quantize_gemma4_ct_embedding.py`](quantize_gemma4_ct_embedding.py) requantizes only them). An int8 embedding gave
  +55k tokens on the RTX 3090; on the Radeon VII only int4/int4 fit. Check that the vision weights are included
  (`--vision-only`).
- **Qwen3.8-27B:** GPTQ W4A16 with int4 embedding and lm_head and an int8 MTP head
  (`JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB`, revision `mtp-int8`; needs `vllm.patch`); AWQ int4 for TP2;
  AutoRound with an int8 embedding in RAM in hyperqwen.
- **Flash-Next:** [`../vllm-rtx3070/quantize_flash_next_dense.py`](../vllm-rtx3070/quantize_flash_next_dense.py)
  quantizes the dense projections to int8 (round-to-nearest, 0.8 % error vs 14 % for int4; 8.0 -> 4.83 GiB) and the
  lm_head to int4 (KL 0.0042).
- **Embedding models:** `quantization: fp8` (weight-only) halves Qwen3-Embedding-4B without a retrieval loss.
- W4A8 (hyperqwen `INT8_ACT=int8`): pp +23-63 % but chat 136 -> 40 t/s with DFlash2; rejected for generation-bound use.

## CUDA graphs and compilation

- `compilation-config: {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [...]}` with exactly the
  batch sizes that occur: `max-num-seqs` x (1 + drafted tokens). [1] for one sequence without speculative decoding,
  [1, 2, 3] for MTP 2, [1, 2, 3, 4, 6, 8, 9, 12] for 4 sequences with MTP 2. Eager mode is ~2x slower (7.4 vs
  15.3 t/s).
- Keep the compile caches on a volume (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, the `vllm-cache` volume):
  gfx906 first start ~6 min, later 1.5-3 min. `TORCHINDUCTOR_COMPILE_THREADS=1` when the RAM is nearly all pinned.

## Scheduler: `max-num-seqs`, `max-num-batched-tokens`

- `max-num-batched-tokens` (the prefill chunk) 2048 on Ampere, 1024 on the 1-GPU A4000, 512 on gfx906 and for Gemma
  4 31B. Larger chunks were not faster (4096 on the RTX 3090 Qwen: same pp, 139k vs 147k context; 8192 at TP2: same
  pp, -0.5 GiB of KV) or cost half the context (RTX 3070 Gemma: +8 % pp).
- Gemma 4 encodes each image or video in one chunk (bidirectional attention), so the chunk limits the media size
  (512: a video of 6 frames).
- `max-num-seqs`: 1 for a single user on a tight GPU, 4 for agents and PaperQA2, 8 at TP2. DFlash2 reserves pages for
  8 verified tokens per sequence; it is as fast as MTP for 1-2 agents, MTP faster at 4.

## Prefix caching

- `enable-prefix-caching: true` everywhere. Hybrid (Gated DeltaNet) models checkpoint their states only at block
  boundaries (`mamba-cache-mode: align`; blocks of 784-3200 tokens), so part of a returning prompt is recomputed.
- Test returns with [`../benchmark/vllm_prefix_return.py`](../benchmark/vllm_prefix_return.py) and concurrent agents
  with [`../benchmark/vllm_concurrent_agents.py`](../benchmark/vllm_concurrent_agents.py).

## CPU KV offloading

- `kv-offloading-size: <GiB>` with `kv-offloading-backend: native` (pinned RAM): a 25k-token conversation returned
  after eviction in 1.7-4 s instead of 26 s (RTX 3090 Qwen), 0.2 s instead of 27 s (radeon-vii Gemma), without a
  GPU memory or speed cost on single-GPU setups. Size it by the bytes per token (Qwen3.8-27B ~36 KB; hyperqwen
  ~100 KB with its recurrent snapshots, so 32-40 GiB).
- **Gotchas:** needs `vllm.patch` for `expandable_segments` (`enable-cumem-allocator` needs more memory); Gemma 4 with
  its MTP drafter needed a scheduler fix in `vllm.patch`; its GPU buffers caused an OOM on the first request on the
  1-GPU A4000; the second TP rank fails `cudaHostRegister`; pointless when the GPU cache is as large as the CPU cache;
  incompatible with the Flash-Next runtime. Use `docker compose stop` instead of `docker rm -f` (unpinning takes a
  while), and give hyperqwen its own `shm_size` instead of `ipc: host`.

## Speculative decoding

- **MTP:** `speculative-config: {"method": "mtp", "num_speculative_tokens": 2}` (Gemma 4 needs its assistant drafter
  as `"model"`). A4000 TP2 Qwen chat 37.5 -> 55.4 t/s; RTX 3070 Gemma +19 %. 3 tokens were faster only for edits.
  Rejected where the drafter costs too much context (Gemma 4 31B on the RTX 3090: 8192 tokens; Qwen on the Radeon
  VII: 9.2k).
- **DFlash2** (hyperqwen `SPEC=dflash2`): a 7-token block drafter plus lookup drafting from the context. RTX 3090
  chat 136 vs 83 t/s with MTP, code edits 290 vs 112 t/s, 80 t/s at 100k; PaperQA2 on the A4000 1.5x stock vLLM.
- **N-gram drafting alone and MTP are bad for MoE models with experts on the CPU or behind PCIe:** verifying more
  tokens reads more experts (flash-next on the RTX 3070: edits 1.7x, everything else 2-4x slower).
- Measure with the `chat`, `depth` and `edit` tests of `llama_cpp_bench_http.py`; the raw generation test with
  repeated text gives ~98 % acceptance.

## Tensor parallelism (2x RTX A4000)

- Without working PCIe P2P (no NVLink, AMD IOMMU): `NCCL_P2P_DISABLE=1` (NCCL hung) and
  `disable-custom-all-reduce: true` (the custom all-reduce spun forever). `iommu=pt` is the probable fix.
  `NCCL_PROTO=LL` cost 16 % pp.
- Split an embedding server on the same GPUs with TP too, so that the LLM gets an equal KV cache on both GPUs.
- TP2 vLLM vs llama.cpp: pp 1.4x, 4 concurrent 8k requests 1.6x; long-context tg better only with int8 + CUSTOM.

## MoE and weights on the CPU

- **Stock UVA:** `offload-backend: uva`, `cpu-offload-gb: <upper bound>`, `cpu-offload-params: [experts, embed_tokens]`:
  the GPU reads only the selected experts over PCIe (Gemma 4 26B-A4B on 8 GB: 26.6 t/s, PCIe-bound; pp 2-2.5x slower
  than llama.cpp). `offload-backend: prefetch` copies whole layers, which is wrong for MoE decoding.
- `vllm.patch` fixes UVA: exactly sized pinned memory (PyTorch rounds up to the next power of two: 47.6 -> 64 GiB),
  the Marlin MoE repack without full GPU copies, and quantized embeddings.
- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (Flash-Next): the 512 MiB default OOMed on the first request.
- **Flash-Next runtime with CPU experts** (flash-next): hot experts on the GPU (`QWEN38_HOT_ONLY`,
  `QWEN38_GPU_SHARE_FRAC=0.3`), cold experts computed on the CPU with AVX2 int8 activations
  (`QWEN38_CPU_EXPERTS_INT8=1` from `cpu-moe.patch`: +27 % chat, KL equal to the noise), 1 thread per CCX
  (`QWEN38_CPU_EXPERTS_CPUS`; 8 beat 16-48), `QWEN38_STREAM_GROUP=32` (-300 MiB vs 64; 16 saved another 140 MiB on the
  RTX 3070 for +10k tokens of KV at -3 % prompt processing; prefill chunks of 4096 did not fit in 8 GB even then;
  `expandable_segments` had no effect). Decoding is bound by the CPU memory bandwidth; 1.6x stock UVA. The offline
  `cpu_moe_bench` reproduces the server within 7 %.
- **Embedding model in RAM** (rtx-a4000): `offload-backend: prefetch`, `offload-group-size: 1`,
  `offload-num-in-group: 1`, `offload-prefetch-step: 2` (step 1/2/3: 5895/7153/7560 t/s).

## Multimodal

- **CPU vision towers** (`cpu_vision*.py` and the vision patches; `QWEN38_VISION_DEVICE=cpu`,
  `GEMMA4_VISION_DEVICE=cpu`, `*_VISION_THREADS`): no VRAM for the tower, unchanged text speed, ~1000 tokens and
  7-15 s per image; an encode blocks the other requests. On gfx906, route the CPU math through oneDNN
  ([`../vllm-radeon-vii/cpu_onednn.py`](../vllm-radeon-vii/cpu_onednn.py), 100x faster).
- Limits: `limit-mm-per-prompt: {"image": 64, "video": 1}`,
  `mm-processor-kwargs: {"max_pixels": 1048576, "videos_kwargs": {"max_pixels": 12582912}}`,
  `media-io-kwargs: {"video": {"max_frames": 64}}`, `VLLM_VIDEO_LOADER_BACKEND=qwen3_vl`, `skip-mm-profiling: true`.

## Embedding servers

- `runner: pooling`, `max-model-len: 8192`, explicit `kv-cache-memory-bytes`, `max-num-seqs` 8-16. Qwen3-Embedding-4B
  retrieved as well as 8B; 0.6B is 6x faster at R@10 0.892 vs 0.922. vLLM: ~7000 t/s vs ~3200 for one llama.cpp
  slot.
- CPU image (`vllm/vllm-openai-cpu`): `dtype: float32` on CPUs without BF16 (Zen 2: BF16 5x, FP16 6x slower),
  `VLLM_CPU_KVCACHE_SPACE` (1 GiB is enough for bidirectional models, which keep no KV cache); do not give it GPUs.
  Threads: `VLLM_CPU_OMP_THREADS_BIND=nobind` with `OMP_NUM_THREADS` on a machine shared with other users
  (rtx-a4000); on a dedicated machine the default `auto` (one thread per physical core) was 5-12 % faster than
  `nobind` with 32 threads (zen2-embeddinggemma2). Keep `torch.compile` (`enforce-eager` -25 %).
- **Causal pooling models (Qwen3-Embedding) get the prefix cache by default:** repeated benchmark texts are then served
  from it (4B: 175 -> 1030 t/s), so `llama_cpp_bench_http.py` sends a random `cache_salt` with vLLM embedding requests.
  The cache costs nothing on the CPU. The KV cache (`VLLM_CPU_KVCACHE_SPACE`, FP32: 224 KiB per token for 0.6B,
  288 KiB for 4B) must hold `max-num-batched-tokens` (16384 was 6 % faster than 8192). Matryoshka:
  `hf-overrides: {"is_matryoshka": true}` (any `dimensions`).
- **EmbeddingGemma 2 on the CPU needs vLLM > 0.32.0:** v0.32.0's CPU attention ran its `ENCODER_ONLY` layers as
  causal (cosine 0.77-0.91 with sentence-transformers), fixed by vllm-project/vllm#63738 (nightly from 2026-10-27).
  Check new embedding models against sentence-transformers; the CPU attention only honours `ENCODER_ONLY` through
  the fix or the `EncoderOnlyAttention` wrapper. Matryoshka `dimensions` need
  `hf-overrides: {"matryoshka_dimensions": [...]}`. On the Threadripper 3970X: 4300-4700 t/s, 2x llama.cpp.

## Images and patches

- Prefer the generic image [`./Dockerfile`](Dockerfile) (stock vLLM + [`vllm.patch`](vllm.patch) + pth-flash-attn),
  built by `.github/workflows/deploy-docker-vllm.yml`; keep it compatible with all configurations. Separate images
  only for other base images (gfx906, the Flash-Next runtime, hyperqwen), each with its own workflow and the base
  pinned by digest.
- gfx906 (`vllm-gfx906.patch`): exllama `gptq_gemm` allocated a K x N scratch buffer on every call (2.4 GiB for the
  Qwen lm_head); the patch chunks only matrices with scratch above 512 MiB (chunking everything above 64 MiB cost
  40 % tg).

## Power and the host

- RTX 3070: a 1600 MHz clock lock or 170 W limit is needed, as with llama.cpp. LXC does not permit
  `ulimits: memlock: -1`. Faster RAM and the `performance` governor are untried for CPU-expert runtimes.

## vLLM or llama.cpp?

- **vLLM won:** Qwen3.8-27B on the RTX 3090 (+52 % context, 65-155 % faster tg; hyperqwen faster still), throughput
  and concurrency on 2x A4000 (PaperQA2 2.2-2.5x), Qwen3.8-Flash-Next (pp 2.4-2.8x, flat tg, video), returning
  conversations (CPU KV offloading).
- **llama.cpp won:** Gemma 4 26B-A4B on the Radeon VII (1.9x context, 1.5-3x faster) and the RTX 3070 (CPU expert
  kernels), Gemma 4 31B on the RTX 3090 (1.5x context, 2x tg), Gemma 4 26B-A4B on the RTX 3090 beyond ~6k tokens,
  long prompts with Qwen on the Radeon VII, CPU-only machines.
- Recurring reasons vLLM loses: ~1-1.35 GiB of runtime overhead on 16 GB cards, Triton attention scaling worse with
  the context on GPUs without tensor cores, page padding of sliding-window models, and the MTP drafter not fitting.
  llama.cpp also starts in ~15 s and unloads when idle; vLLM holds its VRAM and starts in 1.5-7 min.
- **Embeddings on the CPU (Threadripper 3970X):** vLLM was 2x faster than llama.cpp for EmbeddingGemma 2 and 1.5x for
  Qwen3-Embedding-0.6B, but 1.7x slower for Qwen3-Embedding-4B, where llama.cpp's Q6_K kernels beat vLLM's FP32 (no
  BF16 on Zen 2). The compute-bound small models favor vLLM's batching, larger ones llama.cpp's quantized kernels.

## Methodology

- Test servers: [`../benchmark/vllm_test_server.sh`](../benchmark/vllm_test_server.sh) `IMAGE CONFIG ENV [PORT] -- args`
  with the production container stopped (`GPUS`, `DOCKER_ARGS`, `CACHE_VOLUME` to reuse the compile caches); it
  prints the KV cache size, the maximum concurrency and the VRAM.
- Speed: `llama_cpp_bench_http.py --tests generation prompt chat depth edit concurrency --server-config <yaml>`
  (vLLM is detected; timing from the stream, `cache_salt` against prefix-cache hits, draft acceptance from
  `/metrics`).
- Quality: `llama_cpp_quality_http.py --tests retrieval --logprobs N --reference <label>` with speculative decoding
  off, against a noise floor from a lossless change (e.g. `max-num-batched-tokens` 2048 -> 1024: KL 0.0020).
- A single run varies ±5-10 % (chat ±10 %): repeat, and call smaller differences noise.
- Comment every YAML key with the measured reason, record rejected variants and a "not tried" list, keep the summary
  table at the start of the README (with the llama.cpp comparison), and store the results in `../benchmark/results/`.
