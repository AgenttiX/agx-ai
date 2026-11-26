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
   BF16) with `attention-backend: CUSTOM` (pth-flash-attn, Ampere) or `TRITON_ATTN` with split-KV decoding for
   speculative decoding (upstream since vLLM v0.33.0; `vllm.patch` before). This made tg flat over the context instead of falling to a third at 100k tokens.
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
- **`int4_per_token_head` on pth-flash-attn** (hyperqwen's `pthfa-int4/`, 2026-11-23; vLLM itself reads it only with
  Triton, 4.5-5.8x slower on sm_86): 1.95x the tokens of int8 in the same memory (Qwen3.8-27B with DFlash2: 262128
  instead of 139264 tokens of context in 5.95 GB) at the same speed (decode attention 2.2x faster, but the 8-token
  verify is compute-bound). KL divergence 12x the noise floor (retrieval 121 vs 123/132): an opt-in, not the default.
- **Triton split-KV fix** (`vllm.patch` until v0.31.0, upstream since v0.33.0): the 3D split-KV kernel for speculative-decoding verification batches
  (≤8 query tokens). Qwen MTP at 32k 23.7 -> 64 t/s; Gemma at 100k 16.7 -> 84 t/s; identical answers.
- **Triton prefill tuning for sm_86** (head size 256): pp at 96k 395 -> 582 t/s. The same tuning on gfx906 was
  faster in a microbenchmark but slower in the server and lost a request: verify kernel wins in the server.
- **gfx906:** `VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS=128` (default 16) fills the 60 CUs with few KV heads:
  Gemma at 32k 30.9 -> 45 t/s, short chats -4 %. The gfx906 CUSTOM flash attention (fp16 KV only) gave a third of the
  context and was rejected. `VLLM_USE_V2_MODEL_RUNNER=0` with CPU KV offloading (memory fault otherwise).
- **FP8 KV:** FlashInfer FP8 allows only piecewise CUDA graphs with MTP (chat 51 vs 81 t/s), rejected. Triton FP8
  needs sm_89+; the Flash-Next QSA kernel reads e4m3 as uint8 on Ampere (`vllm.patch` until v0.31.0, upstream since
  v0.33.0). For hybrid models FP8
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
- **No vLLM checkpoint of a finetune:** quantize the BF16 checkpoint with llm-compressor's GPTQ into the format of the
  checkpoint the configuration expects (`../vllm-rtx3090-hyperqwen/quantize_w4a16.py`: compressed-tensors W4A16,
  int4 symmetric group 128, static act order; Terminal-RSI-27B in 40 min on the RTX 3090 with the model in RAM,
  164/176 retrieval answers vs 166 for the base model's quant). llm-compressor 0.15 keeps the MTP module (BF16).
- An int8 lm_head (HyperQwen's `prepare`) costs 526 MiB at the peak more than the GPTQ int4 one of its "fast"
  variant, which needs the drafter pipeline (captured hidden states) for each model.
- **MoE fine-tunes without a 4-bit checkpoint** (Phai-A1-35B-1M, a Qwen3.6-35B-A3B fine-tune, rtx3090):
  [`quantize_qwen35_moe_w4a16.py`](quantize_qwen35_moe_w4a16.py) uses llm-compressor's `model_free_ptq`
  (round-to-nearest, 1.5 min on the GPU, little RAM): int4 experts, int8 dense projections and lm_head, int4
  embedding. Unlike `oneshot`, it also quantizes the MTP layer (whose 256 experts would take 1.6 GB in BF16). Retrieval
  152/176 vs 154 for llama.cpp's imatrix IQ4_XS; vLLM was 17-56 % faster than llama.cpp but fit 98304 instead of 151552
  tokens.
- **Fine-tunes published without their MTP head:** [`graft_mtp.py`](graft_mtp.py) copies the base model's MTP tensors
  into a checkpoint directory of symlinks (Phai-A1-35B-1M: 43-65 % acceptance with 3 drafts, 99 % in code edits).
- **An int8 lm_head converted at load time** (flash-next, 2026-11-24, `QWEN38_INT8_SKINNY=1`): when the drafter
  needs the target's lm_head unquantized (Flash-Next's reduced MTP draft vocabulary is a view of it), keep its
  method class and convert only the weight after loading, with an int8 kernel for both uses: 0.64 instead of 1.27 GB
  read per use, -1.3 ms per decode step with MTP 3 (the head and 3 drafts). One scale per row: KL divergence 1.5
  times the noise floor; one per row and 128 inputs: at the noise floor. The smaller BF16 projections (3-7 MB) were
  not faster in int8 (latency-bound at 4 tokens).

## CUDA graphs and compilation

- `compilation-config: {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [...]}` with exactly the
  batch sizes that occur: `max-num-seqs` x (1 + drafted tokens). [1] for one sequence without speculative decoding,
  [1, 2, 3] for MTP 2, [1, 2, 3, 4, 6, 8, 9, 12] for 4 sequences with MTP 2. Eager mode is ~2x slower (7.4 vs
  15.3 t/s).
- The Flash-Next runtime does not compile (`mode: 3` fails at startup with "auto_functionalized was not removed",
  from its custom ops that write into their inputs), so it runs with `mode: 0`; its GPU part per layer is mostly
  weight reads near the bandwidth (W8A16 Marlin at 590-770 GB/s, the skinny Triton GEMVs) anyway. The elementwise
  kernels that compiling could fuse take ~1.8 ms of a ~55 ms decode step (2026-11-24, see
  [its README](../vllm-rtx3090-flash-next/README.md#torchcompile)).
- Measure kernels in CUDA graphs (`torch.cuda.CUDAGraph` with 10-20 calls, replayed), not with launches: small
  decode GEMVs are 3-15 µs, of the order of a launch. A torch profile of a served model (`--profiler-config`)
  shows the kernels inside the graphs with their grids, which identifies them.
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
- Test returns with [`../agx_ai/benchmark/vllm/prefix_return.py`](../agx_ai/benchmark/vllm/prefix_return.py) and concurrent agents
  with [`../agx_ai/benchmark/vllm/concurrent_agents.py`](../agx_ai/benchmark/vllm/concurrent_agents.py).

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
- **DFlash2 transfers to finetunes:** the drafter of Qwen3.8-27B accepted Terminal-RSI-27B's tokens better than the
  base model's (35-45 % vs 26-33 %), and beat the finetune's own MTP head (int8, 3 drafts) by 36 % in chat and 2x in
  edits (RTX 3090).
- **N-gram drafting alone and MTP are bad for MoE models with experts on the CPU or behind PCIe:** verifying more
  tokens reads more experts (flash-next on the RTX 3070: edits 1.7x, everything else 2-4x slower).
- **External Qwen3.5 draft models do not work in v0.33.0** (`method: draft_model`, e.g.
  nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft): the draft config is rewritten into an MTP head, and once that is skipped, the
  v2 runner's draft speculator does not keep the draft's Gated DeltaNet state (acceptance 1-12 %); the v1 runner
  rejects M-RoPE drafts. See [`../vllm-rtx3090`](../vllm-rtx3090/README.md#external-draft-model-qwen35-4b-qwen38-27b-draft-2026-11-23).
- **The rejection sampler sorts the vocabulary for top-k/top-p** when fewer than 8 rows are verified (vLLM 0.31-0.33:
  `apply_top_k_top_p` uses its Triton kernel only from 8 rows): with MTP 3 (4 rows of 248320 logits, top_k 20,
  top_p 0.95) 919 µs per step on an RTX 3090, with the Triton kernel 248 µs and the same tokens kept. flash-next
  lowers the threshold to 2 rows (`flash-next.patch`): -0.6 ms per decode step. Check it in a profile of the step's
  tail (`--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`, `POST /start_profile`).
- **The number of drafts with experts on the CPU** (flash-next, 2026-11-26, estimated from a decode trace): each
  further token per step adds its own experts (~2.9 more CPU experts per layer in code edits, ~4 ms per step with
  the draft), so it pays off only where the drafts are nearly always accepted: in edits (0.996 / 0.99 / 0.96 at
  positions 1-3) 5 drafts would give an estimated +21 %, 7 +34 %; in chats (positions 1-3: ~0.5 / 0.3 / 0.2) 3 stays
  the best (2 or 4: −0.5 / −3 %). vLLM has variable-K scheduling (`num_spec_tokens_to_schedule`), chosen by batch
  size; an adaptive depth needs a choice by the recent acceptance. Read the per-position acceptance from vLLM's
  `SpecDecoding metrics` log lines.
- Measure with the `chat`, `depth` and `edit` tests of `bench_http.py`; the raw generation test with
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
- **A second GPU for the cold experts** (flash-next, 2026-11-23): a worker in a ROCm container keeps 135 experts per
  layer in the Radeon VII's 16 GB and computes them on request of the CPU pool over `/dev/shm`
  (`rvii-moe.patch`, `rvii_experts/`, `QWEN38_RVII=1`; ~70 µs per layer, run in parallel with the CPU's ~1 ms):
  +32 % chat, +64-69 % with context, +14 % edits. It only paid off with `QWEN38_GPU_SHARE_FRAC=0` (the GPU share took
  the Radeon VII's experts first) and with the experts ranked by their **measured** decode use
  (`QWEN38_EXPERT_USAGE`; the runtime's static ranking put only 11.8 % of the routed experts on the GPU's 24 hot
  ones, the measured one 24.8 %). Two GPUs of different vendors cooperate through a host-memory protocol, not NCCL.
  - **The second GPU in prefill too** (2026-11-25, `QWEN38_RVII_PREFILL=112`, `rvii_prefill.py`): prefill streams
    every cold expert to the RTX 3090 in each chunk, and the PCIe copy is the bottleneck; the Radeon VII computes 112
    of its experts per layer instead, so they are not streamed (the GPU sends x and the routing to host memory and
    waits for the partial sum on the GPU, without host round trips). +12.5 % prompt processing at 32k. The other GPU
    must not become the critical path: balance the number of experts (96 and 135 were slower than 112), send the
    partial sums in BF16 (as precise as the main GPU's own per-expert BF16 outputs; FP32 took PCIe bandwidth from the
    stream), and write the kernels for the hardware: rocBLAS reached only ~7 TFLOPS on gfx906 at expert sizes, an
    own kernel (int4 weights converted to FP16 in registers, `v_dot2_f32_f16`, activations as scalar loads, 8 tokens
    per wave for occupancy) 10-13.
  - **Re-rank the placement automatically** (2026-11-25, `RVII_AUTO_RERANK=1`, `rvii_experts/rerank.py`): the
    runtime counts the decode requests per expert all the time (`QWEN38_EXPERT_USAGE` in `/dev/shm`, negligible cost);
    the worker's supervisor decays them (72-hour half-life), and when another set would serve 2 % more of the
    requests, repacks the second GPU's experts in the background and restarts its worker while the server is idle
    (2-20 s of CPU-only cold experts). The main GPU's hot set follows at its next start (`QWEN38_RANKINGS_AUTO`).
    **Benchmarks train it too:** after a day of mostly benchmark traffic, the code-edit test ran at 104.6 instead of
    81.5 t/s, as the CPU got 14.2 instead of 21.9 experts per layer in it, while chats changed by +3 %. Compare
    configurations with the same ranking, and expect real use to move it elsewhere.
- **CPU expert kernel on Zen 2** (flash-next, 2026-11-24, `cpu-moe-opt.patch`): find the bottleneck with a decode
  trace first (`QWEN38_DECODE_TRACE`, `cpu_moe_bench/analyze_trace.py` and the replay mode of `cpu_moe_bench`): the
  CPU part was ~60 % of a step and the RAM was idle while the GPU computed (~0.2 ms between the layers, ~40 %).
  - **Loads, not prefetch instructions, to fill the L3 cache:** on Zen 2, lines brought in by
    `prefetcht0/t2/nta` are not kept in the L3 when the L2 evicts them (a 2.4 MB read afterwards: ~30 GB/s, as
    from RAM; after plain loads: ~114 GB/s). With loads, the idle pool threads read the experts that the next layer
    used in the previous step (`QWEN38_CPU_EXPERTS_PREFETCH=1`, ~45 % of them are used again) and compute them first:
    -9 % CPU time per step. More than 1 per thread did not fit in the gap.
  - **int8 GEMV with AVX2:** sum the `vpmaddubsw` products of 4 rows in int16 (exact for 4-bit weights and int8
    activations), and keep the accumulators of 4 tokens in registers (8 columns at a time, with an empty `asm`
    barrier so that GCC does not hoist all products and spill them): 4-token experts 7.0 -> 9.0 GB/s per core,
    -5.5 % CPU time; bit-exact. `-mtune=znver2` (portable, unlike `-march=znver2`, which gave the same) another
    ~2 %; clang 21 and GCC 15 were not faster than the image's GCC 13.
  - **Loader threads:** a second core per CCX (`QWEN38_CPU_EXPERTS_LOADER_CPUS`) reads the expert that its pool
    thread is computing with plain loads, ahead of it, into the shared L3: -5 % CPU time, +2.5-6 % t/s, as the
    kernel's loop alone reaches ~61 of the 71 GB/s that plain loads get. On the pool cores' SMT siblings only -1 %.
  - **Pre-gating for the prefetch** (`QWEN38_CPU_EXPERTS_PREGATE=1`): the GPU, which waits for the CPU anyway,
    applies the next layer's router to this layer's MoE input and sends the predicted experts; prefetching those
    instead of the previous step's: 61 instead of 42 % of the prefetched experts used, -3.5 % CPU time, +2.4 % t/s
    in code edits. Measure the prediction offline first (`QWEN38_DECODE_XDUMP`, `cpu_moe_bench/pregate_eval.py`:
    70 vs 50 % with 7 experts per layer).
  - Not better: column slices of an expert for load balance (strided rows halve the RAM bandwidth with 8 threads,
    as the hardware prefetchers read the neighbouring slices' lines), a row-streaming loop order, longer prefetch
    distances, 12-16 threads. Also (2026-11-26) splitting a layer's largest expert between two threads in
    contiguous halves (gate/up by their inputs, down by its outputs): ±0.3 % to +5 % CPU time in the trace replay,
    as the slowest thread waits for the RAM like the others, and a split half loses the expert's prefetch into one
    CCX's L3. The kernel reaches 61-63 GB/s of the 71 GB/s that plain loads get with the same
    access pattern.
- **Other programs on the CPU pool's cores** (flash-next, 2026-11-24): 64 busy threads of a desktop program cut
  decode to 40 %, as each layer waits for the slowest of the pool's 8 threads. Nice values do not help across
  cgroups (Docker's `system.slice` and the desktop's `user.slice` have the same CPU weight); the real-time policy
  SCHED_FIFO for the pool's threads does (`QWEN38_CPU_EXPERTS_FIFO=10`, `cap_add: SYS_NICE`): 88-94 % of the
  unloaded speed with the same load, and the cores are free between requests, as the threads sleep. Keep every other
  thread of the serving processes off the real-time threads' CPUs (`taskset`), and make the runtime's single-slot
  queues wait instead of failing (the PLE offload's `put_nowait` killed the engine under load).
- **Embedding model in RAM** (rtx-a4000): `offload-backend: prefetch`, `offload-group-size: 1`,
  `offload-num-in-group: 1`, `offload-prefetch-step: 2` (step 1/2/3: 5895/7153/7560 t/s).

## Multimodal

- **CPU vision towers** (`cpu_vision*.py` and the vision patches; `QWEN38_VISION_DEVICE=cpu`,
  `GEMMA4_VISION_DEVICE=cpu`, `*_VISION_THREADS`): no VRAM for the tower, unchanged text speed, ~1000 tokens and
  7-15 s per image; an encode blocks the other requests. On gfx906, route the CPU math through oneDNN
  ([`../vllm-radeon-vii/cpu_onednn.py`](../vllm-radeon-vii/cpu_onednn.py), 100x faster).
- **Vision tower on a second GPU** (hyperqwen, 2026-11-23): `QWEN38_VISION_DEVICE=rvii` sends the normalized pixels
  to a service on the Radeon VII (FP16, `rvii_vision/server.py`, Unix socket), with the CPU as fallback: 1.1 instead
  of 5.9 s for a 1-megapixel image, no VRAM of the main GPU. On gfx906, compute Qwen3-VL's patch embedding (a
  `Conv3d` with kernel = stride) as a GEMM: MIOpen's naive `Conv3d` kernel took 2.0 of 3.1 s.
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
  from it (4B: 175 -> 1030 t/s), so `bench_http.py` sends a random `cache_salt` with vLLM embedding requests.
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

- Temperatures: `bench_http` records the min, mean and max of the GPUs' (with their memory) and the CPU's sensors
  (`agx_ai/benchmark/temps.py`). The RTX 3090's GDDR6X reached 98 °C in Flash-Next decode benchmarks and 108 °C
  with a 260k-token prefill (it throttles at ~110 °C; the core ~80 °C, the Radeon VII at most 46 °C); its power profiles and those of the Radeon VII are in
  [`../vllm-rtx3090-flash-next/README.md`](../vllm-rtx3090-flash-next/README.md#power-profiles).
- Host settings for CPU-expert runtimes (need root, not measured yet): L3 cache allocation (resctrl CAT) that keeps
  most of each CCX's L3 for the expert threads, reserved 1 GiB huge pages for the expert arena
  (`QWEN38_EXPERT_STORE_HUGETLB=1G` in flash-next), and the `performance` energy-performance preference; commands in
  [`../vllm-rtx3090-flash-next/HOST_TUNING.md`](../vllm-rtx3090-flash-next/HOST_TUNING.md).
- RTX 3070: a 1600 MHz clock lock or 170 W limit is needed, as with llama.cpp. LXC does not permit
  `ulimits: memlock: -1`. Faster RAM and the `performance` governor are untried for CPU-expert runtimes.
- GeForce RTX 30 series: [gpu-unlock](../nvidia-unlock/README.md) removes the half-rate cap of FP16/BF16 tensor-core
  math with FP32 accumulation. On the RTX 3090 with HyperQwen (W4A16 Marlin), prompt processing was 40-52 % faster
  and generation unchanged except +11-13 % past 100k tokens of context
  (see [`../vllm-rtx3090-hyperqwen/README.md`](../vllm-rtx3090-hyperqwen/README.md#gpu-unlock)).

## Block-diffusion models (DiffusionGemma)

DiffusionGemma 26B-A4B (rtx3090, [`config-diffusiongemma-26b-a4b.yaml`](../vllm-rtx3090/config-diffusiongemma-26b-a4b.yaml)),
natively supported by vLLM 0.33.0:

- The checkpoint: cyankiwi's AWQ int4 (compressed-tensors) takes 15.36 GiB without the vision tower (fp16 dense MLPs
  and tied embedding). It does not fit radeon-vii's 16 GB. NVFP4 and FP8 need matrix units that Ampere lacks.
- `generation-config: vllm` (the checkpoint limits `max_tokens` to 256), `hf-overrides` with the entropy-bound sampler
  and `diffusion-config: {"canvas_length": 256}` as in Google's recipe.
- The int8 KV cache with `TRITON_ATTN` works. The KV cache of 3.7 GB holds 141030 tokens (98304 of context), as the
  sliding-window layers do not grow with the context. `gpu-memory-utilization: 0.88` ran out of memory after the CUDA
  graph capture; set `kv-cache-memory-bytes`.
- Requests must not have `temperature`, `seed`, `min_p` or `min_tokens` (HTTP 400, also inside a stream).
- Speed on rtx3090: 281 t/s end to end in chat, 148 t/s from the second block on at 32k tokens of context, prompt
  processing 2100-2500 t/s: as fast in chat as autoregressive Gemma 4 26B-A4B with MTP, but with 2.4 times slower
  prompt processing and 60 % of its context. 2.7 times as fast as llama.cpp's draft implementation.
- The stream sends each 256-token block as one chunk, so compare the end-to-end speed (`output_tps` in
  `bench_http`) rather than the generation speed between chunks.
- Not tried: `max-num-seqs` > 1, FULL CUDA graphs.

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

- Test servers: [`../agx_ai/benchmark/vllm/test_server.py`](../agx_ai/benchmark/vllm/test_server.py) `IMAGE CONFIG ENV [PORT] -- args`
  with the production container stopped (`GPUS`, `DOCKER_ARGS`, `CACHE_VOLUME` to reuse the compile caches); it
  prints the KV cache size, the maximum concurrency and the VRAM.
- Speed: `bench_http.py --tests generation prompt chat depth edit concurrency --server-config <yaml>`
  (vLLM is detected; timing from the stream, `cache_salt` against prefix-cache hits, draft acceptance from
  `/metrics`).
- Quality: `quality_http.py --tests retrieval --logprobs N --reference <label>` with speculative decoding
  off, against a noise floor from a lossless change (e.g. `max-num-batched-tokens` 2048 -> 1024: KL 0.0020).
- A single run varies ±5-10 % (chat ±10 %): repeat, and call smaller differences noise.
- Comment every YAML key with the measured reason, record rejected variants and a "not tried" list, keep the summary
  table at the start of the README (with the llama.cpp comparison), and store the results in `../benchmark/results/`.
