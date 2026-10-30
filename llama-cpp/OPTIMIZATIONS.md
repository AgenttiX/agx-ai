# llama.cpp optimizations in this repository

A catalog of the llama.cpp settings that were tried in the configurations of this repository, what they did,
and when they were rejected. The measurements and their full context are in the README files of the
configurations; this file collects them by topic, so that a new configuration can start from what is known.
The Claude Code skill [`optimize-llama-cpp`](../.claude/skills/optimize-llama-cpp/SKILL.md) applies them.

Abbreviations: tg = generation, pp = prompt processing, ub = `ubatch-size`, ncmoe = `n-cpu-moe`,
acceptance = MTP draft acceptance. Configurations: [radeon-vii](../llama-cpp-radeon-vii),
[rtx-a4000](../llama-cpp-rtx-a4000), [rtx3070](../llama-cpp-rtx3070), [rtx3090](../llama-cpp-rtx3090),
[t550](../llama-cpp-t550), [t480](../llama-cpp-t480), [l14](../llama-cpp-l14), [frame](../llama-cpp-frame),
[windows](../llama-cpp-windows).

## Order of priority

What has given the largest gains, roughly in order:

1. **All layers on the GPU** (`n-gpu-layers = all`). Dense models lose a lot from partial offload (Gemma 12B with
   48/44/40 layers on the GPU: 53/36/29 t/s; Gemma 4 31B on the RTX 3090: a q8_0 KV cache that let all 60 layers
   fit doubled tg). For MoE models that do not fit, keep `n-gpu-layers = all` and move experts with `n-cpu-moe`
   instead (rtx3070: +60 % chat speed compared with `ngl 13`).
2. **Flash attention** (`flash-attn = on`): 14-50 % faster almost everywhere, and needed for a quantized V cache.
3. **MTP speculative decoding** (`spec-type = draft-mtp`) with a short draft (`spec-draft-n-max = 2`): +18-25 %
   on GPUs.
4. **Context from the memory that is left**: the KV cache type, the vision encoder's placement, the ubatch size and
   the number of expert layers on the CPU are all trades between speed and context. Measure each one's cost per
   token and in MiB, and spend the VRAM where it gains the most.

## KV cache type (`cache-type-k`, `cache-type-v`)

- **Quality:** q8_0 for K, V or both is within the noise floor (no larger than a change of the ubatch size) for
  every model tested (Qwen3.8-27B, Gemma 4 26B-A4B, Gemma 4 31B, E4B), on CUDA and ROCm. q4_0 is clearly above it:
  2-17x the noise KLD for Qwen, 4-30x for Gemma, and the first answer token is affected the most. Retrieval
  accuracy did not change even with q4_0, so accuracy alone is too blunt to see the loss. Use q8_0, not q4_0,
  unless the context is otherwise far too small.
- **Speed:** a quantized cache slows tg more the longer the context: rtx3090 Qwen3.8-27B -2 to -5 % up to 16k,
  -10 % at 36k, -15 % at 60k; rtx3070 -18 % at 70k; t550 -9 % at 30k; t480 -10 %. pp is only 2-3 % slower.
  q4_0 is no faster than q8_0.
- **Only matched types are fast.** The CUDA build of the ggml-org images handles f16/f16, bf16/bf16, q8_0/q8_0 and
  q4_0/q4_0 efficiently. Other types and mixed pairs (q8_0/q4_0) fall back to the CPU for parts of the attention:
  17 t/s pp at depth 16k. On ROCm gfx906, mixed pairs were 5-13 % slower. K-only or V-only q8_0 (with f16 for the
  other) does work and was measured on radeon-vii (V-only q8_0 was the fastest quantized variant there).
- **CUDA saves less than the size suggests:** for batches of more than one token, the CUDA flash attention converts
  a quantized cache to f16 in a temporary buffer that grows with the context, and the model and the MTP drafter
  each reserve one. Gemma 4 26B-A4B on the RTX 3090 saves only 7 % per token with q8_0; on the RTX 3070 q8_0 saves
  a fixed ~670 MiB; on the T550 with MTP it saves nothing. Measure the VRAM before choosing.
- **Where q8_0 was chosen:** Qwen3.8-27B (rtx3090: 64k -> 102k context; radeon-vii: 28k -> 47k), Gemma 4 31B (all
  layers fit), Gemma 4 26B-A4B on radeon-vii and t550, Qwen3.8-Flash-Next. **f16 kept:** Gemma 4 26B-A4B on the
  RTX 3090 and 3070 (little saving, 14 % slower), t480 (slower).
- `cache-type-k-draft`/`cache-type-v-draft` for the MTP drafter made no difference (radeon-vii).
- The memory per token depends on the architecture: Qwen3.8 has a KV cache in only 16 of 64 layers (the rest are
  Gated DeltaNet with a fixed state of ~450 MiB per slot), Gemma 4 only in the global-attention layers (the others
  have a 1024-token sliding window, which grows with the ubatch size). Read the per-token cost from two loads
  instead of computing it.

## Context size and VRAM headroom

- **Probe:** load, then send one prompt of `ctx-size - 1500` tokens; record the VRAM after loading and the peak.
  Step the context in multiples of 1024-4096 tokens, extrapolate by the measured cost per token, and confirm the
  final value with the full benchmark, a concurrent or mixed load, and, for routers, unload/reload cycles
  (`POST /models/unload`). On the rtx3070, 10-20 consecutive near-full prompts were also run.
- **Loading is not enough:** several contexts loaded but died on the first long prompt (radeon-vii 75000 at 20k
  tokens; rtx-a4000 106496 in `cudaGraphInstantiate`; 67584 on the A4000 died under load).
- **Headroom targets:** radeon-vii peak ≤ ~16300 of 16368 MiB (16331 crashed later; within ~50 MiB of the total is
  too close); rtx3070 ~7650-7720 of 7839 MiB (7751+ too tight); rtx3090 ~21.5 GiB for llama-server so that ≥1 GiB
  stays free next to the desktop; T550 (Windows WDDM) ~3940 of 4096 MiB.
- **Windows spills silently:** beyond the dedicated VRAM, WDDM moves memory to shared memory and tg drops (18 vs
  30 t/s) without an error. Watch `Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage","\GPU Adapter Memory(*)\Shared Usage"`
  (150-190 MiB shared is normal).
- Record the cost per 1024 tokens and the alternatives in the preset comments (e.g. "With f16 caches, the
  maximum would be ~64000"), so that later changes can be re-planned without re-probing.
- `fit = off` everywhere: the sizes are set by hand from the measurements.

## Experts and tensors on the CPU (`n-cpu-moe`, `override-tensor`, `no-host`, `no-op-offload`)

- **`n-cpu-moe` instead of partial `n-gpu-layers`** for MoE models. Each expert layer moved to the CPU frees its
  size (Gemma 4 26B-A4B: ~408 MiB ≈ 16-19k tokens) at a cost of ~2-5 % chat speed (rtx3070: ncmoe 25 -> 28 grew
  the context from 77824 to 122880). This was cheaper than a q8_0 KV cache for the same context (-10 % vs -18 % at
  70k).
- **Expert offload suffers from a busy CPU:** with `stress-ng` on all threads, tg fell 78 % with ncmoe 1 but only
  4-5 % with everything on the GPU (radeon-vii). On a desktop or a shared host, prefer everything on the GPU. On
  the 2-GPU A4000, ncmoe 6 was slower than splitting the model over both GPUs (94 vs 135 t/s).
- **Bandwidth-bound:** with most experts on the CPU, more GPU layers help little (Qwen3.8-Flash-Next on the RTX
  3090: 5 more layers on the GPU = +7 %), and the IQ quant types are slow to decode on the CPU (~13 GB/s vs ~41 GB/s
  for Q4_K experts).
- **Op offload:** during pp, the CPU-resident experts are copied to the GPU for each ubatch, so a larger ubatch
  speeds up pp. `no-op-offload = 1` avoids the PCIe traffic but makes pp ~12x slower; use it only if the GPU is
  unstable.
- **`no-host = true`** repacks the CPU weights (AVX2/ARM layouts): tg +16-40 %, but pp runs on the CPU (3-3.7x
  slower) and it makes an anonymous copy of the CPU weights (12.5 GiB; swapped on WSL). It wins only when the
  answers are longer than the new prompt tokens (break-even: prompt ≈ 1.1-1.3x the answer, t480).
- **Very small GPUs:** `override-tensor = ffn_|token_embd|output=CPU` keeps only the attention projections and the
  KV cache on a 2 GB MX150 (t480): a 5.8k-token document in 145 s instead of 412 s CPU-only.
- **Do not mix CPU and Vulkan layers on an iGPU** (l14): tg 0.17-0.5 t/s.
- Gemma 4 26B-A4B with offload beat a dense Gemma 12B that fits (rtx3070); on a CPU, the 26B-A4B (4B active)
  out-generates E4B and 12B (t480).

## Batch sizes (`ubatch-size`, `batch-size`)

- The ubatch size is a trade between pp and VRAM (compute buffers): rtx3090 dense models ub 1024 = +1-5 % pp for
  +240-480 MiB, so they use 512; the MoE Gemma 4 26B-A4B gains +22-34 % from 2048 for ~870 MiB, so it uses 2048;
  radeon-vii Gemma 256/512/1024 = 1133/1501/1851 t/s; t550 26B 1024 = +30 % over 512, 2048 no faster. MoE models and
  configurations with experts on the CPU gain the most.
- Lowering the ubatch is a way to free VRAM (radeon-vii Qwen: ub 256 = -116 MiB, 1024 = +250 MiB vs 512).
- **Gemma vision encodes an image in one ubatch**, so the ubatch caps the image tokens (ub 512: 512 instead of
  1120 tokens per image).
- **Crashes:** T550 flash attention + MTP at ub 512 crashed on long prompts (CUDA VMM, `cuMemSetAccess`): ub 256;
  Adreno 750 ub 512 hangs the GPU (`vk::DeviceLostError`): ub 128; radeon-vii Vulkan ub 2048: `DeviceLostError`.
- `batch-size` stays at the default 2048, except 4096 for embedding servers.

## Flash attention (`flash-attn`)

- `on` (explicitly, though `auto` enables it on CUDA and ROCm). Off was slower everywhere it was measured: rtx3090
  -14-22 % pp and -33 % tg, Gemma 4 31B -20-50 %, radeon-vii -35 % tg at 16k, rtx3070 39 vs 47 t/s, Vulkan 73 vs
  85-89 t/s, T550 52 vs 84 t/s pp and +230 MiB.
- **Exception:** Intel iGPU (l14, Vulkan): flash attention is slow and hangs the GPU, so it is off. Adreno: no
  difference.

## Speculative decoding (MTP)

- `spec-type = draft-mtp` with the model's MTP drafter. **`spec-draft-n-max = 2` is the usual optimum**
  (radeon-vii, rtx3090 Qwen and Gemma 26B-A4B); Gemma 4 31B accepts more and is fastest with 4 (3 for the
  uncensored one). Measure 1-4 (or 5) with the realistic `chat` and `depth` tests, two alternating runs each.
- Gains on GPUs: +18-25 % (radeon-vii Qwen), Gemma 4 31B +40 % with 4. Costs VRAM: the drafter (500-900 MiB) and its
  own compute buffers.
- **Not always worth it:** slower on the t480 CPU, on the l14 (dense models on the CPU, E4B, Vulkan 26B) and for
  Qwen3.8-Flash-Next with the drafter on the CPU (verification selects more distinct experts). Faster on the l14
  CPU 26B (+10-20 %), frame E4B (+6-7 %), t550 E4B (~+50 %), the rtx3070 CPU model (+45 %).
- Benchmarks with repeated text or temperature 0 overstate the acceptance (85-90 %); `llama bench` has no
  speculative decoding at all.
- The Unsloth drafter of Gemma 4 26B-A4B also works with an uncensored finetune; give it by snapshot path
  (`model-draft`), since `hf-repo-draft` picks the main model's file.
- **Drafter load bug** ([llama.cpp#29044](https://github.com/ggml-org/llama.cpp/issues/29044)): when VRAM is nearly
  full, the drafter fails to load (`vector::_M_range_check`); `tensor-split = 1` on a single GPU works around it.
- `device-draft = CUDA1` places the drafter on another GPU.

## Multiple GPUs (`split-mode`, `tensor-split`, `device`)

- `split-mode = tensor` with `tensor-split = 1,1` for Qwen3.8-27B on 2x RTX A4000 (needs flash attention and an
  f16 cache). Gemma 4 26B-A4B crashes with tensor split (`fattn.cu:762`), and so does the embedding model
  (`ggml-backend-meta.cpp` assert): use `split-mode = layer` for them.
- Layer split keeps several compute-buffer copies (pipeline parallelism), so it needs a smaller ubatch (1024). The
  last GPU holds the output layer: use an uneven split (4,3) when 1:1 OOMs on it.
- NCCL (since ~b11500) on a board with broken PCIe P2P: `NCCL_P2P_DISABLE=1` and `shm_size: 2g` (64 MiB gives
  "unhandled system error"). Its buffers cost ~1.5 GiB per GPU.
- `device = CUDA1` / `CUDA0` puts one model on each GPU; `device = none` keeps a CPU model off the GPUs.

## Threads (`threads`, `threads-batch`)

- Fewer threads than the CPU has is usually best for tg, and pp may want more: rtx3070 host 12 of 16 (4/8/12/16 =
  35/48/59/56 t/s); t480 and frame `threads = 4`, `threads-batch = 8`; t550 8 (16 uses E-cores and halves tg); l14
  8-10 (LP E-cores slow it down); rtx3090 Flash-Next 24 = 32 (bandwidth-bound). CPU pinning (`-C`, `--cpu-strict`)
  did not help, except ~3 % for the compute-bound embedding model on the Threadripper (`cpu-range = 0-31`,
  `cpu-strict = 1` and the `-batch` variants).
- **Never the SMT siblings for compute-bound work:** EmbeddingGemma 2 on the Threadripper 3970X with 8/16/32/64
  threads = 1000/1650/2300/880 t/s (500-token texts).
- In an LXC container, check the cpuset (`/sys/fs/cgroup/cpuset.cpus.effective`) to avoid oversubscribing SMT
  siblings.
- Two models on one CPU at once (rtx3070 GPU + CPU models): 16 threads each collapsed the GPU model to 3-4 t/s; use
  8 each.

## Vision encoder (`no-mmproj-offload`, `no-mmproj`, `image-min-tokens`)

- The vision encoder takes ~1.1-1.3 GiB of VRAM. On the CPU (`no-mmproj-offload = 1`), that is ~22-26k tokens of
  context (rtx3090: +22-28 % for Qwen and Gemma 4 26B-A4B, +58 % and +138 % for the Gemma 4 31B models), and each
  image takes 3-5 s longer (a 20-s video ~80-90 s on radeon-vii).
- Keep the encoder on the GPU when the context is already large, and move it when the context is otherwise too
  small. Put the other choice's `ctx-size` in the preset as a comment.
- On WSL with memory-mapped experts, the CPU encoder evicted experts from the page cache (tg 9.3 vs 19.7 t/s), so
  t550 Docker uses `no-mmproj = true`.
- Qwen-VL warns that grounding needs `image-min-tokens = 1024` (left as a comment).
- Test images with the `image` test of `llama_cpp_bench_http.py`; the media peaks should stay below the text
  stress peaks.

## Loading (`load-mode`)

- `dio` for models entirely on the GPU (radeon-vii, rtx3090); it also works with `n-cpu-moe` without warnings on
  radeon-vii.
- `none` (pinned RAM) for hybrid CPU+GPU models, to avoid the "tensor overrides to CPU are used with mmap enabled"
  warning. On Windows, it counts against the commit limit, so the t550 does not use it.
- `mmap` where models are unloaded when idle (rtx-a4000): reloading from the page cache takes seconds instead of
  ~50 s.
- `mmap` + `mlock` for CPU-only models; in LXC this needs `lxc.prlimit.memlock: unlimited` and compose `ulimits`.
  In presets, this is `load-mode = mmap+mlock` (b12004 has no `mlock` preset key) and compose `ulimits: memlock: -1`.

## Router, slots and caches (`config.ini`)

- `kv-unified = 1` shares one KV pool between the slots; `parallel` matches the client (1 for a single user,
  2 for Gemma 4 26B-A4B on the RTX 3090, 4 for PaperQA2).
- `models-max = 1` (one model at a time), 2 for an LLM and an embedding model or a GPU and a CPU model. The router
  cannot make GPU models exclusive: unload one (`/models/unload`) before loading another, or it OOMs.
- `cache-ram = 16384` (default 8192) keeps the prompt caches of evicted slots in RAM: restoring a 35k-token
  conversation took 1.0 s instead of 19 s (~0.7-1 GB each). Set it to 0 when RAM is short (t550 WSL).
- `ctx-checkpoints = 0` on Vulkan (l14): the SWA checkpoints cost ~10 s per request; follow-up turns still reuse
  the cache.
- `sleep-idle-seconds = 600` unloads idle models (rtx-a4000).
- The ROCm backend sampler does not support `top-k` (harmless error).

## Quantization

- Prefer the largest quant that fits with a useful context; on gfx906 UD-IQ4_XS was even faster than the smaller
  UD-Q3_K_XL and UD-IQ3_S.
- QAT models (Gemma 4) are worth it: E4B QAT was ~39 % faster, accepted more drafts (66 vs 52 %) and had 2.7x the
  context. Unsloth UD-Q4_K_XL beat Google's Q4_0 GGUF (frame).
- On the CPU, the IQ types are decode-bound with AVX2; Q2_0 without an AVX2 kernel was 2.5x slower. Q4_0 is
  repacked on the CPU at load (more RAM).

## Embedding servers

- `embedding = 1`, `pooling = last` (Qwen3-Embedding), `parallel = 1`, `ctx-size = 4096`, `batch-size = 4096`,
  `ubatch-size = 2048` (rtx-a4000). The compute buffer grows ~600 MiB per 1024 ubatch. Qwen3-Embedding-4B retrieved
  as well as 8B at 1.6x the speed and 1.5 GiB less VRAM per GPU.
- **Bidirectional models (EmbeddingGemma 2):** an input has to fit into one ubatch, so `ubatch-size = batch-size =
  ctx-size = 8192`. On the Threadripper 3970X CPU (zen2-embeddinggemma2): 8 slots (the texts of a request share a
  ubatch; 4 was as fast for 500-token texts, 16 slower), BF16 = F16 = F32 = Q8_0 within 5 % (compute-bound), flash
  attention off 2.4x slower, 2200-2300 t/s. vLLM's CPU image was 2x faster for the same model.
- **Routers list the whole Hugging Face cache:** a request for another cached model loads it and, with
  `models-max = 1`, evicts the embedding model. Dedicated embedding routers set `LLAMA_ARG_MODELS_AUTOLOAD=false`
  (the `load-on-startup` model still loads). `llama_cpp_bench_http.py --tests embedding` triggered this until
  2026-10-31.

## Backends, images and builds

- Compare backends on the same GGUF with `llama_cpp_bench_container.py --image` and `llama_cpp_test_server.sh`.
  On radeon-vii, Vulkan (RADV) was ~10 % faster in short chats but slower beyond ~6k tokens and ~20 % slower in pp,
  so the gfx906 ROCm image ([`mixa3607/llama.cpp-gfx906`](https://github.com/mixa3607/ML-gfx906)) stays. On the
  l14, Vulkan beat SYCL for MoE models.
- Pin build tags and measure updates A/B in alternating order: compute buffers change between builds (b11176 needed
  3.2 GiB vs 2.0 GiB for Flash-Next; b11552 peaked lower for Gemma), which can break a tight context.
- CUDA images: `NVIDIA_DISABLE_REQUIRE=1` if the image needs a newer CUDA than the driver; the images set
  `LLAMA_ARG_PORT=8080`, which overrides `config.ini`. Pascal/Turing: keep a `cuda-cache` volume with
  `CUDA_CACHE_MAXSIZE=4G` (JIT; restart 343 -> 121 s). Windows: the CUDA 12.4 build for Turing with an older driver;
  native was faster than WSL (+4-11 % tg, 4-6x faster loading).
- ARM big.LITTLE (frame): do not build with `GGML_NATIVE` (it detects the small core).

## Power, clocks and the OS

- RTX 3070 "fell off the bus" (Xid 79) under long full-load prompts: a 170 W power limit or a clock lock
  (`nvidia-smi -lgc 210,1600`: -25 % power, -6 % pp, same tg), set at boot by
  [`../llama-cpp-rtx3070/gpu-power-limit`](../llama-cpp-rtx3070/gpu-power-limit).
- Laptops: check the actual power limit (`nvidia-smi -q -d POWER`; the T550 was capped at 20 W instead of 30 W, -35 %)
  and close background apps ([`../llama-cpp-t550/free-ram.ps1`](../llama-cpp-t550/free-ram.ps1)).

## Methodology

- Benchmark the serving configuration on a test server
  ([`../benchmark/llama_cpp_test_server.sh`](../benchmark/llama_cpp_test_server.sh), production container stopped)
  with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py) `--tests generation prompt chat depth image`.
  Judge speed by the `chat` and `depth` tests; the default generation test uses repeated text that flatters MTP.
- Raw model speed (quants, ubatch, KV types, builds): `llama_cpp_bench_container.py` (`llama bench`).
- Quality of a setting that saves memory: [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py)
  against an f16 reference with MTP off, and a noise floor from lossless changes (another ubatch size). Compare the
  KLD (mean, 99 %, first token) and the changed top tokens with the noise, not just the accuracy. `llama-perplexity`
  KLD is unusable for some instruction-tuned models (Gemma 4 26B-A4B).
- Noise is ±1-4 % on desktops and up to ~10 % on laptops: repeat twice in alternating order. Turn off the LiteLLM
  health check during benchmarks (see the benchmark README).
- Document every non-default setting in the preset with the measured reason and the rejected alternatives, keep the
  summary table at the start of the README, and store the results in `../benchmark/results/`.
