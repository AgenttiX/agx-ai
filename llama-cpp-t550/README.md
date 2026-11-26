# llama.cpp on the NVIDIA T550 Laptop GPU

Configuration for the NVIDIA T550 Laptop GPU (4 GB, Turing TU117, 30 W) under Docker Desktop with WSL 2 on Windows.
The model and its settings are in [`preset.ini`](preset.ini), server-wide settings in [`../llama-cpp/config.ini`](../llama-cpp/config.ini)
and the CUDA/WSL workarounds in [`docker-compose.yml`](docker-compose.yml).
The benchmark results of this computer are saved with the hostname `t550` (`--hostname t550` for the scripts in
[`../agx_ai/benchmark`](../agx_ai/benchmark/README.md)).

Five models, one loaded at a time (`models-max = 1` in [`../llama-cpp/config.ini`](../llama-cpp/config.ini); requesting the other model unloads
the loaded one):

- `google/gemma-4-e4b-qat`, the default model, loaded at startup: Gemma 4 E4B QAT (`unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL`),
  entirely on the GPU, with its MTP drafter, flash attention, an f16 KV cache, `ctx-size = 69632`, a single slot
  and the vision/audio encoder (mmproj) on the CPU.
- `google/gemma-4-e4b-qat-long`, loaded on request: the same model for long documents, with q8_0 K and V caches and
  no MTP, `ctx-size = 98304`. Generation is 35-50 % slower (see
  [KV cache quantization](#gemma-4-e4b-qat-kv-cache-quantization-and-context-2026-11-26)).
- `google/gemma-4-26b-a4b-qat`, loaded on request: Gemma 4 26B-A4B QAT
  (`unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL`), with the MoE experts in RAM and everything else on the GPU, MTP,
  q8_0 K and V caches, `ctx-size = 73728`, a single slot and the vision encoder on the CPU (see
  [Gemma 4 26B-A4B QAT](#gemma-4-26b-a4b-qat)). Slower but much stronger than the E4B.
- `qwen/qwen3.5-4b`, loaded on request: Qwen3.5-4B (`unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL`), entirely on the GPU,
  f16 KV cache, `ctx-size = 28672`, the vision encoder on the CPU (see [Qwen3.5-4B and 9B](#qwen35-4b-and-9b-2026-11-26)).
- `qwen/qwen3.5-9b`, loaded on request: Qwen3.5-9B (`unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL`), 20 of 32 layers on the
  GPU, q8_0 K and V caches, `ctx-size = 16384`, the vision encoder on the CPU.

[`preset.ini`](preset.ini) is set up for running llama.cpp [natively on Windows](#natively-on-windows-2026-09-30),
which is faster and has room for the 26B-A4B's vision encoder. Where Docker/WSL 2 needs different settings, the
Docker/WSL 2 settings are kept in the preset as comments, marked "Docker/WSL 2"; uncomment them (and comment out the
native ones next to them) to use the preset with Docker.

## Summary (2026-11-25)

Natively on Windows with llama.cpp b12250 (CUDA 12.4 build), GPU power limit 20 W; Docker with
`server-cuda13-b12242`. tg = generation, pp = prompt processing; "chat" = the chat test of
[`bench_http.py`](../agx_ai/benchmark/README.md), the realistic generation speed.

| Model | Context | VRAM (peak) | RAM (llama-server) | tg: chat / at 16k / at 32k | pp: 4096 tokens / at 16k / at 32k | Notes |
|---|---|---|---|---|---|---|
| Gemma 4 E4B QAT UD-Q4_K_XL, MTP 2 | 69632 | 3883 MiB | ~4.2 GB | 30.6 (native, after a reboot), 32.9 (Docker) / 25.1 (native), 26.9 (Docker) / 19.3 t/s | 245 (native), 253 (Docker) / 212 (native), 215 (Docker) / 180 t/s | Docker is as fast as native; builds after b12250 generate up to 29 % slower at long contexts (see [below](#llamacpp-update-2026-11-25)) |
| Gemma 4 E4B QAT UD-Q4_K_XL, q8_0 K+V, no MTP (`-long`, 2026-11-26) | 98304 | 3897 MiB | ~4.2 GB | 19.9 / 14.5 / 11.9 t/s (8.1 at 64k, 6.1 at 94k) | 228 / 201 / 173 t/s (133 at 64k, 110 at 94k) | 41 % more context, generation 35-50 % slower; quality as f16 (see [below](#gemma-4-e4b-qat-kv-cache-quantization-and-context-2026-11-26)) |
| Gemma 4 26B-A4B QAT UD-Q4_K_XL, MTP 2, experts in RAM | 73728 | 3867 MiB | ~16.2 GB | 6.2 (7.9 with b11262 and more free RAM on 2026-09-30) / 6.0 / - t/s | 87 / 82 / - t/s | RAM-bound: keep ~2 GB free for the page cache (see [below](#gemma-4-26b-a4b-qat-with-b12250-natively-2026-11-25)); 4-5 times slower than the E4B, but much stronger; in Docker only without the vision encoder (RAM of the WSL 2 VM) |
| Qwen3.5-4B UD-Q4_K_XL (2026-11-26) | 28672 | 3835 MiB | ~4 GB | 13.3 / 12.6 (22k) / - t/s | 141 / 108 (22k) / - t/s | No MTP; slower than the E4B (see [Qwen3.5-4B and 9B](#qwen35-4b-and-9b-2026-11-26)) |
| Qwen3.5-9B UD-Q4_K_XL, 20 of 32 layers on the GPU (2026-11-26) | 16384 | 3833 MiB | ~6.5 GB | 3.5 / 3.0 (14k) / - t/s | 66 / 61 (14k) / - t/s | CPU-bound; the 26B-A4B is faster and stronger here |

## llama.cpp update (2026-11-25)

Updated from b11262 (native, CUDA 12.4) and b11151 (Docker) to **b12250** (native) and **b12242** (Docker,
`ghcr.io/ggml-org/llama.cpp:server-cuda13-b12242`, the newest image at or before b12250). Both are now pinned:
[`docker-compose.yml`](docker-compose.yml) names the image tag, and `%LOCALAPPDATA%\llama.cpp\current.txt` the native
build (`..\llama-cpp-windows\install-llama-cpp.ps1 -Build b12250 -Backend cuda-12.4`). The newest build, b12686, is
faster for prompts too but slower for generation at long contexts:

| Gemma 4 E4B QAT, production preset, native | b11262 | **b12250** | b12686 |
|---|---|---|---|
| Chat (MTP) | 24.9 t/s | **27.8 t/s** | 21.0-32.0 t/s |
| 32k-token code context: pp / tg (MTP) | 94.8 / 19.1 t/s | **180.4 / 19.3 t/s** | 142.2 / 13.6 t/s |
| 64k-token code context: pp / tg (MTP) | (b11151 in Docker: 81 / 16.7 t/s) | | 108.6 / 11.6 t/s |
| `llama bench` pp4096 (`-fa 1 -ub 256`) | 137-145 t/s | 241 t/s | 230-247 t/s |
| `llama bench` tg64 at depth 0 / 16384, ratio (without MTP) | 0.79, 0.79 | 0.78 | 0.71-0.72 |
| Peak VRAM | 3861 MiB | 3883 MiB | 3879 MiB |

- **Prompt processing is ~70-90 % faster since b12100** (`llama bench` pp4096: b11900 145 t/s, b12100 249 t/s). The
  likely cause is the fused FFN gate/up matmuls + GLU in MMQ
  ([ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948), 2026-10-25): Gemma's gated GELU
  FFN no longer needs separate gate and up matmuls and a separate GLU kernel. On this GPU, 67 % of the prompt time is
  in the quantized matmuls (MMQ with dp4a, as TU117 has no tensor cores) and 25-30 % in flash attention at 16k tokens
  of context (Nsight Systems).
- **Generation at long contexts got slower after b12250** (b12400 and b12686 measured): through the server with MTP
  29 % at 32k tokens (13.6 vs 19.1 t/s) and 30 % at 64k tokens; without MTP (`llama bench -d 32768`) 16 %. At depth
  0 all builds generate equally fast (23-24 t/s without MTP). The Turing configurations of the flash-attention and
  MMVQ kernels did not change between these builds, and the commit could not be narrowed down further on this
  laptop (see the measurement noise below). b12250 has the faster prompts without this regression.
- **ubatch** (b12686, `llama bench` pp4096): 128 / 256 / 512 / 1024 = 227 / 231 / 227 / 226 t/s, so the
  `ubatch-size = 256` that MTP needs costs nothing.
- **Docker** with b12242 (16k-token context: pp 215 t/s, tg 26.9 t/s; chat 32.9 t/s; 4096-token prompt 253 t/s;
  peak VRAM 3895 MiB) is as fast as native b12250. The image now sets `LLAMA_ARG_PORT=8080`, which overrides
  `port = 9931` of `config.ini`, so [`docker-compose.yml`](docker-compose.yml) sets `LLAMA_ARG_PORT` back.
- The CUDA 13.5 build of b12211 and later (the default of `install-llama-cpp.ps1`) still fails with the driver
  596.52 (CUDA 13.2): `the provided PTX was compiled with an unsupported toolchain`. Use `-Backend cuda-12.4`.

### Can the kernels be optimized for this GPU?

- **Native Turing kernels would not help.** The Windows CUDA 12.4 build and the Docker CUDA 13 image both contain
  Turing only as compute_75 PTX (native SASS only for sm_86, sm_89 and newer; `cuobjdump --list-elf`/`--list-ptx`).
  The driver JIT-compiles this PTX, so the Turing code paths are already used (the MMVQ Turing parameter table, the
  MMA flash-attention kernels), and the result is cached. A build with `CMAKE_CUDA_ARCHITECTURES=75-real` would only
  save the JIT at the first start.
- **Generation is close to the memory bandwidth.** The GDDR6 runs at 12 Gbps on a 64-bit bus (96 GB/s, also at the
  20 W limit). The E4B reads ~2.5 GB of GPU weights per token, so the ceiling is ~38 t/s without MTP, and
  `llama bench` reaches 23-24 t/s (~63 %). The remaining time per token includes the per-layer embeddings, which
  Gemma 4 E-models compute on the CPU (a busy CPU slowed generation from 23 to 15 t/s), and small kernels.
- **Prompt processing is compute-bound** on the dp4a MMQ kernels at the 20 W power limit (the GPU runs at
  ~1.5-1.8 GHz and reports the software power cap during prompts). TU117 has no tensor cores, so the int8/FP16 MMA
  paths of newer GPUs do not apply. The largest gain, the fused gate/up MMQ, came with the update.
- **The remaining kernel-level opportunity** is the flash-attention decode path at long contexts (the regression
  above shows that it is sensitive to the kernel changes): with the 512-wide global-attention heads, it uses the
  MMA kernel, which on TU117 runs on the FP16 units instead of tensor cores. Changing it means patching and building
  llama.cpp (no MSVC on this computer; a Docker build would work), which was not attempted.

### Measurement noise on this laptop

The GPU's clock follows the shared CPU/GPU power budget and the temperature: `llama bench` tg at depth 0 varied
between 15 and 24 t/s for the same build within an hour, and the chat test between 21 and 32 t/s. Compare builds
in alternating order, with the ratio of tg at depth to tg at depth 0 from the same run, and confirm with the server
benchmark (which repeats each request). Stop other llama.cpp processes first: a leftover server makes everything
4 times slower without an error. At the end of the session on 2026-11-25, the GPU ran at only ~510 MHz at 18.7 W
with the software power cap active (`clocks_event_reasons.active` 0x4 in `nvidia-smi`), on mains power, and the
E4B generated only 18-19 t/s in chat, so that run was discarded. Check the clocks with
`nvidia-smi --query-gpu=power.draw,clocks.sm,clocks_event_reasons.active --format=csv` before benchmarking.

## Hardware and software

| | |
|---|---|
| OS | Windows 11 25H2 (build 26200), Docker Desktop 4.88.1 with the WSL 2 backend (kernel 6.18) |
| CPU | Intel Core i7-1260P (Alder Lake, 4 performance + 8 efficiency cores, 16 threads) |
| RAM | 32 GB DDR4-3200 (2 x 16 GB SO-DIMM, dual channel); the WSL 2 VM gets 15.5 GiB of it (WSL's default of half the RAM) and 4 GiB of swap |
| iGPU | Intel Iris Xe Graphics (96 EU, shares the system RAM), drives the display |
| dGPU | NVIDIA T550 Laptop GPU: Turing TU117 (sm_75, no tensor cores), 4 GB GDDR6 (64-bit), PCIe link x4 (nvidia-smi: gen 4, max width x16), driver 596.52 (CUDA 13.2) |
| dGPU power limit | 30 W (default and maximum). On 2026-09-27 and -28 the platform capped it at 20 W ("Current Power Limit" in `nvidia-smi -q -d POWER`, while "Requested Power Limit" was 30 W), which made the GPU ~35 % slower (see the results of 2026-09-27 below). |
| llama.cpp | Natively: b12250, CUDA 12.4 build (since 2026-11-25; b11262 before). Docker: `ghcr.io/ggml-org/llama.cpp:server-cuda13-b12242` (since 2026-11-25; b11151 before) |

## Gemma 4 E4B QAT: results

Measured with [`agx_ai/benchmark/bench_http.py`](../agx_ai/benchmark/README.md) (results in
`benchmark/results/t550.jsonl`), llama.cpp build b11151.
tg = token generation, pp = prompt processing, acc = accepted MTP draft tokens.

### Gemma 4 E4B QAT UD-Q4_K_XL, ctx 69632, f16 KV cache, MTP `spec-draft-n-max = 2` (2026-09-28, build b11151 in Docker; for the current builds see the [summary](#summary-2026-11-25))

GPU power limit 20 W (see above). "Chat" = 12 chat prompts of up to 512 tokens,
"16k/64k context" = questions about 16k/64k tokens of Python code, up to 512 generated tokens.

| Test | Result |
|---|---|
| Generation, 1 request, 256 tokens | 36.3 t/s, acc 66 % |
| Prompt processing, 4096 tokens | 141 t/s |
| Chat | 30.1 t/s (27.3-37.2), acc 54 % |
| 16k context | pp 120 t/s, tg 25.3 t/s, acc 62 % |
| 64k context | pp 81 t/s, tg 16.7 t/s, acc 60 % |
| Peak GPU memory | 3875 MiB |

### Gemma 4 E4B QAT UD-Q4_K_XL, ctx 65536, MTP `spec-draft-n-max = 2` (2026-09-27)

During these tests, `nvidia-smi -q -d POWER` showed a GPU power limit ("Current Power Limit") of 20 W instead of
the 30 W of the earlier tests, with the laptop on mains power and the Windows "Balanced" power plan.
This makes the GPU slower: `llama bench` generated 15.7 t/s with the non-QAT model, compared with 25 t/s on
2026-09-24. The speeds also varied between server starts, so the ranges below are over several starts.
Compare the two models by the `llama bench` rows, which were measured back to back under the same conditions.

| Test | Result |
|---|---|
| Generation, 1 request, 256 tokens | 28.5-38.6 t/s over four server starts, acc 66 % |
| Prompt processing, 4096 tokens | 138-143 t/s |
| Chat request with a 62.1k-token prompt (ctx 65536) | pp 80 t/s, tg 15.8 t/s, acc 56 % |
| Chat request with a 58.1k-token prompt (ctx 61440) | pp 77 t/s, tg 15.3 t/s, acc 61 % |
| Peak GPU memory in the benchmark | 3807 MiB |
| `spec-draft-n-max = 1` | tg 27.7 t/s, acc 67 % |
| `spec-draft-n-max = 3` | tg 28.8-35.2 t/s, acc 57 %; slower than 2 in both of two alternating pairs of restarts (35.2 vs 38.6, 28.8 vs 34.5 t/s) |
| `llama bench`, QAT UD-Q4_K_XL, no MTP (`-fa 1 -ub 256`) | pp512 152 t/s, pp4096 141 t/s, tg128 21.8 t/s |
| `llama bench`, non-QAT UD-Q4_K_XL, no MTP (`-fa 1 -ub 256`) | pp512 153 t/s, pp4096 141 t/s, tg128 15.7 t/s |

Compared with the non-QAT model, the QAT model generates ~39 % faster without MTP (smaller weights to read per
token), processes prompts at the same speed, has a higher MTP draft acceptance (66 % vs 52 %), and leaves room for
a 2.7 times larger context.

### Previous configuration: Gemma 4 E4B UD-Q4_K_XL (non-QAT), ctx 24576, MTP `spec-draft-n-max = 2` (2026-09-24)

GPU power limit 30 W.

| Test | Result |
|---|---|
| Generation, 1 request, 256 tokens | 37.9 t/s, acc 52 % |
| Prompt processing | 153 t/s for a 4096-token prompt, 123 t/s for a 22.9k-token prompt |
| Chat request with a 22.9k-token prompt | tg 27.3 t/s, acc 58 % |
| `spec-draft-n-max = 1` / `3` | tg 36.6 t/s (acc 77 %) / 32.5 t/s (acc 40 %) |
| `llama bench`, no MTP (`-fa 1`) | pp512 162 t/s, pp4096 151 t/s, tg128 25 t/s, i.e. MTP gave ~50 % faster generation |

## Gemma 4 E4B QAT: VRAM budget

Measured with the Windows "GPU Adapter Memory" performance counters, as nvidia-smi does not show the shared memory.

- Windows lets the model use up to ~3940 MiB of the 4096 MiB dedicated memory. Beyond that, WDDM silently moves
  memory to shared system memory ("Shared Usage" grows) instead of failing, and generation slows down: with flash
  attention off, ctx 32768 of the non-QAT model put 1100 MiB into shared memory and ran at 18 t/s instead of 30 t/s.
  Check for this in PowerShell with:
  ```powershell
  (Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage","\GPU Adapter Memory(*)\Shared Usage").CounterSamples
  ```
  150-190 MiB of shared usage is normal, as the pinned `CUDA_Host` buffers count as shared memory.
- Gemma 4 E4B keeps its per-layer embeddings on the CPU (1872 MiB for QAT UD-Q4_K_XL, 2288 MiB for non-QAT
  UD-Q4_K_XL), so only part of the GGUF file goes to the GPU: 2493 MiB for QAT UD-Q4_K_XL (4.0 GB file) and
  3026 MiB for non-QAT UD-Q4_K_XL (4.9 GB file). The MTP drafter takes ~70 MiB for QAT (Q4_0 drafter) and
  ~110 MiB for non-QAT (Q8_0 drafter). The compute buffers take ~70 MiB with `ubatch-size = 256`.
- The global-attention KV cache costs 16 KiB per token (512 MiB at ctx 32768), and the sliding-window cache ~30 MiB.
- Prompt processing needs ~15-130 MiB on top of the memory used after loading.

## Gemma 4 E4B QAT: context size

Probes with a prompt that nearly fills the context, containing a "secret word" in the middle that the model has
to find, followed by a generated summary. Peak = dedicated GPU memory.

| Model | ctx-size | Prompt | After loading | Peak | Result |
|---|---|---|---|---|---|
| QAT | 32768 | short | 3243 MiB | | loads |
| QAT | 61440 | 58.1k tokens | 3719 MiB | 3733 MiB | stable |
| QAT | 65536 | 62.1k tokens | 3787 MiB | 3801 MiB | stable; two unload/reload cycles through the router API |
| QAT | **69632** | 67.0k tokens | 3855 MiB | 3869 MiB | stable (pp 73 t/s, tg 13.2 t/s, found the word); the largest context that stays below the limit |
| non-QAT | 10000 | 8.8k tokens | | 3565 MiB | stable |
| non-QAT | 24576 | 22.9k tokens | 3677 MiB | 3815 MiB | stable; two unload/reload cycles through the router API |
| non-QAT | 32768 | 29.8k tokens | 3813 MiB | 3941 MiB | works, but at the ~3940 MiB limit, leaving no margin for other programs that use the GPU |

Each 4096 tokens cost 64 MiB. The next step, 73728, would peak at ~3930 MiB, i.e. at the limit. 69632 leaves ~70 MiB.

## Gemma 4 E4B QAT: KV cache quantization and context (2026-11-26)

Retested natively with b12250, to fit more context with a quantized KV cache. The earlier finding still holds:
**with MTP, a quantized cache saves no VRAM**, and without MTP it costs a lot of generation speed. The default model
therefore keeps its f16 cache, MTP and ctx 69632, and the q8_0 configuration is a second model,
`google/gemma-4-e4b-qat-long`, for documents that do not fit otherwise.

**Why:** the global-attention layers of Gemma 4 have a head size of 512, and llama.cpp's CUDA flash attention has
vector kernels (which read a quantized cache directly) only for head sizes up to 256 (`FATTN_VEC_CASES_ALL_D` in
`fattn.cu`). For head size 512, it uses the tile kernel, which converts the layer's whole quantized K and V to f16 in
a temporary buffer (`ggml_cuda_flash_attn_ext_get_alloc_size`): for prompts, for MTP's verification batches, and
for every generated token. The model and the MTP drafter each reserve this buffer (4 KiB per token of context), so
q8_0 costs 8.5 + 4 + 4 = 16.5 KiB per token with MTP, as much as f16 (16 KiB). Without MTP it costs 12.5 KiB, but
the conversion makes generation slower the longer the context.

**VRAM** after loading (`ubatch-size = 256`; the KV buffer includes the ~30 MiB sliding-window cache):

| KV cache | MTP | ctx-size | KV buffer | Compute buffers (model + drafter) | After loading |
|---|---|---|---|---|---|
| f16 | yes | 69632 | 1118 MiB | 88 + 44 MiB | 3863 MiB |
| q8_0 K+V | yes | 69632 | 594 MiB | 354 + 313 MiB | 3889 MiB |
| q5_0 K+V | yes | 81920 | 450 MiB | 408 + 367 MiB | 3857 MiB |
| q4_0 K+V | yes | 86016 | | | 3827 MiB |
| q4_0 K+V | yes | 90112 | 404 MiB | 444 + 403 MiB | 3881 MiB |
| q8_0 K+V | no | **98304** | 832 MiB | 480 MiB | 3893-3897 MiB |
| q8_0 K+V | no | 106496 | 900 MiB | 516 MiB | over the limit (~3990 MiB by the buffers) |

ctx 98304 with q8_0 and no MTP held a 94k-token prompt at 3897 MiB, and the shared GPU memory of llama-server
stayed at its normal 150 MiB (sampled every 20 s with the Windows GPU counters), i.e. nothing moved to shared memory.

**Speed** with [`bench_http.py`](../agx_ai/benchmark/README.md), GPU power limit 20 W (t/s):

| Configuration | Chat | Prompt, 4096 tokens | 16k context: pp / tg | 32k context: pp / tg | 64k context: pp / tg |
|---|---|---|---|---|---|
| f16, MTP 2, ctx 69632 (default) | 30.6 | 245 | 212 / 25.1 | 180 / 19.3 | (b12686: 109 / 11.6) |
| q8_0 K+V, no MTP, ctx 98304 (`-long`) | 19.9 | 228 | 201 / 14.5 | 173 / 11.9 | 133 / 8.1 |
| q4_0 K+V, MTP 2, ctx 86016 | 16.9 (acceptance 54 %) | | 180 / 10.9 | 174 / 11.7 | |

At 94k tokens, the `-long` configuration processed the prompt at 110 t/s and generated 6.1 t/s. q4_0 with MTP is
slower still, as every MTP step converts the cache, and q4_0 is measurably lossy (below), so it is not used.

**Quality** with [`quality_http.py`](../agx_ai/benchmark/quality_http.py) (greedy retrieval questions, MTP off, see
the method in the next section), q8_0 K+V at ctx 98304 against f16 at ctx 69632:

| KV cache | 31k tokens: single / two-step | 61k tokens | 90k tokens | Same answer as f16 | Mean KLD | 99 % KLD | Top token changed |
|---|---|---|---|---|---|---|---|
| f16 | 32/32, 9/12 | 32/32, 9/12 | (does not fit) | | | | |
| q8_0 K+V | 32/32, 10/12 | 32/32, 9/12 | 32/32, 9/12 | 86/88 | 0.00027 | 0.0070 | 2/570 |

The mean KLD is below the noise floor measured in 2026-09 (0.00056 for another `ubatch-size`), i.e. the q8_0 cache is
as good as f16, also at 90k tokens.

## Gemma 4 E4B QAT: KV cache quantization (2026-09-28)

Tested with the QAT model at ctx 65536: a q8_0 K and V cache (`cache-type-k`/`cache-type-v = q8_0`) is about as good
as f16, but on this GPU it saves no VRAM when MTP is on, and it is slower at long contexts. The KV cache therefore
stays at f16.

**VRAM** after loading (ctx 65536, `ubatch-size = 256`, the rest as in [`preset.ini`](preset.ini)):

| KV cache | KV buffers | Compute buffers (model + MTP drafter) | Total with MTP | Total without MTP |
|---|---|---|---|---|
| f16 | 1054 MiB | 83 + 42 MiB | 3787 MiB | 3699 MiB |
| V q8_0 | 807 MiB | 193 + 167 MiB | 3775 MiB | |
| K q8_0 | 807 MiB | 194 + 169 MiB | 3779 MiB | |
| K+V q8_0 | 560 MiB | 324 + 297 MiB | 3789 MiB | 3447 MiB |

The compute buffers grow with the context size, most likely because the CUDA flash attention converts a quantized
cache back to f16 for prompt processing (the T550 has no tensor cores), in a temporary buffer of about one layer's
cache over the whole context. Both the model and the MTP drafter reserve such a buffer, so together they eat the whole
saving, at any context size (at ctx 32768: 3229 MiB with K+V q8_0,
3243 MiB with f16). A smaller `ubatch-size` hardly changes it (ub 128: 3735 MiB). Without MTP, only the model's
buffer remains, and K+V q8_0 costs ~12 KiB per token instead of 16 KiB, which would allow a context of ~95k tokens.

**Quality** with [`../agx_ai/benchmark/quality_http.py`](../agx_ai/benchmark/quality_http.py): 132 greedy
retrieval questions (single and two-step lookups in Python source) at 9k, 31k and 61k tokens of context, MTP off
(needed for the token probabilities), each compared with the f16 run. The noise floor is a setting that is lossless in
principle (another `ubatch-size`). KLD is the KL divergence of the answer-token distributions from the reference.

| KV cache | Correct | Same answer as f16 | Mean KLD | 99 % KLD | Top token changed |
|---|---|---|---|---|---|
| f16 (reference) | 128/132 | | | | |
| Noise: f16, ub 128 | 128/132 | 131/132 | 0.00056 | 0.015 | 1/852 |
| K+V q8_0 | 129/132 | 129/132 | 0.00073 | 0.028 | 3/842 |
| K+V q4_0 (to show that the test detects a lossy cache) | 126/132 | 127/132 | 0.0046 | 0.147 | 5/826 |

All single lookups were right in every run (96/96); the differences are in the two-step lookups. K+V q8_0 is
slightly above the noise floor (1.3 times the mean KLD) and far below q4_0, i.e. its effect on quality is negligible.

**Speed** with [`../agx_ai/benchmark/bench_http.py`](../agx_ai/benchmark/README.md) at ctx 65536, GPU power limit 20 W
(t/s; "chat" = 12 chat prompts of up to 512 tokens, "30k context" = questions about 30k tokens of code):

| KV cache | MTP | Generation | Prompt, 4096 tokens | Chat | 30k context: prompt / generation | Peak VRAM |
|---|---|---|---|---|---|---|
| f16 | yes | 30.5 | 140 | 24.2 | 104 / **18.7** | 3807 MiB |
| K+V q8_0 | yes | 30.6 | 139 | 28.7 | 103 / 17.0 | 3811 MiB |
| f16 | no | 23.3 | 145 | 21.3 | 108 / 16.3 | 3717 MiB |
| K+V q8_0 | no | 21.7 | 137 | 20.6 | 102 / 12.4 | 3469 MiB |

With MTP, the q8_0 cache is 9 % slower at 30k tokens of context and not faster otherwise (the chat numbers vary with
the draft acceptance). The only way to use its saving, turning MTP off for a ~95k context, would make generation
at 30k tokens of context a third slower (12.4 vs 18.7 t/s).

## Gemma 4 26B-A4B QAT

Measured on 2026-09-28 and -29 with the GPU power limit at 20 W, build b11151. "Chat" = the chat prompts of
[`../agx_ai/benchmark/bench_http.py`](../agx_ai/benchmark/README.md) with up to 512 generated tokens (the realistic
generation speed), "generation" = its raw-completion test, which repeats one paragraph and is therefore much faster
for a mixture-of-experts model (consecutive tokens use the same experts) and flatters MTP (88 % draft acceptance).

### How the hardware is used

The UD-Q4_K_XL GGUF is 13.3 GiB: 12.0 GiB of MoE experts (408 MiB in each of the 30 layers) and 1.3 GiB of everything
else (attention, shared FFN, router, embeddings/output). The MTP drafter is 225 MiB, the vision encoder 1.1 GiB.

- **VRAM (3.8 GiB):** everything but the experts (`n-gpu-layers = all`, `n-cpu-moe = 30`), the KV cache and the MTP
  drafter. At ctx 50176 with an f16 KV cache and ub 1024: weights 1323 MiB, KV cache 980 MiB (global attention)
  + 400 MiB (sliding window, grows with `ubatch-size`), drafter 225 MiB, compute buffers 570 + 220 MiB.
  There is no room for expert layers on the GPU at a useful context (408 MiB each).
- **RAM:** the experts stay memory-mapped from the GGUF file (llama.cpp's `CPU_Mapped` buffer), i.e. in the page cache
  of the WSL 2 VM. The CPU runs them for generation. For prompts, llama.cpp copies them to the GPU for every ubatch
  ("op offload"), so a larger `ubatch-size` makes prompts faster.
- **The WSL 2 VM has 15.5 GiB** (see [Hardware](#hardware-and-software)), and the model needs ~14 GiB of it: the page
  cache holds the whole 13.6 GiB file, plus ~0.8 GiB for the process. Anything more pushes experts out of the page
  cache, and they are re-read from the Windows file system through the slow 9P file share while generating:
  - The vision encoder on the CPU (`no-mmproj-offload = 1`, 1.1 GiB): generation 9.3 t/s instead of 19.7 t/s
    (ub 1024). In Docker, the model therefore runs without it (`no-mmproj = true`, commented out in the preset); use
    the E4B for images and audio, give the VM more memory (below), or run llama.cpp
    [natively on Windows](#natively-on-windows-2026-09-30), where it fits.
  - llama-server's prompt cache in RAM (`cache-ram`, 8 GiB by default): after a few chat requests it had grown to
    1.2 GiB and evicted experts. With `cache-ram = 0`, the single slot still reuses its own KV cache for the next turn of
    the same conversation.

  Windows itself keeps only ~2.5-3.5 GB free while the model is loaded (the VM uses 15.4 GB, the rest ~13 GB).

### Settings

Each variant on a fresh server, the others as in [`preset.ini`](preset.ini) (ctx 50000, f16 KV cache, no mmproj):

| Setting | Chat tg | Generation | Prompt, 4096 tokens | Notes |
|---|---|---|---|---|
| `threads` 4 / 6 / **8** / 12 / 16 | 7.2 / 7.4 / **7.7** / 7.0 / 4.0 | 18.6 / 18.7 / 19.6 / 20.3 / 12.8 | 94 / 87 / 88 / 87 / 88 | 12 threads: 70 t/s for prompts in a rerun with `cache-ram = 0`; 16 threads use the efficiency cores and are much slower |
| MTP off / `spec-draft-n-max` 1 / **2** / 3 | 5.9 / 7.3 / **7.7** / 7.7 | | | draft acceptance in chat 79 / 72 / 63 %; MTP takes 450 MiB of VRAM |
| `no-host = true` (repacked experts), `threads-batch = 12` | 8.9 | | 28 | the GPU cannot use the repacked layout, so prompts run on the CPU; the repacked copy is regular memory (12.5 GiB), which pushed 2 GiB into swap |
| `ubatch-size` 512 / **1024** / 2048 (8192-token prompt) | | | 71 / ~90 / 93 | ub 2048 needs 3897 MiB of VRAM already at ctx 32768 |

The repacked layout (`no-host`) would generate 16 % faster, but prompts are usually much longer than answers
(system prompts, tool definitions, documents), so op offload with 3x faster prompts is the better trade.

### KV cache quantization

Unlike for the E4B, q8_0 K and V caches save VRAM here: at ctx 50176, 3323 MiB instead of 3801 MiB after loading
(the KV cache shrinks from 1380 to 733 MiB, only the MTP drafter's compute buffer grows, by 165 MiB).
Quality with [`../agx_ai/benchmark/quality_http.py`](../agx_ai/benchmark/quality_http.py), 88 greedy retrieval
questions at 9k and 41k tokens of context, MTP off, compared with f16 (see the E4B section for the method):

| KV cache | Correct | Same answer as f16 | Mean KLD | 99 % KLD | Top token changed |
|---|---|---|---|---|---|
| f16 (reference) | 67/88 | | | | |
| Noise: f16, ub 512 | 69/88 | 81/88 | 0.0043 | 0.094 | 7/484 |
| K+V q8_0 | 72/88 | 80/88 | 0.0049 | 0.190 | 8/477 |

q8_0 is within the noise of this model (whose noise floor is ~8 times that of the E4B). All single lookups were right
in all runs but one; the two-step lookups (service -> database -> region, thinking off) are hard for this model at any
setting. Speed at ctx 50000 with MTP: chat 7.8 t/s with both, 16k-token code context pp 71 / 72 t/s and tg 7.1 / 6.9 t/s
(f16 / q8_0). The q8_0 caches are therefore used, and the saved VRAM goes to the context.

### Context size

With q8_0 caches, each token costs ~21 KiB of VRAM, as the compute buffers of both the model and the drafter grow with
the context:

| ctx-size | After loading | Peak | Result |
|---|---|---|---|
| 50176 | 3323 MiB | 3361 MiB | |
| **73728** | 3799 MiB | 3873 MiB | 71k-token retrieval test: 32/32 single and 7/12 two-step lookups; a 70.9k-token prompt: pp 52 t/s, tg 8.3 t/s |
| 90112 | 3926 MiB | | at the limit, and 390 MiB more shared memory: memory moved to shared memory |

### Results

The configuration in [`preset.ini`](preset.ini) (ctx 73728, q8_0 K and V caches, MTP `spec-draft-n-max = 2`,
ub 1024, 8 threads, no vision encoder), through the production server, GPU power limit 20 W, 2026-09-29
(results in `benchmark/results/t550.jsonl`):

| Test | Result |
|---|---|
| Chat | 7.2 t/s (6.3-9.1), acc 72 % |
| 16k context | pp 74 t/s, tg 7.2 t/s, acc 70 % |
| 64k context | pp 51 t/s, tg 7.0 t/s, acc 75 % |
| Prompt processing, 4096 tokens | 93 t/s |
| Generation, 256 tokens (repeated paragraph) | 8.4-18.4 t/s in four runs, acc 84 % |
| Peak GPU memory | 3837 MiB (3873 MiB in the 71k-token retrieval test) |
| Switching models (request to the unloaded model until its answer) | E4B -> 26B-A4B 155 s, 26B-A4B -> E4B 93 s |

The chat and long-context generation speeds were stable at 7-8 t/s over two days, but the repeated-paragraph generation
test varied between 8.4 and 18.4 t/s with these settings (8.4 at 03:50, 18.4 at 04:05), probably due to
background activity on the CPU, which runs the experts. Compared with the E4B, the 26B-A4B generates 4 times slower in chat
(7.2 vs 30.1 t/s) and processes prompts ~1.5 times slower (93 vs 141 t/s for 4096 tokens).

### Using the vision encoder: more memory for the WSL 2 VM

To run the 26B-A4B with its vision encoder, the VM needs about 17-18 GiB (not tested). WSL 2 gives it half of the
RAM by default; `%USERPROFILE%\.wslconfig` sets another limit:

```ini
[wsl2]
memory=18GB
```

Then shut down WSL (this stops Docker Desktop and all WSL distributions) and start Docker Desktop again:

```powershell
wsl --shutdown
```

Windows then has ~13.7 GB left, about what it uses now with the usual background apps, so close those before loading
the 26B-A4B: [`free-ram.ps1`](free-ram.ps1) shows them and their memory use (Slack, Teams, OneDrive, PowerToys,
SyncTrayzor, ...; 2.8 GB on 2026-09-28), and `.\free-ram.ps1 -Stop` closes them. It also prints the command for
stopping the PostgreSQL services. Then replace `no-mmproj = true` with `no-mmproj-offload = 1` in [`preset.ini`](preset.ini).

## Natively on Windows (2026-09-30)

llama.cpp can also run without Docker and WSL 2, with the generic scripts in [`../llama-cpp-windows`](../llama-cpp-windows)
and this computer's configuration in [`windows/`](windows): [`windows/config.ini`](windows/config.ini) (as
[`../llama-cpp/config.ini`](../llama-cpp/config.ini), but with `host = 127.0.0.1`) and the same [`preset.ini`](preset.ini) and `../llama-cpp/llama-cpp.env`
as Docker. This removes the 15.5 GiB memory limit of the WSL 2 VM and the slow file share (see
[Gemma 4 26B-A4B QAT](#gemma-4-26b-a4b-qat)).

```powershell
docker compose stop                                   # in this directory: the GPU and port 9931 are needed
..\llama-cpp-windows\install-llama-cpp.ps1 -Build b12250 -Backend cuda-12.4   # see "llama.cpp update" above
.\windows\start.ps1                                    # runs in the foreground, Ctrl+C stops it
..\llama-cpp-windows\stop-llama-server.ps1           # or stop it from another window
```

- The CUDA 13.4 build of b11262 (the CUDA version of the Docker image) starts with the driver 596.52 (CUDA 13.2) and
  finds the GPU, but fails when loading a model: `CUDA error: the provided PTX was compiled with an unsupported
  toolchain`. The Windows builds have no native Turing kernels, and the driver cannot compile the kernels of a newer
  CUDA version. The CUDA 12.4 build of the same release works. A driver that supports CUDA 13.4 would allow the
  CUDA 13.4 build.
- The `tensor-split = 1` workaround for the MTP drafter ([ggml-org/llama.cpp#29044](https://github.com/ggml-org/llama.cpp/issues/29044))
  is needed natively too: without it, the E4B's drafter fails to load (`invalid vector subscript`, the MSVC wording of
  `vector::_M_range_check`). The cause is not WSL but that the driver reports 0 bytes of free VRAM once the model has
  filled it (the E4B does; the 26B-A4B still leaves some free when its drafter loads, so it loads without the
  workaround). It makes no difference in speed (26B-A4B, 12 chat requests and a 4096-token prompt: 8.6 t/s and
  94.9 t/s with it, 8.0/8.5 t/s and 88.6/90.3 t/s in two runs without it; prompt processing varies by this much
  between runs anyway), so both models keep it.
- The E4B loaded in ~50 s (Docker: 80-120 s). First speed check (build b11262 with CUDA 12.4, ctx 69632, GPU power
  limit 20 W): generation 30.8 t/s, chat 27.2 t/s, prompt processing 143.5 t/s for 4096 tokens, 3851 MiB of VRAM after
  loading and 3861 MiB peak, i.e. about the same as in Docker (36.3 / 30.1 / 141 t/s, 3855 MiB, build b11151), within
  the variation between runs seen so far.

### Gemma 4 26B-A4B QAT natively vs. Docker (2026-09-30)

The same preset as in Docker, except that the vision encoder runs on the CPU natively (`no-mmproj-offload = 1`)
and is left out in Docker (`no-mmproj = true`, see [Gemma 4 26B-A4B QAT](#gemma-4-26b-a4b-qat)). GPU power limit 20 W
in both. Native: build b11262 with CUDA 12.4; Docker: build b11151 with CUDA 13.4 (2026-09-29).

| Test | Native, with the vision encoder | Docker, without it |
|---|---|---|
| Chat | **7.9 t/s** (6.2-10.0), acc 71 % | 7.2 t/s (6.3-9.1), acc 72 % |
| 16k context | pp 75.5 t/s, tg **8.0 t/s** | pp 74 t/s, tg 7.2 t/s |
| 64k context | pp 51.7 t/s, tg **7.3 t/s** | pp 51 t/s, tg 7.0 t/s |
| Prompt processing, 4096 tokens | 91.8 t/s | 92.5 t/s |
| Peak GPU memory | 3833 MiB | 3837 MiB |
| Loading (after the download) | 28-45 s, 15 s when the file is still in Windows' file cache | ~3 min |

Natively, generation is 4-11 % faster, prompts are as fast (they are limited by copying the experts to the GPU), the
model loads 4-6 times faster, and the vision encoder fits in RAM: image input works (a 113-token prompt with a small
image took 40 s). The generation test with the repeated paragraph (7.7 t/s) ran while the model was still being read
from the disk and is not comparable.

RAM: the llama-server process uses ~15.7 GB (working set), and Windows had only ~1 GB free and ~0.8 GB of standby
cache left with the usual background apps and Docker Desktop's idle VM (1.4 GB) running, but the page reads stayed at
~0/s, i.e. nothing that the model needs was paged out. Closing the background apps ([`free-ram.ps1`](free-ram.ps1))
and Docker Desktop gives it ~4 GB more headroom, e.g. for PyCharm.

Other settings, natively (chat with 6 requests, prompt processing with 4096 tokens):

| Setting | Chat | Prompt | Notes |
|---|---|---|---|
| **`threads = 8`, memory-mapped weights** (the preset) | 7.9 t/s | 92 t/s | |
| `threads = 12` / `16` | 7.7 / 7.2 t/s | 90 / 82 t/s | as in WSL 2, 8 threads are the best |
| `load-mode = none` | 8.6 t/s | 95 t/s | the experts are copied to pinned host memory (13.2 GiB of "shared GPU memory"), which Windows cannot page out and which counts against the commit limit: 41.2 of 42.3 GB were committed (the page file had to grow). Too close to the limit for other programs, so not used. |

### Gemma 4 26B-A4B QAT with b12250, natively (2026-11-25)

The same preset, GPU power limit 20 W, with Docker Desktop's idle WSL 2 VM (1.9 GB) and the usual background apps
running:

| Test | b12250 | b11262 (2026-09-30) |
|---|---|---|
| Chat (6 requests) | 6.2 t/s (5.6-7.5), acc 70 % | 7.9 t/s (6.2-10.0), acc 71 % |
| 16k context: pp / tg | **82.1** / 6.0 t/s | 75.5 / 8.0 t/s |
| Prompt processing, 4096 tokens | 86.8 t/s | 91.8 t/s |
| Peak GPU memory | 3867 MiB | 3833 MiB |

RAM was the limit in these runs: the llama-server process had a 16.2 GB working set, 6.8 GB of it private memory
(the rest is the memory-mapped experts), Windows had only 0.5-0.6 GB available, and pages were read from the disk
(~60/s) during generation, i.e. some experts were re-read. The private memory was the same with b11262 (6.4 GB
while loading), so it is not new with the build. Right after the b12250 run, b11262 under the same conditions was
no faster: chat 6.4 t/s (5.3-7.9), acc 71 %, and a 4096-token prompt at 67.4 t/s, i.e. b12250 processes prompts
~29 % faster (the gate/up fusion in MMQ also covers the shared FFN and the expert offload) and generates as fast.
The lower speeds than on 2026-09-30 come from the RAM, not from the build.

To give the 26B-A4B its full speed natively, keep ~2 GB free for the page cache: close the background apps
([`free-ram.ps1`](free-ram.ps1)) and Docker Desktop, or, after using the Docker configuration, at least drop the
page cache of its WSL 2 VM, which keeps the E4B's files (7.1 GB after the Docker E4B benchmarks):

```powershell
docker run --rm --privileged --entrypoint sh ghcr.io/ggml-org/llama.cpp:server-cuda13-b12242 -c "sync; echo 3 > /proc/sys/vm/drop_caches"
```

## Qwen3.5-4B and 9B (2026-11-26)

Natively with b12250, GPU power limit 20 W. Both are Qwen3.5 hybrid models: only every fourth of their 32 layers has
attention with a KV cache (32 KiB per token with f16), the others are Gated DeltaNet with a fixed state (25-50 MiB),
so the context costs little VRAM and generation hardly slows down with it. The GGUFs have no MTP layer, so there is
no speculative decoding, and their 248k-token vocabulary makes the compute buffer ~1 MiB per ubatch token, so both use
`ubatch-size = 128` (which also processed the 4B's prompts faster than 256: 147-151 vs 135 t/s).

### Choosing the quantization

KLD against the Q8_0 of the same model with `llama-perplexity` (6 chunks of 2048 tokens of Python standard-library
source, the second half of each scored), and the generation speed with `llama bench` (tg128, all on the GPU, compared
at similar GPU clocks, as these varied a lot, see [Measurement noise](#measurement-noise-on-this-laptop)):

| Quant (unsloth) | Size | Mean KLD | 99 % KLD | Same top token | PPL | tg |
|---|---|---|---|---|---|---|
| 4B Q8_0 with ub 256 (noise floor) | 4.17 GiB | 0.0005 | 0.003 | 99.3 % | 3.447 | |
| 4B Q4_0 | 2.41 GiB | 0.0329 | 0.268 | 93.7 % | 3.531 | 17-18 t/s at ~600 MHz |
| 4B IQ4_NL | 2.40 GiB | 0.0240 | 0.194 | 94.8 % | 3.480 | 18.7 t/s (one clean run) |
| **4B UD-Q4_K_XL** | 2.71 GiB | 0.0133 | 0.105 | 96.0 % | 3.483 | 13.0-13.6 t/s at ~420 MHz, 15.0 at higher clocks |
| 4B UD-Q5_K_XL | 3.03 GiB | 0.0050 | 0.040 | 97.4 % | 3.454 | 11.4-11.5 t/s at ~400 MHz (-15 %) |
| 9B Q8_0 with ub 256 (noise floor) | 8.87 GiB | 0.0004 | 0.003 | 99.3 % | 3.140 | |
| **9B UD-Q4_K_XL** | 5.56 GiB | 0.0086 | 0.082 | 96.7 % | 3.170 | |
| 9B UD-IQ3_XXS | 3.74 GiB | 0.0796 | 0.776 | 90.3 % | 3.390 | |

- **4B: UD-Q4_K_XL.** Q4_0 and IQ4_NL generate a little faster on this GPU (llama.cpp's Turing MMVQ parameters give
  the K-quants half the warps), but have 2-2.5 times its KLD. UD-Q5_K_XL is 2.6 times closer to Q8_0 but 15 % slower
  and leaves room for ~18k tokens of context instead of 28k. Profiled with Nsight Systems, the 4B's generation is
  entirely in the matrix-vector kernels at ~50 % of the memory bandwidth (the E4B's mostly Q4_0 weights reach ~63 %).
- **9B: UD-Q4_K_XL with 12 layers on the CPU.** Its GPU part alone would be 5.1 GiB, so it does not fit. The largest
  quant that fits entirely on the GPU, UD-IQ3_XXS, has 9 times the KLD (PPL +7.9 %), i.e. it is clearly damaged.
  With 12-14 layers on the CPU, generation is limited by the CPU (the GPU waits at 11-15 W): `llama bench` tg64 with
  18 / 19 / 20 layers on the GPU = 4.1 / 4.5 / 5.3 t/s. 20 layers fit with q8_0 caches and 16k tokens (or with f16
  and 8k). Threads 4 / 6 / 8 / 12 made no clear difference (4.0-4.6 t/s, noisy); 8 as for the 26B-A4B.

### Results

| Test | Qwen3.5-4B UD-Q4_K_XL | Qwen3.5-9B UD-Q4_K_XL |
|---|---|---|
| Chat | 13.3 t/s (12.9-14.1) | 3.5 t/s (3.4-3.6) |
| Prompt processing, 4096 tokens | 140.6 t/s | 66.3 t/s |
| Long context: pp / tg | 22k tokens: 107.7 / 12.6 t/s | 14k tokens: 60.7 / 3.0 t/s |
| Generation, 256 tokens | 14.6 t/s | 3.1 t/s |
| Peak GPU memory | 3710 MiB at ctx 24576; 3835 MiB at ctx 28672 with a 26.5k-token prompt (pp 102 t/s; tg 3.0-8.2 t/s, as the GPU dropped to its clock floor) | 3833 MiB |

The 9B generates slower through the server than with `llama bench` or `llama-completion` (5.0-5.3 t/s with the same
weights, layers, threads, KV types and sampling). The vision encoder is not the cause (3.3 t/s without it).
`prio = 2` (a high process priority for the model process, which the router otherwise starts in the background,
where Windows may throttle it) gave +8 % in one run (3.75-3.92 t/s in chat), so it explains only part of the gap,
and it is not in the preset. The 4B, entirely on the GPU, loses less (13.3 vs ~15 t/s).

Both are slower than the Gemma 4 models here: the E4B generates twice as fast as the 4B thanks to MTP and its Q4_0
weights, and the 26B-A4B (6-8 t/s, with its experts in RAM) is faster than the 9B and much stronger. The Qwen models
are mainly useful as a second opinion from another model family.

## Other findings

- Gemma 4 E4B was chosen over E2B, as it fits with a large context. E2B UD-Q4_K_XL takes only 1482 MiB of VRAM for
  the weights and would allow a larger context, but it is a much weaker model.
- `flash-attn = on` with MTP and the default `ubatch-size` of 512 crashes on prompts longer than a few thousand tokens,
  as the CUDA memory pool cannot grow: `CUDA error: device not ready` in `cuMemSetAccess`
  (`ggml_cuda_pool_vmm::alloc`). `ubatch-size = 256` fixes this. `flash-attn = off` also works, but needs ~230 MiB
  more VRAM (the V cache is padded to 1024 due to the different head sizes) and processed an 8.8k-token prompt at
  52 t/s instead of 84 t/s (both measured with a cold CUDA JIT cache).
- For the E4B, a q8_0 KV cache does not help on this GPU (for the 26B-A4B it does), see [KV cache quantization](#gemma-4-e4b-qat-kv-cache-quantization-2026-09-28).
  With the non-QAT model at ctx 45056, it already used 3927 MiB after loading, memory was moved to shared memory,
  and prompt processing ran at 47 t/s.
- Without `tensor-split = 1`, the MTP drafter fails to load on WSL 2 (and natively, see above)
  (`vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)`), see
  [ggml-org/llama.cpp#29044](https://github.com/ggml-org/llama.cpp/issues/29044).
- The image requires CUDA >= 13.4, but the driver 596.52 provides CUDA 13.2. `NVIDIA_DISABLE_REQUIRE=1` and loading
  the WSL `libcuda` instead of the image's forward-compatibility `libcuda` make it work, see `docker-compose.yml`.
- The image has no native kernels for Turing (sm_75), so the CUDA driver JIT-compiles them on first use. With the
  `cuda-cache` volume, a restart of the container took 121 s instead of 343 s until the model was loaded, and the
  first request 0.8 s instead of 8.6 s.
- Loading takes 1-2 minutes, as the ~5 GB of model and mmproj files are read through the Windows file system.
