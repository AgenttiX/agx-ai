# llama.cpp on the NVIDIA T550 Laptop GPU

Configuration for the NVIDIA T550 Laptop GPU (4 GB, Turing TU117, 30 W) under Docker Desktop with WSL 2 on Windows.
The model and its settings are in [`preset.ini`](preset.ini), server-wide settings in [`config.ini`](config.ini)
and the CUDA/WSL workarounds in [`docker-compose.yml`](docker-compose.yml).
The benchmark results of this computer are saved with the hostname `t550` (`--hostname t550` for the scripts in
[`../benchmark`](../benchmark/README.md)).

Two models, one loaded at a time (`models-max = 1` in [`config.ini`](config.ini); requesting the other model unloads
the loaded one):

- `google/gemma-4-e4b-qat`, loaded on request: Gemma 4 E4B QAT (`unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL`),
  entirely on the GPU, with its MTP drafter, flash attention, an f16 KV cache, `ctx-size = 69632`, a single slot
  and the vision/audio encoder (mmproj) on the CPU.
- `google/gemma-4-26b-a4b-qat`, the default model, loaded at startup: Gemma 4 26B-A4B QAT
  (`unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL`), with the MoE experts in RAM and everything else on the GPU, MTP,
  q8_0 K and V caches, `ctx-size = 73728`, a single slot and the vision encoder on the CPU (see
  [Gemma 4 26B-A4B QAT](#gemma-4-26b-a4b-qat)). Slower but much stronger than the E4B.

[`preset.ini`](preset.ini) is set up for running llama.cpp [natively on Windows](#natively-on-windows-2026-09-30),
which is faster and has room for the 26B-A4B's vision encoder. Where Docker/WSL 2 needs different settings, the
Docker/WSL 2 settings are kept in the preset as comments, marked "Docker/WSL 2"; uncomment them (and comment out the
native ones next to them) to use the preset with Docker.

## Hardware and software

| | |
|---|---|
| OS | Windows 11 25H2 (build 26200), Docker Desktop 4.88.1 with the WSL 2 backend (kernel 6.18) |
| CPU | Intel Core i7-1260P (Alder Lake, 4 performance + 8 efficiency cores, 16 threads) |
| RAM | 32 GB DDR4-3200 (2 x 16 GB SO-DIMM, dual channel); the WSL 2 VM gets 15.5 GiB of it (WSL's default of half the RAM) and 4 GiB of swap |
| iGPU | Intel Iris Xe Graphics (96 EU, shares the system RAM), drives the display |
| dGPU | NVIDIA T550 Laptop GPU: Turing TU117 (sm_75, no tensor cores), 4 GB GDDR6 (64-bit), PCIe link x4 (nvidia-smi: gen 4, max width x16), driver 596.52 (CUDA 13.2) |
| dGPU power limit | 30 W (default and maximum). On 2026-09-27 and -28 the platform capped it at 20 W ("Current Power Limit" in `nvidia-smi -q -d POWER`, while "Requested Power Limit" was 30 W), which made the GPU ~35 % slower (see the results of 2026-09-27 below). |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda13`, build b11151 |

## Gemma 4 E4B QAT: results

Measured with [`benchmark/llama_cpp_bench_http.py`](../benchmark/README.md) (results in
`benchmark/results/t550.jsonl`), llama.cpp build b11151.
tg = token generation, pp = prompt processing, acc = accepted MTP draft tokens.

### Gemma 4 E4B QAT UD-Q4_K_XL, ctx 69632, f16 KV cache, MTP `spec-draft-n-max = 2` (current, 2026-09-28)

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

**Quality** with [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py): 132 greedy
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

**Speed** with [`../benchmark/llama_cpp_bench_http.py`](../benchmark/README.md) at ctx 65536, GPU power limit 20 W
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
[`../benchmark/llama_cpp_bench_http.py`](../benchmark/README.md) with up to 512 generated tokens (the realistic
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
Quality with [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py), 88 greedy retrieval
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
[`config.ini`](config.ini), but with `host = 127.0.0.1`) and the same [`preset.ini`](preset.ini) and `llama-cpp.env`
as Docker. This removes the 15.5 GiB memory limit of the WSL 2 VM and the slow file share (see
[Gemma 4 26B-A4B QAT](#gemma-4-26b-a4b-qat)).

```powershell
docker compose stop                                   # in this directory: the GPU and port 9931 are needed
..\llama-cpp-windows\install-llama-cpp.ps1 -Backend cuda-12.4
.\windows\start.ps1                                    # runs in the foreground, Ctrl+C stops it
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
