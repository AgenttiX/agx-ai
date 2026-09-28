# llama.cpp on the NVIDIA T550 Laptop GPU

Configuration for the NVIDIA T550 Laptop GPU (4 GB, Turing TU117, 30 W) under Docker Desktop with WSL 2 on Windows.
The model and its settings are in [`preset.ini`](preset.ini), server-wide settings in [`config.ini`](config.ini)
and the CUDA/WSL workarounds in [`docker-compose.yml`](docker-compose.yml).
The benchmark results of this computer are saved with the hostname `t550` (`--hostname t550` for the scripts in
[`../benchmark`](../benchmark/README.md)).

Model: Gemma 4 E4B QAT (`unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL`) with its MTP drafter,
flash attention, an f16 KV cache, `ctx-size = 69632`, a single slot and the vision/audio encoder (mmproj) on the CPU.

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

## Results

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

## VRAM budget

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

## Context size

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

## KV cache quantization (2026-09-28)

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

## Other findings

- Gemma 4 E4B was chosen over E2B, as it fits with a large context. E2B UD-Q4_K_XL takes only 1482 MiB of VRAM for
  the weights and would allow a larger context, but it is a much weaker model.
- `flash-attn = on` with MTP and the default `ubatch-size` of 512 crashes on prompts longer than a few thousand tokens,
  as the CUDA memory pool cannot grow: `CUDA error: device not ready` in `cuMemSetAccess`
  (`ggml_cuda_pool_vmm::alloc`). `ubatch-size = 256` fixes this. `flash-attn = off` also works, but needs ~230 MiB
  more VRAM (the V cache is padded to 1024 due to the different head sizes) and processed an 8.8k-token prompt at
  52 t/s instead of 84 t/s (both measured with a cold CUDA JIT cache).
- A q8_0 KV cache does not help on this GPU, see [KV cache quantization](#kv-cache-quantization-2026-09-28).
  With the non-QAT model at ctx 45056, it already used 3927 MiB after loading, memory was moved to shared memory,
  and prompt processing ran at 47 t/s.
- Without `tensor-split = 1`, the MTP drafter fails to load on WSL 2
  (`vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)`), see
  [ggml-org/llama.cpp#29044](https://github.com/ggml-org/llama.cpp/issues/29044).
- The image requires CUDA >= 13.4, but the driver 596.52 provides CUDA 13.2. `NVIDIA_DISABLE_REQUIRE=1` and loading
  the WSL `libcuda` instead of the image's forward-compatibility `libcuda` make it work, see `docker-compose.yml`.
- The image has no native kernels for Turing (sm_75), so the CUDA driver JIT-compiles them on first use. With the
  `cuda-cache` volume, a restart of the container took 121 s instead of 343 s until the model was loaded, and the
  first request 0.8 s instead of 8.6 s.
- Loading takes 1-2 minutes, as the ~5 GB of model and mmproj files are read through the Windows file system.
