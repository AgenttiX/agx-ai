# llama.cpp on the NVIDIA T550 Laptop GPU

Configuration for the NVIDIA T550 Laptop GPU (4 GB, Turing TU117, 30 W) under Docker Desktop with WSL 2 on Windows.
The model and its settings are in [`preset.ini`](preset.ini), server-wide settings in [`config.ini`](config.ini)
and the CUDA/WSL workarounds in [`docker-compose.yml`](docker-compose.yml).

Model: Gemma 4 E4B QAT (`unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL`) with its MTP drafter,
flash attention, `ctx-size = 65536`, a single slot and the vision/audio encoder (mmproj) on the CPU.

## Results

Measured with [`benchmark/llama_cpp_bench_http.py`](../benchmark/README.md) (results in
`benchmark/results/t550.jsonl`), llama.cpp build b11151.
tg = token generation, pp = prompt processing, acc = accepted MTP draft tokens.

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
| QAT | **65536** | 62.1k tokens | 3787 MiB | 3801 MiB | stable; two unload/reload cycles through the router API |
| non-QAT | 10000 | 8.8k tokens | | 3565 MiB | stable |
| non-QAT | 24576 | 22.9k tokens | 3677 MiB | 3815 MiB | stable; two unload/reload cycles through the router API |
| non-QAT | 32768 | 29.8k tokens | 3813 MiB | 3941 MiB | works, but at the ~3940 MiB limit, leaving no margin for other programs that use the GPU |

For the QAT model, 65536 keeps the same ~140 MiB margin below the limit as ctx 24576 did for the non-QAT model.
Each further 4096 tokens would cost 64 MiB.

## Other findings

- Gemma 4 E4B was chosen over E2B, as it fits with a large context. E2B UD-Q4_K_XL takes only 1482 MiB of VRAM for
  the weights and would allow a larger context, but it is a much weaker model.
- `flash-attn = on` with MTP and the default `ubatch-size` of 512 crashes on prompts longer than a few thousand tokens,
  as the CUDA memory pool cannot grow: `CUDA error: device not ready` in `cuMemSetAccess`
  (`ggml_cuda_pool_vmm::alloc`). `ubatch-size = 256` fixes this. `flash-attn = off` also works, but needs ~230 MiB
  more VRAM (the V cache is padded to 1024 due to the different head sizes) and processed an 8.8k-token prompt at
  52 t/s instead of 84 t/s (both measured with a cold CUDA JIT cache).
- A q8_0 KV cache (`cache-type-k/v = q8_0`) does not help: with flash attention, the quantized cache needs f16
  conversion buffers, so it saves much less than half of the KV cache. The non-QAT model at ctx 45056 already used
  3927 MiB after loading, memory was moved to shared memory, and prompt processing ran at 47 t/s.
- Without `tensor-split = 1`, the MTP drafter fails to load on WSL 2
  (`vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)`), see
  [ggml-org/llama.cpp#29044](https://github.com/ggml-org/llama.cpp/issues/29044).
- The image requires CUDA >= 13.4, but the driver 596.52 provides CUDA 13.2. `NVIDIA_DISABLE_REQUIRE=1` and loading
  the WSL `libcuda` instead of the image's forward-compatibility `libcuda` make it work, see `docker-compose.yml`.
- The image has no native kernels for Turing (sm_75), so the CUDA driver JIT-compiles them on first use. With the
  `cuda-cache` volume, a restart of the container took 121 s instead of 343 s until the model was loaded, and the
  first request 0.8 s instead of 8.6 s.
- Loading takes 1-2 minutes, as the ~5 GB of model and mmproj files are read through the Windows file system.
