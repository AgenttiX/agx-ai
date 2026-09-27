# llama.cpp on the ThinkPad T480

Runs Gemma 4 26B-A4B QAT with llama.cpp on the ThinkPad T480:
Intel Core i7-8550U (4 cores, 8 threads, Kaby Lake R), Intel UHD Graphics 620, 32 GB DDR4-2400 (dual channel),
NVIDIA GeForce MX150 (2 GB GDDR5, Pascal GP108, PCIe 3.0 x4, driver 580 / CUDA 13.0).
The model runs on the CPU, and the MX150 is used for the attention and for prompt processing.

- [docker-compose.yml](docker-compose.yml): the container
- [config.ini](config.ini): server-wide settings
- [preset.ini](preset.ini): the model and its settings

## Setup
```sh
docker compose up -d
```
- The model is downloaded from Hugging Face on the first start (~15 GB with the vision encoder).
- The image is `server-cuda` (CUDA 12.8), not `server-cuda13`: CUDA 13 dropped Pascal, and the NVIDIA 580 driver,
  the last one that supports the MX150, provides only CUDA 13.0.
- The image has no native kernels for the MX150 (sm_61), so the CUDA driver JIT-compiles them from PTX on first use.
  They are cached in the `cuda-cache` volume (~50 MB), so only the first start after an image update is slow.
- Loading takes ~1.5 min. The llama-server process uses ~15.6 GB of RAM (14 GB of it the memory-mapped weights)
  and ~1.7-1.8 GB of VRAM, so only one model fits at a time (`models-max = 1` in [config.ini](config.ini)).
- The MX150 must not be used by other programs (e.g. games through PRIME offload) while the model is loaded.

## When this configuration is faster
The configuration trades generation speed for prompt processing speed:
compared to running the model with the repacked (AVX2-optimized) weights on the CPU,
prompts are processed ~3.7x faster, but answers are generated ~25-30 % slower.
**This setup is faster overall when the new prompt tokens of a request are at least about 1.1x the answer length**
(measured with a 5.8k-token document; ~1.3x for short chats, see the derivation below).
This is the case for most real use: system prompts, tool definitions, documents and web search results
are usually much longer than the answer. With multi-turn chats, only the new tokens of each turn count,
as the server reuses the cached prompt.

For mostly short chats without long prompts, the alternative is to add these to the model in [preset.ini](preset.ini):
```ini
no-host = true
threads = 8
```
(the last row of the [comparison table](#chat-requests-through-the-server)).

Derivation from the table below (seconds per token = 1 / speed):
- After the 5.8k-token document: generating takes 1/5.1 - 1/6.6 = 0.045 s more per answer token,
  and the prompt takes 1/18.0 - 1/66.2 = 0.040 s less per prompt token, so the break-even is at 0.045/0.040 = ~1.1.
- Short chat: 1/5.4 - 1/7.6 = 0.054 s more per answer token vs. the same 0.040 s less per prompt token: ~1.3.
- Compared to running on the CPU only: ~1.4x for short chats (1/5.4 - 1/8.3 = 0.065 vs. 1/16.1 - 1/66.2 = 0.047),
  and faster in any case after a long prompt, as the CPU attention slows down with the context (see below).

## How the hardware is used
Gemma 4 26B-A4B has only ~4B active parameters per token, so on this CPU it generates faster than the much weaker E4B
and ~2.7x faster than the 12B (llama bench, CPU image, UD-Q4_K_XL, -fa 1):

| Model | pp512 (4 / 8 threads) | tg64 (4 / 8 threads) |
|---|---|---|
| E4B | 18.7 / 20.1 | 7.8 / 7.6 |
| 12B | 6.9 / 7.2 | 3.2 / 3.2 |
| 26B-A4B | 12 (noisy) / 19.9 | 7.4 / 8.7 |

The dense 31B (17.3 GB) would generate only ~1.5 t/s (estimated from the 12B).
The 26B-A4B weights (13.3 GiB) do not fit in the 2 GB of the MX150, so they are split as follows:

- **Q/K/V attention projections (381 MiB) and the KV cache in VRAM.** The CPU attention is slow with Gemma 4's large
  heads, so generation on the CPU slowed down by ~40 % already at a few thousand tokens of context
  (all weights in RAM: tg 6.7 t/s at depth 0, 4.1 at 2048, 3.3 at 6144; a q8_0 KV cache or flash attention off
  did not help).
- **All other weights in RAM**, in the CUDA host buffer type (the default for tensors overridden to the CPU),
  which keeps them memory-mapped from the GGUF file in their original layout (~14 GB of page cache).
  For batches of >= 32 tokens (prompt processing), llama.cpp copies them over PCIe to the MX150 and runs the
  matrix multiplications there ("op offload"), which is 3-4x faster than the CPU. The copy takes most of the time,
  so a larger `ubatch-size` is faster (pp2048 with all weights in RAM: 60 t/s with ub 512, 79 t/s with 1024,
  and 2048 did not fit in VRAM).
- **The cost:** the CPU cannot use the repacked (AVX2-optimized) Q4_0 layout for these weights, so generation is
  ~30 % slower than on the plain CPU. `no-host = true` bypasses the host buffer and repacks the weights into a copy
  in RAM, but the MX150 cannot use the repacked layout, so the prompt then runs on the CPU as well.
  The non-repacked kernels are fastest with 4 threads (tg128 6.6 t/s vs. 5.9 with 6 or 8), whereas the prompt
  processing and the repacked CPU kernels prefer 8.
- **No MTP:** speculative decoding with the MTP drafter (`spec-type = draft-mtp`, `spec-draft-n-max = 2`) does not pay
  off on this CPU, as verifying the draft tokens costs about as much as it saves (MTP vs. none, draft acceptance
  58-71 %): CPU only 7.3 vs. 8.3 t/s, all weights in RAM (host buffer) 4.9 vs. 4.7 t/s with a short prompt,
  but 3.0 vs. 3.4 t/s after a long one.

## Performance
Measured on 2026-09-27, build b11176 (tg = token generation, pp = prompt processing, in t/s).

### Chat requests through the server
Thinking off. "Short" = 41-token question and a ~260-token answer,
"doc" = 5777-token document with a question and a 300-token answer (the time includes both).

| Configuration | Short tg | Doc pp | Doc tg | Doc time | VRAM |
|---|---|---|---|---|---|
| CPU only (`device = none`), 8 threads | 8.3 | 16.1 | 5.0 | 412 s | - |
| All weights in host buffer (`n-gpu-layers = 0`), ub 1024 | 5.8 | 64.6 | 3.5 | 174 s | < 1.5 GiB |
| Attention on the MX150, rest in host buffer, ub 512 | 5.6 | 50.6 | 5.2 | 172 s | 1765 MiB (ctx 32768) |
| **Attention on the MX150, rest in host buffer, ub 1024 (this configuration)** | 5.4 | 66.2 | 5.1 | 145 s | 1797 MiB (ctx 24576) |
| Attention on the MX150, rest repacked (`no-host`), 8 threads | 7.6 | 18.0 | 6.6 | 367 s | 1549 MiB (ctx 32768) |

[benchmark/llama_cpp_bench_http.py](../benchmark/README.md) with this configuration
(results in `benchmark/results/agx-t480-kubuntu.jsonl`): tg 6.7 t/s (256 tokens, short prompt),
pp 79.8 t/s (4096 tokens), peak VRAM 1797 MiB.

Image input works (the vision encoder runs on the CPU): a 103-token prompt with a 256x256 image took 13 s.
Two unload/reload cycles through the router API freed and reallocated the VRAM cleanly.

### Context size
VRAM with ub 1024 and an f16 KV cache (the MX150 has 1958 MiB free for CUDA):

| ctx-size | VRAM | Notes |
|---|---|---|
| 16384 | 1621 MiB | |
| 24576 | 1797 MiB peak | A 22.7k-token prompt: pp 47.5 t/s (8 min), tg 5.0 t/s, found a password hidden in the middle |
| 32768 | - | Needs ub 512 (1765 MiB), which makes prompts ~25 % slower |

A q8_0 KV cache would allow more context, but made generation ~10 % slower (ctx 32768, ub 512: short tg 5.1 t/s).

### Other findings
- The UHD Graphics 620 iGPU (Vulkan, `server-vulkan` image, all layers, -fa 0) is slower than this configuration
  in both respects: pp512 25.6 t/s, tg64 4.6 t/s. It shares the memory bandwidth with the CPU, so it cannot add
  generation speed, and mixing the CPU and Vulkan in one model was very slow on the ThinkPad L14
  ([../llama-cpp-agx-l14](../llama-cpp-agx-l14)).
- The whole non-expert part (attention, shared FFN, router, output matrix: 1323 MiB) does not fit in VRAM with
  useful compute buffers: with the MoE experts on the CPU (`n-cpu-moe`), ub 512 ran out of memory, and ub 256 gave
  pp 45 t/s and tg 6.9 t/s (8 threads, experts in the host buffer) or pp 25 t/s and tg 8.4 t/s (`no-host`),
  but left no VRAM for the context.
- Speeds vary by ~10 % between runs, and the i7-8550U runs at ~1.4-2.4 GHz and 70 °C under sustained load
  (power profile "balanced").
