# llama.cpp on the Radeon VII

llama-server for the Radeon VII (gfx906, 16 GB HBM2) on agx-z2e, serving Gemma 4 26B-A4B QAT (`UD-Q4_K_XL`)
with MTP speculative decoding. The settings are in [`preset.ini`](preset.ini); this file has the benchmarks behind them.

## Context size and ubatch size (2026-09-26)

Radeon VII (16368 MiB, nothing else on it), `mixa3607/llama.cpp-gfx906:v0.3.0-rocm-7.14`,
[`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py) (256 generated tokens, MTP on, 2 slots),
prompt = ctx-size - 1500 tokens, peak VRAM from `amd-smi`:

| Setup | Generation | Prompt processing | 2 concurrent | Peak VRAM |
|---|---|---|---|---|
| ub 512, ctx 75000 (previous) | - | OOM crash | - | 16344 MiB at load |
| ub 512, ctx 51200 | 127.6 t/s | 923 t/s (49700) | 98.5 t/s | 15997 MiB |
| ub 1024, ctx 32768 | 126.7 t/s | 1192 t/s (31268) | 109.7 t/s | 15988 MiB |
| ub 1024, ctx 45000 | 128.6 t/s | 1068 t/s (43500) | 108.4 t/s | 16277 MiB |
| ub 1024, ctx 49152 | 126.2 t/s | 1032 t/s (47652) | 104.4 t/s | 16353 MiB |
| **ub 1024, ctx 51200 (current)** | 127.3 t/s | 1011 t/s (49700) | 108.6 t/s | 16356 MiB |
| ub 1024, ctx 53248 / 56000 | - | OOM crash | - | - |

ctx 51200 was also stress-tested with 3 full-context prompts and 2 concurrent 24000-token prompts (peak 16357 MiB).

`llama bench` pp2048 with ub 256/512/1024/1536: 1133/1501/1851/1842 t/s
(at depth 16384: 879/1073/1203/1151 t/s).

## MTP draft length (2026-09-27)

Six real chat prompts (code, explanations, translation, maths) twice each, with the preset's sampling and
512 generated tokens. The HTTP benchmark's repeated-paragraph prompt at temperature 0 overstates the draft acceptance.

| `spec-draft-n-max` | 1 | **2 (current)** | 3 | 4 |
|---|---|---|---|---|
| Generation | 120.6 t/s | 123.3 t/s | 118.0 t/s | 114.0 t/s |
| Draft acceptance | 0.78 | 0.69 | 0.61 | 0.56 |

The ranking is the same on Vulkan and on the `-mxxm` image.

## Backend and image comparison (2026-09-27)

The current preset with MTP n-max 2. "Chat" is the chat test above; "depth" is Python source code as context followed
by a question, 512 generated tokens. Generation / prompt processing in t/s:

| Image | Chat | Depth 6.4k | Depth 16k | Depth 36.5k | `llama bench` pp2048 / tg128 |
|---|---|---|---|---|---|
| **gfx906 `v0.3.0-rocm-7.14` (current)** | 123.3 | 109.3 / 1412 | 96.8 / 1299 | 86.6 / 1089 | 1848 / 99.5 |
| gfx906 `b10951-rocm-7.14-mxxm` | 125.6 | - | HTTP 500 | - | 1882 / 102.0 |
| gfx906 `v0.5.0-rocm-7.14` | - | - | - | - | 1763 / 100.8 |
| ggml-org `full-vulkan` b11176 (RADV) | 135.9 | 105.2 / 1143 | 93.0 / 1070 | 77.0 / 865 | 1478 / 106.0 |

- Vulkan is ~10 % faster for short chats but slower from ~6k tokens of context on, and its prompt processing is
  ~20 % slower throughout, so ROCm stays.
- On Vulkan, flash attention is faster (tg128 at depth 16384: 85-89 vs 73 t/s without it), and ub 2048 crashes with
  `vk::DeviceLostError`.
- The `-mxxm` image uses more VRAM (16340 MiB after load) and failed on a 16k prompt at this ctx-size.
