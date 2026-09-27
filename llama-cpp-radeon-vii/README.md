# llama.cpp on the Radeon VII

llama-server for the Radeon VII (gfx906, 16 GB HBM2) on agx-z2e, serving Gemma 4 26B-A4B QAT (`UD-Q4_K_XL`)
with MTP speculative decoding. The image is `mixa3607/llama.cpp-gfx906:v0.5.0-rocm-7.14` (see
[`docker-compose.yml`](docker-compose.yml)), and the settings are in [`preset.ini`](preset.ini). This file has the
benchmarks behind them. The tools are in [`../benchmark`](../benchmark); the commands are at the end.

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
512 generated tokens (the `chat` test of `llama_cpp_bench_http.py`). The default generation test's repeated-paragraph
prompt at temperature 0 overstates the draft acceptance. Image v0.3.0.

| `spec-draft-n-max` | 1 | **2 (current)** | 3 | 4 |
|---|---|---|---|---|
| Generation | 120.6 t/s | 123.3 t/s | 118.0 t/s | 114.0 t/s |
| Draft acceptance | 0.78 | 0.69 | 0.61 | 0.56 |

The ranking is the same on Vulkan and on the `-mxxm` image.

## Backend and image comparison (2026-09-27)

The current preset with MTP n-max 2. "Chat" is the chat test above; "depth" is Python source code as context followed
by a question, 512 generated tokens (the `depth` test). Generation / prompt processing in t/s:

| Image | Chat | Depth 6.4k | Depth 16k | Depth 36.5k | `llama bench` pp2048 / tg128 |
|---|---|---|---|---|---|
| gfx906 `v0.3.0-rocm-7.14` (previous) | 123.3 | 109.3 / 1412 | 96.8 / 1299 | 86.6 / 1089 | 1848 / 99.5 |
| gfx906 `b10951-rocm-7.14-mxxm` | 125.6 | - | HTTP 500 | - | 1882 / 102.0 |
| **gfx906 `v0.5.0-rocm-7.14` (current)** | see [below](#image-update-to-v050-2026-09-27) | | | | 1763 / 100.8 |
| ggml-org `full-vulkan` b11176 (RADV) | 135.9 | 105.2 / 1143 | 93.0 / 1070 | 77.0 / 865 | 1478 / 106.0 |

- Vulkan is ~10 % faster for short chats but slower from ~6k tokens of context on, and its prompt processing is
  ~20 % slower throughout, so ROCm stays.
- On Vulkan, flash attention is faster (tg128 at depth 16384: 85-89 vs 73 t/s without it), and ub 2048 crashes with
  `vk::DeviceLostError`.
- The `-mxxm` image uses more VRAM (16340 MiB after load) and failed on a 16k prompt at this ctx-size.

## Image update to v0.5.0 (2026-09-27)

Both images with the current preset, measured back to back with the same `llama_cpp_bench_http.py` run (commands
below). Generation in t/s, prompt processing in parentheses; draft acceptance is given for the chat test:

| Test | `v0.3.0-rocm-7.14` | **`v0.5.0-rocm-7.14` (current)** |
|---|---|---|
| Generation, 256 tokens (repeated text, temperature 0) | 127.1 | 138.2 |
| Prompt processing, 49700 tokens (full context) | (1052) | (1030) |
| Chat, 12 requests | 122.7, acceptance 69 % | 129.8, acceptance 70 % |
| Code context of 6432 tokens | 111.2 (1450) | 114.9 (1430) |
| Code context of 16032 tokens | 96.7 (1369) | 101.8 (1348) |
| Code context of 36532 tokens | 86.7 (1132) | 90.6 (1115) |
| 2 concurrent x 24000-token prompts, per request | 108.4-113.7 | 110.4-111.0 |
| Peak VRAM (amd-smi) | 16256 MiB | 16324 MiB |
| VRAM after loading | 15992 MiB | 16080 MiB |
| `llama bench` pp2048 / tg128 | 1848 / 99.5 | 1768 / 101.1 |
| `llama bench` pp2048 / tg128 at depth 16384 | 1197 / 89.0 | 1167 / 90.0 |

v0.5.0 generates 4-6 % faster with MTP at every depth, and its prompt processing is 1-2 % slower. The generation gain
is larger than in `llama bench` (+1-2 %), which has no speculative decoding. v0.5.0 handles the full-context prompt,
but it uses ~70-90 MiB more VRAM, which leaves only 44 MiB free at the peak; if it runs out of memory, lower `ctx-size`.

## Commands

```sh
cd ../benchmark
# Serving speed: generation, full-context prompt, chat, depth, 2 concurrent long prompts
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt concurrency chat depth --prompt-tokens 49700 --concurrency 2 \
    --concurrent-prompt-tokens 24000 --repeat 2 --depth 6400 16000 36500 --label "..."
# Raw model speed in the running container
./llama_cpp_bench_container.py --container llama-cpp-radeon-vii --url http://localhost:9932 \
    --env-file ../llama-cpp-radeon-vii/llama-cpp.env --label "..." -- -ngl 999 -fa 1 -ub 1024 -p 2048 -n 128 -d 0,16384 -r 3
# Raw model speed with another image or backend
./llama_cpp_bench_container.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \
    --image ghcr.io/ggml-org/llama.cpp:full-vulkan --label "..." -- -ngl 999 -fa 1 -ub 1024 -p 2048 -n 128 -d 0,16384
# Serving speed with another image or a preset variant (stop the production container first)
docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml stop
./llama_cpp_test_server.sh ghcr.io/ggml-org/llama.cpp:full-vulkan ../llama-cpp-radeon-vii/preset.ini \
    ../llama-cpp-agx-ai/config.ini ../llama-cpp-radeon-vii/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 2 --depth 6400 16000 36500 --label "..."
docker rm -f llama-test && docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml start
```
