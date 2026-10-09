# llama.cpp on agx-z2e (RTX 3090)

llama-server on the RTX 3090 of agx-z2e, Mika's desktop computer. The Radeon VII in the same computer runs its own
server, see [`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii).

[`docker-compose.yml`](docker-compose.yml) runs `ghcr.io/ggml-org/llama.cpp:server-cuda13` (build b10689) on port 9931
as a router with one of the presets below, which is selected by commenting the volume lines in or out. It uses the
server-wide settings of [`../llama-cpp-agx-ai/config.ini`](../llama-cpp-agx-ai/config.ini), which is identical to
[`config.ini`](config.ini) here, and the API key of [`../llama-cpp-radeon-vii/llama-cpp.env`](../llama-cpp-radeon-vii).

All models run entirely on the GPU with flash attention and MTP speculative decoding. The context of each one is as
large as possible while llama-server stays at ~21 GiB of VRAM (peak 21520-21556 MiB in the benchmarks with
full-context prompts), so that ≥1 GiB of the 24 GiB stays free with the desktop session using ~1.9 GiB.

| Preset | Model | KV cache | Context | Chat generation | Prompt processing |
|---|---|---|---|---|---|
| [`preset-qwen3.8-27b.ini`](preset-qwen3.8-27b.ini) (default) | Qwen3.8-27B `UD-Q4_K_M` | q8_0 | 102400 | 49 t/s, 36 t/s at 60k, 26 t/s at 100k | 1140 t/s at 16k, 775 t/s at 100k |
| [`preset-qwen3.8-27b-uncensored.ini`](preset-qwen3.8-27b-uncensored.ini) | Qwen3.8-27B Uncensored `IQ4_XS` | q8_0 | 118784 | 55 t/s, 38 t/s at 60k, 26 t/s at 116k | 1190 t/s at 16k, 750 t/s at 116k |
| [`preset-gemma-4-26b-a4b-qat.ini`](preset-gemma-4-26b-a4b-qat.ini) | Gemma 4 26B-A4B QAT `UD-Q4_K_XL`, 2 slots | f16 | 165888 | 216 t/s, 148 t/s at 70k, 104 t/s at 163k | 4450 t/s at 16k, 2000 t/s at 163k |
| [`preset-gemma-4-31b-qat.ini`](preset-gemma-4-31b-qat.ini) | Gemma 4 31B QAT `UD-Q4_K_XL` | q8_0 | 61440 | 77 t/s, 48 t/s at 59k | 1020 t/s at 16k, 750 t/s at 59k |
| [`preset-gemma-4-31b-qat-uncensored.ini`](preset-gemma-4-31b-qat-uncensored.ini) | Gemma 4 31B QAT Uncensored `Q4_K_M` | q8_0 | 38912 | 51 t/s, 40 t/s at 36k | 920 t/s at 16k, 790 t/s at 36k |

The vision encoder (mmproj) is on the GPU for the Qwen models and Gemma 4 26B-A4B and on the CPU
(`no-mmproj-offload = 1`) for the Gemma 4 31B models. Each preset has the context size for the other choice as a
comment; see [Vision encoder on the CPU](#vision-encoder-on-the-cpu-no-mmproj-offload).

The rest of this file has the measurements behind the settings. They were made with the tools in
[`../benchmark`](../benchmark/README.md) on 2026-10-09 unless noted otherwise, with test servers from
[`../benchmark/llama_cpp_test_server.sh`](../benchmark/llama_cpp_test_server.sh) and the production container stopped.
The results are in `../benchmark/results/agx-z2e-kubuntu*.jsonl` (labels starting with "z2e").
VRAM figures are those of the llama-server process (`nvidia-smi --query-compute-apps`), i.e. without the desktop.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-38-generic), KDE Plasma desktop (Wayland) on the RTX 3090 |
| CPU | AMD Ryzen Threadripper 3970X (Zen 2, 32 cores, 64 threads) |
| RAM | 128 GB DDR4-2666 (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPU | NVIDIA GeForce RTX 3090, which also drives the desktop (~1.2-2.0 GiB of VRAM, ~1.9 GiB on 2026-10-09); driver 610.43.02 (CUDA 13.3), power limit 350 W (default) |
| GPU architecture | Ampere (GA102, sm_86), PCIe 4.0 x16 |
| VRAM | 24 GB GDDR6X (24576 MiB) |
| Other GPUs | AMD Radeon VII ([`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii)), NVIDIA GeForce GTX TITAN (Kepler, not supported by the driver, unused) |

## Memory

VRAM of llama-server after loading (the peak with a full-context prompt is only 20-75 MiB higher on CUDA), with
ub 512 and MTP on, as a base and the cost of each setting:

| | Qwen3.8-27B | Qwen3.8-27B Uncensored | Gemma 4 26B-A4B | Gemma 4 31B | Gemma 4 31B Uncensored |
|---|---|---|---|---|---|
| Weights (GGUF) | 15.3 GiB | 14.6 GiB | 13.3 GiB | 16.1 GiB | 17.4 GiB |
| Base: K+V q8_0, mmproj on the GPU | 18496 MiB at 32k | 17770 MiB at 32k | 17088 MiB at 64k | 20108 MiB at 16k | 21444 MiB at 16k |
| Per 1024 tokens of context, K+V f16 | 69 MiB | 69 MiB | 22 MiB | ~120 MiB | ~120 MiB |
| Per 1024 tokens of context, K+V q8_0 | 43 MiB | 43 MiB | 21 MiB | 61 MiB | 61 MiB |
| Per 1024 tokens of context, K+V q4_0 | 27 MiB | 27 MiB | 16 MiB | 41 MiB | 41 MiB |
| Vision encoder (mmproj) on the GPU | 1138 MiB | 1138 MiB | 1292 MiB | 1298 MiB | 1298 MiB |
| MTP drafter | 896 MiB (f16) | 814 MiB | 596 MiB | 486 MiB | 486 MiB |
| ub 256 / 1024 / 2048 instead of 512 | -122 / +242 / +766 MiB | -122 / +242 / +766 MiB | -144 / +288 / +868 MiB | -206 / +480 / +1514 MiB | -206 / - / - |

- The Qwen models have a KV cache in only 16 of their 64 layers (the others are Gated DeltaNet layers with a
  fixed-size state), and the Gemma models in only the global-attention layers (5 of 30 or 10 of 60; the others are
  sliding-window layers of 1024 tokens). All five have 262144 tokens of trained context.
- A quantized KV cache saves less than its size suggests: the CUDA flash attention converts the quantized cache of a
  layer back to f16 in a temporary buffer for prompt processing (and for the multi-token batches of MTP), and the
  model and the MTP drafter each reserve such a buffer, which grows with the context. With Gemma 4 26B-A4B, whose
  KV cache is small, a q8_0 cache therefore saves only 7 % per token.
- With an f16 cache, Gemma 4 31B fits only up to ~20k tokens (21106 MiB at 16k), and the uncensored one not even at
  16k.

## KV cache type

### Supported types

`llama bench` with Qwen3.8-27B (`ghcr.io/ggml-org/llama.cpp:full-cuda13-b10689`, the build of the server image), t/s:

| K / V | pp512 / tg32 | pp2048 / tg64 at depth 16384 |
|---|---|---|
| f16 / f16 | 1298 / 41.9 | 1129 / 40.1 |
| bf16 / bf16 | 1286 / 41.6 | |
| q8_0 / q8_0 | | 1112 / 37.9 |
| q4_0 / q4_0 | | 1111 / 37.0 |
| q8_0 / f16, f16 / q8_0 | 926 / 39.7, 820 / 39.4 | |
| q8_0 / q4_0 | | 17 / 7.4 |
| q4_1, q5_0, q5_1, iq4_nl (K = V) | 818, 487, 461, 362 / 39-40 | |

This build's CUDA flash attention handles only equal f16 (or bf16), q8_0 and q4_0 K and V types efficiently
(apparently it was not built with `GGML_CUDA_FA_ALL_QUANTS`). Other types and mixed pairs fall back to the CPU for parts of the attention, which is
catastrophic at long contexts (q8_0/q4_0: 17 t/s prompt processing at depth 16384). The candidates are therefore
f16, q8_0 and q4_0.

### Speed

Qwen3.8-27B at ctx 65536 (the largest context where f16 fits), ub 512, MTP, with
`llama_cpp_bench_http.py --tests generation prompt chat depth`. Means of two runs; generation in t/s, prompt
processing in parentheses:

| KV cache | Chat | Depth 6.4k | Depth 16k | Depth 36.5k | Depth 60k | Prompt 60000 | VRAM |
|---|---|---|---|---|---|---|---|
| f16 | 49.0 | 55.5 (1154) | 51.6 (1133) | 46.3 (1022) | 43.5 (913) | 934 t/s | 21606 MiB |
| q8_0 | 48.2 (-2 %) | 52.5 (-5 %) | 50.1 (-3 %) | 41.9 (-10 %) | 36.8 (-15 %) | 913 t/s | 19902 MiB |
| q4_0 | 48.6 (-1 %) | 53.2 (-4 %) | 46.7 (-9 %) | 42.4 (-8 %) | 36.9 (-15 %) | 911 t/s | 18878 MiB |

The two runs agreed within 1-4 %. A quantized cache slows down generation more the longer the context is, as
reading it needs dequantization (and the f16 conversion of the whole cache for MTP's multi-token verification
batches); prompt processing is only 2-3 % slower. q4_0 is no faster than q8_0. For Gemma 4 26B-A4B, the earlier
measurement in its preset showed the same: tg128 at depth 16384 122 t/s with q8_0 vs 142 t/s with f16.

### Quality

With [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py), the method of
[`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii/README.md#entirely-on-the-gpu-with-a-q8_0-v-cache-2026-09-28):
greedy retrieval questions (single and two-step lookups of facts hidden in Python source) with the top-20 token
probabilities of every answer token, on test servers without MTP (drafted tokens have no probabilities) and with the
vision encoder on the CPU, each compared with an f16 cache. The noise floor is a different ubatch size, which is
lossless in principle and only reorders floating-point operations. KLD is the KL divergence of the answer-token
distributions from the f16 run, "first token" the KLD of the first answer token, and "top token changed" the
positions where the most likely token differs.

| Model (contexts of the questions) | Configuration | Correct | Same answer | Mean KLD | 99 % KLD | First token KLD | Top token changed |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (9k/33k/61k/76k, ctx 77824) | f16 (reference) | 166/176 | - | - | - | - | - |
| | Noise: ub 256 / ub 1024 | 165 / 164 | 175 / 174 | 0.00075 / 0.00097 | 0.012 / 0.020 | 0.0027 / 0.0044 | 1 / 2 |
| | **q8_0** | 166 | 174 | **0.00051** | 0.017 | 0.0019 | 2 |
| | q4_0 | 164 | 171 | 0.00190 | 0.043 | 0.0082 | 5 |
| Qwen3.8-27B Uncensored (same) | f16 (reference) | 165/176 | - | - | - | - | - |
| | Noise: ub 256 / ub 1024 | 166 / 164 | 175 / 175 | 0.00038 / 0.00034 | 0.008 / 0.008 | 0.0012 / 0.0014 | 1 / 1 |
| | **q8_0** | 164 | 175 | **0.00029** | 0.007 | 0.0011 | 1 |
| | q4_0 | 165 | 174 | 0.00215 | 0.066 | 0.0087 | 2 |
| Gemma 4 26B-A4B (same, 1 slot) | f16 (reference) | 145/176 | - | - | - | - | - |
| | Noise: ub 256 / ub 1024 | 145 / 145 | 166 / 167 | 0.0028 / 0.0024 | 0.067 / 0.063 | 0.011 / 0.008 | 10 / 9 |
| | q8_0 | 147 | 159 | 0.0041 | 0.095 | 0.013 | 17 |
| | q4_0 | 145 | 166 | 0.0195 | 0.372 | 0.081 | 10 |
| Gemma 4 31B (9k/17k/29k, ctx 30720) | f16 (reference) | 128/132 | - | - | - | - | - |
| | Noise: ub 256 / ub 1024 | 128 / 128 | 132 / 132 | 0.00047 / 0.00021 | 0.014 / 0.005 | 0.0014 / 0.0005 | 0 / 0 |
| | **q8_0** | 128 | 132 | **0.00058** | 0.019 | 0.0023 | 0 |
| | q4_0 | 130 | 130 | 0.00694 | 0.036 | 0.039 | 2 |
| Gemma 4 31B Uncensored (9k/21k, ctx 22528) | f16 (reference) | 86/88 | - | - | - | - | - |
| | Noise: ub 256 / ub 1024 | 86 / 86 | 88 / 88 | 0.00041 / 0.00027 | 0.005 / 0.003 | 0.0011 / 0.0015 | 0 / 0 |
| | **q8_0** | 86 | 88 | **0.00029** | 0.007 | 0.0012 | 0 |
| | q4_0 | 88 | 86 | 0.00183 | 0.048 | 0.0093 | 2 |

The Gemma 4 31B runs use shorter contexts because the f16 reference has to fit on the GPU. All single lookups were
right in every run except one or two of Gemma 4 26B-A4B's 128 (in the noise runs too); the differences in the
"Correct" column are in the two-step lookups, which are hard for the models with thinking off.

- **q8_0 is within the noise for every model**, except for Gemma 4 26B-A4B, where it is slightly above it
  (1.5 times the noise KLD, 17 vs 9-10 changed top tokens). The same comparison on the Radeon VII put this model's
  q8_0 cache within the noise, so the effect is marginal at most.
- **q4_0 is clearly above the noise for every model**: 2-6 times the noise KLD for the Qwen models, 4-30 times for
  the Gemma models, and the first answer token, where the model decides what it retrieved, is affected the most.
  The retrieval accuracy is nevertheless unchanged, so the loss is small, but it is measurable.

### Choice

- **Qwen3.8-27B (both): q8_0.** The quality is unaffected, and the context grows from ~64k (f16) to 102400
  (118784 for the smaller uncensored GGUF). The price is slower generation at long contexts: 2-5 % up to 16k tokens,
  10 % at 36k and 15 % at 60k. The f16 cache could not reach those lengths anyway beyond 64k. For mostly short
  conversations where 64k suffices, `cache-type-k/v = f16` with `ctx-size = 63488` would be 3-15 % faster.
- **Gemma 4 26B-A4B: f16.** q8_0 saves only 7 % of the memory per token (~10k tokens of context), is 14 % slower
  and slightly above the noise. q4_0 saves 30 % but is clearly lossy.
- **Gemma 4 31B (both): q8_0**, which was already used so that all layers fit on the GPU; it is within the noise.

## ubatch size

`llama bench` pp2048 with the KV cache type of the preset, t/s at depth 0 / depth 16384:

| ub | Qwen3.8-27B | Qwen3.8-27B Uncensored | Gemma 4 26B-A4B | Gemma 4 31B | Gemma 4 31B Uncensored (q4_0) |
|---|---|---|---|---|---|
| 256 | 1252 / 1047 | 1325 / 1105 | 3224 / 2696 | 1219 / 905 | 1139 / 856 |
| 512 | 1293 / 1080 | 1371 / 1135 | 4459 / 3442 | 1237 / 908 | 1157 / 860 |
| 1024 | 1309 / 1117 | 1383 / 1170 | 5459 / 3918 | 1279 / 955 | 1199 / 906 |
| 2048 | 1317 / 1132 | 1388 / 1184 | 5972 / 4204 | 1289 / 970 | 1199 / 915 |

For the dense models, 1024 is only 1-5 % faster than 512 and costs 240-480 MiB (6-8k tokens of context), so they
use 512 (Gemma 4 31B used 1024 before). For the mixture-of-experts Gemma 4 26B-A4B, larger batches matter: 2048 is
22-34 % faster than 512 for 868 MiB (~40k tokens of context), so it keeps 2048. The `batch-size` stays at the
default 2048.

## MTP draft length

Chat test (12 requests of up to 512 tokens) and depth test at 16k tokens, generation in t/s:

| `spec-draft-n-max` | Qwen3.8-27B | Qwen3.8-27B Uncensored | Gemma 4 26B-A4B | Gemma 4 31B | Gemma 4 31B Uncensored |
|---|---|---|---|---|---|
| 1 | 50.2 / 46.4 | 53.9 / 50.4 | 211.0 / 172.3 | 56.1 / 45.3 | 45.2 / 36.6 |
| 2 | **49.5 / 51.6** | **55.7 / 53.1** | **219.4 / 177.2** | 64.2 / 57.5 | 50.6 / 43.6 |
| 3 | 46.7 / 49.2 | 54.0 / 52.0 | 215.9 / 165.7 | 67.7 / 62.5 | **51.0 / 44.8** |
| 4 | | | | **77.9 / 67.5** | 51.7 / 43.6 |
| 5 | | | | 70.9 / 62.2 | 49.9 / 41.9 |

A repeated comparison of 3 and 4 for Gemma 4 31B (two runs each, alternating) confirmed 4: chat 70.9-74.6 vs 78.3-78.4
t/s, depth 16k 62.5-62.8 vs 68.2 t/s, depth 36k 54.0-54.1 vs 57.1-57.3 t/s. The Gemma 4 31B drafter is accepted
more often than the others (64 % of 3 drafted tokens vs 44 % for Qwen), so longer drafts pay off for it. For the
uncensored Gemma 4 31B, 2-4 are within ~2 %; it uses 3. The others use 2.

## Vision encoder on the CPU (`no-mmproj-offload`)

The vision encoder takes 1138 MiB (Qwen) or ~1295 MiB (Gemma) of VRAM. On the CPU, it frees that for context:

| Model | Context with the encoder on the GPU | On the CPU | Gain |
|---|---|---|---|
| Qwen3.8-27B | **102400** | 129024 | +26 % |
| Qwen3.8-27B Uncensored | **118784** | 145408 | +22 % |
| Gemma 4 26B-A4B | **165888** | 212992 | +28 % |
| Gemma 4 31B | 38912 | **61440** | +58 % |
| Gemma 4 31B Uncensored | 16384 | **38912** | +138 % |

(Bold: the preset's choice.) The cost is the time to encode an image, measured with the `image` test of
[`llama_cpp_bench_http.py`](../benchmark/README.md) (a 1280x960 PNG, mean of 2-3 requests):

| Model | Prompt tokens | Prompt processing incl. the image, encoder on the GPU | On the CPU |
|---|---|---|---|
| Qwen3.8-27B | 1261 | 1.8 s | 6.7 s |
| Gemma 4 26B-A4B | 291 | 1.7 s | 4.2 s |
| Gemma 4 31B | 291 | 0.8 s | 3.4 s |

On the CPU, each image takes 3-5 s longer (with the CPU otherwise idle; longer when it is busy). For the Gemma 4 31B
models, the context is otherwise too small, so they keep the encoder on the CPU, as before. For the others, the
context is large already, and the encoder stays on the GPU for fast image input; the presets have the larger
`ctx-size` as a comment.

## Context size

Stress test: load, then one prompt of ctx-size - 1500 tokens (for Gemma 4 26B-A4B with its 2 slots also two
concurrent prompts of half that), peak VRAM of llama-server. The target is a peak of at most ~21500 MiB.

| Model | Setting | ctx-size | VRAM after loading | Peak | Prompt processing |
|---|---|---|---|---|---|
| Qwen3.8-27B | q8_0, mmproj GPU | 90112 / 94208 / 98304 | 20930 / 21106 / 21282 MiB | 20954 / 21130 / 21306 MiB | 810 / 792 / 779 t/s |
| | q8_0, mmproj CPU | 118784 / 122880 / 126976 | 21024 / 21200 / 21376 MiB | 21048 / 21224 / 21400 MiB | 720 / 710 / 700 t/s |
| Qwen3.8-27B Uncensored | q8_0, mmproj GPU | 106496 / 110592 / 114688 | 20908 / 21084 / 21260 MiB | 20944 / 21120 / 21296 MiB | 779 / 765 / 754 t/s |
| | q8_0, mmproj CPU | 135168 / 139264 / 143360 | 21002 / 21178 / 21354 MiB | 21038 / 21214 / 21390 MiB | 699 / 689 / 678 t/s |
| Gemma 4 26B-A4B | f16, ub 2048, mmproj GPU | 147456 / 155648 / 163840 | 20880 / 21104 / 21328 MiB | 20954 / 21178 / 21402 MiB | 2166 / 2096 / 2025 t/s |
| | f16, ub 2048, mmproj CPU | 212992 / 221184 / 229376 | 21380 / 21604 / 21828 MiB | 21454 / 21678 / 21902 MiB | 1689 / 1644 / 1602 t/s |
| Gemma 4 31B | q8_0, mmproj CPU | 49152 / 53248 / 57344 | 20746 / 20988 / 21230 MiB | 20764 / 21008 / 21248 MiB | 815 / 784 / 763 t/s |
| | q8_0, mmproj GPU | 28672 / 32768 / 36864 | 20834 / 21076 / 21318 MiB | 20854 / 21096 / 21338 MiB | 941 / 915 / 886 t/s |
| Gemma 4 31B Uncensored | q8_0, mmproj CPU | 28672 / 30720 / 32768 | 20872 / 20994 / 21114 MiB | 20892 / 21014 / 21132 MiB | 895 / 882 / 869 t/s |
| | q8_0, mmproj GPU | 16384 / 18432 | 21444 / 21566 MiB | 21464 / 21586 MiB | 974 / 964 t/s |

The context sizes of the presets are extrapolated from these by the memory per token and confirmed by the final
benchmarks below (peaks 21520-21556 MiB). With the desktop using 1935 MiB, that leaves 1085-1121 MiB free. The
desktop's share varies (1.2-2.2 GiB was seen); if it grows beyond ~2 GiB, less than 1 GiB remains, but llama-server
does not need more than its peak.

Gemma 4 26B-A4B has 2 slots with a unified KV cache (`kv-unified` in `config.ini`): both slots share the 165888 tokens,
so two long conversations at the same time cannot both be kept in VRAM. When a slot switches to a conversation saved
in the host RAM prompt cache (`cache-ram`) while the other slot holds a long context, restoring it fails ("failed to
find N available cells in kv cache") and the prompt is processed again; this is logged as an error but harmless.

## Results of the current presets

Each preset as it is, on a test server, with
`llama_cpp_bench_http.py --tests generation prompt chat depth image --repeat 2` (prompt = ctx-size - 1500 tokens).
Generation in t/s, prompt processing in parentheses; "generation" is the raw-completion test with repeated text at
temperature 0, which flatters MTP, "chat" the realistic one:

| Test | Qwen3.8-27B | Qwen3.8-27B Uncensored | Gemma 4 26B-A4B | Gemma 4 31B | Gemma 4 31B Uncensored |
|---|---|---|---|---|---|
| ctx-size | 102400 | 118784 | 165888 | 61440 | 38912 |
| Generation, 256 tokens | 61.3 | 72.3 | 210.3 | 76.9 | 52.1 |
| Chat, 12 requests (range) | 49.2 (38-56) | 54.8 (44-68) | 216.3 (197-241) | 76.9 (63-103) | 51.2 (43-62) |
| Draft acceptance (chat) | 54 % | 53 % | 71 % | 59 % | 67 % |
| Depth 6.4k | 54.7 (1156) | 62.2 (1212) | 183.1 (4410) | 72.7 (1060) | 48.3 (983) |
| Depth 16k | 51.9 (1137) | 53.5 (1191) | 175.7 (4449) | 68.1 (1022) | 43.9 (923) |
| Depth 36.5k | 41.7 (1029) | 46.9 (1073) | 169.1 (3903) | 55.8 (876) | 39.7 (793) |
| Depth 60k (59k for Gemma 4 31B) | 36.2 (917) | 37.7 (954) | | 48.4 (749) | |
| Depth 70k | | | 148.4 (3127) | | |
| Depth 100k / 116k / 120k | 25.9 (776) | 26.3 (751) | 121.3 (2405) | | |
| Depth 163k | | | 103.5 (2008) | | |
| Full-context prompt | 773 t/s (100900) | 749 t/s (117284) | 2027 t/s (164388) | 758 t/s (59940) | 836 t/s (37412) |
| 2 concurrent requests | | | 231 t/s in total, 154-156 each | | |
| Image 1280x960, prompt processing | 1.8 s | 1.8 s | 1.7 s | 3.4 s (CPU) | 3.5 s (CPU) |
| Peak VRAM of llama-server | 21556 MiB | 21542 MiB | 21532 MiB | 21520 MiB | 21526 MiB |

No errors other than the harmless prompt cache message above were logged.

## Earlier results (from the presets)

These were in the preset files before 2026-10-09. RTX 3090 with the desktop session using ~1.2 GiB, build b10689,
`benchmark/llama_cpp_bench_http.py` (256 generated tokens, MTP on); the peak VRAM includes the desktop.

### Gemma 4 26B-A4B QAT (2026-09-26)

2 slots. With ub 2048, ctx 200000 peaked at 23.7 of 24.6 GiB, which was too tight with a desktop; 163840 was used.

| Setup | Generation | Prompt 45k | 2 concurrent | Peak VRAM |
|---|---|---|---|---|
| ub 512, ctx 200000 | 196.0 t/s | 2973 t/s | 184 t/s | 21767 MiB |
| ub 1024, ctx 200000 | 193.5 t/s | 3291 t/s | 204 t/s | 22358 MiB |
| ub 2048, ctx 200000 | 194.6 t/s | 3523 t/s | 206 t/s | 23690 MiB |
| ub 2048, ctx 163840 (used until 2026-10-09) | 194.0 t/s | - | 204 t/s | 22780 MiB |
| ub 2048, ctx 131072 | 195.6 t/s | 3564 t/s | 205 t/s | 21860 MiB |
| ub 2048, ctx 100000 | 196.1 t/s | 3604 t/s | 206 t/s | 21022 MiB |

The KV cache takes ~26 MiB per 1000 tokens of context. With ctx 163840, a 150000-token prompt was processed at
1997 t/s without running out of memory (the peak VRAM above is from that run). Flash attention off was 14-22 % slower
in prompt processing and 33 % slower in generation at depth 16384. A q8_0 KV cache was slower (tg128 at depth 16384:
122 vs 142 t/s).

### Gemma 4 31B QAT (2026-09-25/26)

| Setup | Generation | Prompt 16k | Prompt 45k | Peak VRAM |
|---|---|---|---|---|
| ngl 53, f16 KV (the preset before) | 29.1 t/s | 801 t/s | - | 21505 MiB |
| ngl all, q8_0 KV, ub 512 | 63.1 t/s | 971 t/s | 773 t/s | 22146 MiB |
| ngl all, q8_0 KV, ub 1024 (used until 2026-10-09, ctx 50000) | 62.1 t/s | - | 817 t/s | 22620 MiB |
| ngl all, q8_0 KV, ub 2048 | 62.4 t/s | - | 845 t/s | 23645 MiB |

With an f16 KV cache, only 53 of the 60 layers fit; the q8_0 cache lets all of them fit, which doubles the generation
speed. ub 1024 sped up prompt processing by 6 % vs 512 for +530 MiB; 2048 by 9 % (3 % vs 1024) but peaked at
23.6 of 24.6 GiB. `llama bench`, ngl 53, pp2048/tg128 at depth 16384: flash attention on 660/13.2 t/s, off
530/11.6 t/s (off was 20-50 % slower).

### Gemma 4 31B QAT Uncensored (2026-09-26)

| Setup | Generation | Prompt | Peak VRAM |
|---|---|---|---|
| ngl 50, f16 KV, ctx 50000 (the preset before) | 24.1 t/s | 677 t/s 16k | 22360 MiB |
| ngl 57, q8_0 KV, ctx 50000 | 35.9 t/s | 654 t/s 45k | 22834 MiB |
| ngl all, q8_0 KV, ctx 50000 | 49.7 t/s | 728 t/s 45k | 23872 MiB |
| ngl all, q8_0 KV, ctx 32768 (used until 2026-10-09) | 48.5 t/s | 780 t/s 30k | 22983 MiB |

With all layers and q8_0 KV, ctx 50000 peaked at 23.9 of 24.6 GiB, which was too tight with a desktop.

### Qwen3.8-27B (2026-09-26)

Measured with 2 slots (the preset has had 1 slot since 2026-09-27), ctx 50000, f16 KV:

| Setup | Generation | Prompt 45k | 2 concurrent | Peak VRAM |
|---|---|---|---|---|
| ub 512 (used until 2026-10-09) | 63.8 t/s | 919 t/s | 63.3 t/s | 22331 MiB |
| ub 1024 | 62.3 t/s | 943 t/s | 63.1 t/s | 22689 MiB |

ub 1024 gave +2.6 % prompt processing (within the noise) for +360 MiB, so it was not used. `llama bench`
pp2048/tg128 at depth 16384: flash attention on 1031/36.8 t/s, auto 1003/33.6 t/s, off 827/25.7 t/s (off was 20 %
slower in prompt processing and 30 % slower in generation).

## Commands

```sh
cd ../benchmark
# Stop the production server, then start a test server on port 9934 with a preset or a variant of it
docker compose -f ../llama-cpp-agx-z2e/docker-compose.yml stop
NAME=llama-test-cuda DOCKER_ARGS="--gpus all" ./llama_cpp_test_server.sh ghcr.io/ggml-org/llama.cpp:server-cuda13 \
    ../llama-cpp-agx-z2e/preset-qwen3.8-27b.ini ../llama-cpp-agx-ai/config.ini ../llama-cpp-radeon-vii/llama-cpp.env 9934
# Serving speed with a full-context prompt, chat, depth and an image
./llama_cpp_bench_http.py --url http://localhost:9934 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth image --prompt-tokens 100900 --repeat 2 --depth 6400 16000 36500 60000 100000 \
    --label "..."
# Output quality of a KV cache type (on test servers without MTP, see ../benchmark/README.md)
./llama_cpp_quality_http.py --url http://localhost:9934 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \
    --tests retrieval --label "..." --reference "z2e Qwen3.8-27B quality f16 ub 512 (reference)"
docker rm -f llama-test-cuda && docker compose -f ../llama-cpp-agx-z2e/docker-compose.yml start
# Raw model speed (llama bench) of a GGUF with the same build
./llama_cpp_bench_container.py --image ghcr.io/ggml-org/llama.cpp:full-cuda13-b10689 --docker-arg=--gpus=all \
    --model /root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/.../Qwen3.8-27B-UD-Q4_K_M.gguf \
    --keep-loaded --label "..." -- -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -ub 256,512,1024,2048 -p 2048 -n 0 -d 0,16384
```
