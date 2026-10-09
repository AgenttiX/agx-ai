# llama.cpp on the Radeon VII

llama-server for the Radeon VII (gfx906, 16 GB HBM2) on agx-z2e. The image is
`mixa3607/llama.cpp-gfx906:v0.6.0-rocm-7.14` (see [`docker-compose.yml`](docker-compose.yml), where the model is
selected). There are two presets, one model at a time, both entirely on the GPU with MTP speculative decoding and
q8_0 K and V caches:

| Preset | Model | Context | Chat generation | Prompt processing |
|---|---|---|---|---|
| [`preset-gemma-4-26b-a4b-qat.ini`](preset-gemma-4-26b-a4b-qat.ini) (default) | Gemma 4 26B-A4B QAT `UD-Q4_K_XL` | 88064 | ~114 t/s, 54-78 t/s at 36-85k tokens of context | ~1180 t/s at 16k, ~740 t/s at 85k |
| [`preset-qwen3.8-27b.ini`](preset-qwen3.8-27b.ini) | Qwen3.8-27B `UD-IQ4_XS` | 47104 | ~30 t/s, ~24 t/s at 30-43k tokens of context | ~240 t/s at 16k, ~210 t/s at 45k |

The Gemma context is larger than the 77824 of [`../llama-cpp-agx-ai`](../llama-cpp-agx-ai), so that the same agentic
workloads can be split between the two. Qwen3.8-27B is a dense model, so it is 4 times slower than the
mixture-of-experts Gemma; see [Qwen3.8-27B](#qwen38-27b-2026-10-09). This file has the benchmarks behind the
settings. The tools are in [`../benchmark`](../benchmark); the commands are at the end.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-38-generic) |
| CPU | AMD Ryzen Threadripper 3970X (32 cores, 64 threads) |
| RAM | 128 GB DDR4-2666 (8 x 16 GB Kingston 9965745-002.A00G unbuffered DIMMs, ECC), 125.6 GiB visible to the OS |
| NPU | none |
| GPU | AMD Radeon VII, used only for this server (the RTX 3090 in the same machine runs [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e)) |
| GPU architecture | GCN 5.1 (Vega 20, gfx906), 60 CUs, PCIe 3.0 x16 |
| VRAM | 16 GB HBM2 (16368 MiB) |

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
| ub 1024, ctx 51200 (used until 2026-09-28) | 127.3 t/s | 1011 t/s (49700) | 108.6 t/s | 16356 MiB |
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

Both images with the preset of that time (ctx 51200, ub 1024, no CPU offload), measured back to back with the same `llama_cpp_bench_http.py` run (commands
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

## Context size 77824 (2026-09-28)

`ctx-size = 77824` needs ~620 MiB more VRAM than 51200 (the KV cache grows by ~240 MiB per 10240 tokens) and does not
load with the previous settings ("failed to create MTP context"). The variants below free memory by quantizing the KV
cache (`cache-type-k`/`cache-type-v = q8_0`), by moving the experts of the first N layers to the CPU (`n-cpu-moe`), or
with a smaller `ubatch-size` (smaller compute buffers). Each was loaded with `llama_cpp_test_server.sh`, stress-tested
with one 76324-token prompt, and then measured with the `chat` and `depth` tests. Generation in t/s, prompt processing
in parentheses; the first row is the previous configuration at ctx 51200:

| Variant | Chat | Depth 6.4k | Depth 16k | Depth 36.5k | Depth 70k | Peak VRAM |
|---|---|---|---|---|---|---|
| ctx 51200, ub 1024 (previous) | 129.8 | 114.9 (1430) | 101.8 (1348) | 90.6 (1115) | - | 16324 MiB |
| f16 KV, ub 1024 | - | - | - | - | - | does not load |
| K q8_0, ub 512 | 117.1 | 103.9 (1290) | 93.3 (1214) | 79.8 (1035) | 66.5 (823) | 16225 MiB |
| K q8_0, ub 768 | 117.6 | 106.3 (1368) | 94.7 (1278) | 81.6 (1080) | 66.9 (850) | 16359 MiB |
| V q8_0, ub 512 (used on 2026-09-28) | 121.7 | 106.2 (1282) | 96.2 (1209) | 77.9 (1030) | 68.0 (818) | 16257 MiB |
| K+V q8_0, ub 512 | 112.5 | 98.8 (1268) | 89.3 (1186) | 77.3 (1011) | 65.0 (806) | 16085 MiB |
| K+V q8_0, ub 1024 | crashed during the tests | | | | | 16331 MiB |
| n-cpu-moe 2, ub 1024 | 106.5 | 96.3 (1296) | 90.7 (1236) | 80.1 (1037) | 66.8 (801) | 16135 MiB |
| n-cpu-moe 1, K q8_0, ub 1024 | 110.2 | 95.4 (1364) | 86.2 (1283) | 79.2 (1072) | 62.4 (835) | 16322 MiB |
| n-cpu-moe 1, ub 512 (used on 2026-09-28 night) | 120.6 | 108.6 (1159) | 97.6 (1101) | 85.0 (953) | 72.2 (768) | 16118 MiB |
| n-cpu-moe 1, ub 512, 16 threads | 116.8 | 108.0 (1158) | 95.8 (1100) | 84.6 (953) | 72.0 (768) | 16125 MiB |

- One layer's experts on the CPU with an f16 KV cache generates fastest at every depth from 6.4k on, and it has the
  most headroom of the variants that fit. A q8_0 KV cache costs more the longer the context is (77-80 vs 85 t/s at
  36.5k), as the quantized cache is slower to read in flash attention. The price is ~10 % slower prompt processing
  than with a q8_0 cache at the same ub 512, probably because the CPU layer's expert weights are copied to the GPU for
  every ubatch.
- Compared with ctx 51200, the extra context costs ~4-8 % in generation at the same depths and ~15-20 % in prompt
  processing (ub 512 and the CPU offload).
- Offloading more layers (n-cpu-moe 2) slows short-context generation by ~12 %; halving the CPU threads (default 32)
  does not help.
- Variants with a peak within ~50 MiB of the 16368 MiB total are too close to crashing (K+V q8_0 with ub 1024 did).
- `load-mode = dio` works with `n-cpu-moe` (no mmap warning, unlike mmap on agx-ai).

The production container with n-cpu-moe 1 and ub 512 (image v0.5.0), measured with the full command below: generation
(256 tokens) 121.4 t/s, two 76324-token prompts at 784 t/s, chat 117.0 t/s, depth 6.4k/16k/36.5k/70k
105.9/95.6/85.2/70.9 t/s (prompt 1157/1103/953/768 t/s), peak 16122 MiB, no errors in the log. The chat test
varies by ~±3 % between runs.

## Entirely on the GPU with a q8_0 V cache (2026-09-28)

This computer is also the desktop where the code written by the agents runs, so the CPU is often busy. With
`n-cpu-moe = 1`, the default 32 threads take up to 26 % of the 64 hardware threads during generation (`llama bench`
tg128 at depth 16384, sampled from `/proc/stat`; 3 % with everything on the GPU). The configuration is therefore the
fastest one that runs entirely on the GPU at ctx 77824: a q8_0 V cache with ub 512 (see the table above). Mixed K/V
types run on the GPU too (this build's `FA_QUANTS` lists only equal pairs, but the host CPU stays at ~3 %).

Quality, with [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py): each configuration on a
fresh test server at ctx 77824 (MTP on, as served), 176 greedy retrieval questions at 9k/33k/61k/76k tokens of
context (32 single lookups and 12 two-step lookups each), compared with the f16 reference:

| Configuration | Correct | Same answer as the reference | Greedy outputs identical (12) | Mean common prefix |
|---|---|---|---|---|
| f16, n-cpu-moe 1, ub 512 (reference) | 146/176 | - | - | - |
| Same, on a new server | 146/176 | 176/176 | 12 | 1.00 |
| Noise: f16, ub 256 | 143/176 | 160/176 | 6 | 0.56 |
| Noise: f16, n-cpu-moe 2 | 144/176 | 167/176 | 0 | 0.09 |
| V q8_0, all on GPU | 146/176 | 169/176 | 0 | 0.08 |
| K q8_0, all on GPU | 144/176 | 167/176 | 0 | 0.08 |
| K+V q8_0, all on GPU | 146/176 | 166/176 | 0 | 0.06 |
| K+V q4_0, all on GPU | 145/176 | 163/176 | 0 | 0.08 |

All single lookups were correct in every configuration; the two-step lookups (thinking off) are hard for the model
(3-6 of 12 per context length) and make up all the differences. None of the quantized caches is worse than the
noise of settings that are lossless in principle, but neither is q4_0, so this test is not sensitive enough to
separate them; see the token-probability comparison below. Greedy outputs diverge after any numerical change
(0 of 12 identical already with n-cpu-moe 2), so they do not measure quality either.

The more sensitive comparison: the same 176 questions with the top-20 token probabilities of every answer token
(`--logprobs`), on test servers without MTP (drafted tokens have no probabilities; this does not change the main
model's distributions), all on the GPU, compared with the f16 cache position by position while the answers agree.
KLD is the KL divergence of the answer-token distributions from the reference; "first token" is the first answer
token, where the model decides what it retrieved; "Δ log p" is the change of the log-probability of the reference's
token:

| Configuration (ctx 77824, ub 512, no MTP) | Correct | Same answer | Mean KLD | 99 % KLD | First token KLD | Top token changed | Mean \|Δ log p\| |
|---|---|---|---|---|---|---|---|
| f16 (reference) | 144/176 | - | - | - | - | - | - |
| Noise: f16, ub 256 | 146/176 | 166/176 | 0.0102 | 0.181 | 0.034 | 10/998 | 0.0200 |
| Noise: f16, ub 1024 | 147/176 | 162/176 | 0.0080 | 0.198 | 0.027 | 14/974 | 0.0184 |
| V q8_0 | 146/176 | 166/176 | 0.0064 | 0.158 | 0.023 | 10/996 | 0.0149 |
| K q8_0 | 144/176 | 169/176 | 0.0057 | 0.103 | 0.021 | 7/1002 | 0.0150 |
| K+V q8_0 | 146/176 | 164/176 | 0.0084 | 0.182 | 0.030 | 12/988 | 0.0178 |
| K+V q4_0 | 145/176 | 164/176 | 0.0216 | 0.597 | 0.084 | 12/974 | 0.0315 |

q8_0 K and V caches, alone or together, change the answer distributions no more than a different ubatch size does
(which is lossless in principle and only reorders floating-point operations): they are within the noise. The q4_0
cache is clearly above it (2-3 times the KLD, 3 times the 99th percentile), which shows that the comparison can detect
a lossy cache; its retrieval accuracy is nevertheless unchanged. The effect of q8_0 on quality is therefore
negligible here.

`llama-perplexity`'s KL divergence ([`../benchmark/llama_cpp_kld.py`](../benchmark/llama_cpp_kld.py)) is not usable
for this model: on raw text (Python source, also wrapped in the chat template) its perplexity is ~290-370, and a
different ubatch size or flash attention off changes 30 % of the top tokens (mean KLD 0.36), the same as a q8_0 cache.
The same happens with the upstream Vulkan build, so it is the model, not the gfx906 build.

The production container with this configuration, measured with the full command below: generation (256 tokens)
144.1 t/s, two 76324-token prompts at 821 t/s, chat 122.8 t/s, depth 6.4k/16k/36.5k/70k 107.7/96.4/78.3/68.1 t/s
(prompt 1282/1208/1029/818 t/s), peak 16266 MiB, no errors in the log. Compared with n-cpu-moe 1 and an f16
cache, generation is about the same up to 16k tokens and 4-8 % slower at 36-70k, and prompt processing is 7-11 %
faster.

### With a busy CPU

`llama_cpp_bench_http.py --tests chat depth --repeat 1 --depth 16000 60000` on a test server, once with an idle CPU
and once while `stress-ng --cpu 64 --cpu-method matrixprod` keeps all 64 hardware threads busy (at normal priority,
like a compile or test run of the agents' code). Generation in t/s, prompt processing in parentheses:

| Configuration | CPU | Chat | Depth 16k | Depth 60k |
|---|---|---|---|---|
| ctx 88064, K+V q8_0, all on GPU (current) | idle | 114.0 | 90.8 (1183) | 65.5 (856) |
| | 64 threads busy | 108.6 (-5 %) | 86.8 (1177) (-4 %) | 62.7 (854) (-4 %) |
| ctx 77824, f16, n-cpu-moe 1 | idle | 122.3 | 99.2 (1101) | 76.1 (815) |
| | 64 threads busy | **27.5 (-78 %)** | **26.2 (1080) (-74 %)** | **25.1 (809) (-67 %)** |

With experts on the CPU, generation collapses to a quarter when the CPU is busy: the default 32 threads have to
synchronize for every token and are preempted by the other work. Entirely on the GPU, only the CPU's share of driving
the GPU and sampling slows down (4-5 %). Prompt processing is unaffected in both cases.

## Context size 88064 with q8_0 K and V caches (2026-09-28)

As the q8_0 K cache does not degrade quality either (above), both caches are q8_0 and the context is as large as the
memory allows. Stress test: load, then one prompt of ctx-size - 1500 tokens (MTP on, as served):

| ctx-size | VRAM after loading | Prompt processing | Peak VRAM |
|---|---|---|---|
| 77824 | 15907 MiB | 805 t/s | 16053 MiB |
| 86016 | 16073 MiB | 765 t/s | 16222 MiB |
| **88064 (current)** | 16113 MiB | 756 t/s | 16262 MiB |
| 90112 | 16155 MiB | 747 t/s | 16301 MiB |
| 94208 | 16237 MiB | 730 t/s | 16331 MiB |
| 98304 / 102400 | 16319 / 16323 MiB | crashes | - |

Memory grows by ~20 MiB per 1024 tokens of context, only a little less than the ~24 MiB with f16 caches (at ctx
30720-51200), so most of the growth is not in the K and V caches themselves; a q8_0 cache for the MTP drafter
(`cache-type-k-draft`/`cache-type-v-draft`) does not change it. 88064 keeps the peak of the
previous configuration, whose full benchmark ran without errors; 90112 and 94208 survive the stress prompt but with
less headroom (a configuration that peaked at 16331 MiB in the stress test crashed later in the benchmarks).

The production container with this configuration, measured with the full command below (prompt 86564 tokens and
depth up to 85k): generation (256 tokens) 124.9 t/s, two 86564-token prompts at 759 t/s, chat 112.8 t/s, depth
6.4k/16k/36.5k/70k/85k 100.5/90.0/77.8/66.1/53.8 t/s (prompt 1265/1183/1009/805/737 t/s), peak 16298 MiB, no
errors in the log. Then the retrieval test at 9k/33k/61k/86k tokens of context (MTP on): 144/176 correct, all
single lookups right except 1 at 61k and 2 at 86k (the f16 reference had 32/32 at 76k; the ub 256 noise run 31/32).
Compared with only the V cache at q8_0 and ctx 77824, generation is ~7 % slower up to 16k tokens of context and 0-3 %
slower beyond, and prompt processing 1-2 % slower. If the extra context is not needed, `cache-type-k = f16` with
`ctx-size = 77824` is the faster choice.

## Image update to v0.6.0 (2026-10-09)

Both images with [`preset-gemma-4-26b-a4b-qat.ini`](preset-gemma-4-26b-a4b-qat.ini) (ctx 88064, K+V q8_0, ub 512),
on a test server, measured twice each in alternating order (v0.5.0, v0.6.0, v0.5.0, v0.6.0) with the full command
below. Means of the two runs; generation in t/s, prompt processing in parentheses:

| Test | `v0.5.0-rocm-7.14` | **`v0.6.0-rocm-7.14` (current)** | Change |
|---|---|---|---|
| Generation, 256 tokens (repeated text, temperature 0) | 119.0 | 124.2 | +4 % |
| Prompt processing, 86564 tokens (full context) | (759) | (761) | 0 % |
| Chat, 12 requests | 110.3, acceptance 68 % | 113.7, acceptance 68 % | +3 % |
| Code context of 6.4k tokens | 97.4 (1262) | 101.1 (1260) | +4 % |
| Code context of 16k tokens | 87.3 (1182) | 90.5 (1180) | +4 % |
| Code context of 36.5k tokens | 75.9 (1006) | 77.9 (1008) | +3 % |
| Code context of 70k tokens | 65.5 (802) | 66.5 (805) | +2 % |
| Code context of 85k tokens | 51.9 (731) | 53.7 (737) | +3 % |
| VRAM after loading / peak (amd-smi) | 16113 / 16298 MiB | 16122 / 16306 MiB | +9 MiB |
| `llama bench` pp2048 / tg128 (K+V q8_0, ub 512) | 1463 / 90.0 | 1463 / 89.6 | 0 % |
| `llama bench` pp2048 / tg128 at depth 16384 | 1051 / 75.3 | 1052 / 76.3 | 0 / +1 % |

v0.6.0 generates 2-4 % faster with MTP at every depth (the two v0.6.0 runs agreed within 1 %, the v0.5.0 runs within
1-3 %), with the same prompt processing and 9 MiB more VRAM. As with v0.5.0, the gain is not visible in `llama bench`,
which has no speculative decoding. During these tests, a benchmark on the RTX 3090 ran at the same time; it uses
other hardware but adds some CPU load.

## Qwen3.8-27B (2026-10-09)

[`preset-qwen3.8-27b.ini`](preset-qwen3.8-27b.ini): Qwen3.8-27B (dense, 64 layers, of which 16 have full attention
with a KV cache and 48 are Gated DeltaNet layers with a fixed-size state; 262144 tokens of trained context), entirely
on the GPU. Image v0.6.0. Like the Gemma preset, it uses the GPU only, so that the CPU stays free for the desktop and
the agents' code (see [Entirely on the GPU](#entirely-on-the-gpu-with-a-q8_0-v-cache-2026-09-28)).

### Quantization of the weights

The `UD-Q4_K_M` of [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e) is 16.5 GB and does not fit. The candidates are
unsloth's smaller quants of the same repository. `llama bench`, K+V q8_0, ub 512; VRAM after loading
the server with K+V q8_0, ub 512, MTP and the vision encoder on the CPU (amd-smi):

| Quant | Size | pp2048 / tg128 | At depth 16384 | VRAM at ctx 16384 / 32768 |
|---|---|---|---|---|
| **`UD-IQ4_XS` (current)** | 13.26 GiB | **267 / 25.5** | **225 / 21.9** | 14683 / 15387 MiB |
| `UD-Q3_K_XL` | 12.23 GiB | 251 / 22.4 | 219 / 19.8 | 13627 / 14331 MiB |
| `UD-IQ3_S` | 11.20 GiB | 239 / 21.0 | 203 / 18.6 | 12573 / 13277 MiB |

The largest quant is also the fastest on gfx906, so the smaller ones would only buy context (each GiB ~24k tokens)
at the price of both quality and speed. `UD-IQ4_XS` it is.

### Memory

At ctx 16384 with the settings of the preset (K+V q8_0, ub 512, MTP, vision encoder on the CPU), the server uses
14683 MiB after loading. Changes from there:

| Change | VRAM |
|---|---|
| Context, per 1024 tokens: K+V f16 / q8_0 / q4_0 | +72 / +44 / +28 MiB |
| K+V f16 / q4_0 instead of q8_0 at ctx 16384 | +452 / -256 MiB |
| Vision encoder (mmproj, BF16) on the GPU (without `no-mmproj-offload`) | +1138 MiB |
| No MTP | -813 MiB |
| ub 256 / 1024 instead of 512 | -116 / +250 MiB |

Prompt processing and generation add ~180 MiB on top of the memory after loading for a full-context prompt, and up to
~300 MiB in the full benchmark.

### MTP draft length

Chat test (12 requests) and the depth test at 16k tokens, K+V q8_0, ctx 32768. Generation in t/s:

| `spec-draft-n-max` | Chat | Draft acceptance (chat) | Depth 16k |
|---|---|---|---|
| no MTP | 25.4 | - | 21.8 |
| 1 | 29.7 | 65 % | 23.0 |
| **2 (current)** | **29.9** | 53 % | **27.3** |
| 3 | 26.2 | 43 % | 24.3 |
| 4 | 24.5 | 37 % | 21.8 |

MTP with 2 drafted tokens is 18 % faster in chat and 25 % faster at 16k tokens of context than no speculative
decoding, for 813 MiB (~18k tokens of context).

### KV cache type

`llama bench` at depth 16384, ub 512, t/s:

| K \ V | f16 | q8_0 | q4_0 |
|---|---|---|---|
| f16 | 229 / **22.9** | 229 / 21.1 | 229 / 20.6 |
| q8_0 | 229 / 21.0 | 229 / **22.2** | 229 / 20.1 |
| q4_0 | 229 / 20.3 | 229 / 20.0 | 229 / **23.1** |

(pp2048 / tg128.) Prompt processing does not depend on the cache type. Equal K and V types generate fastest; mixed
types are 5-13 % slower, so they are not worth it. On the server (ctx 32768, MTP), generation in t/s, prompt
processing in parentheses:

| KV cache | Generation, 256 tokens | Chat | Depth 6.4k | Depth 16k | Depth 30k | Peak VRAM |
|---|---|---|---|---|---|---|
| f16 | ran out of memory during the benchmark | | | | | |
| **q8_0 (current)** | 34.6 | 29.5 | 28.2 (248) | 27.4 (239) | 24.6 (225) | 15673 MiB |
| q4_0 | 31.3 | 30.1 | 28.9 (249) | 26.9 (238) | 25.3 (226) | 15157 MiB |

Quality with [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py), as for Gemma above:
132 greedy retrieval questions at 9k/17k/29k tokens of context with the top-20 token probabilities, on test
servers without MTP at ctx 30720 (the f16 reference does not fit at a longer one), compared with the f16 cache:

| Configuration | Correct | Same answer | Mean KLD | 99 % KLD | First token KLD | Top token changed | Mean \|Δ log p\| |
|---|---|---|---|---|---|---|---|
| f16, ub 512 (reference) | 124/132 | - | - | - | - | - | - |
| Noise: f16, ub 256 | 124/132 | 132/132 | 0.00016 | 0.005 | 0.0007 | 0/744 | 0.0016 |
| Noise: f16, ub 1024 | 123/132 | 131/132 | 0.00036 | 0.010 | 0.0015 | 1/740 | 0.0025 |
| **K+V q8_0 (current)** | 123/132 | 131/132 | 0.00028 | 0.010 | 0.0012 | 1/740 | 0.0019 |
| K+V q4_0 | 124/132 | 130/132 | 0.00269 | 0.070 | 0.0110 | 2/735 | 0.0081 |

All single lookups were right in every run (96/96). The q8_0 cache is within the noise of a different ubatch size;
q4_0 is 7-17 times above it (q4_0 is above the noise for this model on the RTX 3090 too, see
[`../llama-cpp-agx-z2e/README.md`](../llama-cpp-agx-z2e/README.md)), although its retrieval accuracy is unchanged.
Qwen3.8's distributions are much more stable than Gemma 4 26B-A4B's (noise KLD 0.0002-0.0004 vs ~0.01).

The cache is therefore q8_0: with it, the context can be ~47k tokens, compared with ~28k with f16. q4_0 would allow
~63k tokens at the same speed, if the small quality loss is acceptable.

### ubatch size

`llama bench` pp2048, K+V q8_0, t/s:

| ub | 128 | 256 | **512 (current)** | 1024 | 2048 |
|---|---|---|---|---|---|
| Depth 0 | 209 | 250 | **267** | 279 | 281 |
| Depth 16384 | 184 | 217 | **229** | 238 | 239 |

1024 is 4 % faster than 512 but needs 250 MiB more, i.e. ~6k tokens less context.

### Vision encoder on the CPU (`no-mmproj-offload`)

The vision encoder takes 1138 MiB on the GPU, i.e. ~26k tokens of context. On the CPU, an image takes a few seconds
longer to process. The `image` test of [`../benchmark/llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py)
(a 1280x960 PNG, 1261 prompt tokens with the image, mean of 3):

| Vision encoder | Prompt processing incl. the image | Whole request (32 generated tokens) |
|---|---|---|
| GPU (ctx 16384) | 6.7 s | 8.0 s |
| **CPU (current)** | 11.0 s | 12.4 s |

So the CPU costs ~4 s per image (with the CPU otherwise idle), which is worth the 26k tokens of context here.

### Context size

Stress test: load, then one prompt of ctx-size - 1500 tokens (MTP on, as served), peak from amd-smi:

| ctx-size | VRAM after loading | Prompt processing | Peak VRAM |
|---|---|---|---|
| 45056 | 15915 MiB | 213 t/s | 16095 MiB |
| **47104 (current)** | 16003 MiB | 211 t/s | 16183 MiB |
| 49152 | 16091 MiB | 209 t/s | 16272 MiB |

In the full benchmark (below), ctx 49152 peaked at 16346 MiB, only 22 MiB below the total, so the context is
47104, whose full benchmark peaked at 16298 MiB, the same as the Gemma preset. Both ran without errors.

The production configuration on a test server, measured with the full command below (prompt 45604 tokens, depth up to
43k):

| Test | Result |
|---|---|
| Generation, 256 tokens (repeated text, temperature 0) | 34.7 t/s, draft acceptance 71 % |
| Prompt processing, 45604 tokens (full context) | 211 t/s |
| Chat, 12 requests | 29.9 t/s (25.5-35.1), draft acceptance 53 % |
| Code context of 6.4k / 16k / 30k / 43k tokens | 28.1 / 27.3 / 24.3 / 24.5 t/s (prompt 249 / 240 / 225 / 212 t/s) |
| Image, 1280x960 | 11.2 s prompt processing |
| Peak VRAM | 16298 MiB |

For comparison, the same model in `UD-Q4_K_M` on the RTX 3090 of [`../llama-cpp-agx-z2e`](../llama-cpp-agx-z2e)
generates ~49 t/s in the chat test and processes prompts at ~1100 t/s.

## Commands

```sh
cd ../benchmark
# Serving speed: generation, two full-context prompts (ctx-size - 1500), chat, depth
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 86564 --repeat 2 --depth 6400 16000 36500 70000 85000 --label "..."
# The same for Qwen3.8 (preset-qwen3.8-27b.ini), with the image test
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth image --prompt-tokens 45604 --repeat 2 --depth 6400 16000 30000 43000 --label "..."
# Output quality compared with a reference run (on a test server; see ../benchmark/README.md)
./llama_cpp_quality_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --label "..." \
    --reference "Radeon VII ctx 77824 f16 KV, n-cpu-moe 1, ub 512 (reference)"
# Raw model speed in the running container
./llama_cpp_bench_container.py --container llama-cpp-radeon-vii --url http://localhost:9932 \
    --env-file ../llama-cpp-radeon-vii/llama-cpp.env --label "..." -- -ngl 999 -ncmoe 1 -fa 1 -ub 512 -p 2048 -n 128 -d 0,16384 -r 3
# Raw model speed with another image or backend
./llama_cpp_bench_container.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \
    --image ghcr.io/ggml-org/llama.cpp:full-vulkan --label "..." -- -ngl 999 -ncmoe 1 -fa 1 -ub 512 -p 2048 -n 128 -d 0,16384
# Serving speed with another image or a preset variant (stop the production container first)
docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml stop
# (for a preset variant, edit a copy of the preset and pass it instead)
./llama_cpp_test_server.sh ghcr.io/ggml-org/llama.cpp:full-vulkan ../llama-cpp-radeon-vii/preset-gemma-4-26b-a4b-qat.ini \
    ../llama-cpp-agx-ai/config.ini ../llama-cpp-radeon-vii/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 2 --depth 6400 16000 36500 70000 --label "..."
docker rm -f llama-test && docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml start
```
