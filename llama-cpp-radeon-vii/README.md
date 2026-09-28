# llama.cpp on the Radeon VII

llama-server for the Radeon VII (gfx906, 16 GB HBM2) on agx-z2e, serving Gemma 4 26B-A4B QAT (`UD-Q4_K_XL`)
entirely on the GPU, with MTP speculative decoding, q8_0 K and V caches and a context of 88064 tokens, more than the
77824 of [`../llama-cpp-agx-ai`](../llama-cpp-agx-ai), so that the same agentic workloads can be split between the two. The image is `mixa3607/llama.cpp-gfx906:v0.5.0-rocm-7.14` (see
[`docker-compose.yml`](docker-compose.yml)), and the settings are in [`preset.ini`](preset.ini). This file has the
benchmarks behind them. The tools are in [`../benchmark`](../benchmark); the commands are at the end.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-34-generic) |
| CPU | AMD Ryzen Threadripper 3970X (32 cores, 64 threads) |
| RAM | 125.6 GiB visible to the OS, DDR4; module size and frequency not yet recorded (`sudo dmidecode -t memory`) |
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

## Commands

```sh
cd ../benchmark
# Serving speed: generation, two full-context prompts (ctx-size - 1500), chat, depth
./llama_cpp_bench_http.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 86564 --repeat 2 --depth 6400 16000 36500 70000 85000 --label "..."
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
# (for a preset variant, edit a copy of preset.ini and pass it instead)
./llama_cpp_test_server.sh ghcr.io/ggml-org/llama.cpp:full-vulkan ../llama-cpp-radeon-vii/preset.ini \
    ../llama-cpp-agx-ai/config.ini ../llama-cpp-radeon-vii/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 2 --depth 6400 16000 36500 70000 --label "..."
docker rm -f llama-test && docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml start
```
