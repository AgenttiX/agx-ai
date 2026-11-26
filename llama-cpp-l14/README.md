# llama.cpp on the ThinkPad L14 Gen 5

llama.cpp router server for the ThinkPad L14 Gen 5:
Intel Core Ultra 5 125U (2 P-cores with HT + 8 E-cores + 2 LP E-cores),
Intel Graphics iGPU (Meteor Lake, 4 Xe cores), Meteor Lake NPU (NPU 3720), 32 GB RAM shared by all of them.
The models and their settings are in [preset.ini](preset.ini), and the server-wide settings in [`../llama-cpp/config.ini`](../llama-cpp/config.ini).

Current configuration (build b12723, Vulkan on the iGPU, ctx 50000, `parallel = 1`;
tg = token generation in chat and with a 3.5k-token prompt, pp = prompt processing of 3.5k tokens, t/s):

| Model                      | Quant (Unsloth) | Context | RAM (total in use)  | tg         | pp      | Notes                                                    |
|----------------------------|-----------------|---------|---------------------|------------|---------|----------------------------------------------------------|
| Gemma 4 26B-A4B QAT        | UD-Q4_K_XL      | 50000   | 24.8 GB at 45.5k    | 8.5-9.2    | 125-129 | Loaded at startup; no MTP, no checkpoints                |
| Gemma 4 12B QAT            | UD-Q4_K_XL      | 50000   | -                   | 3.4-3.6    | 62      | No MTP, no checkpoints                                   |
| Gemma 4 E4B QAT            | UD-Q4_K_XL      | 50000   | -                   | 7.5-7.9    | 275-288 | OpenVINO (../openvino-l14-e4b) generates 1.8x as fast    |
| Qwen3.5-9B                 | UD-Q4_K_XL      | 50000   | ~24 GB at 46k       | 4.9-5.7    | 77-78   | MTP 2; OpenVINO (../openvino-l14) generates 2.2x as fast |

The RAM column is the whole computer's use (with the desktop session), as the iGPU shares the RAM.

All models run on the iGPU with Vulkan and flash attention off, as that gives the best performance overall:
compared to the CPU, prompt processing is 3-4x faster, while token generation is 30-40 % slower
(see the 8k and 32k results below). The iGPU is therefore faster whenever the prompt is longer than ~1.5-2.5x the answer,
which is the case for most real use (system prompts, documents, tool definitions).
For short chats without a system prompt, the CPU would be faster
(`device = none`, `threads = 8`, `flash-attn = on`; the numbers are below).

Everything is limited by the memory bandwidth, which the CPU, iGPU and NPU share,
so splitting a model between them does not add bandwidth,
and mixing CPU and Vulkan in one model is very slow (see below).

OpenVINO Model Server runs Qwen3.5-9B on the iGPU or Qwen3-8B on the NPU ([../openvino-l14](../openvino-l14)),
Gemma 4 E4B on the iGPU or the NPU ([../openvino-l14-e4b](../openvino-l14-e4b)),
and EmbeddingGemma 2 on the NPU ([../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2)).
On the iGPU, OpenVINO generates Gemma 4 E4B 1.8x as fast as llama.cpp (14 vs 7.8 t/s), but it cannot run the 26B A4B
(see [../openvino-l14/README.md](../openvino-l14/README.md#using-the-cpu-the-igpu-and-the-npu-together)).
Do not run the LLM servers at the same time as this server:
together they exhausted the 32 GB of RAM and the kernel killed the desktop session.

The sections below were measured on 2026-09-25/26 (build b11176, tg = token generation, pp = prompt processing, t/s)
unless otherwise noted; see [Update to b12723](#update-to-b12723-2026-11-26) and [Qwen3.5-9B](#qwen35-9b-2026-11-26)
for the current numbers.

## llama bench
`-fa 1`, no speculative decoding, UD-Q4_K_XL:

| Backend (image)                   | E4B pp512 | E4B tg64 | 12B pp512 | 12B tg64 | 26B-A4B pp512 | 26B-A4B tg64 |
|-----------------------------------|-----------|----------|-----------|----------|---------------|--------------|
| CPU 4 threads (`server`)          | -         | -        | 11        | 3.9      | -             | -            |
| CPU 6 threads                     | 31        | 10.0     | 12        | 4.5      | -             | -            |
| CPU 8 threads                     | 35        | 10.7     | -         | -        | 34            | 12.6         |
| CPU 10 threads                    | 37        | 10.6     | 15        | 5.0      | 37            | 12.1         |
| CPU 12 threads                    | -         | -        | 11        | 4.7      | -             | -            |
| Vulkan iGPU (`server-vulkan`)     | 128       | 8.3      | 51        | 3.7      | 97            | 9.7          |
| SYCL iGPU (`server-intel`)        | 135       | 8.5      | 53        | 4.7      | 65            | 6.6          |
| OpenVINO iGPU (`server-openvino`) | 260       | 10.0     | 107       | 5.5      | fails ("failed to decode prompt batch") | |
| OpenVINO CPU                      | 52        | 6.0      | -         | -        | -             | -            |
| OpenVINO NPU                      | fails     | fails    | fails     | fails    | fails         | fails        |

- Flash attention on Vulkan is slow on this iGPU at long contexts, and the i915 driver kills GPU jobs that run for more
  than a few seconds (heartbeat 2.5 s, preempt timeout 7.5 s in `/sys/class/drm/card1/engine/*/`),
  so long flash attention kernels crashed the server:
  "context reset due to GPU hang" in the kernel log and "ggml_vulkan: device lost" in llama.cpp.
  E4B pp8192 with `-ub 512`: 139 t/s with flash attention off, and a GPU hang with it on.
- OpenVINO on the iGPU (`GGML_OPENVINO_DEVICE=GPU`, `device = OPENVINO0`) is broken for Gemma 4 in llama-server:
  chat answers ignore the question, and prompts longer than a few hundred tokens fail with
  "ov::Exception ... Argument shapes are inconsistent" (also with `GGML_OPENVINO_DISABLE_KV_SLICE=1` or `swa-full`).
  It cannot run the 26B-A4B MoE at all and does not support multimodal input.
- Mixing the CPU and Vulkan in one model is very slow, as the CPU weights are then in Vulkan host memory,
  which is slow for the CPU to read:
  - E4B with `n-gpu-layers 0` (CPU weights, Vulkan op offload for prompts): tg 0.17 t/s,
    with `--no-host` 2.2 t/s and pp 15 t/s.
  - 26B-A4B with the experts on the CPU (`n-cpu-moe 30`): tg 0.5 t/s, with `--no-host` tg 5.8 t/s and pp 27 t/s.
  - The server with CPU weights and op offload took 552 s for 193 tokens.
- With flash attention off, the context checkpoints that llama-server saves for the sliding-window attention layers
  of Gemma 4 took ~10 s per request (12B: a 5-token prompt took 10.9 s, and 0.5 s with `ctx-checkpoints = 0`;
  `cache-ram = 0` did not help). Without checkpoints, follow-up turns of a chat still reuse the cached prompt
  (12B, 4271-token first turn: the next turns processed only 29-32 new tokens in 1.2-1.6 s).
  `swa-full = true` (which would also avoid checkpoints) fails to allocate its memory with ctx 50000.
- A smaller `ubatch-size` costs little on Vulkan (E4B pp2048 with flash attention: 71 t/s with 128, 73 t/s with 512).

## llama-server over HTTP
Measured with [agx_ai/benchmark/bench_http.py](../agx_ai/benchmark/bench_http.py)
(results in `benchmark/results/agx-l14-kubuntu.jsonl`).
The CPU backend is as fast in the OpenVINO image as in the plain CPU image (`server`).

- 26B-A4B CPU 8 threads, MTP 2: tg 15.2 t/s (acc 85 %), pp 31 t/s for 2048 tokens (CPU image);
  tg 15.8 t/s, pp 30 t/s (OpenVINO image).
  256-token runs, MTP 2 vs. none: 13.1-14.1 vs. 11.9 t/s, so MTP gives ~10-20 %.
- 26B-A4B Vulkan all layers, fa on, MTP 2: tg 10.7 t/s (acc 85 %), pp 48 t/s.
- 12B CPU 8 threads: no MTP tg 4.9 t/s (4.7 in the OpenVINO image), MTP 1 4.7 t/s, MTP 2 4.2 t/s, MTP 3 4.9 t/s;
  pp 13 t/s. MTP does not help the dense 12B on the CPU, as verifying the draft costs as much as it saves.
- E4B CPU 8 threads: no MTP tg 13.8 t/s, MTP 2 tg 10.0 t/s (acc 48 %); pp 35-36 t/s for 2048 tokens.
- 8192-token prompt (the generation test uses a short prompt), Vulkan all layers:

  | Model   | CPU (fa on)      | Vulkan fa on, ub 128 | Vulkan fa off, ub 512           |
  |---------|------------------|----------------------|---------------------------------|
  | E4B     | pp 31.5, tg 11.2 | pp 27.2, tg 7.7      | pp 129, tg 7.5                  |
  | 12B     | pp 11.6, tg 4.7  | pp 11.3, tg 3.4      | pp 41.5, tg 3.4                 |
  | 26B-A4B | pp 25.5, tg 11.7 | pp 17.5, tg 10.9     | pp 76, tg 9.4 (MTP 2, acc 77 %) |

  With a 32768-token prompt, Vulkan fa off: E4B pp 89 t/s, 26B-A4B pp 62 t/s, no GPU hangs.
  The Vulkan fa off numbers above include the ~10 s checkpoint overhead per request (see above).
- 26B-A4B Vulkan fa off, `ctx-checkpoints = 0`: MTP 2 tg 8.6 t/s (acc 63 %), no MTP tg 8.9 t/s;
  pp 122 t/s for 512 tokens.
- Final configuration, chat requests with a 3.5k-token prompt: pp 106 (26B-A4B), 155 (E4B), 55 (12B) t/s;
  tg 8.1-8.9, 7.3-7.8, 3.3-3.5 t/s. Image input works (the mmproj runs on the iGPU).
  26B-A4B with a 45.5k-token prompt: found a password hidden in the middle, pp 55 t/s (14.5 min), tg 4.5 t/s,
  24.8 GB of RAM used in total (19.7 GB of it shared with the iGPU), no GPU hangs.

The results vary by ~10 % between runs, and the first benchmark after loading a model can be much slower
(26B: tg 5.3 t/s) while the pages that other programs pushed to swap are read back.

## Update to b12723 (2026-11-26)
The image is pinned to the digest of build b12723, as there are no build-tagged `server-vulkan` images after b12282.
The newer images set `LLAMA_ARG_PORT=8080`, which overrides `port = 9931` of `config.ini`,
so [docker-compose.yml](docker-compose.yml) sets it back.
`bench_http.py --tests generation prompt chat depth --prompt-tokens 3500 --depth 3500 --repeat 1`,
same preset as before (fa off, no MTP, `ctx-checkpoints = 0`):

| Model   | Generation | pp 3.5k   | Chat          | Depth 3.5k: pp / tg |
|---------|------------|-----------|---------------|---------------------|
| E4B     | 8.0        | 275 (155) | 7.9 (7.3-7.8) | 288 / 7.5           |
| 12B     | 3.6        | 62 (55)   | 3.6 (3.3-3.5) | 62 / 3.4            |
| 26B-A4B | 9.2        | 129 (106) | 9.2 (8.1-8.9) | 125 / 8.5           |

The b11176 numbers of the final configuration are in parentheses: prompts are 20-80 % faster and generation 3-10 %
faster, with no GPU hangs.

## Qwen3.5-9B (2026-11-26)
[unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF), b12723, Vulkan, fa off, ctx 50000.
The model is hybrid: 24 Gated DeltaNet (recurrent) layers and 8 full-attention layers, so its KV cache is small.
`bench_http.py --tests generation prompt chat depth image --prompt-tokens 3500 --depth 3500 --repeat 1`
(the image test is a 1280x960 image, 1219 prompt tokens):

| Variant                 | Generation | pp 3.5k | Chat (acceptance) | Depth 3.5k: pp / tg | Image   |
|-------------------------|------------|---------|-------------------|---------------------|---------|
| UD-Q4_K_XL              | 4.6        | 81      | 4.6               | 82 / 4.5            | 37 s    |
| UD-Q4_K_XL, MTP 1       | -          | -       | 5.6 (76 %)        | 76 / 4.6            | 38 s    |
| UD-Q4_K_XL, MTP 2       | 5.7-5.8    | 77      | 5.7 (62 %)        | 77-78 / 4.8-4.9     | 38 s    |
| UD-Q4_K_XL, MTP 3       | -          | -       | 4.9 (53 %)        | 77 / 4.4            | 38 s    |
| Q4_0                    | 4.5        | 105     | 4.5               | 106 / 4.4           | 33 s    |
| Q4_0, MTP 2             | 5.8        | 99      | 5.0 (59 %)        | 100 / 4.3           | 34 s    |

- UD-Q4_K_XL with MTP 2 was chosen. Q4_0 processes prompts ~30 % faster but does not generate faster,
  and its drafts were accepted less often.
- MTP 2 is +24 % in chat and +7 % at depth over no MTP. The greedy answers to 80 arithmetic questions were all correct
  with MTP 1-3 and without it.
- **Context checkpoints are needed** for this model, unlike for Gemma 4 (see above): with the default checkpoints,
  the follow-up turns of a 3-turn chat on a 5k-token document processed 27 new tokens in 3 s,
  while with `ctx-checkpoints = 0` every turn reprocessed the whole 5k tokens (63 s),
  as the state of the recurrent layers cannot be rolled back to the common prefix.
- The vision encoder: `hf` picks `mmproj-BF16`, which warns that the graph "uses unsupported operators by the backend"
  but works (38 s per image). `mmproj-F16` took 252 s per image, and afterwards the server answered text with garbage
  (0/80 arithmetic answers). The BF16 encoder on the CPU (`no-mmproj-offload = true`) took 100 s.
- Final configuration: tg 5.8 (generation), 5.7 (chat), 4.9 (3.5k depth) and 2.9 t/s (16k depth);
  pp 77 t/s (3.5k) and 66 t/s (16k). A 46k-token prompt found a password hidden in the middle,
  pp 45 t/s (17 min), tg 2.1 t/s, with 7.5 GB of RAM still available.
- OpenVINO Model Server runs the same model over twice as fast on the iGPU (tg 12.2-12.7 t/s, pp 149 t/s, INT4,
  [../openvino-l14](../openvino-l14)), so it is the better choice for Qwen3.5-9B on this laptop.

Commands, from the repository root (the test server runs on port 9933):
```bash
DOCKER_ARGS="--group-add 992" uv run -m agx_ai.benchmark.llama_cpp.test_server \
  ghcr.io/ggml-org/llama.cpp@sha256:4e88890069ad09665bf3b8f430f49b308198b5326329693022fef8abc25d2930 \
  llama-cpp-l14/preset.ini llama-cpp/config.ini llama-cpp/llama-cpp.env
uv run -m agx_ai.benchmark.bench_http --url http://localhost:9933 --env-file llama-cpp/llama-cpp.env \
  --model qwen/qwen3.5-9b --no-embedding --repeat 1 --tests generation prompt chat depth image \
  --prompt-tokens 3500 --depth 3500 16000 --label "L14 b12723: Qwen3.5-9B final (UD-Q4_K_XL, MTP 2)"
```

## NPU with llama.cpp
OpenVINO backend, `GGML_OPENVINO_DEVICE=NPU`:
- Gemma 4 does not run on this NPU. E4B UD-Q4_K_XL (unsloth QAT), gemma-4-E4B_q4_0-it (Google's QAT Q4_0) and
  Q4_K_M (bartowski, validated on the Lunar Lake NPU in llama.cpp `docs/backend/OPENVINO.md`) all compile for ~10 min
  and then fail even with ctx 512: "L0 pfnAppendGraphExecute result: ZE_RESULT_ERROR_UNINITIALIZED ...
  driver is not initialized" followed by "llama_decode: failed to decode, ret = -3". 12B and 26B-A4B fail as well.
  The kernel driver (intel_vpu, firmware vpu_37xx 20260219) and the container's user-mode driver (1.38) are fine,
  as small models work.
- The NPU compiles a static graph for the whole context, so its per-token cost grows with `ctx-size`.
  Qwen3-0.6B Q4_K_M (llama completion, 64 tokens): ctx 256 tg 14.8, ctx 1024 tg 0.8, ctx 4096 tg 0.2 t/s.
- Qwen3 Q4_K_M at ctx 1024 (llama completion), tg on the NPU / OpenVINO CPU / OpenVINO iGPU:
  0.6B 0.8 / 19.3 / 19.3, 1.7B 0.8 / 12.9 / 15.8, 4B 0.6 / 6.2 / 9.8 t/s.
- Through the router: Qwen3-4B on the NPU tg 2.0 t/s at ctx 512 and 0.6 t/s at ctx 1024,
  ~50 s to load (NPU compilation, `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` stays empty on the NPU).
  The same model on the CPU (`device = none`) generates 16 t/s.
  Qwen3-0.6B on the NPU at ctx 512 (tg 3.4 t/s) answered "Kemi" and "Turku" for the capital of Finland.
  The Qwen3-4B NPU model is kept commented out at the end of [preset.ini](preset.ini).
- EmbeddingGemma 2 works on the NPU only for inputs of the length that was compiled first,
  see [../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2).
- In short, the NPU works with llama.cpp only for small models with a tiny context,
  and even then it is several times slower than the CPU.
  For real use of the NPU, see OpenVINO GenAI / OpenVINO Model Server with the NPU-optimized models
  (e.g. OpenVINO/Qwen3-8B-int4-cw-ov), which use a stateful pipeline instead
  ([../openvino-l14](../openvino-l14)), and [../openvino-l14-e4b](../openvino-l14-e4b).

## Memory and threads
- The 26B-A4B llama-server process used ~20.6 GB RSS on the CPU (weights, the CPU_REPACK copy of the Q4_0 tensors,
  mmproj, MTP drafter and KV cache), so only one model fits at a time (`models-max = 1` in [`../llama-cpp/config.ini`](../llama-cpp/config.ini)).
  Requesting another model through the router unloads the loaded one.
- Threads (CPU): 8-10 is the sweet spot.
  12 threads was slower, presumably as the slow LP E-cores hold the others back.
