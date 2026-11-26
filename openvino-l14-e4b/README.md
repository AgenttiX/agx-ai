# Gemma 4 E4B on the ThinkPad L14 Gen 5 (iGPU or NPU)

Runs Gemma 4 E4B on the ThinkPad L14 Gen 5 (Intel Core Ultra 5 125U) with the stock OpenVINO Model Server (OVMS),
on the iGPU (default) or on the NPU (Meteor Lake NPU 3720).
On the iGPU, the official INT4 model is used as is. The official model does not compile for this NPU,
so for the NPU [convert_gemma4.py](../agx_ai/openvino/convert_gemma4.py) recompresses its language model
to symmetric INT4 in a container ([../openvino-npu-convert](../openvino-npu-convert)), and two NPU settings work around compiler bugs (see below).

| Model | Device | Context | GPU/NPU memory | RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B, INT4 ([OpenVINO/gemma-4-E4B-it-int4-ov](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov), asymmetric, AWQ) | **iGPU (default)** | 8.5k tokens tested (2 GB KV cache) | shared RAM | 13 GB | 14.1–14.4 t/s (chat test 13.6) | 0.5 s for short prompts, 3.9 s for 778 tokens, 122 t/s at 4.1k and 79 t/s at 8.5k tokens | image: 2.5–2.6 s to the first token |
| Gemma 4 E4B, INT4_SYM group 128 ([convert_gemma4.py](../agx_ai/openvino/convert_gemma4.py)) | NPU (`ovms-gemma4-npu`) | prompt 1024 + answer | shared RAM | 6 GB RSS | 9.0–9.6 t/s | 3.3 s for any prompt up to 1024 tokens (235 t/s at 778 tokens) | image: 6.0–6.4 s to the first token; leaves the iGPU and the CPU free |

- The iGPU generates 1.5x as fast as the NPU, gets the first token 7x sooner for short prompts and 2.5x sooner for
  images, and has no 1024-token prompt limit. The NPU leaves the iGPU and the CPU free, e.g. for
  [../llama-cpp-l14](../llama-cpp-l14), and uses less RAM.
- The best way to use the NPU at the same time is to run the embeddings there
  ([../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2)): the generation on the iGPU then slows
  down by 10–20 %, but by half with the embeddings on the iGPU. Splitting the model itself between the devices or
  speculative decoding with the NPU did not help, see
  [../openvino-l14/README.md](../openvino-l14/README.md#using-the-cpu-the-igpu-and-the-npu-together).
- The symmetric INT4 model for the NPU gives wrong answers on the iGPU (28/80 arithmetic questions correct,
  77/80 on the CPU), probably due to a bug of the GPU plugin, so the iGPU uses the official model (74–76/80).
- Compared to the previous configuration on the NPU (the third-party server
  [ryugyosoft/gemma-4-E4B-it-npu](https://huggingface.co/ryugyosoft/gemma-4-E4B-it-npu), see the Git history),
  the NPU variant generates 10 % faster, processes long prompts faster and uses a third of the RAM,
  and supports tool calls and thinking, with the same answer quality,
  but short prompts take 3.3 s instead of 0.34 s to the first token, as the NPU always processes a full
  1024-token prompt, and the third-party server also reused the KV cache of the previous turn.
- OpenVINO on the CPU ran the channel-wise INT4 version at 13.4–14.6 t/s (OpenVINO GenAI), and llama.cpp runs E4B on
  the iGPU at 7.8 t/s ([../llama-cpp-l14](../llama-cpp-l14)).

## Hardware and software
- Computer: Lenovo ThinkPad L14 Gen 5 (Intel)
- OS: Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0
- CPU: Intel Core Ultra 5 125U (Meteor Lake, 2 P-cores with HT + 8 E-cores + 2 LP E-cores)
- RAM: 32 GB (2 x 16 GB DDR5-5600 SO-DIMM), shared by the CPU, the iGPU and the NPU
- NPU: Intel AI Boost (Meteor Lake NPU 3720), kernel driver intel_vpu, user-mode driver 1.38.0 in the OVMS image
- GPU: Intel Graphics (Meteor Lake iGPU, Xe-LPG architecture, 4 Xe cores), no dedicated VRAM (uses the shared RAM),
  kernel driver i915
- OVMS 2026.5.0 (weekly build of 2026-11-19, OpenVINO 2026.5.0), NNCF 3.4.0 and OpenVINO 2026.4.1 in the converter

## Usage
```sh
docker compose up -d
```
- On the first start, OVMS downloads [OpenVINO/gemma-4-E4B-it-int4-ov](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov)
  (4.4 GB) to the Docker volume `openvino-l14-e4b_models` and compiles it for the iGPU (~30 s, cached in the same
  volume). Later starts take a few seconds.
- The NPU variant is started instead with `docker compose pull` and `docker compose up -d ovms-gemma4-npu`
  (stop the other one first, as they use the same port). It starts the `convert` container first, which downloads the
  model files (13 GB: the language model of
  [OpenVINO/gemma-4-E4B-it-fp16-ov](https://huggingface.co/OpenVINO/gemma-4-E4B-it-fp16-ov) and the rest of
  [OpenVINO/gemma-4-E4B-it-int4-ov](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov))
  to the Hugging Face cache of the host (`~/.cache/huggingface/hub`, of which only the `hub` directory is mounted),
  converts the model to the same Docker volume (6.1 GB) and exits
  (12 min in total with the download at ~20 MB/s; the compression itself takes ~30 s and 12 GB of RAM).
  OVMS starts after it and compiles the model for the NPU in ~3 min (cached in the same volume, 4.3 GB),
  and later starts take 6-8 s.
  Later starts skip the conversion, as long as the revisions, the settings and the library versions are the same
  (`export.json`).
- The API is OpenAI-compatible at http://localhost:9934/v3 (also `/v1`, e.g. `/v1/chat/completions`).
  The model id is `google/gemma-4-e4b-it` (`google/gemma-4-e4b-it-npu` on the NPU).
- Thinking is off by default and can be enabled per request with `"chat_template_kwargs": {"enable_thinking": true}`.
  Thinking and tool calls are returned in the separate `reasoning_content` and `tool_calls` fields (tested on both).
  With a tool call, `content` is `"<eos>"`.
- Images are given as base64 data URLs (image URLs would need `--allowed_media_domains`).
  Two images in one request worked too (on the NPU).
- On the iGPU, prompts of 8.5k tokens worked. A 16k-token prompt (with the symmetric INT4 model) made the i915 driver
  reset the GPU ("GPU hang", see [../openvino-l14/README.md](../openvino-l14/README.md#igpu-settings)), after which
  OVMS failed every request until it was restarted, although it still reported itself as healthy.
  On the NPU, prompts are limited to 1024 tokens, and the answer can continue for at least 128 tokens after that.
- On the iGPU, the server uses 13 GB of RAM (`free` before and after). On the NPU, it uses 6 GB (RSS, 10 GB at the
  peak of the first start).
  It can run at the same time as [../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2):
  with embeddings computed continuously on the NPU, the generation on the iGPU slows from 13.8 to 11.5–12.9 t/s,
  but on the NPU from 8.3 to 3.9 t/s, as they share the NPU.
  Do not run this at the same time as [../llama-cpp-l14](../llama-cpp-l14)
  or [../openvino-l14](../openvino-l14), as the RAM would run out.

The converter image, which is shared with [../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2),
is described in [../openvino-npu-convert](../openvino-npu-convert/README.md).
The converted revisions are set in the `command` of the `convert` service in [docker-compose.yml](docker-compose.yml).

## Performance
Measured on 2026-11-25 with [`agx_ai/benchmark/chat_stream.py`](../agx_ai/benchmark/chat_stream.py) `--no-thinking`
(the speeds, and the times to the first token for prompts of 26-29 and 775-778 tokens and for a 291-293-token
prompt with a 1280x960 image), and two ad-hoc answer checks that are not in the repository
(40 short questions of arithmetic and knowledge, and 80 seeded 2-3-digit arithmetic questions, greedy;
on the iGPU only the latter, with other seeds than on the NPU):

| Configuration | Generation | TTFT short / 778 tok. | Image | RAM (docker stats) | 40 questions | Arithmetic |
|---|---|---|---|---|---|---|
| **iGPU: official INT4, 256-token chunks (default)** | 14.1–14.4 t/s | 0.5 s / 3.9 s | 2.5–2.6 s | 13 GB (`free`) | | 74-76/80 |
| iGPU: INT4_SYM group 128 (the NPU model), 512-token chunks | 13.7–14.2 t/s | 0.5 s / 3.3 s | 2.3–2.7 s | | | 28/80 (CPU: 77/80) |
| **NPU: INT4_SYM group 128, OVMS 2026.5 (`ovms-gemma4-npu`)** | 9.0–9.6 t/s | 3.3 s / 3.3 s | 6.0–6.4 s | 6 GB | 39-40/40 | 74-76/80 |
| INT4_SYM group 128, OVMS 2026.4.1 | 9.0–9.7 t/s | 5.7 s / 5.7 s | 8.3–8.6 s | 8 GB | | |
| INT4_SYM channel-wise, OVMS 2026.5 | 11.2–11.5 t/s | 2.1 s / 2.1 s | 4.8–5.1 s | 5 GB | 34/40 | |
| INT4_SYM channel-wise with AWQ and scale estimation | 11.1–11.9 t/s | 2.1 s / 2.1 s | 4.8–5.1 s | 10 GB | 38/40 | 67/80 |
| INT4_SYM group 128, prompt limit 512 (`MIN_RESPONSE_LEN` 512) | 8.8–9.0 t/s | 1.9 s / - | | | | |
| Previous: third-party server, 1024-token context | 8.2–8.5 t/s | 0.34 s / 4.3 s | 3.6–3.8 s | 15–18.5 GB | 39/40 | 75/80 |

- On the iGPU, the chat test of [`agx_ai/benchmark/bench_http.py`](../agx_ai/benchmark/bench_http.py)
  (`--tests chat --repeat 1`, results in `../benchmark/results/agx-l14-kubuntu.jsonl`) ran at 13.6 t/s,
  and the prompt processing slowed down from 177 t/s at 1.4k tokens to 122 t/s at 4.1k and 79 t/s at 8.5k tokens.
  The symmetric INT4 model gave the right answers to the 8 short questions of `chat_stream.py` in the first run on the
  iGPU, but only 4-5 of them in later runs, and 28/80 arithmetic questions, so the iGPU uses the official model.
- The group size of 128 was chosen for the quality on the NPU: the channel-wise quantization is 20 % faster on the NPU
  but made clearly more arithmetic mistakes, also with data-aware AWQ and scale estimation
  (64 samples of 256 tokens, 26 min and 22 GB of RAM for the compression).
  The misses in the 40 questions of the previous server and of the final run of this one were the formatting of
  H₂O as LaTeX. Each question starts with a random nonce, so repeated runs vary by a few answers
  (two runs of this configuration: 76/80 and 74/80).
- An INT4 LM head (`all_layers=True`, 10 % smaller model) was not faster than the default INT8 one.
- The prompt is always processed as a full static 1024-token graph (3.3 s), as the chunked prefill
  that would process short prompts faster (0.6 s with 256-token chunks) gives wrong output on this NPU (see below).

## Notes on the NPU settings
These were found by testing on 2026-11-25. The official model and the default OVMS settings fail in several ways
on NPU 3720 (Meteor Lake). The NPU plugin of OpenVINO (NPUW) has Gemma 4 support, which seems to target newer NPUs.
- [OpenVINO/gemma-4-E4B-it-int4-ov](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov) has asymmetric INT4
  weights (group size 64, AWQ), and the NPU compiler rejects them ("Unsupported DequantizeOp configuration ... u4").
  OVMS 2026.4.1 (NPU driver 1.35) compiled it for over 30 min without finishing, and the 2026.5 weekly failed in 1 min.
  Hence [convert_gemma4.py](../agx_ai/openvino/convert_gemma4.py) compresses the FP16 language model to symmetric INT4 (OpenVINO does not
  publish an NPU-specific `-cw-ov` version of E4B, only of E2B).
- With the default `NPUW_LLM_GENERATE_HINT` (`FAST_COMPILE`), NPUW compiles the generation model as folded,
  repeated layer subgraphs (`NPUW_UNFOLD_IREQS`), which output garbage on this NPU.
  The same happens with Intel's NPU model of E2B
  ([OpenVINO/gemma-4-E2B-it-int4-cw-ov](https://huggingface.co/OpenVINO/gemma-4-E2B-it-int4-cw-ov)),
  so it is not caused by the conversion. `BEST_PERF` compiles the model as one graph (`NPUW_ONLINE_PIPELINE` `NONE`),
  which gives the same output as the CPU. The prefill model has to be compiled as one graph too, which is set with
  `++NPUW_LLM_PREFILL_CONFIG` (`++` adds to the default configuration instead of replacing it).
  The nested configuration is given as an OpenVINO map string (`{KEY:VALUE}`), as OVMS does not accept a nested JSON
  object in `--plugin_config`. Other prefill and attention hints (`NPUW_LLM_PREFILL_HINT` `STATIC`,
  `NPUW_LLM_*_ATTENTION_HINT` `PYRAMID`, `DYNAMIC` or `HFA`) did not fix the output.
- `--max_prompt_len` above 1024 enables chunked prefill (1024-token chunks by default), which failed in all
  variants: with chunks of 1024, the NPU compiler crashed ("Destination is not inside single cluster" for a
  1536 = 1024 + 512 sliding-window buffer), and with chunks of 256 or 512, or 2048 with the default chunks,
  it compiled but the first token was garbage. The same chunked prefill on the CPU through NPUW
  (`NPUW_DEVICES` `CPU`) was correct, so the NPU computes it wrongly.
  `NPUW_LLM_ENABLE_SWA_KV_CACHE_SHRINK` `NO` did not help. The OpenVINO 2027.0 nightly (2026-11-24) behaved the same.
- The third-party server works around a bug of NPU 3720 with the 512-wide attention heads of Gemma 4's
  full-attention layers by splitting them into 256-wide halves (`split_sdpa_v` in its `npu_fixes.py`).
  Applied to the IR, this was not needed with the settings above (the output matched the CPU),
  and it made the generation 10 % slower. It also needs `NPUW_LLM_OPTIMIZE_V_TENSORS` `NO`,
  as NPUW then copies the KV cache of those layers with the wrong axis ("KV slice ... out of range for extent 512").
  With the default (partitioned) prefill and 256-token chunks, the split made the chunked prefill coherent,
  but the answers still differed from the CPU, and in OVMS most short answers were empty.
- Speed and memory: `NPU_COMPILER_DYNAMIC_QUANTIZATION` for the prefill did not change anything.
  The OVMS 2026.4.1 release works with the same settings and generates as fast,
  but its prefill takes 5.7 s instead of 3.3 s, so the 2026.5 weekly build is used, pinned by its digest.
