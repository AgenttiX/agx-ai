# OpenVINO Model Server with Qwen3-8B on the ThinkPad L14 Gen 5

OpenVINO Model Server (OVMS) for the ThinkPad L14 Gen 5 (Intel Core Ultra 5 125U), serving Qwen3-8B on the iGPU,
or alternatively on the NPU (Meteor Lake NPU 3720).
The llama.cpp server for the CPU and iGPU is in [../llama-cpp-l14](../llama-cpp-l14),
Gemma 4 E4B (also with OVMS) in [../openvino-l14-e4b](../openvino-l14-e4b),
and EmbeddingGemma 2 on the NPU in [../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2).
How the CPU, the iGPU and the NPU can be used together is investigated [below](#using-the-cpu-the-igpu-and-the-npu-together).

| Model | Device | Context | GPU/NPU memory | RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3-8B, INT4 channel-wise ([OpenVINO/Qwen3-8B-int4-cw-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov)) | **iGPU (default)** | ~28k tokens (2 GB KV cache) | shared RAM | 10 GB | 13.8–14.0 t/s (chat test 13.5) | 0.5 s for short prompts, 4.8 s for 781 tokens, 91 t/s at 4.1k and 57 t/s at 8.5k tokens | |
| the same | NPU (`ovms-qwen3-npu`) | prompt 8192 + answer | shared RAM | 13.6 GB | 7.3–8.0 t/s (chat test 7.3), 5.2 t/s after an 8k prompt | 5.4 s for any prompt up to 1k tokens, ~120 t/s at 8k | leaves the iGPU and the CPU free |

- The iGPU generates 1.8x as fast as the NPU, and short prompts get the first token 10x sooner.
  The NPU processes prompts of several thousand tokens faster (about 2x at 8k tokens), and leaves the iGPU free,
  e.g. for [../llama-cpp-l14](../llama-cpp-l14).
- The CPU generates at 8.5–8.9 t/s with OpenVINO, but takes 28 s for a 780-token prompt.
- llama.cpp on the CPU generates 16 t/s with Qwen3-4B Q4_K_M (see [the model comparison](#model-comparison-on-the-npu)).

## Hardware and software
- Computer: Lenovo ThinkPad L14 Gen 5 (Intel)
- OS: Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0
- CPU: Intel Core Ultra 5 125U (Meteor Lake, 2 P-cores with HT + 8 E-cores + 2 LP E-cores)
- RAM: 32 GB (2 x 16 GB DDR5-5600 SO-DIMM), shared by the CPU, the iGPU and the NPU
- GPU: Intel Graphics (Meteor Lake iGPU, Xe-LPG architecture, 4 Xe cores), no dedicated VRAM (uses the shared RAM),
  kernel driver i915
- NPU: Intel AI Boost (Meteor Lake NPU 3720), kernel driver intel_vpu, user-mode driver 1.38.0 in the OVMS image
- OVMS 2026.5.0 (weekly build of 2026-11-19, OpenVINO 2026.5.0), the same image as in
  [../openvino-l14-e4b](../openvino-l14-e4b)

## Usage
```sh
docker compose up -d
```
- The model is downloaded from Hugging Face to `/home/mika/.cache/ovms` on the first start.
- The API is OpenAI-compatible at http://localhost:9933/v3 (also `/v1`, e.g. `/v1/chat/completions`).
  The model id is `qwen/qwen3-8b` (`qwen/qwen3-8b-npu` on the NPU).
- The NPU variant is started instead with `docker compose up -d ovms-qwen3-npu` (stop the other one first, as they use
  the same port).
- Thinking can be disabled per request with `"chat_template_kwargs": {"enable_thinking": false}`.
  This is recommended for interactive use, as Qwen3 often thinks for over 1000 tokens.
- Thinking and tool calls are returned in the separate `reasoning_content` and `tool_calls` fields (tested).
- The server uses 10 GB of RAM on the iGPU (`free` before and after; `docker stats` shows 14.6 GB including the page
  cache). Do not run this at the same time as [../llama-cpp-l14](../llama-cpp-l14) or
  [../openvino-l14-e4b](../openvino-l14-e4b): together they exhausted the 32 GB of RAM and the kernel killed
  the desktop session. It can run with [../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2)
  on the NPU.

## iGPU settings
Measured on 2026-11-25 with [`agx_ai/benchmark/chat_stream.py`](../agx_ai/benchmark/chat_stream.py) `--no-thinking`
(short prompts and a 781-token prompt), the chat test of [`agx_ai/benchmark/bench_http.py`](../agx_ai/benchmark/bench_http.py)
(`--tests chat --repeat 1`, six prompts with thinking, up to 512 tokens; results in
`../benchmark/results/agx-l14-kubuntu.jsonl`) and an ad-hoc long-prompt test (a password hidden in a repeated text).
- `--cache_size 2`: with the default dynamic KV cache, prompts of a few thousand tokens stalled: a 3.4k-token prompt
  had not finished after 6 minutes, with one CPU thread at 100 % and the iGPU idle, both with continuous batching
  and with the stateful pipeline (`--pipeline_type LM`). OpenVINO GenAI without OVMS processed a 4.5k-token prompt in
  27 s on the same iGPU.
- `--max_num_batched_tokens 256`: the prompt is processed in chunks of this size. The i915 driver resets GPU jobs that
  do not finish in time (`/sys/class/drm/card1/engine/*/preempt_timeout_ms` 7500 and `heartbeat_interval_ms` 2500),
  and the kernel log showed "Fence expiration time out" for OVMS with longer jobs. With 512-token chunks, OVMS crashed
  once with a "GPU hang" (`CL_OUT_OF_RESOURCES`) while the NPU was in use, and its prompt processing was not faster.
  Gemma 4 E4B also hung once at a 16k-token prompt with 256-token chunks, so the context is not unlimited in practice.
- The prompt processing slows down with the length of the context, from 185 t/s at 1.4k tokens to 57 t/s at 8.5k tokens.

| Configuration | Generation (chat test) | Short prompt | 781 tokens | 4.1k tokens | 8.5k tokens |
|---|---|---|---|---|---|
| **iGPU, continuous batching, 256-token chunks (this)** | 13.8–14.0 t/s (13.5) | 0.5 s | 4.8 s | 45 s | 150 s |
| iGPU, the same with 512-token chunks | 14.0–14.2 t/s (13.5) | 0.5 s | 4.9 s | 48 s | 160 s |
| iGPU, 512-token chunks, draft model on the CPU (see below) | (13.5) | | | | |
| iGPU, 512-token chunks, draft model on the iGPU | (12.7) | | | | |
| iGPU, stateful pipeline (OpenVINO GenAI without OVMS) | 14.5 t/s | 0.4 s | 4.3 s | 27 s (4.5k) | |
| NPU, `max_prompt_len` 4096 | 7.7–8.5 t/s (7.3) | 5.3 s | 5.3 s | | |
| CPU (OpenVINO GenAI) | 8.5–8.9 t/s | 0.7–1.3 s | 28 s | | |

## Using the CPU, the iGPU and the NPU together
Investigated on 2026-11-25 with the OVMS 2026.5 weekly build. In short: no way of splitting one model between the
devices made it faster, as all of them share the same memory bandwidth, and the speculative decoding with the NPU
gives wrong output. The fastest way to use all of them is to run each model on the device that suits it best
at the same time: the LLM on the iGPU and the embeddings on the NPU.

- **Speculative decoding with the main model on the NPU** and a draft model on the CPU or the iGPU
  (`--draft_source_model OpenVINO/Qwen3-0.6B-int4-ov --draft_device CPU`, OpenVINO GenAI's stateful "FastDraft",
  which requires the NPU for one of the models): up to 2–4x faster for some prompts (19–35 t/s), but the output is
  wrong. The answers do not stop at the end of the turn ("Jupiter\n\n**Answer:**  \nJupiter..."), tokens are lost
  ("39" instead of "391"), and OVMS ignores `max_tokens`. The same happened with OpenVINO GenAI's Python API without
  OVMS, also with one draft token at a time, so the NPU computes the verification of several tokens at once wrongly,
  like the chunked prefill of Gemma 4 E4B ([../openvino-l14-e4b](../openvino-l14-e4b)).
  `NPUW_LLM_GENERATE_HINT` `BEST_PERF` failed to compile.
- **The draft model on the NPU** (Qwen3-0.6B INT8) and the main model on the CPU: 0.25 t/s, 13–21 s to the first
  token, and the same wrong end of the answers.
- **Speculative decoding on the iGPU** (continuous batching, which uses another implementation than the NPU) with
  Qwen3-0.6B INT4 as the draft model, on the CPU or on the iGPU: the output is correct, and code was generated
  50 % faster (20–22 t/s), but prose was slower (10–13 t/s), so the chat test was not faster:
  13.5 t/s with the draft on the CPU and 12.7 t/s with it on the iGPU, compared to 13.5 t/s without.
- Gemma 4's own MTP drafters (e.g. [google/gemma-4-E4B-it-assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant))
  run only on the CPU and the iGPU in OpenVINO GenAI ("Gemma4 MTP stateful speculative decoding currently supports
  CPU/GPU only"). There is no exporter for them yet: the target model needs extra outputs, and the pair has to be
  stitched by hand. According to
  [openvinotoolkit/openvino.genai#4594](https://github.com/openvinotoolkit/openvino.genai/issues/4594), this
  experimental pipeline spends most of its time trimming the KV cache, and a continuous batching version is in
  development. Not tested.
- Splitting the layers of a model between the devices (`HETERO`) cannot be faster than the fastest device alone,
  as the layers run one after another and the devices share the memory bandwidth.
  The NPU plugin can offload parts of its graphs to the CPU (`NPUW_DEVICES`), but not to the iGPU.
- **Running models on different devices at the same time** (measured with Gemma 4 E4B on the iGPU and
  EmbeddingGemma 2, 500-token texts continuously): with the embeddings on the NPU, the generation slowed from
  13.8 to 11.5–12.9 t/s and the embeddings ran at 2000–2100 t/s. With the embeddings also on the iGPU, the generation
  slowed to half (6.4 t/s), and the embeddings ran at 1600–2500 t/s.
  So the NPU adds throughput when it runs another model, even though the memory bandwidth is shared.
- Embeddings on the NPU and the iGPU behind one endpoint (round robin over two servers) were not faster than the
  iGPU alone, as the NPU is 5x slower for short texts (see
  [../openvino-l14-embeddinggemma2](../openvino-l14-embeddinggemma2)).
- Gemma 4 26B A4B (QAT) cannot run with OpenVINO on this laptop:
  - Google's QAT GGUFs can be loaded with the GGUF frontend of OpenVINO GenAI (preview, `GGUF_READER` `FRONTEND`),
    which supports Gemma 4 and MoE models, but the iGPU failed to compile the MoE layers of
    [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) UD-Q4_K_XL
    ("No layout format available for gathermatmul"), and with
    [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
    it ran out of memory (over 22 GB of RAM + 8 GB of swap, limited with `docker run --memory`), as did the CPU with
    the UD-Q4_K_XL. The converted model uses a `GatherMatmul` operation of the GGUF frontend, so it cannot be saved
    as an IR for OVMS either.
  - The official INT4 model ([OpenVINO/gemma-4-26b-a4b-it-int4-ov](https://huggingface.co/OpenVINO/gemma-4-26b-a4b-it-int4-ov),
    not QAT, 15 GB) crashed the GPU runtime while loading ("Abort was called at 90 line in file:
    .../linear_stream.h"), and with `GPU_ENABLE_LARGE_ALLOCATIONS` it used over 30 GB of RAM.
    Converting the QAT weights ([google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized))
    would need over 52 GB of RAM for the BF16 weights.
  - A draft model on the NPU would need to be much faster than the 26B A4B on the iGPU or the CPU (9–13 t/s with
    llama.cpp), which no Gemma 4 model is on this NPU (E4B: 9 t/s), and the draft model support with the NPU gives
    wrong output (see above). Use [../llama-cpp-l14](../llama-cpp-l14) for this model.

## Model comparison on the NPU
Measured on 2026-09-26 with OVMS 2026.4.0, `max_prompt_len` 4096, greedy decoding, thinking off.
TTFT = time to first token; decode = generation speed after the first token;
quality = correct answers to 8 short reasoning and knowledge questions when asked to answer with the result only
(a rough check that mostly measures following the answer format).

| Model (Hugging Face)                         | Size | Load  | Decode      | TTFT short / 1.1-1.6k | RAM   | Quality |
|----------------------------------------------|------|-------|-------------|-----------------------|-------|---------|
| OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov    | 3.8B | 21 s  | 13.5 t/s    | 3.8 s / 7.4 s         | 6 GB  | -       |
| FluidInference/qwen3-4b-int4-ov-npu          | 4B   | 25 s  | 11.3 t/s    | 3.8 s / 7.5 s         | 5 GB  | 4/8     |
| OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov | 7B   | 33 s  | 9.8 t/s     | 6.0 s / 11.6 s        | 6 GB  | -       |
| OpenVINO/Qwen3-8B-int4-cw-ov                 | 8B   | 30 s  | 8.1-8.7 t/s | 5.7 s / 11.1 s        | 11 GB | 4/8     |
| llmware/phi-4-npu-ov                         | 14B  | 115 s | 4.3 t/s     | 10.7 s / 20.7 s       | 11 GB | 4/8     |
| Echo9Zulu/Qwen3-14B-int4_sym-ov (group-wise) | 14B  | 100 s | 3.3 t/s     | 19.6 s / 37.7 s       | 13 GB | 6/8     |

Phi-3.5 and Mistral were run with an earlier, lenient version of the quality check: 2/4 and 3/4 correct.

- Qwen3-8B was chosen as the best balance: according to published benchmarks it is clearly smarter than the
  3.8-7B models and roughly on par with Phi-4, while generating ~2x faster than the 14B models,
  which would take minutes for a typical 500-token thinking phase.
  Qwen3-14B is the smartest option if ~3.5 t/s is acceptable.
- The load time is the NPU compilation; later starts use the compiled blobs in the cache directory.
- For comparison, llama.cpp on the CPU generates 16 t/s with Qwen3-4B Q4_K_M,
  and the NPU is several times slower than the CPU with llama.cpp's OpenVINO backend.
- llama.cpp's OpenVINO backend compiles a static graph for the whole context, which makes it unusable on the NPU
  beyond ~512 tokens of context (see [../llama-cpp-l14/README.md](../llama-cpp-l14/README.md)).
  OVMS uses OpenVINO GenAI's NPU LLM pipeline instead,
  which handles the KV cache inside the NPU model ("stateful") and processes the prompt in chunks.
  It requires models with symmetric INT4 weights, preferably channel-wise (`-cw-ov` on Hugging Face).

## NPU settings
- `GENERATE_HINT=BEST_PERF` made decoding slower (6.0 vs 8.7 t/s), and `NPUW_LLM_PREFILL_CHUNK_SIZE=256` crashed OVMS.
- `max_prompt_len` (the NPU prompt size limit) 8192: a 7987-token prompt took 66 s to the first token (~120 t/s),
  and then decoded at 5.2 t/s; short prompts are as fast as with 4096.
  The container used 13.6 GB of RAM after loading and 17.0 GB after a 7.8k-token prompt
  (docker stats, including the page cache), with 14.9 GB still available.
- `max_prompt_len` 16384 works too (a 16166-token prompt: 202 s to the first token, then 5.6 t/s,
  and it found a password hidden in the middle), but the container grew to 26.2 GB
  with only 2.7 GB of RAM left available, which risks running out of memory with a desktop session.
- These were measured with OVMS 2026.4.0. With the 2026.5 weekly build, the NPU generated at 7.7–8.5 t/s with the
  same 5.3 s to the first token (`max_prompt_len` 4096).
