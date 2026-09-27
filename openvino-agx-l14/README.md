# OpenVINO Model Server on the NPU of the ThinkPad L14 Gen 5

OpenVINO Model Server (OVMS) for the NPU of the ThinkPad L14 Gen 5 (Intel Core Ultra 5 125U, Meteor Lake NPU 3720),
serving Qwen3-8B.
The llama.cpp server for the CPU and iGPU is in [../llama-cpp-agx-l14](../llama-cpp-agx-l14),
and Gemma 4 E4B on the NPU in [../openvino-agx-l14-npu-e4b](../openvino-agx-l14-npu-e4b).

llama.cpp's OpenVINO backend compiles a static graph for the whole context, which makes it unusable on the NPU
beyond ~512 tokens of context (see [../llama-cpp-agx-l14/README.md](../llama-cpp-agx-l14/README.md)).
OVMS uses OpenVINO GenAI's NPU LLM pipeline instead,
which handles the KV cache inside the NPU model ("stateful") and processes the prompt in chunks.
It requires models with symmetric INT4 weights, preferably channel-wise (`-cw-ov` on Hugging Face).

## Usage
```sh
docker compose up -d
```
- The model is downloaded from Hugging Face to `/home/mika/.cache/ovms` on the first start.
- The API is OpenAI-compatible at http://localhost:9933/v3 (e.g. `/v3/chat/completions`).
  The model id is `qwen/qwen3-8b-npu`.
- Thinking can be disabled per request with `"chat_template_kwargs": {"enable_thinking": false}`.
  This is recommended for interactive use, as Qwen3 often thinks for over 1000 tokens (2-3 min at ~8 t/s).
- Thinking and tool calls are returned in the separate `reasoning_content` and `tool_calls` fields (tested).
- Do not run this at the same time as [../llama-cpp-agx-l14](../llama-cpp-agx-l14):
  together they exhausted the 32 GB of RAM and the kernel killed the desktop session.
  Stop one with `docker compose down` before starting the other.

## Model comparison
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
  The NPU's advantage is that it leaves the CPU and iGPU free.
- Gemma 4 does not run on the NPU with llama.cpp or OVMS.
  The third-party server ryugyosoft/gemma-4-E4B-it-npu runs it with rewritten model graphs at 8.1-8.6 t/s,
  but with a 1024-token context, see [../openvino-agx-l14-npu-e4b](../openvino-agx-l14-npu-e4b).

## Settings
- `GENERATE_HINT=BEST_PERF` made decoding slower (6.0 vs 8.7 t/s), and `NPUW_LLM_PREFILL_CHUNK_SIZE=256` crashed OVMS.
- `max_prompt_len` (the NPU prompt size limit) 8192: a 7987-token prompt took 66 s to the first token (~120 t/s),
  and then decoded at 5.2 t/s; short prompts are as fast as with 4096.
  The container used 13.6 GB of RAM after loading and 17.0 GB after a 7.8k-token prompt
  (docker stats, including the page cache), with 14.9 GB still available.
- `max_prompt_len` 16384 works too (a 16166-token prompt: 202 s to the first token, then 5.6 t/s,
  and it found a password hidden in the middle), but the container grew to 26.2 GB
  with only 2.7 GB of RAM left available, which risks running out of memory with a desktop session.
