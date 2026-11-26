# EmbeddingGemma 2 on the NPU (or the iGPU) of the ThinkPad L14 Gen 5

Serves the text embeddings of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
on the NPU of the ThinkPad L14 Gen 5 with OpenVINO Model Server (OVMS).
There is no OpenVINO version of the model from Google or Intel,
so [convert_embeddinggemma2.py](../agx_ai/openvino/convert_embeddinggemma2.py) exports the text model from Google's weights
to OpenVINO IR in a container ([../openvino-npu-convert](../openvino-npu-convert/README.md)).
The vision and audio encoders are not exported.

| Model | Device | Context | NPU memory | RAM | 100-token texts | 500-token texts | Notes |
|---|---|---|---|---|---|---|---|
| EmbeddingGemma 2, FP32 IR compiled in FP16 ([convert_embeddinggemma2.py](../agx_ai/openvino/convert_embeddinggemma2.py)) | **NPU (default)** | 512 tokens per input | shared RAM | 1.8 GiB | 692 t/s | 3518 t/s | 0.15 s per input of any length, as the NPU pads every input to 512 tokens |
| the same | iGPU (`ovms-embeddinggemma2-gpu`) | 512 tokens per input | shared RAM | | 3594 t/s | 3076 t/s | 0.07 s for a 100-token input; slows down an LLM on the iGPU by half |

- 12 % faster than the previous export (FP16 IR with the default attention masks, 618 and 3133 t/s),
  and closer to the reference (cosine similarity 0.99999 instead of 0.99997), see [Performance](#performance).
- The NPU is the default, as it leaves the iGPU to the LLM servers ([../openvino-l14](../openvino-l14) and
  [../openvino-l14-e4b](../openvino-l14-e4b)), which then run 10-20 % slower while embeddings are computed
  on the NPU, but at half speed with the embeddings on the iGPU (see
  [../openvino-l14/README.md](../openvino-l14/README.md#using-the-cpu-the-igpu-and-the-npu-together)).
  When no LLM runs on the iGPU, the iGPU variant is faster, 5x for short texts.
  The embeddings of the iGPU and the NPU agree (cosine similarity 0.99998-1.00000 for 5 texts).
- Faster than the CPU of this laptop for document chunks (927 t/s with dynamic shapes),
  but slower for short queries (0.12 s), and it leaves the CPU free.
  [../vllm-zen2-embeddinggemma2](../vllm-zen2-embeddinggemma2) on a Threadripper 3970X is faster (4650-5000 t/s).

## Hardware and software
- Computer: Lenovo ThinkPad L14 Gen 5 (Intel)
- OS: Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0
- CPU: Intel Core Ultra 5 125U (Meteor Lake, 2 P-cores with HT + 8 E-cores + 2 LP E-cores)
- RAM: 32 GB (2 x 16 GB DDR5-5600 SO-DIMM), shared by the CPU, the iGPU and the NPU
- NPU: Intel AI Boost (Meteor Lake NPU 3720), kernel driver intel_vpu, user-mode driver 1.38.0 in the OVMS image
- GPU: Intel Graphics (Meteor Lake iGPU, Xe-LPG architecture, 4 Xe cores), no dedicated VRAM (uses the shared RAM),
  kernel driver i915
- OVMS 2026.5.0 (weekly build of 2026-11-19, the same as in [../openvino-l14-e4b](../openvino-l14-e4b)),
  OpenVINO 2026.4.1 and transformers 5.24.1 in the converter

## Usage
```sh
docker compose pull
docker compose up -d
```
- On the first start, the `convert` container downloads the model files (1.5 GB) to the Hugging Face cache of the host
  (`~/.cache/huggingface/hub`, of which only the `hub` directory is mounted),
  exports the text model to the Docker volume `openvino-l14-embeddinggemma2_models` (1.1 GB) and exits (~1 min).
  OVMS starts after it and compiles the model for the NPU in ~15 s (cached in the same volume, 0.5 GB).
  Later starts skip the export, as long as the revision, the export settings and the library versions are the same
  (`export.json`).
- The API is OpenAI-compatible at http://localhost:9935/v3/embeddings (also `/v1/embeddings`).
  The model id is `google/embeddinggemma-2-npu` (`google/embeddinggemma-2` on the iGPU).
  The embeddings have 768 dimensions and are normalized.
- The iGPU variant is started instead with `docker compose up -d ovms-embeddinggemma2-gpu` (stop the other one first,
  as they use the same port). It compiles the model for the iGPU in a few seconds.
  Truncating them to 512, 256 or 128 dimensions (Matryoshka) has to be done by the client (and renormalized),
  as OVMS does not support the `dimensions` parameter.
- The inputs are limited to 512 tokens, and longer inputs are rejected with HTTP 400 ("Input length ... longer than
  allowed 512"), so the documents have to be chunked accordingly (~1500-2000 characters of English text).
  `--truncate true` would truncate them instead.
- Use the task prefixes of the [model card](https://huggingface.co/google/embeddinggemma-2#1-task-instruction-prefixes),
  e.g. `task: search result | query: {query}` for queries and `title: none | text: {document}` for documents.
- The container uses 1.8 GiB of RAM. It can run at the same time as [../openvino-l14-e4b](../openvino-l14-e4b)
  (tested), but they share the NPU (see below).

The converter image, which is shared with [../openvino-l14-e4b](../openvino-l14-e4b),
is described in [../openvino-npu-convert](../openvino-npu-convert/README.md).
The exported revision is set in the `command` of the `convert` service in [docker-compose.yml](docker-compose.yml).

## Performance
Measured on 2026-11-25 with [../agx_ai/benchmark/bench_http.py](../agx_ai/benchmark/bench_http.py)
`--tests embedding --embed-texts 16` (results in `../benchmark/results/agx-l14-kubuntu.jsonl`), `--max_length 512`.
The accuracy is the cosine similarity to the embeddings of transformers in FP32 on the CPU (mean pooling and
normalization like sentence-transformers) for 28 texts of 22-493 tokens (15 queries and 13 documents),
an ad-hoc check that is not in the repository:

| Export, OVMS | 100-token texts | 500-token texts | Single text | Cosine similarity min / mean | RAM |
|---|---|---|---|---|---|
| **FP32 IR, additive masks, 2026.5 weekly (this)** | **692 t/s** | **3518 t/s** | 0.15 s | 0.99999 / 1.00000 | 1.8 GiB |
| FP16 IR, additive masks, 2026.5 weekly | 628 t/s | 3186 t/s | 0.16 s | 0.99997 / 0.99999 | 2.2 GiB |
| FP16 IR (previous export), 2026.5 weekly | 618 t/s | 3131 t/s | 0.16 s | 0.99997 / 0.99999 | |
| FP16 IR (previous export), 2026.4.1 | 618 t/s | 3133 t/s | 0.16 s | 0.99997 / 0.99999 | 2 GiB |
| FP16 IR (previous), INT8 weights (NNCF `INT8_SYM`) | 625 t/s | 3157 t/s | 0.16 s | 0.99967 / 0.99982 | |
| FP16 IR (previous), INT8 weights and activations (NNCF PTQ, 96 calibration texts) | 676 t/s | 3418 t/s | 0.15 s | 0.99597 / 0.99816 | |

Measured on 2026-11-25 with OVMS 2026.5 weekly and the FP32 IR on other devices (`--tests embedding --embed-texts 16`,
and an ad-hoc test of 12 concurrent requests of 16 texts each, 2 or 4 at a time, over one or two servers):

| Device | 100-token texts | 500-token texts | Single text | Concurrent, 100 / 500-token texts |
|---|---|---|---|---|
| NPU (this) | 692 t/s | 3518 t/s | 0.15 s | 449 / 2208 t/s |
| iGPU (dynamic shapes) | 3594 t/s | 3076 t/s | 0.07 s (100 tokens), 0.16 s (500 tokens) | 4353 / 3912 t/s |
| CPU (dynamic shapes) | 905 t/s | 688 t/s | 0.12 s (100 tokens), 0.81 s (500 tokens) | |
| NPU and iGPU, round robin over two servers | | | | 898 / 4345 t/s |

- The round robin over the NPU and the iGPU waits for the slower NPU, so it was slower than the iGPU alone for short
  texts and only 11 % faster for long ones. A load balancer that sends more requests to the iGPU could do better,
  but the iGPU is better left to the LLMs (see above).

Measured on 2026-10-14 with OVMS 2026.4.1 and the previous export:

| Device, `--max_length` | 100-token texts | 500-token texts | Single text |
|------------------------|-----------------|-----------------|-------------|
| NPU, 512               | 606-612 t/s     | 3067-3085 t/s   | 0.16 s      |
| NPU, 1023              | 128 t/s         | 643 t/s         | 0.78 s      |
| CPU (dynamic shapes)   | 837 t/s         | 927 t/s         | 0.12-0.78 s |

- The NPU runs a static graph, so every input is padded to `--max_length`, and each input takes the same time
  regardless of its length.
- The NPU is saturated by one request: the inputs of a request are run one by one, and 2 or 4 parallel requests
  did not increase the throughput (6.3 inputs/s with the previous export), neither with
  `PERFORMANCE_HINT` `THROUGHPUT`.
- The INT8 quantization of the activations was 9 % faster, but changed the top result for 1 of the 15 queries,
  so it was not used. The INT8 weights did not help, as the model is not limited by the weights on the NPU.
- With Gemma 4 E4B generating on the NPU at the same time ([../openvino-l14-e4b](../openvino-l14-e4b)),
  the embeddings ran at 1869 t/s (500-token texts), and with embeddings computed continuously,
  the generation slowed from 8.3 to 3.9 t/s.

## Notes
- The export ([convert_embeddinggemma2.py](../agx_ai/openvino/convert_embeddinggemma2.py)) wraps the text model of `EmbeddingGemma2Model`, which includes the projection
  to 768 dimensions, and outputs the per-token embeddings. OVMS does the mean pooling and the normalization
  (`--pooling MEAN`), like the sentence-transformers configuration of the model (`modules.json`).
- Optimizations of the export, found by profiling the model on the NPU (`PERF_COUNT`) with a static 1 x 512 shape:
  - The IR is saved with FP32 weights (`compress_to_fp16=False`, 1.1 GB instead of 0.6 GB).
    With FP16 weights, the decompression nodes of the weights stopped the constant folding after OVMS reshaped
    the model to the static shape, so e.g. the rotary embeddings (`Sin` and `Cos`) were computed on the slow SHAVE
    cores on every inference. The NPU compiles the model in FP16 anyway, and the result is also more accurate.
  - The attention masks are given to the model as additive float masks: the padding as a vector over the keys
    and the sliding window as a constant band. The default masks of transformers are boolean 512 x 512 masks built
    at run time (`BitwiseAnd` and `Select` on the SHAVE cores). The window of 512 covers all 512 tokens,
    so here both masks are the padding mask. This made the inference 2 % faster.
  - A static IR (1 x 512, constants folded at the conversion) was as fast as the FP32 IR,
    but OVMS pads the inputs only for a dynamic model, and failed with it.
  - Did not help: replacing the `Power` of the RMSNorms with `Multiply` and `Sqrt`,
    and the NPU options `NPU_TURBO`, `NPU_TILES`, `NPU_DPU_GROUPS`, `optimization-level=3` and the performance hints.
    The plugin configuration of OVMS (`--plugin_config`) is also given to the CPU plugin (tokenizer),
    which rejects the NPU-specific options.
- The text config inherits `max_position_embeddings` 262144 from Gemma 4, which OVMS uses as the input limit,
  so the export sets it to 8192, the context length on the model card.
- OVMS handles the NPU in two ways:
  - `--max_length` below 1024: the model is reshaped to a static shape of 1 x `max_length`, and the inputs are padded.
    This is the configuration used here. 1023 works, but is 5x slower than 512 (see above).
  - `--max_length` 1024 or more: the model is compiled with the NPUW LLM pipeline (`NPUW_TEXT_EMBED`).
    The NPU plugin of OpenVINO 2026.4 then took the expand and reshape of the grouped-query attention for a KV cache,
    handled the model as a causal decoder and failed to compile it ("Failed to check SDPA nodes").
    The export therefore repeats the key/value heads with a Gather (`repeat_kv` in [convert_embeddinggemma2.py](../agx_ai/openvino/convert_embeddinggemma2.py)),
    after which 2048 and 8192 compile (31 s and 381 s, 13 GB of RAM for 8192) and work for single inputs,
    but requests with several inputs fail in OVMS: its CPU post-processing for the NPU gets the hidden states
    padded to `max_length` but the unpadded attention masks ("Input shape dimension equal 5 cannot be broadcasted
    to 2048"). As most clients send batches, this is not usable until OVMS fixes it.
    The OVMS 2026.5 weekly build of 2026-11-19 still has this bug.
- llama.cpp ([ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), build 11610,
  `server-openvino` image, `GGML_OPENVINO_DEVICE=NPU`): the model loads in 20 s, but the NPU graph is compiled for the
  first input length (the warm-up, or the first request with `--no-warmup`), and every input of another length fails
  ("The input tensor size is not equal to the model input type: got [1,1,1,15] expecting [1,1,1,2]").
  The prompts of LLMs are padded to the prefill chunk size on the NPU, but the embedding inputs are not.
  `GGML_OPENVINO_PREFILL_CHUNK_SIZE` and `-ub` did not help. Hence there is no llama.cpp configuration.
  Note that `OPENVINO0` is now the CPU and `OPENVINO1` the NPU in this build.
- vLLM does not support the Intel NPU: its OpenVINO plugin (vllm-openvino) supports only CPUs and GPUs,
  and only text generation.
- [wesjos/embeddinggemma-2-openvino](https://huggingface.co/wesjos/embeddinggemma-2-openvino) is a third-party
  OpenVINO export, but it outputs the pooled embeddings, which OVMS does not support, and it was not used.
