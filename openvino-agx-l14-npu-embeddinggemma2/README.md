# EmbeddingGemma 2 on the NPU of the ThinkPad L14 Gen 5

Serves the text embeddings of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
on the NPU of the ThinkPad L14 Gen 5 with OpenVINO Model Server (OVMS).
There is no OpenVINO version of the model from Google or Intel, so [convert.py](convert.py) exports the text model
from Google's weights to OpenVINO IR in a container ([Dockerfile](Dockerfile)).
The vision and audio encoders are not exported.

## Hardware and software
- OS: Ubuntu 26.04.1 LTS (Kubuntu), kernel 7.0.0
- CPU: Intel Core Ultra 5 125U (Meteor Lake, 2 P-cores with HT + 8 E-cores + 2 LP E-cores)
- RAM: 32 GB (2 x 16 GB DDR5-5600 SO-DIMM), shared by the CPU, the iGPU and the NPU
- NPU: Intel AI Boost (Meteor Lake NPU 3720), kernel driver intel_vpu, user-mode driver 1.38.0 in the OVMS image
- GPU: Intel Graphics (Meteor Lake iGPU, Xe-LPG architecture, 4 Xe cores), no dedicated VRAM (uses the shared RAM)

## Usage
```sh
docker compose pull
docker compose up -d
```
- On the first start, the `convert` container downloads the model files (1.5 GB) to the Hugging Face cache of the host
  (`~/.cache/huggingface/hub`, of which only the `hub` directory is mounted),
  exports the text model to the Docker volume `openvino-agx-l14-npu-embeddinggemma2_models` (0.6 GB) and exits.
  OVMS starts after it and compiles the model for the NPU in ~10 s (cached in the same volume, 0.5 GB).
  Later starts skip the export, as long as the revision and the library versions are the same (`export.json`).
- The API is OpenAI-compatible at http://localhost:9935/v3/embeddings (also `/v1/embeddings`).
  The model id is `google/embeddinggemma-2-npu`. The embeddings have 768 dimensions and are normalized.
  Truncating them to 512, 256 or 128 dimensions (Matryoshka) has to be done by the client (and renormalized),
  as OVMS does not support the `dimensions` parameter.
- The inputs are limited to 512 tokens, and longer inputs are rejected with HTTP 400 ("Input length ... longer than
  allowed 512"), so the documents have to be chunked accordingly (~1500-2000 characters of English text).
  `--truncate true` would truncate them instead.
- Use the task prefixes of the [model card](https://huggingface.co/google/embeddinggemma-2#1-task-instruction-prefixes),
  e.g. `task: search result | query: {query}` for queries and `title: none | text: {document}` for documents.
- The container uses ~2 GB of RAM. It can run at the same time as [../openvino-agx-l14-npu-e4b](../openvino-agx-l14-npu-e4b)
  on the NPU (tested), but they share the NPU (see below).

The converter image is built by
[deploy-docker-embeddinggemma2-ov-convert.yml](../.github/workflows/deploy-docker-embeddinggemma2-ov-convert.yml).
To build it locally instead, run:
```sh
docker build -t ghcr.io/agenttix/agx-ai/embeddinggemma2-ov-convert:latest .
```
The Python dependencies are locked in [uv.lock](uv.lock). To update it, run in Ubuntu 26.04 with its system Python
(see [pyproject.toml](pyproject.toml)):
```sh
uv lock
```
The OpenVINO version of the converter should match that of the OVMS image in [docker-compose.yml](docker-compose.yml).

## Performance
Measured on 2026-10-14 with [../benchmark/llama_cpp_bench_http.py](../benchmark/llama_cpp_bench_http.py)
`--tests embedding` (results in `../benchmark/results/agx-l14-kubuntu.jsonl`), OVMS 2026.4.1:

| Device, `--max_length` | 100-token texts | 500-token texts | Single text |
|------------------------|-----------------|-----------------|-------------|
| NPU, 512               | 606-612 t/s     | 3067-3085 t/s   | 0.16 s      |
| NPU, 1023              | 128 t/s         | 643 t/s         | 0.78 s      |
| CPU (dynamic shapes)   | 837 t/s         | 927 t/s         | 0.12-0.78 s |

- The NPU runs a static graph, so every input is padded to `--max_length`, and each input takes the same time
  regardless of its length. With 512, the NPU is 3.3x faster than the CPU for document chunks,
  but slower than the CPU for short queries, and it leaves the CPU free.
- With Gemma 4 E4B generating on the NPU at the same time ([../openvino-agx-l14-npu-e4b](../openvino-agx-l14-npu-e4b)),
  the embeddings ran at 2063 t/s (500-token texts), and the generation of one of the requests slowed from 8.3 to 7.0 t/s.
- The embeddings match sentence-transformers (float32 on the CPU) with a cosine similarity of 0.99999-1.00000
  (inputs of 15-289 tokens, on the NPU and the CPU), and the tokenizer gives the same token IDs.

## Notes
- The export ([convert.py](convert.py)) wraps the text model of `EmbeddingGemma2Model`, which includes the projection
  to 768 dimensions, and outputs the per-token embeddings. OVMS does the mean pooling and the normalization
  (`--pooling MEAN`), like the sentence-transformers configuration of the model (`modules.json`).
- The text config inherits `max_position_embeddings` 262144 from Gemma 4, which OVMS uses as the input limit,
  so the export sets it to 8192, the context length on the model card.
- OVMS handles the NPU in two ways:
  - `--max_length` below 1024: the model is reshaped to a static shape of 1 x `max_length`, and the inputs are padded.
    This is the configuration used here. 1023 works, but is 5x slower than 512 (see above).
  - `--max_length` 1024 or more: the model is compiled with the NPUW LLM pipeline (`NPUW_TEXT_EMBED`).
    The NPU plugin of OpenVINO 2026.4 then took the expand and reshape of the grouped-query attention for a KV cache,
    handled the model as a causal decoder and failed to compile it ("Failed to check SDPA nodes").
    The export therefore repeats the key/value heads with a Gather (`repeat_kv` in [convert.py](convert.py)),
    after which 2048 and 8192 compile (31 s and 381 s, 13 GB of RAM for 8192) and work for single inputs,
    but requests with several inputs fail in OVMS: its CPU post-processing for the NPU gets the hidden states
    padded to `max_length` but the unpadded attention masks ("Input shape dimension equal 5 cannot be broadcasted
    to 2048"). As most clients send batches, this is not usable until OVMS fixes it.
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
