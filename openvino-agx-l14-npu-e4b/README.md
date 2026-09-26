# Gemma 4 E4B on the NPU of the ThinkPad L14 Gen 5

Runs Gemma 4 E4B on the NPU of the ThinkPad L14 Gen 5 (Intel Core Ultra 5 125U, Meteor Lake NPU 3720)
with the third-party server [ryugyosoft/gemma-4-E4B-it-npu](https://huggingface.co/ryugyosoft/gemma-4-E4B-it-npu),
entirely in a container ([Dockerfile](Dockerfile)).
The server rewrites the Gemma 4 graphs for the NPU and manages the KV cache in Python,
which neither llama.cpp nor OpenVINO Model Server can do for Gemma 4 on this NPU
(see [../openvino-agx-l14](../openvino-agx-l14) and [../llama-cpp-agx-l14](../llama-cpp-agx-l14)).

## Setup
The model files are ~15.5 GB.
```sh
docker compose pull
docker compose run --rm download
docker compose up -d
```
The image is built by [deploy-docker-gemma4-npu.yml](../.github/workflows/deploy-docker-gemma4-npu.yml).
To build it locally instead, run:
```sh
docker build -t ghcr.io/agenttix/agx-ai/gemma4-npu:latest .
```
The Python dependencies are locked in [uv.lock](uv.lock). To update it, run in Ubuntu 26.04 with its system Python
(see [pyproject.toml](pyproject.toml)):
```sh
uv lock
```

- The first start compiles the model for the NPU (~10 min, 578-632 s measured).
  Later starts load the compiled blobs from `models/npu_cache` (9 GB) in ~30 s.
  The blobs depend on the NPU driver version in the image, so a driver update recompiles them,
  and the old blobs have to be deleted by hand.
- Memory: the container peaked at ~24 GB of RAM while compiling and uses ~15-18.5 GB when loaded
  (the 64-token prefill decoder is loaded only with >= 24 GB of RAM, see `block_sizes()` in `gemma4_npu.py`).
- Do not run this at the same time as [../llama-cpp-agx-l14](../llama-cpp-agx-l14)
  or [../openvino-agx-l14](../openvino-agx-l14), as the RAM would run out.

## Usage
- The API is OpenAI-compatible at http://localhost:9934/v1 (`/v1/chat/completions`, streaming, image input),
  and a chat UI is at http://localhost:9934/. The model id is `gemma-4-e4b-it-npu`.
- The context is limited to 1024 tokens (prompt and answer together, a static KV cache).
  Longer prompts are rejected with HTTP 400.
- Tool calls and thinking are not supported by the server.

## Performance
Measured on 2026-09-26 (default decoder, not `--fast`):
- Generation 8.1-8.6 t/s. Time to the first token 0.9 s for an 18-token prompt
  and 3.7-3.8 s for a 684-token prompt (~185 t/s), after which generation slows to 5.8-5.9 t/s.
- Image input works: a 293-token prompt with an image took 3.7-3.8 s to the first token.
- 5/8 correct answers to the short questions used for the models in [../openvino-agx-l14](../openvino-agx-l14)
  (Qwen3-8B: 4/8), i.e. about as fast as Qwen3-8B with OVMS, with image input, but with a much shorter context.
- The published image gave the same results as a local build.

## Code review
Code review of the pinned commit `e120bd9` (2026-09-26): nothing malicious was found
(no shell commands, `eval`, pickle, hidden downloads or data exfiltration; the web UI escapes the model output).
Notes:
- The repository was new (created 2026-09-24, 0 downloads) and the model IRs were built by the author's scripts,
  so the weights cannot be verified against Google's release.
- `server.py` downloads `image_url` http(s) URLs given by the client, and the API has no authentication,
  so the port is published only on localhost.
- The upstream `requirements.txt` uses `--extra-index-url`, which is open to dependency confusion.
  Here the extra indexes are restricted to their packages ([pyproject.toml](pyproject.toml), [uv.lock](uv.lock)),
  and the Dockerfile checks the code files and the NPU driver against the checksums of the reviewed versions.
- OpenVINO includes telemetry (`openvino-telemetry`).
  In the container, it logs "Could not create directory for storing client ID. No data will be sent."
