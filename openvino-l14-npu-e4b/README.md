# Gemma 4 E4B on the NPU of the ThinkPad L14 Gen 5

Runs Gemma 4 E4B on the NPU of the ThinkPad L14 Gen 5 (Intel Core Ultra 5 125U, Meteor Lake NPU 3720)
with the third-party server [ryugyosoft/gemma-4-E4B-it-npu](https://huggingface.co/ryugyosoft/gemma-4-E4B-it-npu),
entirely in a container ([Dockerfile](Dockerfile)).
The server rewrites the Gemma 4 graphs for the NPU and manages the KV cache in Python,
which neither llama.cpp nor OpenVINO Model Server can do for Gemma 4 on this NPU
(see [../openvino-l14](../openvino-l14) and [../llama-cpp-l14](../llama-cpp-l14)).

## Setup
```sh
docker compose pull
docker compose up -d
```
At startup, [start.py](start.py) downloads the model files (~15.5 GB) of the reviewed commit
to the Hugging Face cache of the host (`~/.cache/huggingface/hub`) if they are missing, and then starts the server.
Without a network connection, the cached files are used.
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

- The first start compiles the model for the NPU (~10 min, 578-653 s measured).
  Later starts load the compiled blobs (9 GB) from the Docker volume `openvino-l14-npu-e4b_npu-cache` in 20-30 s.
  The Dockerfile patches `gemma4_npu.py` to use this volume instead of the model directory, which is in the Hugging Face cache.
  The blobs depend on the NPU driver and OpenVINO versions in the image, so updating either recompiles them.
  To delete the old blobs, recreate the volume with `docker compose down -v` before the update.
  The blobs are readable only by the user that created them, so the volume should be recreated also when changing
  `user` in [docker-compose.yml](docker-compose.yml).
- Memory: the container peaked at ~24 GB of RAM while compiling and uses ~15-18.5 GB when loaded
  (the 64-token prefill decoder is loaded only with >= 24 GB of RAM, see `block_sizes()` in `gemma4_npu.py`).
- Do not run this at the same time as [../llama-cpp-l14](../llama-cpp-l14)
  or [../openvino-l14](../openvino-l14), as the RAM would run out.

## Usage
- The API is OpenAI-compatible at http://localhost:9934/v1 (`/v1/chat/completions`, streaming, image input),
  and a chat UI is at http://localhost:9934/. The model id is `gemma-4-e4b-it-npu`.
- The context is limited to 1024 tokens (prompt and answer together, a static KV cache).
  Longer prompts are rejected with HTTP 400.
- Tool calls and thinking are not supported by the server.

## Performance
Measured on 2026-10-14 with [bench.py](bench.py) (default decoder, not `--fast`),
OpenVINO 2026.4.1, transformers 5.19.0 and NPU driver 1.38.0:
- Generation 8.2-8.6 t/s. Time to the first token 0.34 s for a 26-29-token prompt
  and 4.2-4.3 s for a 777-token prompt (~180 t/s), after which generation stays at 8.5 t/s.
- A 290-token prompt with a 640x480 image took 3.5-3.8 s to the first token.
- 8/8 correct answers to the short questions in `bench.py` (a different set from the one below).
- The OpenVINO nightly 2026.5.0.dev20261013, which upstream recommends, gave the same speeds and the same answers,
  so the release from PyPI is used.
- PyTorch is used only by the image processor of transformers for resizing the input image,
  and every model component runs on the NPU with OpenVINO, so a GPU build of PyTorch would not help.

Measured on 2026-09-26 with OpenVINO 2026.5.0b1 and transformers 5.5.4:
- Generation 8.1-8.6 t/s. Time to the first token 0.9 s for an 18-token prompt
  and 3.7-3.8 s for a 684-token prompt (~185 t/s), after which generation slows to 5.8-5.9 t/s.
- Image input works: a 293-token prompt with an image took 3.7-3.8 s to the first token.
- 5/8 correct answers to the short questions used for the models in [../openvino-l14](../openvino-l14)
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
- The container has network access, as it downloads the model files at startup.
  Only the `hub` directory of the Hugging Face cache is mounted, so the container cannot read e.g. a token
  in `~/.cache/huggingface`, but it can modify the other models in the cache.
- The upstream `requirements.txt` uses `--extra-index-url`, which is open to dependency confusion.
  Here OpenVINO comes from PyPI, and the PyTorch CPU index is restricted to its packages ([pyproject.toml](pyproject.toml), [uv.lock](uv.lock)),
  and the Dockerfile checks the code files and the NPU driver against the checksums of the reviewed versions.
