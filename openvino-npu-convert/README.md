# OpenVINO converter for the Intel NPU

A container image with the converters of [`../agx_ai/openvino`](../agx_ai/openvino),
which prepare models for OpenVINO Model Server (OVMS) on the Intel NPU.
It is used by these configurations, whose `docker-compose.yml` runs it as a `convert` service before OVMS:

| Configuration | Converter | What it does |
|---|---|---|
| [`../openvino-l14-e4b`](../openvino-l14-e4b) | [`convert_gemma4.py`](../agx_ai/openvino/convert_gemma4.py) | Gemma 4 E4B with the language model recompressed to symmetric INT4 for NPU 3720 |
| [`../openvino-l14-embeddinggemma2`](../openvino-l14-embeddinggemma2) | [`convert_embeddinggemma2.py`](../agx_ai/openvino/convert_embeddinggemma2.py) | Export of the text model of EmbeddingGemma 2 to OpenVINO IR |

- The converter and its arguments (e.g. the model revisions) are given as the `command` in the `docker-compose.yml`
  of the configuration, e.g. `python -m agx_ai.openvino.convert_gemma4 /models/... --int4-revision ...`.
- The models are downloaded to the Hugging Face cache of the host, and the result is written to a Docker volume
  that OVMS reads. The conversion is skipped if the volume already contains a conversion of the same revisions with
  the same settings and library versions (`export.json`, see [`export.py`](../agx_ai/openvino/export.py)).
- The image is based on Ubuntu 26.04 (Python 3.14), with the dependencies installed by uv from [`uv.lock`](uv.lock)
  (OpenVINO, openvino-tokenizers, NNCF, PyTorch for the CPU and transformers).
  The same dependencies are in the `openvino` extra of [`../pyproject.toml`](../pyproject.toml), so that the
  converters can be run, linted and type-checked on the host after `uv sync --extra openvino`.
- OpenVINO and openvino-tokenizers should match the OpenVINO version of the OVMS images,
  so Dependabot does not update them (see [`../.github/dependabot.yml`](../.github/dependabot.yml)).
  The OVMS images are currently a 2026.5 weekly build, which is not on PyPI, so the converter uses 2026.4.1.

The image is built by
[`deploy-docker-openvino-npu-convert.yml`](../.github/workflows/deploy-docker-openvino-npu-convert.yml).
The build context is the root of the repository, which [`Dockerfile.dockerignore`](Dockerfile.dockerignore)
limits to the needed files. To build it locally instead, run in the root of the repository:
```sh
docker build -f openvino-npu-convert/Dockerfile -t ghcr.io/agenttix/agx-ai/openvino-npu-convert:latest .
```
To update the lock file, run in this directory in Ubuntu 26.04 with its system Python (see [`pyproject.toml`](pyproject.toml)):
```sh
uv lock
```
