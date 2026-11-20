# llama.cpp for DiffusionGemma

Docker images of llama.cpp with support for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(`google/diffusiongemma-26B-A4B-it`), Google's block-diffusion language model based on Gemma 4 26B-A4B. Instead of one
token at a time, it fills a canvas of 256 tokens at once and refines it over up to 48 denoising steps (typically 15-25
per block for chat), each of which is a forward pass over the whole canvas.

DiffusionGemma is not in upstream llama.cpp (2026-11-20): `llama-server` fails with "unknown model architecture:
'diffusion-gemma'" ([ggml-org/llama.cpp#32682](https://github.com/ggml-org/llama.cpp/issues/32682)). There are two
draft pull requests:

- [ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423) (Unsloth): a CLI and servers that talk
  over stdin/stdout to a driver (Unsloth Studio), no HTTP server.
- [ggml-org/llama.cpp#24427](https://github.com/ggml-org/llama.cpp/pull/24427) (NVIDIA): `llama-diffusion-gemma-server`,
  an OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/props`, `/metrics`, `/slots`, an API
  key) with the denoising loop and the sampling on the GPU. These images build it.

The images build the head of #24427 (commit `dd0cf04`) with
[`server-endpoints.patch`](server-endpoints.patch), which adds to the server:

- the llama-server endpoints `/completion` (with `n_predict`), `/tokenize` and `/detokenize`, so that
  [`../agx_ai/benchmark/bench_http.py`](../agx_ai/benchmark/bench_http.py) works with it, and
- the model and context parameters of the command line that the server ignored: `--n-cpu-moe` and `-ot` (needed on the
  Radeon VII), `--cache-type-k`/`-v`, `--flash-attn`, `--no-kv-offload`, `--tensor-split`, `--split-mode` and the
  threads.

| Image | Dockerfile | Base | Used by |
|---|---|---|---|
| `ghcr.io/agenttix/agx-ai/llama-cpp-diffusion-gemma` | [`Dockerfile`](Dockerfile) | `nvidia/cuda:12.8.1` (sm 86, 89, 120) | [`../llama-cpp-rtx3090/docker-compose-diffusiongemma.yml`](../llama-cpp-rtx3090/docker-compose-diffusiongemma.yml) |
| `ghcr.io/agenttix/agx-ai/llama-cpp-diffusion-gemma-gfx906` | [`Dockerfile.gfx906`](Dockerfile.gfx906) | `mixa3607/rocm-gfx906:10.1-complete` (the ROCm of [`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii)) | [`../llama-cpp-radeon-vii/docker-compose-diffusiongemma.yml`](../llama-cpp-radeon-vii/docker-compose-diffusiongemma.yml) |

Both are built by
[`../.github/workflows/deploy-docker-llama-cpp-diffusion-gemma.yml`](../.github/workflows/deploy-docker-llama-cpp-diffusion-gemma.yml).
The CUDA-named optimizations of the pull request (GPU sampling, device self-conditioning, the device denoising loop)
also work with HIP on the Radeon VII (`/props` reports them on).

## Using the server

The model is a GGUF of Unsloth, e.g. `unsloth/diffusiongemma-26B-A4B-it-GGUF` `Q4_K_M` (15.6 GiB), given with
`--model` as a path (the server does not download). The server has one slot. Differences from `llama-server`:

- The KV cache is full-size in all 30 layers, also in the 25 sliding-window ones, as the canvas attends to the whole
  prompt: ~200 KiB per token with f16 (1600 MiB for 8192 tokens), about half that with q8_0. Autoregressive
  Gemma 4 26B-A4B needs this only in its 5 global layers.
- The server keeps an F16 copy of the token embedding on the GPU (1.4 GiB) for the self-conditioning, in addition to
  the weights. If it does not fit, the server falls back to a 2.75 GiB F32 copy, and then to a slow path on the CPU.
- `max_tokens` is rounded up to whole canvases of 256 tokens. `usage.completion_tokens` is the canvas tokens,
  `timings.predicted_n` the answer tokens. Streaming sends the whole answer in one chunk at the end.
- The sampling follows the model's own schedule (entropy-bounded, temperature 0.4-0.8), and `temperature` is ignored.
  `seed` works.
- The raw `/completion` of a repeated paragraph (the `generation` test of `bench_http`) stops after a few tokens:
  without the chat template, the model fills the canvas with end-of-text tokens. Use the `chat` and `depth` tests.

Benchmarks: [`../llama-cpp-rtx3090/README.md`](../llama-cpp-rtx3090/README.md#diffusiongemma) and
[`../llama-cpp-radeon-vii/README.md`](../llama-cpp-radeon-vii/README.md#diffusiongemma).
