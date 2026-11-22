# llama.cpp with the DSpark drafter of Inkling 2 Mini

Images of llama-server for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) with its DSpark
block drafter, used by [`../llama-cpp-rtx3090/docker-compose-inkling.yml`](../llama-cpp-rtx3090/docker-compose-inkling.yml)
and [`../llama-cpp-radeon-vii/docker-compose-inkling.yml`](../llama-cpp-radeon-vii/docker-compose-inkling.yml).

Upstream llama.cpp runs the model since [ggml-org/llama.cpp#32694](https://github.com/ggml-org/llama.cpp/pull/32694)
with the fix [#32704](https://github.com/ggml-org/llama.cpp/pull/32704) (release b12611, first in the release images
with b12627 on 2026-11-22). Its DSpark drafter (`--spec-type draft-dspark` with Unsloth's
[`dspark-Inkling-2-Mini-Q8_0.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), 0.4 GB) is in the open
[#32218](https://github.com/ggml-org/llama.cpp/pull/32218) and in Unsloth's fork, which publishes binaries but no
server image (its Docker images are Unsloth Studio). [`dspark.patch`](dspark.patch) is the commit of #32218
(`0a9105d`), which applies to b12627 unchanged. The images build b12627 with it:

- [`Dockerfile`](Dockerfile): CUDA 13.5 for sm_86/89/120, like upstream's `.devops/cuda.Dockerfile` (`server` target),
  but only llama-server and without the embedded web UI (`config.ini` has `no-ui`). Both runtime images have ffmpeg,
  which llama-server needs to decode videos (`video_url` parts), as upstream's images do.
- [`Dockerfile.gfx906`](Dockerfile.gfx906): ROCm in mixa3607's image for gfx906, as
  [`../llama-cpp-diffusion-gemma/Dockerfile.gfx906`](../llama-cpp-diffusion-gemma/Dockerfile.gfx906). The ROCm image of
  [`../llama-cpp-radeon-vii`](../llama-cpp-radeon-vii) (b12230) predates the model, and upstream's Vulkan image
  generates only 15 t/s on the Radeon VII (this one: 26 t/s without the drafter).

Built by [`../.github/workflows/deploy-docker-llama-cpp-inkling.yml`](../.github/workflows/deploy-docker-llama-cpp-inkling.yml).
Drop them for upstream's images once #32218 is merged. The drafter logs `set_token_state: state size mismatch` twice
at startup (the warm-up), and works afterwards (87 % of the drafts accepted in code edits).

Benchmarks: [`../llama-cpp-rtx3090/README.md`](../llama-cpp-rtx3090/README.md#inkling-2-mini) and
[`../llama-cpp-radeon-vii/README.md`](../llama-cpp-radeon-vii/README.md#inkling-2-mini-2026-11-22).
