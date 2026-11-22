# llama.cpp with fast rollback for draft models

A llama.cpp server image (CUDA) for speculative decoding with a separate draft model (`spec-type = draft-simple`)
when the target or the draft model has recurrent layers, such as Qwen3.8-27B and NVIDIA's
[`Qwen3.5-4B-Qwen3.8-27B-Draft`](https://huggingface.co/nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft) (both with Gated DeltaNet
layers). It is upstream release b12627 with [`draft-rollback.patch`](draft-rollback.patch), built by
[`Dockerfile`](Dockerfile) and [`../.github/workflows/deploy-docker-llama-cpp-draft.yml`](../.github/workflows/deploy-docker-llama-cpp-draft.yml).
Used by [`../llama-cpp-rtx3090/docker-compose-draft.yml`](../llama-cpp-rtx3090/docker-compose-draft.yml), an
experimental configuration: on the RTX 3090, this draft model is slower than the MTP head of Qwen3.8-27B (see
[its results](../llama-cpp-rtx3090/README.md#external-draft-model-2026-11-23)).

## The problem

A recurrent state cannot be rolled back by removing tokens from a KV cache. llama.cpp keeps per-token snapshots of
the recurrent states (`n_rs_seq`) for that, but only for the drafters that live in the target's context (MTP, EAGLE3,
DFlash, ...), not for `draft-simple`. With a separate draft model, b12627 (and master on 2026-11-22) therefore:

1. copies the recurrent states of the target and the draft model to host memory before every verification, and
2. after a partially accepted draft (most of them), copies them back and runs the accepted tokens through the target
   a second time ("replay").

With Qwen3.8-27B on the RTX 3090, a step took ~116 ms instead of ~50 ms: 21 t/s in chat with 6 drafted tokens, less
than without speculative decoding (34.5 t/s).

## The patch

[`draft-rollback.patch`](draft-rollback.patch) (13 lines, against b12627):

- `common_params_speculative::need_n_rs_seq()` also gives the target `n_max` recurrent-state snapshots for
  `draft-simple`, as for MTP: no host checkpoint of the target and no replay.
- The draft context gets the same number of snapshots (`need_n_rs_seq_dft()`), so that the draft model's copy of the
  verification batch can be rolled back after the target has accepted part of it.
- The draft model decodes its proposals one token per batch, which the snapshots cannot undo, so the server keeps its
  (recurrent-only) host checkpoint of the draft context around the drafting also when the draft context has snapshots.

With it, chat on the RTX 3090 went from 21.4 to 35.7 t/s (6 drafts), at the cost of the snapshots (~1.2 GiB for both
models with 6 drafts). The remaining overhead is the drafting itself (~5 ms per token for the 4B draft, 20 % of a
target step), the draft model's pass over the verification batch, and its host checkpoint (see
`TAG_SPEC_AVOID_DRAFT_REEVAL` in `server-context.cpp`). The patch does not change anything for models without recurrent
layers (the snapshots are clamped to 0 by `llm_arch_supports_rs_rollback`) or for the other speculative types.

## Draft model on another GPU

For the test of the draft model on the Radeon VII (2026-11-23), the same image was built with the Vulkan backend as
well, so that one llama-server has `CUDA0` (RTX 3090) and `Vulkan0` (Radeon VII, RADV) and the preset can put the draft
on the Radeon VII with `device = CUDA0` and `device-draft = Vulkan0`. Not kept, as it was slower than no speculative
decoding; to reproduce, change the `Dockerfile`:

- build packages: `libvulkan-dev glslc spirv-headers`,
- `cmake`: `-DGGML_VULKAN=ON` next to `-DGGML_CUDA=ON`,
- runtime packages: `libvulkan1 mesa-vulkan-drivers`,
- `docker run` (or the compose file): `--device /dev/dri --group-add video` next to the NVIDIA GPU, with the
  production server of the Radeon VII stopped.

## Build

```sh
docker build -t ghcr.io/agenttix/agx-ai/llama-cpp-draft:latest .
# Only for the RTX 3090 (faster)
docker build --build-arg CUDA_ARCHITECTURES=86 -t ghcr.io/agenttix/agx-ai/llama-cpp-draft:latest .
```

Drop the image once upstream llama.cpp gives `draft-simple` rollback snapshots for recurrent models.
