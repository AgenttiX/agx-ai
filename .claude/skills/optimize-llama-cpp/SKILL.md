---
name: optimize-llama-cpp
description: Optimize a llama.cpp (llama-server) configuration of this repository for maximum speed and context size while keeping the output quality, by measuring each setting (KV cache type, context, n-cpu-moe, ubatch, flash attention, MTP, vision encoder placement, threads, split mode) with the scripts in ./benchmark. Use when asked to tune, optimize or create a llama.cpp preset or configuration, or to fit a larger context or a model on a GPU.
---

# Optimize a llama.cpp configuration

The goal: the fastest configuration with the largest context that fits the hardware safely, without a measurable
loss of output quality. Every change is measured; nothing is set from assumptions alone.

## Before starting

1. Follow [`AGENTS.md`](../../../AGENTS.md): `git pull --rebase`, `./hostname.py` (if it prints "censored", use the
   hardware model instead of the hostname and pass `--hostname` or `LLAMA_BENCH_HOSTNAME` to the benchmarks).
2. Read [`llama-cpp/OPTIMIZATIONS.md`](../../../llama-cpp/OPTIMIZATIONS.md), the catalog of everything already
   measured in this repository. Do not re-measure what it already settles for the same model, backend and build
   unless something relevant has changed; do measure what is specific to this hardware.
3. Read the configuration to optimize (README, `preset*.ini`, `config.ini`, compose file) and the most similar
   existing configuration (same GPU vendor, VRAM size or model) as a starting point.
4. Record the hardware for the README (see AGENTS.md: OS, CPU, RAM type and frequency, GPU, architecture, VRAM,
   for laptops both GPUs, the dGPU power limit and PCIe width). Check the actual power limit
   (`nvidia-smi -q -d POWER`) and what else uses the GPU (desktop session, other servers).
5. Read [`benchmark/README.md`](../../../benchmark/README.md) and remind the user to turn off the LiteLLM health check
   on the agx-ai server during the benchmarks.

## Work on test servers

Stop the production container and start test servers with
[`benchmark/llama_cpp_test_server.sh`](../../../benchmark/llama_cpp_test_server.sh) and a copy of the preset
in the scratchpad (CUDA: `DOCKER_ARGS="--gpus all"`). Run commands that may take more than ~2 minutes in the
background. Restart the production container at the end, and before that whenever the user needs the server.

Measure:
- **Serving speed:** `llama_cpp_bench_http.py --no-embedding --tests generation prompt chat depth image --repeat 2
  --depth 6400 16000 36500 ... --prompt-tokens <ctx-size - 1500> --label "..."`. Judge generation by `chat` and
  `depth` (realistic content); the default `generation` test uses repeated text that overstates MTP. Add `edit` for
  coding use, `concurrency` for several slots, `--mixed` if an embedding model shares the server.
- **Raw model speed** (quants, ubatch, KV types, builds, backends): `llama_cpp_bench_container.py` (`llama bench`
  with the same `-ngl -fa -ctk -ctv -sm -ts` flags as the preset; it has no speculative decoding).
- **Memory:** the peak VRAM of llama-server during a prompt of `ctx-size - 1500` tokens (the benchmark records it
  with `nvidia-smi`/`amd-smi`; on Windows watch the dedicated and shared GPU memory counters).
- **Quality:** `llama_cpp_quality_http.py --tests retrieval` with MTP off, an f16-KV reference run, and a noise-floor
  run with a lossless change (another `ubatch-size`). A setting is acceptable if its KLD (mean, 99 %, first token)
  and changed top tokens are within the noise floor. Accuracy alone is too blunt.

Runs vary by ±1-4 % on desktops and up to ~10 % on laptops: repeat close results twice in alternating order.

## Order of the optimization

Go through these in order, since the earlier ones change the memory budget of the later ones. For each one, try the
candidates from the catalog, measure, and keep the winner.

1. **Backend, image and build.** Pin the image tag. For AMD, compare ROCm and Vulkan; for older NVIDIA GPUs, pick an
   image whose CUDA version the driver supports. Changing the build later shifts the compute buffers, so re-check a
   tight context after any update.
2. **Model file.** The largest quant that leaves a useful context; prefer QAT checkpoints (Gemma 4) and Unsloth UD
   quants. Consider an MoE model with offload over a smaller dense model.
3. **Placement.** `n-gpu-layers = all` and `flash-attn = on`. If the model does not fit, move MoE experts with
   `n-cpu-moe` (not partial `n-gpu-layers`); for dense models on tiny GPUs, `override-tensor`. Use `load-mode = dio`
   for GPU-only models and `none` for hybrid ones (see the catalog for Windows, WSL and LXC). On multiple GPUs, try
   `split-mode = tensor` and fall back to `layer` if it crashes.
4. **Speculative decoding.** If the model has an MTP drafter: `spec-type = draft-mtp`, `spec-draft-n-max` 1-4
   (usually 2). Check that it pays off on this hardware (not on CPU-heavy setups) and what it costs in VRAM.
5. **ubatch size.** 256-2048 with `llama bench` pp at depth 0 and 16k; weigh the pp gain against the MiB, converted to
   tokens of context. MoE and expert offload benefit most from large batches.
6. **KV cache type.** f16 vs q8_0 (both K and V, matched types only). Measure the real VRAM saving per token (CUDA
   flash attention keeps f16 buffers) and the generation speed at depth, and check the quality. q4_0 only if the
   context would otherwise be far too small, with a quality check.
7. **Vision encoder.** On the GPU for fast images if the context is already large; `no-mmproj-offload = 1` if the
   ~1.1-1.3 GiB is worth more as context. Measure the image time with the `image` test.
8. **Threads** (if anything runs on the CPU): sweep `threads` (and `threads-batch`); fewer than all cores usually wins.
   Mind E-cores, SMT siblings and LXC cpusets.
9. **Context size.** Find the largest `ctx-size` (multiple of 1024) whose peak stays within the headroom: probe with
   prompts of `ctx-size - 1500` tokens, extrapolate by the measured MiB per 1024 tokens, then confirm the final value
   with the full benchmark, concurrent requests and, for routers, unload/reload cycles. Loading is not enough; the
   first long prompt must succeed. Leave a margin: ~1 GiB next to a desktop session, ≥~70-100 MiB on a dedicated
   GPU. If the context is still the limit, go back to steps 3-7 and spend VRAM where a MiB buys the most context per
   speed lost (e.g. one more expert layer on the CPU is cheaper than a q8_0 cache on CUDA).
10. **Server settings:** `parallel` to match the client, `kv-unified`, `cache-ram` for returning conversations,
    `ctx-checkpoints = 0` on Vulkan if checkpoints are slow, `fit = off`, the model's recommended sampling parameters.

Stop when the remaining candidates are within the noise. Leave stable, measured choices over fragile peaks.

## Finish

1. Run the final preset once more with the full benchmark and a full-context prompt, and check the logs for errors.
2. Comment every non-default preset key with the measured reason and the rejected alternatives (e.g. the context
   for the other vision-encoder choice as a commented-out line).
3. Update the configuration's README per AGENTS.md: the summary table at the start (model, context, VRAM, RAM, tg,
   pp, notes, and whether vLLM is better for the same model and hardware, see
   [`vllm/OPTIMIZATIONS.md`](../../../vllm/OPTIMIZATIONS.md)), the hardware, a dated section per experiment with
   tables, and the exact commands.
4. Add new findings that apply beyond this configuration to `llama-cpp/OPTIMIZATIONS.md`.
5. Keep the results in `benchmark/results/`, run `./lint.sh` if Python was changed, restart the production
   container, and commit (directly to main is fine).
