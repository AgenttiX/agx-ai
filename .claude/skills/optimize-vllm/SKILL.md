---
name: optimize-vllm
description: Optimize a vLLM configuration of this repository for maximum speed and context size while keeping the output quality, by measuring each setting (kv-cache-memory-bytes, int8 KV cache and attention backend, CUDA graph sizes, max-num-seqs and prefill chunk, speculative decoding, CPU KV offloading, quantized embeddings, CPU vision towers, tensor parallelism) with the scripts in ./benchmark. Use when asked to tune, optimize or create a vLLM config YAML or configuration, or to fit a larger context or a model on a GPU with vLLM.
---

# Optimize a vLLM configuration

The goal: the fastest configuration with the largest context that fits the hardware safely, without a measurable
loss of output quality. Every change is measured; nothing is set from assumptions alone.

## Before starting

1. Follow [`AGENTS.md`](../../../AGENTS.md): `git pull --rebase`, `./hostname.py` (if it prints "censored", use the
   hardware model instead of the hostname and pass `--hostname` or `LLAMA_BENCH_HOSTNAME` to the benchmarks).
2. Read [`vllm/OPTIMIZATIONS.md`](../../../vllm/OPTIMIZATIONS.md), the catalog of everything already measured in
   this repository, and [`vllm/README.md`](../../../vllm/README.md) for what the generic image and `vllm.patch`
   provide. Do not re-measure what the catalog already settles for the same model, image and GPU architecture unless
   something relevant has changed.
3. Read the configuration to optimize (README, `config*.yaml`, compose file, Dockerfile) and the most similar
   existing configuration (same GPU architecture, VRAM size or model) as a starting point.
4. Record the hardware for the README (see AGENTS.md) and check what else uses the GPU (desktop session, other
   servers) and the actual power limit.
5. Read [`benchmark/README.md`](../../../benchmark/README.md) and remind the user to turn off the LiteLLM health check
   on the agx-ai server during the benchmarks.
6. Check whether llama.cpp is known to be better for this model and hardware (the catalog's "vLLM or llama.cpp?"
   section and [`llama-cpp/OPTIMIZATIONS.md`](../../../llama-cpp/OPTIMIZATIONS.md)), and say so before investing in
   vLLM.

## Work on test servers

Stop the production container and start test servers with
[`benchmark/vllm_test_server.sh`](../../../benchmark/vllm_test_server.sh) and a copy of the YAML in the scratchpad
(`GPUS`, `DOCKER_ARGS="-e ..."`, `CACHE_VOLUME=<production compile cache volume>` to avoid recompiling). Startup
takes 1.5-7 min (longer the first time), so run it in the background. Restart the production container at the end,
and before that whenever the user needs the server.

Measure:
- **Memory:** the startup log ("Available KV cache memory", "GPU KV cache size", "Maximum concurrency", graph
  capture memory) and the peak VRAM per process during a prompt of `max-model-len - 1500` tokens.
- **Serving speed:** `llama_cpp_bench_http.py --no-embedding --tests generation prompt chat depth edit
  --depth 6400 32000 100000 --prompt-tokens <max-model-len - 1500> --server-config <yaml> --repeat 2 --label "..."`.
  Judge generation by `chat`, `depth` and `edit`; the default `generation` test overstates speculative decoding. Add
  `concurrency` (and `vllm_concurrent_agents.py`) when `max-num-seqs` > 1, and `vllm_prefix_return.py` for CPU KV
  offloading.
- **Quality:** `llama_cpp_quality_http.py --tests retrieval --logprobs N` with speculative decoding off, a reference
  run, and a noise floor from a lossless change (e.g. `max-num-batched-tokens` 2048 -> 1024). A setting is acceptable
  if its KL divergence and changed top tokens are within the noise floor. Accuracy alone is too blunt.

A single run varies ±5-10 % (chat ±10 %): repeat close results, alternating the variants.

## Order of the optimization

Go through these in order, since the earlier ones change the memory budget of the later ones. For each one, try the
candidates from the catalog, measure, and keep the winner.

1. **Image.** Prefer the generic image of [`vllm/`](../../../vllm) (stock vLLM + `vllm.patch` + pth-flash-attn). If
   something is missing, extend it there, keeping it compatible with the other configurations (AGENTS.md); a separate
   image (with a workflow in `.github/workflows/`) only for another base image.
2. **Checkpoint.** Int4 weights (Marlin on Ampere, exllama on gfx906) with a quantized embedding and lm_head (BF16
   ones cost ~2.4 GiB for Qwen3.8-27B); for Gemma 4 QAT, the bit-exact repack of `vllm/repack_gemma4_qat.py` with
   `dtype: float16`.
3. **Memory baseline.** `kv-cache-memory-bytes` set explicitly, `gpu-memory-utilization` just below what other GPU
   users leave, `hf-overrides` for `max_position_embeddings`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
   `max-num-seqs` as small as the client allows, `mamba-ssm-cache-dtype: float16` for Gated DeltaNet models, and the
   vision tower either off (`language-model-only`) or on the CPU (the CPU vision towers of the catalog).
4. **KV cache and attention.** `kv-cache-dtype: int8_per_token_head` with `attention-backend: CUSTOM` where its shapes
   are supported (Ampere/Ada, x86-64), else `TRITON_ATTN` with the split-KV fix. Compare tg at depth: it should stay
   nearly flat. FP8 or int4 KV only with a quality check and a measured gain (see the catalog for the GPU
   architecture).
5. **CUDA graphs.** `compilation-config: {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [...]}`
   with exactly the sizes `max-num-seqs` x (1 + drafted tokens) that occur.
6. **Speculative decoding.** MTP with `num_speculative_tokens` 2 (try 1 and 3), or DFlash2 with the hyperqwen image.
   Weigh the drafter's VRAM against the context; skip it for MoE models with experts behind PCIe or on the CPU.
7. **Prefill chunk.** `max-num-batched-tokens` 512-2048 (larger rarely helps, and costs KV for sliding-window
   models); it must hold one image or video for Gemma 4.
8. **Context size.** Raise `max-model-len` to the KV tokens in the log divided by ~1.05-1.1, stress-test with prompts
   of `max-model-len - 1500` tokens (and concurrent requests, embeddings and media if served), and adjust
   `kv-cache-memory-bytes` and `max-model-len` together until the peak keeps the headroom (~1 GiB next to a desktop,
   ~80-250 MiB on a dedicated GPU). A successful startup proves nothing; the first long request must succeed. If the
   context is still the limit, go back to steps 2-6.
9. **Returning conversations.** `enable-prefix-caching: true`, and `kv-offloading-size` with
   `kv-offloading-backend: native` if the RAM allows (measure with `vllm_prefix_return.py`; check the gotchas in the
   catalog first).
10. **Multiple GPUs or CPU offload** if needed: TP with the P2P workarounds, UVA offload of experts and embeddings,
    or the CPU-expert runtime; measure against llama.cpp on the same hardware.

Stop when the remaining candidates are within the noise. Leave stable, measured choices over fragile peaks.

## Finish

1. Run the final configuration once more with the full benchmark and a full-context prompt, and check the logs.
2. Comment every YAML key with the measured reason and the rejected alternatives.
3. Update the configuration's README per AGENTS.md: the summary table at the start (model, context, VRAM, RAM, tg,
   pp, notes, and whether llama.cpp is better for the same model and hardware), the hardware, a dated section per
   experiment with tables, a "not tried" list, and the exact commands.
4. Add new findings that apply beyond this configuration to `vllm/OPTIMIZATIONS.md`.
5. Keep the results in `benchmark/results/`, run `./lint.sh` if Python was changed, restart the production
   container, and commit (directly to main is fine).
