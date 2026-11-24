# GPU kernel benchmarks of Qwen3.8-Flash-Next (RTX 3090)

Microbenchmarks of the RTX 3090's work in a decode step, run in the runtime's image. See
[the runtime's README](../README.md#gpu-kernels-the-radeon-vii-and-cpu-isolation-2026-11-24) for the results.

- [`dense_gemm.py`](dense_gemm.py): the int8 (W8A16) projections of the Gated DeltaNet and attention layers with
  vLLM's Marlin kernel against the runtime's int8 skinny Triton kernel, at 4 tokens and in prefill, and the BF16
  projections of the hyper-connections with several configurations of the BF16 skinny kernel.
- [`bridge_submit.cu`](bridge_submit.cu): how fast the GPU writes a decode request of the CPU experts (40 KB) into
  pinned host memory, with several variants of the runtime's submit kernel.

```sh
docker run --rm --gpus all -v "$PWD/kernel_bench":/kb --entrypoint python3 \
    ghcr.io/agenttix/agx-ai/vllm-flash-next:latest /kb/dense_gemm.py
docker run --rm --gpus all -v "$PWD/kernel_bench":/w -w /w --entrypoint sh \
    ghcr.io/agenttix/agx-ai/vllm-flash-next:latest -c "nvcc -O3 -arch=sm_86 -std=c++17 bridge_submit.cu -o /tmp/b && /tmp/b"
```

Both need ~2 GB of free VRAM, so the production server can keep running (it adds some noise).
