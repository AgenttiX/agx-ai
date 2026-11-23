# Radeon VII expert worker for Qwen3.8-Flash-Next

The [Qwen3.8-Flash-Next runtime](../README.md) keeps 24 "hot" experts per layer on the RTX 3090 and computes the other
("cold") experts with a CPU thread pool during decode, which is the bottleneck (memory bandwidth of DDR4). This worker
keeps the next most used cold experts in the 16 GB of HBM2 of the otherwise idle Radeon VII, and computes them on
request of the CPU pool, in parallel with the CPU computing the rest, over the shared-memory protocol of
[`rvii_protocol.h`](rvii_protocol.h) (`/dev/shm/qwen38-rvii`; both containers use `--ipc host`).

| | |
|---|---|
| Experts | ranks 24..158 of each layer: 135 per layer, 6480 in total, 15.30 GiB (2.42 MiB each). The compose file of the runtime uses the ranking measured on decode ([`rankings-measured.json`](rankings-measured.json), `RVII_RANKINGS`; see [the runtime's README](../README.md#radeon-vii-experts-2026-11-23)), which the runtime uses for its 24 hot experts too; the default is its static ranking ([`rankings.json`](rankings.json), from `/opt/qwen38-3090/configs/static_hot_cache_rankings.json` of the runtime image) |
| VRAM | 15.4 GiB used, ~600 MiB free (`RVII_RESERVE_MIB`, default 640 MiB, is left for the HIP runtime and the buffers) |
| Latency, 4 tokens and 8 experts | 71 µs median, 91 µs p99 (from the request's seq to the response's seq, see below) |
| Error | max \|y - ref\| / max \|ref\| ≈ 1e-5 against an FP64 reference (the CPU pool: 1e-7 in FP32, 1e-2 with int8 activations) |
| CPU | one core spins while requests come; when idle for 100 ms, a poll every 100 µs (~3 % of a core) |

Files:

- [`rvii_protocol.h`](rvii_protocol.h): the protocol (shared with the CPU pool in the runtime's `cpu_moe.cpp`).
- [`prepare.py`](prepare.py): packs the selected experts from the checkpoint into a cache directory, in the layout
  of the kernel (`~/.cache/vllm-models/flash-next-rvii/hot{HOT}-n{N}-{hash of the ranking file}/`: `experts.bin`,
  `slots.bin`, `meta.json`).
  Reads the safetensors shards through mmap with 8 processes: 15 s for 15 GB from the page cache. Checks that the
  zero points are 8 (the checkpoint stores them as `qzeros` = `0x77777777`, i.e. 7 + 1 in the GPTQ v1 format).
- [`rvii_worker.hip`](rvii_worker.hip): the worker (HIP, gfx906).
- [`bench_rvii.py`](bench_rvii.py): a test and benchmark client that plays the CPU side.
- [`Dockerfile`](Dockerfile), [`entrypoint.sh`](entrypoint.sh): the image `ghcr.io/agenttix/agx-ai/flash-next-rvii-experts`
  (built by [`../../.github/workflows/deploy-docker-flash-next-rvii-experts.yml`](../../.github/workflows/deploy-docker-flash-next-rvii-experts.yml)),
  based on the [vLLM gfx906 image](../../vllm-radeon-vii) for its ROCm and Python with numpy.

## Build and run

```sh
docker build -t ghcr.io/agenttix/agx-ai/flash-next-rvii-experts:latest .
docker stop llama-cpp-radeon-vii  # the Radeon VII is needed whole
docker run -d --name rvii-experts --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro -v ~/.cache/vllm-models:/root/.cache/vllm-models \
  ghcr.io/agenttix/agx-ai/flash-next-rvii-experts:latest
```

The entrypoint packs the experts into the cache when it does not exist yet (~15 s, ~15 GB of disk per HOT and N),
then the worker loads them into VRAM (4-8 s) and sets `ready = 1`. Settings (environment):

- `RVII_HOT` (default 24): the experts per layer on the RTX 3090 (`QWEN38_HOT_ONLY` of the runtime); the worker takes
  the next ranks.
- `RVII_PER_LAYER`: the experts per layer; default: as many as fit, from `RVII_VRAM_MIB` if set, else from the free
  VRAM minus `RVII_RESERVE_MIB` (default 640). With the default, N depends on the free VRAM, so set it if something
  else uses the GPU (otherwise a new cache is packed).
- `RVII_CACHE` (default `/root/.cache/vllm-models/flash-next-rvii`).
- `RVII_DOWN_UNITS_PER_WAVE` (default 2): (expert, group) units per wave of the down projection (1-4 performed the
  same).

The worker exits cleanly on SIGTERM (`docker stop`): `ready = 0` and the segment is removed. If it crashes,
`ready` stays 1, so the CPU side has to check `heartbeat` (incremented every 20 ms) or `worker_pid`.

Test and benchmark (with the worker running):

```sh
docker run --rm --ipc host -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro --entrypoint python \
  ghcr.io/agenttix/agx-ai/flash-next-rvii-experts:latest /opt/rvii/bench_rvii.py
```

## How it works

- Weights: prepare.py repacks the int4 weights so that each wavefront reads 1 KiB contiguous per load instruction,
  each lane one column (gate and up interleaved, so that one lane has both for SiLU), and 16 or 32 rows (k) per 16-B
  load. The scales stay FP16, the zero point 8 is implicit.
- Math: the activations (x, and SiLU(gate) * up) are quantized per block of 64 values to 24-bit integers relative to
  the block's range (as exact as FP32 relative to the range) in 6 planes of 4 bits, so that the products with the
  int4 weights are exact integer dot products with `v_dot8_u32_u4` (8 products per instruction, without converting
  the weights to float, which made an FP32 version compute bound with 4 tokens); the planes are combined, scaled and
  accumulated in FP32. The error, ~1e-5 of the output's range, comes from the 24-bit rounding of the activations and
  the cancellation in `sum (q - 8) x` (FP32 activations, as tried first: ~2e-6).
- One kernel launch per request, with three roles of workgroups in the order of dispatch: copy (x from the shared
  memory over PCIe, zero-copy, to planes), gate/up + SiLU (64 columns x 4 groups per workgroup, 50 per expert) and
  down (64 output columns per workgroup, its waves take (expert, group) units). A role waits only for the data that
  it needs (flags in VRAM), after it has issued the loads of its weights; the last down workgroups write y straight
  into the shared memory and the response seq. The worker thread does not wait for the kernel: it reads the request
  (and the map), passes the jobs as kernel arguments and goes back to polling.

## Measurements

agx-z2e (Ubuntu 26.04, kernel 7.0, ROCm 7.14 of the vLLM gfx906 image; AMD Ryzen Threadripper 3970X, 128 GB DDR4-2666;
AMD Radeon VII: gfx906 (Vega 20), 60 CUs, 16 GB HBM2 (1 TB/s), PCIe 3.0 x16), 2026-11-23. The latency is from
writing the request's seq to seeing the response's seq, in the Python client (`bench_rvii.py`, 1000 requests each,
random experts of the worker in random layers, each with 1..T tokens, so ~2.5 tokens per expert with T = 4):

| T | J (experts) | p50 µs | p90 µs | p99 µs | weights GB/s (p50) | p50 µs, 1 ms between requests |
|---|---|---|---|---|---|---|
| 1 | 1 | 25 | 33 | 51 | 103 | |
| 1 | 4 | 32 | 35 | 41 | 314 | 40 |
| 1 | 8 | 51 | 52 | 59 | 397 | 59 |
| 1 | 12 | 66 | 69 | 74 | 458 | 77 |
| 1 | 16 | 83 | 85 | 93 | 491 | |
| 4 | 1 | 42 | 51 | 57 | 60 | |
| 4 | 4 | 52 | 58 | 66 | 196 | 60 |
| 4 | 8 | 70 | 79 | 92 | 288 | 81 |
| 4 | 12 | 94 | 104 | 118 | 323 | 108 |
| 4 | 16 | 119 | 127 | 138 | 342 | |

With 1 ms between requests (as in decoding, ~48 layers per ~50 ms step), the GPU's clocks drop between requests, and
the latency is ~15 % higher. A small request is bound by latencies (kernel launch ~5 µs, PCIe reads of x ~2 µs, and
the chain of flags and atomics between the roles), a large one reaches ~0.35-0.5 TB/s of the HBM's 1 TB/s. Both are
well below the CPU pool's time per layer (~1 ms), which this worker shortens.

Tried and dropped:

- FP32 activations with the weights converted to float (as the CPU pool): 1.2-1.3x slower with 4 tokens (compute
  bound on the conversions and FMAs).
- A persistent kernel (polling the shared memory from the GPU, no launches): not faster (the GPU-side chain of PCIe
  reads and flags costs about as much as a launch), and it keeps the GPU busy while decoding.
- Gate/up workgroups that compute all 20 groups of their columns (no cross-workgroup sum): slower (more VGPRs, lower
  occupancy).
