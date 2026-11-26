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
| Tokens per request (2026-11-26) | 1-8 (protocol version 2; 1-4 before), for MTP depths up to 7; a job has at most 4 tokens, and the CPU pool sends an expert with more as two jobs, so that the kernels keep one wave per job token. 8 tokens and 8 jobs: 86 µs median, 116 µs p99; 30 jobs: 256 / 296 µs; error 1e-5 as with 1 and 4 tokens (`bench_rvii.py`) |
| Error | max \|y - ref\| / max \|ref\| ≈ 1e-5 against an FP64 reference (the CPU pool: 1e-7 in FP32, 1e-2 with int8 activations) |
| CPU | one core spins while requests come; when idle for 100 ms, a poll every 100 µs (~3 % of a core), and after 2 s every 1 ms |
| Prefill (2026-11-25) | in prefill chunks, the RTX 3090 sends the routing and x of each layer, and the worker computes the masked ones of its experts (the runtime's `QWEN38_RVII_PREFILL`: the 112 least used of each layer): 42 ms for all 135 experts and 8192 tokens with routing drawn from decode (41k token-expert pairs), 30 ms in a real prompt; see [Prefill](#prefill) |
| Re-ranking (2026-11-25) | with `RVII_AUTO_RERANK=1`, [`rerank.py`](rerank.py) runs the worker and re-ranks its experts from their measured use; see [Automatic re-ranking](#automatic-re-ranking) |

Files:

- [`rvii_protocol.h`](rvii_protocol.h): the protocol (shared with the CPU pool in the runtime's `cpu_moe.cpp`).
- [`prepare.py`](prepare.py): packs the selected experts from the checkpoint into a cache directory, in the layout
  of the kernel (`~/.cache/vllm-models/flash-next-rvii/hot{HOT}-n{N}-{hash of the ranking file}/`: `experts.bin`,
  `slots.bin`, `meta.json`).
  Reads the safetensors shards through mmap with 8 processes: 15 s for 15 GB from the page cache. Checks that the
  zero points are 8 (the checkpoint stores them as `qzeros` = `0x77777777`, i.e. 7 + 1 in the GPTQ v1 format).
- [`rvii_worker.hip`](rvii_worker.hip): the worker (HIP, gfx906).
- [`bench_rvii.py`](bench_rvii.py): a test and benchmark client that plays the CPU side.
- [`bench_prefill.py`](bench_prefill.py): a test and benchmark client of the prefill path that plays the RTX 3090's
  side (routing drawn from [`usage/general.bin`](usage/general.bin)).
- [`rerank.py`](rerank.py): the automatic re-ranking (`RVII_AUTO_RERANK=1`), which runs the worker.
- [`rank_experts.py`](rank_experts.py): builds a ranking from measured expert usage, e.g. for particular topics
  (see [Choosing the experts](#choosing-the-experts)); [`usage/general.bin`](usage/general.bin) holds the counts
  of `rankings-measured.json`.
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
- `RVII_AUTO_RERANK=1`: re-rank the experts from their use ([Automatic re-ranking](#automatic-re-ranking)).
- `RVII_PF_STATS=N`: log the mean time of the prefill requests every N requests.

The worker exits cleanly on SIGTERM (`docker stop`): `ready = 0` at once, then it serves the requests that still come
until none has come for a second (a prefill chunk in progress on the RTX 3090 is finished), and the segments are
removed. If it crashes, `ready` stays 1, so the CPU side has to check `heartbeat` (incremented every 20 ms) or
`worker_pid`.

Test and benchmark (with the worker running):

```sh
docker run --rm --ipc host -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro --entrypoint python \
  ghcr.io/agenttix/agx-ai/flash-next-rvii-experts:latest /opt/rvii/bench_rvii.py
docker run --rm --ipc host -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro --entrypoint python \
  ghcr.io/agenttix/agx-ai/flash-next-rvii-experts:latest /opt/rvii/bench_prefill.py
```

Both send requests of their own: stop the runtime first, or their sequence numbers collide with its.

## Choosing the experts

The ranking file decides the placement: for each layer, the experts in order of use. The runtime keeps ranks 0-23 on
the RTX 3090 (`QWEN38_HOT_ONLY`), this worker ranks 24-158, and the CPU computes the rest. Without the
[automatic re-ranking](#automatic-re-ranking), the placement is fixed while the containers run.

[`rankings-measured.json`](rankings-measured.json) is [`usage/general.bin`](usage/general.bin) (4.9 M decode
requests on chat, raw generation, 16k and 64k tokens of code and retrieval questions, 2026-11-23) ranked by
[`rank_experts.py`](rank_experts.py), with the runtime's static ranking for ties. To favor a topic, collect its usage
and mix it with the general counts (a ranking from one topic only would slow down the others):

```sh
cd ..  # vllm-rtx3090-flash-next
docker compose -f docker-compose.yml -f docker-compose.usage.yml up -d
# ... send prompts of the topic: a few hundred thousand generated tokens ...
cp rvii_experts/usage/collect.bin rvii_experts/usage/my-topic.bin
python3 rvii_experts/rank_experts.py rvii_experts/rankings-my-topic.json \
  rvii_experts/usage/general.bin:0.3 rvii_experts/usage/my-topic.bin:0.7
```

Each collection is normalized per layer before weighting, so the weights decide the mix regardless of the collections'
sizes. The script prints the share of each collection's requests that the RTX 3090, the Radeon VII and the CPU would
compute with the new ranking and the static one (`--report` evaluates an existing ranking against usage files). For
reference, the general collection: 24.8 / 49.8 / 25.4 % with `rankings-measured.json`, 11.8 / 40.8 / 47.3 % with the
static ranking. To use the new ranking, mount it in `../docker-compose.yml` in place of `rankings-measured.json`, and
in the `rvii-experts` service, mount it too and point `RVII_RANKINGS` to it; the entrypoint packs a new cache for it.

Experts specialize by kind of token (code syntax, prose, language) as much as by subject, so broad domains separate
better than subjects within them. In the measurements of [the runtime's README](../README.md#radeon-vii-experts-2026-11-23),
the measured ranking gained 24-28 % in the code tests, of the same kind as part of the collection, and nothing in chats,
which were in neither.

## Automatic re-ranking

With `RVII_AUTO_RERANK=1` (2026-11-25, set in [`../docker-compose.yml`](../docker-compose.yml)), the entrypoint runs
[`rerank.py`](rerank.py), which runs the worker as its child process on a cache of its own, `hot{HOT}-n{N}-auto/` in
`RVII_CACHE` (packed from `RVII_RANKINGS` at the first start), and:

- reads the runtime's decode usage counts every 10 min (`QWEN38_EXPERT_USAGE=/dev/shm/qwen38-usage.bin` in the
  runtime: cumulative since its start, rewritten every 256 decode steps) and accumulates them with an exponential
  decay (half-life `RVII_RERANK_HALF_LIFE_H`, default 72 hours), kept in `usage-state.npz` across restarts;
- every `RVII_RERANK_INTERVAL_S` (default 600), and at once when the RTX 3090's hot experts change (the runtime writes
  them to `/dev/shm/qwen38-hot.bin` at its start), ranks the experts of each layer by their recent use mixed with the
  general collection (`usage/general.bin`, weight `RVII_RERANK_PRIOR`, default 0.2), and writes the ranking to
  `rankings.json` in the cache directory;
- if the N best-ranked experts that the RTX 3090 does not keep would serve at least `RVII_RERANK_MIN_GAIN` (default
  0.02, i.e. 2 % of all decode requests) more than the worker's current ones, and at least
  `RVII_RERANK_MIN_REQUESTS` (default 2e7, ~10000 decode steps) were counted, packs them into a new cache
  (`prepare.py --dir`, at the lowest priority, ~30 s) and, once the runtime has sent no request for
  `RVII_RERANK_IDLE_S` (default 120 s), stops the worker and starts it on the new cache (2 s from the page cache,
  ~10-20 s from the disk; meanwhile the CPU computes the worker's experts, and prefill streams them).

The runtime uses `rankings.json` for its own 24 hot experts at its next start (`QWEN38_RANKINGS_AUTO`, see
[`../rvii-prefill.patch`](../rvii-prefill.patch)); the worker then re-ranks for the new hot set. The cache keeps the
worker's experts in rank order within each layer, as `prepare.py` packs them, so the runtime's prefill still gives the
least used of them to the worker (see [Prefill](#prefill)).

The worker of the new layout loads its experts in its own process, so the decode and prefill protocols are unchanged:
the runtime sees the worker go (`ready = 0`, drained as at `docker stop`) and come back with a new pid and map, and
maps the new prefill segment. The automatic cache takes another ~16 GB of disk (and ~16 GB more while a new layout is
packed), next to the hash-named cache of `RVII_RANKINGS`.

Tested (2026-11-25) with a synthetic usage file that gave the use of ranks 100-158 to ranks 159-218 of each layer and
short intervals: the worker would serve 49.5 % instead of 44.7 % of the requests, so a new layout (2254 experts
changed) was packed, and the worker restarted on it after 10 s without requests (loaded in 2.7 s); `bench_rvii.py`
and `bench_prefill.py` then checked its results against the FP64 reference (decode 1.0e-5, prefill 3.2e-3, as
before), and a `docker stop` took 3.8 s with the counts saved.

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

## Prefill

In a prefill chunk of more than 384 tokens (`QWEN38_CPU_PREFILL_MAX`), the runtime streams the cold experts of every
layer from RAM to the RTX 3090 (`stream_v2.py`), and the PCIe copy is its bottleneck (2.37 of 3.5 s per chunk of 8192
tokens, 2026-11-25). With `QWEN38_RVII_PREFILL=N` (the runtime's [`../rvii_prefill.py`](../rvii_prefill.py)), the
worker computes the N least used of its experts of each layer (its highest slots, minus any hot ones) instead, and the
runtime does not stream them. Per MoE layer, over the second segment of [`rvii_protocol.h`](rvii_protocol.h)
(`/dev/shm/qwen38-rvii-pf`, 85 MB):

1. the RTX 3090 copies the routing (ids, weights) and x (BF16) into the segment (DMA on a side stream, while it
   computes its hot experts) and publishes seq;
2. the worker sorts the (expert, token) pairs of the masked experts by expert (a counting sort on the CPU), copies x
   into VRAM, converts it to FP16, computes the experts (below), converts y to BF16 (as the RTX 3090 rounds each
   expert's output) and copies it into the segment, and answers;
3. the RTX 3090 waits for the answer on the GPU before the layer's reduction (a kernel that polls the segment, with a
   10-s timeout) and adds y to its FP32 accumulator.

The kernels (FP16 activations, FP32 accumulation with `v_dot2_f32_f16`; the int4 weights converted in registers to
FP16, scaled by their group's scale): gate/up in one launch for all pairs, a wave per (expert, 64 columns, 8 tokens),
each lane a column with both its gate and up weights, the activations as wave-uniform scalar loads (SGPR operands),
SiLU(gate) * up to FP16; down in one launch per expert (its tokens are distinct, so the adds to y need no atomics),
a wave per (64 hidden columns, 8 tokens). Error against the FP64 reference: 3e-3 of the output's range (the BF16 y;
4e-4 before the rounding).

Measured with [`bench_prefill.py`](bench_prefill.py) (8192 tokens, routing drawn from the decode usage; the time from
the request's seq to the response's seq):

| Version | 135 experts (41k pairs) | 96 (24k) | 64 (14k) |
|---|---|---|---|
| rocBLAS FP16 GEMMs (dequantized weights per expert, FP32 y) | 69 ms | 48 ms | 33 ms |
| first own kernel (16 tokens per wave, a group's sums in temporaries; 169 VGPRs) | 86 ms | 56 ms | 36 ms |
| scaled FP16 weights (no temporaries; 119 VGPRs, 2 waves per SIMD) | 51 ms | | 24 ms |
| **gate/up 8 tokens and 4 packed rows per step, down 8 tokens (61 and 52 VGPRs)** | **42 ms** | **28 ms** | **20 ms** |

Of the 42 ms: x in 3.4 ms, gate/up ~21 ms (13 TFLOPS), down ~14 ms (9.6 TFLOPS), y out 3.1 ms. rocBLAS reached only
~7 TFLOPS at the experts' sizes (M = 64-1024 tokens; 19 TFLOPS at 8192). Larger token tiles (12-32 per wave) spilled
SGPRs (the compiler hoists the activations of all tokens) and were slower; so were x and y over PCIe by the kernels
instead of copies (+2 ms), and the chunk in 2-4 parts of tokens so that the transfers overlap with the computation
(+2 to +13 ms: the experts' tiles get smaller). The routing of real prompts is less concentrated on these experts than
that of decode: with all 135 masked, a 32k-token prompt gave them ~27k pairs per layer (~100 of the experts were
used), 30 ms. The end-to-end results are in [the runtime's README](../README.md#radeon-vii-in-prefill-and-automatic-re-ranking-2026-11-25).
