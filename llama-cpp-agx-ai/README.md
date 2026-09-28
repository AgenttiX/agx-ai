# llama.cpp on agx-ai

llama.cpp for the `agx-ai` LXC container on the Proxmox host `agx-h12`:
NVIDIA GeForce RTX 3070 (8 GB), AMD EPYC 7302 (16 cores, 32 threads), 128 GB RAM, Supermicro H12SSL-i.
The LXC container is limited to 24 of the 32 threads. Proxmox picks which ones at each boot (currently
1-2, 5-7, 10-25 and 29-31, see `/sys/fs/cgroup/cpuset.cpus.effective`), so the container sometimes gets both threads
of some cores and none of another: during the tests of 2026-09-25, its 24 threads were on only 15 of the 16 cores.

One `llama-server` router ([`docker-compose.yml`](docker-compose.yml), [`config.ini`](config.ini),
[`preset.ini`](preset.ini)) serves two models on port 9931. They are not used at the same time,
so each is tuned for running alone.

| Model | Runs on | Chat generation | Prompt processing |
|---|---|---|---|
| `google/gemma-4-26b-a4b-qat`: [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) UD-Q4_K_XL + MTP, ctx 122880 | GPU, with the experts of 28 of 30 layers on the CPU, core clock locked to 1600 MHz | ~55 t/s, 45-49 t/s at 36-70k tokens of context | ~1490 t/s for a 70k-token prompt, ~1220 t/s for 119k |
| [llmfan46/gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-GGUF](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-GGUF) Q4_0 + Unsloth's MTP drafter, ctx 75000 | CPU only | 17-24 t/s | ~74 t/s |

Both models reuse the prompt cache of a continuing conversation, so only the new part of the conversation is
processed, see [Prompt cache](#prompt-cache).

**The GPU model requires the core clock lock of 1600 MHz or the 170 W power limit**, which are both set by
[`gpu-power-limit/`](gpu-power-limit) on the host. Without them, the GPU crashes under long prompts,
see [Xid 79 crashes](#xid-79-crashes) and [Locking the GPU clock](#locking-the-gpu-clock).

## Setup

### Power limit and clock lock on agx-h12

The power limit and the clock lock reset at every reboot of the host. Install the script and the systemd service on `agx-h12`:

```sh
sudo install -m 755 gpu-power-limit/gpu-power-limit.sh /usr/local/sbin/gpu-power-limit.sh
sudo install -m 644 gpu-power-limit/gpu-power-limit.service /etc/systemd/system/gpu-power-limit.service
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-power-limit.service
systemctl status gpu-power-limit.service
nvidia-smi --query-gpu=power.limit --format=csv
```

The script waits up to 300 s for the driver and the GPU after boot, enables persistence mode (so that the limits are kept
when no program uses the GPU), sets the power limit to 170 W and locks the core clock to 210-1600 MHz. It also creates the NVIDIA device nodes, including
`/dev/nvidia-uvm-tools`, before Proxmox starts the guests. Previously, the autostart of the `agx-ai` container failed
with `TASK ERROR: Device /dev/nvidia-uvm-tools does not exist`, as the node only appears when something first uses
the GPU, so the container can now be set to start at boot. The settings (`POWER_LIMIT_W`, `MAX_CLOCK_MHZ`,
`GPU_BUS_ID`, `WAIT_TIMEOUT_S`) can be overridden in `/etc/default/gpu-power-limit`, e.g.:

```sh
echo 'MAX_CLOCK_MHZ=1700' | sudo tee -a /etc/default/gpu-power-limit
sudo systemctl restart gpu-power-limit.service
```

An empty value leaves that setting unchanged, e.g. `POWER_LIMIT_W=` to lock only the clock. Restarting the service
does not raise a power limit that was set earlier, so after changing it, also run
`sudo nvidia-smi -i 00000000:81:00.0 -pl 240` or reboot.

### Locking the GPU clock

A power limit makes the GPU boost to its highest clocks and voltages and then pull back, which still allows short
power and voltage spikes. Under load, the core clock ranged from 1515 to 1920 MHz (mean ~1760 MHz) both with and
without the 170 W limit. Capping the core clock instead keeps the GPU on the lower, more efficient part of its
voltage-frequency curve all the time. On Linux, this is the closest thing to undervolting, which is not
directly possible. It may allow removing the power limit, or give lower temperatures with it.
The supported range is 210-2100 MHz, and the memory runs at 6801 MHz under load (max 7001 MHz).

Run on `agx-h12` (settings last until a reboot or a reset):

```sh
sudo nvidia-smi -i 00000000:81:00.0 -lgc 210,1600   # lock the core clock to 210-1600 MHz
sudo nvidia-smi -i 00000000:81:00.0 -rgc            # reset the core clock lock
sudo nvidia-smi -i 00000000:81:00.0 -lmc 405,5001   # optional: lower the memory clock (less VRAM heat, slower)
sudo nvidia-smi -i 00000000:81:00.0 -rmc            # reset the memory clock lock
nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv
```

Suggested test sequence, each with 20 consecutive near-full-context prompts while logging power and temperature:
1. 1600 MHz with the 170 W limit: should run cooler and use less power than now. If the prompt processing
   speed drops by less than ~5 %, the clock lock alone is worth keeping.
2. 1600 MHz without the power limit (`-pl 240`): the actual candidate for replacing the power limit.
3. If 2 passes, 1750 MHz without the power limit for more speed. If it fails, 1400 MHz.
4. Optionally the memory clock at 5001 MHz, if the GPU still runs hot.
Keep the setting that passes with the highest prompt processing speed, and set it in
`/etc/default/gpu-power-limit`.

Results with `n-cpu-moe = 25`, ctx 77824 and 20 consecutive 75k-token prompts (the power and temperature without
the clock lock are from the 20-run test with `n-cpu-moe = 24` and 63k-token prompts):

| Setting | Result | Core clock under load | Power | Temperature | pp | Chat tg (English / code / Finnish) |
|---|---|---|---|---|---|---|
| 170 W | passed | 1755 MHz mean, max 1920 | 161 W mean, max 169 | 80 °C mean, max 85 | 1589-1617 t/s | 59 / 62 / 46 t/s |
| 170 W + 1600 MHz (2026-09-26) | 20/20 passed, no PCIe replays | 1590 MHz | 120 W mean, max 128 | 72 °C mean, max 75 | 1507-1511 t/s | 64 / 72 / 46 t/s |
| 240 W (no power limit) + 1600 MHz (2026-09-26) | 20/20 passed, no PCIe replays | 1590 MHz | 120 W mean, max 127 | 71 °C mean, max 73 | 1505-1511 t/s | 62 / 61 / 50 t/s |

The clock lock reduces power by 25 % and temperature by 8-10 °C for 6 % slower prompt processing, and token generation
is not slower, as it is mostly limited by the experts on the CPU. As the GPU stays well below 170 W, the power limit
only cuts short spikes, which the 5 s samples of `nvidia-smi dmon` do not show. Without the power limit, the results
are practically identical, so the clock lock alone prevents the crashes (for comparison, `ubatch-size = 2048` without
any limits crashed on the 5th prompt). [`gpu-power-limit/`](gpu-power-limit) sets both: the clock lock for
the lower temperature, and the power limit as a safety net against spikes, which costs nothing, as the GPU stays
below it. A higher clock (e.g. 1700 MHz) could win back some of the 6 % of prompt processing speed, but 1600 MHz
was kept, as it is more power-efficient and keeps the GPU coolest.

### Memory locking

The CPU model uses `load-mode = mmap+mlock`, but `mlock` fails, as the LXC container limits locked memory to 8 MiB
(`warning: failed to mlock 621395968-byte buffer ... Try increasing RLIMIT_MEMLOCK`). The model still works, but
its memory could be swapped out if the host runs low on RAM. Setting `ulimits: memlock` in
[`docker-compose.yml`](docker-compose.yml) alone makes the container fail to start with
`error setting rlimit type 8: operation not permitted`, as the LXC container does not allow raising the limit.
To enable it, add this to `/etc/pve/lxc/108.conf` on `agx-h12`, restart the `agx-ai` container, and uncomment
`ulimits` in `docker-compose.yml`:

```
lxc.prlimit.memlock: unlimited
```

### Higher-precision MTP drafters

[`preset.ini`](preset.ini) has commented-out `model-draft` lines for the Q8_0 drafter of Unsloth. To download it (or the
F16/BF16 version) into the Hugging Face cache, which is owned by root, as the container created it:

```sh
docker run --rm -v /home/mika/.cache/huggingface/hub:/root/.cache/huggingface/hub python:3-slim \
  sh -c "pip install -q huggingface_hub && hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
ls /home/mika/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/*/MTP/
```

If the repository has been updated since, the file goes into a new snapshot directory, so check the path.

### Docker image and driver

The `server-cuda13` image of build b11176 requires CUDA >= 13.4, but the driver 610.57 of `agx-h12` provides CUDA 13.3,
so Docker refuses to start it. `NVIDIA_DISABLE_REQUIRE=1` in [`docker-compose.yml`](docker-compose.yml) skips the check,
and CUDA minor version compatibility lets it run. The previous image (b10884, CUDA 13.3) is kept locally as
`ghcr.io/ggml-org/llama.cpp:server-cuda13-b10884` for rollback.

## Measurements

Measured on 2026-09-25 and 2026-09-26 with build b11176. tg = token generation, pp = prompt processing,
acc = accepted MTP draft tokens. "Chat" is `/v1/chat/completions` with three natural 400-token answers
(English explanation, Python code, Finnish history), as the repetitive text of
[`llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py) gives an unrealistically high MTP acceptance (~90 %).
Other numbers are from `llama_cpp_bench_http.py` (results in `benchmark/results/agx-ai.jsonl`) or `llama bench`
(without MTP). The helper scripts used for the tests are in `benchmark/scratch/` (gitignored).

### GPU model

**Layer placement.** The old setting of `n-gpu-layers = 13` (13 of 30 layers on the GPU, ctx 75000) was the slowest
option: chat tg 34-43 t/s (acc 41-73 %), pp 1076 t/s for an 8192-token prompt, `llama bench` tg128 39 t/s.
For this MoE model, it is much faster to put all layers on the GPU and keep only the expert tensors of some layers
on the CPU with `n-cpu-moe`, as the attention and shared weights are used for every token but each expert only for some.

`llama bench` (fa on, without KV cache for a long context):

| `n-cpu-moe` | tg128 | pp2048 |
|---|---|---|
| 30 (all experts on CPU) | 45 t/s | 917 t/s |
| 24 | 54 t/s | 1087 t/s |
| 22 | 53 t/s | 1159 t/s |
| 20 | 58 t/s | 1226 t/s |
| 18 | 60 t/s | 1306 t/s |
| 16 | out of memory | |

In the server with ctx 50000, MTP and 2 slots (and the default `ubatch-size = 512`), `n-cpu-moe` 18 ran out of memory
for the KV cache, 20 for the MTP context, 21 peaked at 7551 MiB of the 7839 MiB usable with a 45k-token prompt
(too tight), and 22 used 7111 MiB after loading and 7141 MiB at peak. With `n-cpu-moe = 22`, ctx 50000 and 16 threads:
chat tg 62-68 t/s in English and code (acc 70-79 %) and 54 t/s in Finnish (acc 42 %), i.e. ~60 % faster than before,
pp 1148 t/s for a 45k-token prompt, and 58 t/s in total for two concurrent requests.

**Flash attention** `flash-attn = on` is faster than off (`llama bench` with `n-cpu-moe` 22: tg 53 vs 49 t/s,
pp2048 1159 vs 1082 t/s; at a depth of 16384 tokens tg 47 vs 39 t/s and pp512 1061 vs 916 t/s).
It is set explicitly instead of `auto`.

**MTP draft length.** `spec-draft-n-max` 1, 2 and 3 gave chat tg averages of 64, 62 and 59 t/s, which is within
the noise. Longer drafts help with code but hurt with the Finnish text, where the acceptance is low.

**Threads.** `llama bench` tg128 with `n-cpu-moe` 22: 4 threads 35 t/s, 6: 44, 8: 48, 12: 59, 16: 56
(pp2048 ~1160 t/s regardless), so `threads = 12`.

**Context size (before the power limit)** with `n-cpu-moe` 22 and `ubatch-size = 512` (each extra token costs ~21 KiB of VRAM):
ctx 50000 peaked at 7141 MiB with a 45k-token prompt and ctx 65536 at 7471 MiB with a 62k-token prompt
(tg 74.5 t/s, pp 1091 t/s). ctx 81920 loaded with 7617 MiB, but crashed the GPU, see below.

**Gemma 4 12B instead.** [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)
UD-Q4_K_XL (dense, 6.24 GiB) was tested to fit the whole model on the GPU, but it is not worth it:
- All layers on the GPU with ctx 50000 and f16 KV cache run out of memory for the 800 MiB KV cache.
  With `cache-type-k/v = q8_0` and `ubatch-size = 256` it loads, but its MTP drafter does not fit anymore.
- Without MTP: chat tg 49.5 t/s, pp 1201 t/s for a 45k-token prompt, peak 7547 MiB.
  This is slower than the 26B-A4B with MTP, uses more VRAM, and the model is weaker.
- With partial offloading it is even slower: `llama bench` tg128 53 t/s with all 48 layers on the GPU,
  36 t/s with 44 and 29 t/s with 40.

### Xid 79 crashes

Three times, the GPU crashed during long prompts with `CUDA error: unspecified launch failure`, after which it was lost
until a cold boot (`Unable to determine the device handle for GPU0: 0000:81:00.0: Unknown Error`):
ctx 81920 with a 78.9k-token prompt, ctx 65536 on the third of three consecutive 63k-token prompts
(with the same 7471 MiB peak as the successful runs, so not a VRAM limit), and `ubatch-size = 2048` on the fifth
of consecutive 63k-token prompts. The host kernel log showed
`NVRM: Xid (PCI:0000:81:00): 79, GPU has fallen off the bus`, followed by
`Xid 154, GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)`.
`modprobe -r nvidia`, `pct stop` and the host shutdown all hung, so the host had to be powered off with the button.
This is the same issue as the one described in
[Xid 79 / ACPI 15 GPU crash](https://agx.fi/it/hardware.html#xid-79--acpi-15-gpu-crash).

**PCIe traffic.** During prompt processing, llama.cpp computes the layers whose experts are in system RAM on the GPU
anyway ("op offload") and copies their weights over PCIe for every ubatch. The experts take 408 MiB per layer, so with
`n-cpu-moe` 22 and `ubatch-size = 512`, that is 8.8 GiB per 512 tokens (~17.6 MiB per token), i.e. ~19 GB/s at
pp 1100 t/s, close to the practical limit of PCIe 4.0 x16. Token generation computes those experts on the CPU
and only moves activations (~0.5 MiB per token). The old `n-gpu-layers = 13` setup streamed the weights of
the 17 CPU layers the same way.

Tests on 2026-09-26, each with consecutive 63k-token prompts at ctx 65536 (ASPM and PCIe Spread Spectrum disabled
in the BIOS in all tests):

| Setup | Result | pp | tg (benchmark text) | GPU load |
|---|---|---|---|---|
| `no-op-offload = 1`, `n-cpu-moe` 22 | 10 runs (78 min) passed | 135-144 t/s | 57-64 t/s | PCIe 0.3 GB/s to and 0.1 GB/s from the GPU, 72 W mean, max 96 W, max 70 °C |
| `ubatch-size = 2048`, `n-cpu-moe` 24 | crashed on run 5 (~4.5 min) | 1645-1789 t/s | 56 t/s | PCIe bursts up to 17 GB/s (mean 2.3 GB/s), 160-220 W, 84-88 °C, throttling by power |
| PCIe-only stress test, no llama.cpp | 30 min passed, no PCIe replays | | | 25.0 GB/s one way, 2 x 17.5 GB/s both ways, 19.5 GB/s in 8.8 GiB bursts (33 TB in total), 81 W, max 67 °C |
| `ubatch-size = 2048`, `n-cpu-moe` 24, **170 W limit** | 20 runs (18 min) passed, no PCIe replays | 1690-1745 t/s | 52-56 t/s | PCIe 6.2 GB/s mean, 161 W mean, max 169 W, 80 °C mean, max 85 °C |

So PCIe traffic alone does not seem to be the trigger, but rather the heat or power of full compute load,
possibly combined with the PCIe traffic. 88 °C at ~220 W is hot for an RTX 3070, so the cooling of the card
(dried thermal paste or pads, dust, airflow in the case) may be worth checking.

`no-op-offload = 1` works without the power limit (and without the larger compute buffers, so with `n-cpu-moe = 22`),
but pp is ~12x slower. The ubatch copies the weights once per 2048 tokens instead of per 512
(24 x 408 MiB = 9.6 GiB per 2048 tokens, ~4.8 MiB per token), which also makes pp faster than the default.

**Other settings tried with the 170 W limit** and ctx 65536 (one 63k-token prompt, two concurrent 30k-token prompts
and the chat test each):

| Setup | pp | Chat tg | Peak VRAM |
|---|---|---|---|
| ubatch 2048, `n-cpu-moe` 24 (chosen) | 1747 t/s | 61 / 69 / 47 t/s | 7719 MiB |
| ubatch 1024, `n-cpu-moe` 22 | 1479 t/s | 61 / 71 / 51 t/s | 7813 MiB (too tight) |
| ubatch 4096, `n-cpu-moe` 26 and 27 | out of memory when loading | | |
| ubatch 4096, `n-cpu-moe` 28 | the first long prompt fails with `CUDA error: the resource allocation failed` (not Xid 79) | | 7829 MiB after loading |
| ubatch 2048, KV cache q8_0, `n-cpu-moe` 23 (22 ran out of memory) | 1721 t/s | 62 / 61 / 44 t/s | 7523 MiB |

The quantized KV cache is not faster and costs some quality.

**Context size with the 170 W limit and ubatch 2048.** Each additional layer of experts on the CPU frees ~408 MiB of
VRAM, i.e. room for ~19k more tokens of context (a near-full prompt, two concurrent half-size prompts and the chat test
each):

| `n-cpu-moe` | ctx | pp | Chat tg (English / code / Finnish) | Peak VRAM |
|---|---|---|---|---|
| 24 | 65536 | 1747 t/s (63k tokens) | 61 / 69 / 47 t/s | 7719 MiB |
| 25 | 81920 | 1591 t/s (79k tokens) | 59 / 62 / 46 t/s | 7761 MiB |
| 26 | 98304 | 1472 t/s (95k tokens) | 52 / 63 / 44 t/s | 7801 MiB (too tight) |

Each layer costs ~5 % of chat speed (pp also drops, partly due to the longer prompts). `n-cpu-moe = 25` with
ctx 77824 was chosen for ~19 % more context than before at ~5 % of the speed, with a lower peak than
the previously tested setting: 10 consecutive 75k-token prompts passed at pp 1589-1617 t/s with a peak of 7649 MiB
(7577 MiB after loading). This was later replaced by `n-cpu-moe = 28` and ctx 122880, see below.

**KV cache quantization and a larger context (2026-09-28).** With the 1600 MHz clock lock, the goal was to increase
the context without a significant loss of quality or speed. The tests used the upstream `chat` and `depth` tests of
[`llama_cpp_bench_http.py`](../benchmark/llama_cpp_bench_http.py) (results in `benchmark/results/agx-ai.jsonl`).

*Quality* was measured with `llama perplexity` on WikiText-2 (4 chunks of 32768 tokens, i.e. 65k scored tokens at
long distances), as the KL divergence from the f16 KV cache and the share of tokens with the same top prediction.
The instruction-tuned model predicts raw WikiText badly (PPL ~1780 with any settings, also with 512-token chunks),
but the relative differences are still meaningful. Changing only `ubatch-size` from 2048 to 512 with the f16 cache,
which should not change the quality, calibrates the numerical noise of this MoE model (small numerical differences
change which experts are selected):

| KV cache (K / V) | PPL | Mean KLD | Median KLD | 99.9 % KLD | Same top token |
|---|---|---|---|---|---|
| f16 / f16 (repeated run) | 1780.3 | 0.000 | 0.000 | 0.00005 | 100.0 % |
| f16 / f16 with `ubatch-size = 512` (noise floor) | 1777.8 | 0.099 | 0.035 | 2.40 | 84.7 % |
| q8_0 / q8_0 | 1762.8 | 0.109 | 0.038 | 2.52 | 83.9 % |
| q8_0 / f16 | 1763.5 | 0.107 | 0.037 | 2.57 | 84.1 % |
| f16 / q8_0 | 1766.5 | 0.109 | 0.038 | 2.57 | 84.1 % |
| q4_0 / q4_0 | 1788.5 | 0.275 | 0.135 | 4.25 | 73.6 % |

So q8_0 only adds ~0.01 of KLD to the noise floor, which is not a measurable loss of quality, but q4_0 is clearly
worse. In a retrieval test (8 facts hidden at 5-95 % depth of ~66k tokens of WikiText, 3 different texts,
`benchmark/scratch/needle_test.py`), the f16 cache found 21/24 and q8_0/q8_0 23/24.

*Speed* at ctx 77824 and `n-cpu-moe = 25`:

| KV cache (K / V) | VRAM after loading | Chat tg | tg at 36k tokens | tg at 70k tokens | pp at 70k tokens |
|---|---|---|---|---|---|
| f16 / f16 | 7577 MiB | 56.9 t/s | 52.5 t/s | 49.7 t/s | 1381 t/s |
| q8_0 / f16 | 7267 MiB | 56.8 t/s | 48.7 t/s (-7 %) | 43.6 t/s (-12 %) | 1366 t/s |
| f16 / q8_0 | 7331 MiB | 58.4 t/s | 46.1 t/s (-12 %) | 41.9 t/s (-16 %) | 1370 t/s |
| q8_0 / q8_0 | 6907 MiB | 56.9 t/s | 46.4 t/s (-12 %) | 40.7 t/s (-18 %) | 1359 t/s |

The quantized cache slows down token generation at long contexts, as the attention has to dequantize it
(`llama bench` showed the slowdown for both 1-token steps and the 3-token steps of MTP). It also does not halve
the cost of context: for batches of more than one token, the CUDA flash attention converts the quantized cache to
f16 in a temporary buffer that grows with the context, so each additional token still costs ~22 KiB of VRAM, and q8_0
only saves a fixed ~670 MiB. With q8_0/q8_0 at `n-cpu-moe = 25`, ctx 106496 loads with 7541 MiB and peaks at
7627 MiB, 114688 loads with 7745 MiB (too tight), and 122880 does not load. At ctx 106496, tg was 46.0 t/s at 36k,
39.5 t/s at 70k and 34.2 t/s at 100k tokens.

Putting more experts on the CPU gives the same context at a much smaller cost. Each layer makes room for ~16k tokens
(f16 KV cache, `ubatch-size = 2048`, `prompt chat depth` tests with a 70k-token prompt):

| `n-cpu-moe` | ctx | VRAM after loading / peak | pp at 70k tokens | Chat tg | tg at 36k tokens | tg at 70k tokens |
|---|---|---|---|---|---|---|
| 25 | 77824 | 7577 / 7659 MiB | 1534 t/s | 56.9 t/s | 52.5 t/s | 49.7 t/s |
| 27 | 106496 | 7543 / 7627 MiB | (1321 t/s at 102k) | 56.6 t/s | 49.2 t/s | 48.5 t/s (44.1 at 100k) |
| **28 (chosen)** | **122880** | 7583 / 7667 MiB | 1488 t/s | 54.8 t/s | 48.8 t/s | 44.9 t/s |
| 29 | 139264 | 7625 / 7709 MiB | 1472 t/s | 53.2 t/s | 48.6 t/s | 44.1 t/s |
| 30 (all experts) | 155648 | 7667 / 7751 MiB (too tight) | 1460 t/s | 53.1 t/s | 48.4 t/s | 41.9 t/s |
| 27, `ubatch-size = 1024` | 139264 | 7457 / 7513 MiB | 1224 t/s | 48.5 t/s | 43.8 t/s | 43.1 t/s |

`ubatch-size = 1024` makes room for ~30k tokens, but costs 20 % of pp and also slows down generation, so 2048 is kept.
`n-cpu-moe = 28` with ctx 122880 gives 58 % more context than 77824 for 3-4 % slower pp and chat and up to 10 % slower
generation at long contexts, which is less than what q8_0 costs for a smaller gain. 20 consecutive 119k-token
prompts passed at pp 1219-1232 t/s with a peak of 7655 MiB (111 W mean, max 118 W, 66 °C mean, max 68 °C), as did
two concurrent 59k-token prompts (peak 7655 MiB). In the retrieval test, the model found 23/24 facts at ~57k tokens
and 20/24 at ~112k tokens, so the recall of the model itself decreases somewhat at the longest contexts.

### CPU model

**Before** (separate container, no MTP, default settings): chat tg 16-16.5 t/s, pp 72 t/s for a 4096-token prompt.

**MTP.** No uncensored Gemma 4 26B-A4B on Hugging Face ships its own MTP drafter. The "MTP" release
[HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP](https://huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP)
just bundles Unsloth's drafter for the original model, so the same drafter from `unsloth/gemma-4-26B-A4B-it-qat-GGUF`
(downloaded for the GPU model) is used. It works with this heretic model with about the same acceptance as with
the original model: chat tg 23-24 t/s in English and code (acc 68-77 %) and 18 t/s in Finnish (acc 39 %),
i.e. ~45 % faster in English and ~10 % in Finnish. `llama_cpp_bench_http.py`: tg 27.0 t/s (acc 82 %), pp 72 t/s,
two concurrent requests 10.6 t/s in total. The drafter path is pinned to a snapshot, as there is no `--hf-file`
equivalent for the draft model, and `--hf-repo-draft` would pick the main model file.
Update the path if the Unsloth repository is updated.

**MTP draft length.** `spec-draft-n-max` 1, 2 and 3 gave chat tg averages of 21.0, 21.9 and 23.0 t/s.
3 was fastest only thanks to the code answer (29 t/s) and slowest in Finnish (16.6 t/s),
so 2 was kept as the balanced choice.

**Threads** (`llama bench`, fa on): 8: tg64 14.8 t/s, pp512 58 t/s; 12: 16.5 / 84; 16: 17.7 / 82; 24: 5.4 / 52.
Using all 24 threads of the container oversubscribes the physical cores, as many of the threads are SMT siblings
on the same cores (the tests ran with 24 threads on 15 cores), so `threads = 16`, set explicitly in case the default
changes.

**Flash attention** `flash-attn = on` is slightly faster than off (`llama bench`: tg64 17.1 vs 16.4 t/s,
pp512 80 vs 75 t/s; at a depth of 8192 tokens tg 13.2 vs 12.5 t/s and pp 63 vs 59 t/s).
ctx 75000 has no speed cost, as there is plenty of RAM.

**In the same container as the GPU model**, `device = none` keeps the model and its MTP drafter completely off the GPU:
`nvidia-smi` shows only the GPU model's process (7618 MiB), and the GPU model's peak is the same 7719 MiB as without
the CPU model. Chat tg 21-24 t/s in English and code and 17 t/s in Finnish, pp 74 t/s, i.e. the same as in
the previous CPU-only container.

### Prompt cache

llama-server reuses the KV cache of a previous request when a new prompt starts with the same tokens, so in a
continuing conversation, only the new messages are processed, not the whole history. When other conversations take
over the slots in between, the KV cache of the old conversation is saved to host RAM (`cache-ram`, default 8192 MiB)
and restored when the conversation continues. This works even though Gemma 4 uses sliding window attention and
the reasoning is removed from the history in thinking mode. Test with `benchmark/scratch/cache_test.py`
(a conversation with a ~35k-token history, turn 2 after two other conversations used both slots):

| Model | Turn 1 | Turn 2 |
|---|---|---|
| GPU, 35k-token history | 35188 tokens processed, 19 s | 80-92 tokens processed, 35183 from the cache, 1.0 s |
| GPU, thinking mode | 35191 tokens, 19 s | 22-27 tokens processed, 0.7 s |
| CPU, 2.5k-token history | 2503 tokens processed, 37 s | 97-102 tokens processed, 2.5 s |

The saved state of a ~35k-token conversation takes roughly 0.7-1 GB, so the default of 8192 MiB holds about
8-10 long conversations. As the container has ~95 GB of RAM free with both models loaded,
`cache-ram = 16384` is used for both models.

### Using both models at the same time

The GPU model also uses the CPU for its experts. With 16 threads in both, generating with both at the same time
collapsed to chat tg 3-4 t/s in the GPU model and 1-26 t/s in the CPU model. If the models are to be used
at the same time, use `threads = 8` in both: chat tg 34-51 t/s in the GPU model and 15-21 t/s in the CPU model
while both generate (42-57 t/s and 14-20 t/s alone).
