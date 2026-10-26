# HyperQwen on the RTX 3090 of agx-z2e

[HyperQwen](https://github.com/syv-ai/HyperQwen) for Qwen3.8-27B on the RTX 3090 of agx-z2e, tuned for agentic coding
with one user and one or a few agents. HyperQwen is a patch series for vLLM 0.31.0, a re-quantized checkpoint and two
drafters (MTP and DFlash2). Its image is used as is, with the settings in the `environment` of
[`docker-compose.yml`](docker-compose.yml), and the CPU KV offloading of this repository added. It replaces the
Qwen3.8-27B configuration of [`../vllm-agx-z2e`](../vllm-agx-z2e) on the same GPU and port (9931), with the same model
name (`Qwen/Qwen3.8-27B`) and API key, and is faster for one to three agents at once.

## Current configuration

| Model | Context | VRAM (peak) | RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|---|
| Qwen3.8-27B W4A16 (AutoRound, int4 lm_head), DFlash2 drafter (W4A16) + lookup drafting, int8 KV cache, pth-flash-attn | 139264 | 21478 MiB | 38 GiB: 32 GiB CPU KV cache, ~2 GiB embedding, ~4 GiB processes | chat 136 t/s; 128 / 110 / 80 / 71 t/s at 6.4k / 32k / 100k / 136k tokens of context; **290 t/s when copying code** (edits) | 1220 t/s at 6.4k, 820 t/s at 100k, 740 t/s at 136k | up to 4 sequences; an agent's evicted conversation comes back from RAM (25k tokens: 1.8 s instead of 21 s) |

Compared with the Qwen3.8-27B configuration of [`../vllm-agx-z2e`](../vllm-agx-z2e) as it was before (stock vLLM 0.31
with [`../vllm/vllm.patch`](../vllm/vllm.patch), GPTQ int4, MTP 2, Triton attention), measured on the same day with the
same tests: chat 136 vs 86 t/s (+58 %), generation 128 / 110 / 80 vs 80 / 67 / 41 t/s at 6.4k / 32k / 100k tokens of
context (+61 / +64 / +97 %), code edits 290 vs 107 t/s (2.7 times), prompt processing 822 vs 544 t/s at 100k (+51 %).
That configuration now has two of HyperQwen's optimizations too (see
[Combining the optimizations](#combining-the-optimizations)): 155648 tokens of context, chat 83 t/s, 83 / 79 / 66 t/s,
edits 112 t/s, prompt processing 812 t/s at 100k. llama.cpp is slower for this model on this GPU (see
[`../vllm-agx-z2e/README.md`](../vllm-agx-z2e/README.md)).

For more than ~3 agents at once, `SPEC=mtp` is better (see [Concurrent agents](#concurrent-agents)).

## Hardware

| | |
|---|---|
| Computer | agx-z2e (desktop) |
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-38-generic), KDE Plasma desktop (Wayland) on the RTX 3090 |
| CPU | AMD Ryzen Threadripper 3970X (Zen 2, 32 cores, 64 threads) |
| RAM | 128 GB DDR4-2666 in 4 channels (8 x 16 GB unbuffered ECC DIMMs), 125.6 GiB visible to the OS |
| NPU | none |
| GPU | NVIDIA GeForce RTX 3090, which also drives the desktop (1.0-2.7 GiB of VRAM); driver 610.43.02 (CUDA 13.3), power limit 350 W (default), PCIe 4.0 x16 |
| GPU architecture | Ampere (GA102, sm_86) |
| VRAM | 24 GB GDDR6X (24576 MiB) |

The motherboard and the other GPUs are in [`../vllm-agx-z2e/README.md`](../vllm-agx-z2e/README.md#hardware).

## Setup

```sh
docker compose -f ../vllm-agx-z2e/docker-compose.yml stop   # or ../llama-cpp-agx-z2e, on the same GPU and port
docker compose run --rm prepare   # once: ~22 GB of downloads and a few minutes of re-quantization on the CPU
docker compose up -d
```

`prepare` runs HyperQwen's `docker/prepare.sh` into `~/.cache/vllm-models/hyperqwen`: it downloads
[`dbirks/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) (19.5 GB),
re-quantizes its BF16 lm_head and embedding (2.5 GB each) and its MTP module to int8, and downloads the "fast" variant
(the lm_head and the MTP module in int4 GPTQ, 1 GB, used by default) and the
[DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) re-quantized to W4A16 (1.2 GB). The first start
compiles kernels for ~5 min; later starts take 2-4.5 min (the caches are in the `cache` volume). The image is that of
HyperQwen's commit `ce9615c` (2026-10-22), pinned by digest. The project was called `qwen38-27b-rtx3090` before;
its last image under that name (`ghcr.io/syv-ai/qwen38-27b-rtx3090`) is from 2026-09-17.

The measurements were made on 2026-10-26 with the tools in [`../benchmark`](../benchmark/README.md): `chat`, `depth`
(real Python code as the context and a question about it) and the new `edit` test (a 2.2k-token file output again with
one identifier renamed, without thinking, as a coding agent's edit), all with the server's sampling settings
(temperature 1.0, top-k 20, top-p 0.95). The results are in
[`../benchmark/results/agx-z2e-kubuntu.jsonl`](../benchmark/results/agx-z2e-kubuntu.jsonl) (labels starting with
"z2e HyperQwen", "z2e vllm-rtx3090-hyperqwen" and "z2e vLLM Qwen3.8-27B"). The test servers ran on port 9933 with the
production containers stopped. VRAM figures are those of the vLLM processes, i.e. without the desktop, which used
1.0-1.3 GiB.

## HyperQwen's modes

HyperQwen's launcher has a few ready-made modes (`SPEC` and `CTX`). On this GPU, with the 21 GiB that the other
configurations of this GPU use too (which leaves ~1 GB free next to a desktop of up to 2.7 GiB):

| Configuration | Context | Chat | Gen. 6.4k / 32k / 100k | Edit (short / after 32k) | Prompt 6.4k / 100k | Peak VRAM |
|---|---|---|---|---|---|---|
| [`../vllm-agx-z2e`](../vllm-agx-z2e) (before) | 147456 | 86.3 | 79.6 / 66.7 / 40.5 | 107.1 / 89.0 | 1143 / 544 | 21056 MiB |
| `SPEC=dflash2 CTX=long` (HyperQwen's defaults) | 131072 | 133.2 | 118.7 / 95.7 / 78.7 | 290.8 / 266.7 | 1208 / 818 | 22160 MiB |
| The same with `EMBED_UVA=1` | 131072 | 133.1 | 118.6 / 95.5 / 78.5 | 290.0 / 266.3 | 1205 / 816 | 20960 MiB |
| `SPEC=mtp CTX=long EMBED_UVA=1` | 150000 | 105.9 | 102.3 / 96.9 / 80.8 | 146.5 / 137.6 | 1182 / 801 | 20968 MiB |
| **`SPEC=dflash2 CTX=long EMBED_UVA=1` + 32 GiB CPU KV cache (current)** | 139264 | 136.4 | 128.4 / 109.7 / 79.9 | 290.3 / 266.7 | 1223 / 822 | 21466 MiB |

The current row was measured with the default KV cache (131072 tokens of context, peak 21130 MiB with a 128k-token
prompt: 67.1 t/s, 753 t/s prompt processing). With the larger cache of the configuration, 136k tokens of context gave
70.9 t/s and 743 t/s (peak 21466 MiB), and a final check of the production container chat 134.0 t/s, 125.5 / 101.8 t/s
at 6.4k / 32k, edits 298.8 t/s (peak 21478 MiB). The chat test varies by ±10 % from run to run (the generated text and its
draft acceptance change), the depth test at 32k by about as much.

- **DFlash2** ([Inco's block drafter](https://inco.ai/blog/dflash2/)) drafts 7 tokens in one pass from hidden states
  of the target model, plus tokens copied from the context by HyperQwen's lookup drafting when the model reproduces
  its context. Coding agents do that a lot: edits repeat the code around the change. The `edit` test runs at
  290 t/s with it (97 % of the drafts accepted), twice as fast as with MTP. It also wins in chats and short contexts;
  at 32k-100k tokens of context, MTP is as fast.
- **The pth-flash-attn kernel** ([noon-at-cgn/pth-flash-attn](https://github.com/noon-at-cgn/pth-flash-attn), `PTH_FA=1`,
  HyperQwen's default for this mode on sm_86) reads the int8 KV cache for the target and the drafter. It is the main
  reason why the long contexts are faster than with the patched Triton attention of
  [`../vllm/vllm.patch`](../vllm/vllm.patch) (see [Combining the optimizations](#combining-the-optimizations)).
- **`EMBED_UVA=1`** keeps the int8 token embedding in pinned RAM (the GPU reads the rows of the prompt's tokens over
  PCIe): 1.2 GiB less VRAM, with no measurable cost. Without it, this mode would need 22.2 GiB.
- **The context**: the mode's default KV cache (5.58 GB, `KV_MEM`) holds 136429 tokens for 131072 tokens of context.
  The configuration uses 5.95 GB (146461 tokens) for 139264 tokens of context, which keeps the peak at 21466 MiB with
  a 136k-token prompt and with 4 agents at once. HyperQwen's launcher warns about more than its default; it is
  tuned for a 24 GB GPU without a desktop. Each DFlash2 sequence reserves recurrent-state pages for all of its 8
  verified tokens (MTP: 4), so its pool holds fewer tokens per GB than MTP's (178643 tokens in 6.37 GiB), and the mode
  allows 4 instead of 8 sequences.
- `SPEC=mtp CTX=long` (150000 tokens of context) has no fixed KV cache size by default; `GPU_UTIL=0.85` gave 6.37 GiB.
  `GPU_UTIL` is set to 0.85 in all variants: vLLM checks at startup that this share of the GPU is free, and the default
  of 0.93 would fail next to the desktop. With a fixed KV cache (`KV_MEM`), it has no other effect.
- Not tested: `DFLASH_TOKENS=15` ("reproduction mode", 16 instead of 8 verified tokens per step while the lookup
  drafting fires: HyperQwen measured +50 % when reproducing a document, but -3 to -20 % in mixed long-context work),
  which needs the FlashAttention-2 FP8 or Triton int8 KV cache instead of pth-flash-attn and leaves less context;
  `CTX=huge` (KVarN 4/2-bit KV cache, 240k tokens, about half the speed past 100k); `INT8_ACT=int8` (W4A8 prefill,
  +20-30 % prompt processing, +4 % perplexity).

## Concurrent agents

Each agent had its own 16k- or 32k-token code context already in the prefix cache (as for an agent's next turn), and
all agents asked their question at the same time (512 tokens each, one run per point;
[`../benchmark/vllm_concurrent_agents.py`](../benchmark/vllm_concurrent_agents.py)). DFlash2 ran with the CPU KV cache
(40 GiB at 16k, 32 GiB at 32k), MTP without it:

| Agents x context | DFlash2 (current), per agent | in total | MTP (`SPEC=mtp CTX=long`), per agent | in total |
|---|---|---|---|---|
| 1 x 16k | 99.8 t/s | | 115.4 t/s | |
| 2 x 16k | 103.4 t/s | 131 t/s | 91.2 t/s | 124 t/s |
| 4 x 16k | 57.0 t/s | 147 t/s | 79.8 t/s | 182 t/s |
| 1 x 32k | 113.2 t/s | | 105.6 t/s | |
| 2 x 32k | 86.1 t/s | 124 t/s | 91.5 t/s | 130 t/s |
| 3 x 32k | 82.4 t/s | 106 t/s | 47.4 t/s (two contexts computed again: 26 and 34 s to the first token) | 38 t/s |

DFlash2 is as fast as MTP for 1-2 agents (and twice as fast in edits), MTP is faster for 4. With 3 x 32k, MTP had lost
two of the three cached contexts although they fit in its pool; the current configuration got the third one back from
the CPU KV cache (9.3 s to the first token).

## CPU KV offloading

vLLM's native CPU KV offloading (`KV_OFFLOAD_GB`, `--kv-offloading-size`) keeps the prefix cache blocks that are evicted
from the GPU in pinned RAM. HyperQwen's launcher refuses it with `PTH_FA=1` as untested, so the entrypoint of
[`docker-compose.yml`](docker-compose.yml) removes that refusal from the launcher before starting it. Tested with
[`../benchmark/vllm_prefix_return.py`](../benchmark/vllm_prefix_return.py) (a 25030-token conversation, evicted from
the GPU by other prompts, then asked again; greedy). Without a CPU KV cache (HyperQwen's default), an evicted
conversation is computed again, as cold:

| CPU KV cache | Cold | Again (GPU prefix cache) | After 2 x 100k other tokens | After 1 x 120k other tokens |
|---|---|---|---|---|
| 16 GiB | 21.5 s | 1.7 s | 23.1 s: the cache was full (22.9 GB written) | 1.8 s |
| 40 GiB | 21.4 s | 1.7 s | **1.8 s** (0.89 GB loaded) | |

The answers were identical to the cold one in all cases. Each offloaded block carries a snapshot of the recurrent
(Gated DeltaNet) state, so the CPU cache needs ~100 KB per token, ~2.5 times as much as the GPU's int8 cache: 16 GiB
held less than the 225k tokens of the test. The configuration uses 32 GiB (~330k tokens, e.g. three agents with
100k tokens each). It is pinned at startup, as a file in the container's `/dev/shm` (`shm_size: 36gb`). HyperQwen's
compose file uses the host's `/dev/shm` (`ipc: host`), but its launcher deletes the offload files there that no
process of its own container maps, which could include those of another vLLM container with offloading.

As in [`../vllm-agx-z2e`](../vllm-agx-z2e), the offloading makes no measurable difference to the speed. The launcher
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` with it (vLLM rejects KV connectors with expandable segments;
[`../vllm/vllm.patch`](../vllm/vllm.patch) allows them, HyperQwen does not), with no measurable difference to the VRAM
use either.

## Combining the optimizations

Which optimizations of HyperQwen and of this repository could be combined:

| Optimization | From | Combined | Result |
|---|---|---|---|
| pth-flash-attn kernel for the int8 KV cache | HyperQwen | **in [`../vllm`](../vllm)'s image** (built from its source, on x86-64) | our Qwen3.8-27B configuration: 100k tokens of context 40.5 -> 69.1 t/s, 32k 66.7 -> 80.4 t/s, prompt processing at 100k 544 -> 794 t/s |
| Recurrent state in FP16 (`mamba-ssm-cache-dtype: float16`) | HyperQwen | **in [`../vllm-agx-z2e`](../vllm-agx-z2e)'s Qwen3.8-27B configuration** | 6.5 % more tokens in the same KV cache (158480 -> 168848), context 147456 -> 155648; the speed is unchanged within the noise |
| CPU KV offloading | this repository | **in this configuration** | a 25k-token conversation evicted from the GPU: 1.8 s instead of 21 s |
| Split-KV Triton attention for MTP and larger prefill blocks on sm_86 ([`../vllm/vllm.patch`](../vllm/vllm.patch)) | this repository | not needed | pth-flash-attn is faster at long contexts (69 vs 41 t/s at 100k with our configuration) |
| Int4 embedding and lm_head (JRamirez-UAB's checkpoint) | this repository | not needed | HyperQwen has an int4 lm_head and keeps the int8 embedding in RAM (`EMBED_UVA`), which saves more VRAM |
| DFlash2 with lookup drafting, MTP draft vocabulary, sampler and Marlin patches | HyperQwen | not ported | they need most of HyperQwen's 50 patches; using its image is simpler than porting them |

So the main optimizations of both sides combine: HyperQwen's DFlash2, lookup drafting and pth-flash-attn kernel
with this repository's CPU KV offloading here, and HyperQwen's kernel and FP16 recurrent state in
[`../vllm`](../vllm)'s image and [`../vllm-agx-z2e`](../vllm-agx-z2e)'s Qwen3.8-27B configuration. That configuration
stays useful as the stock-vLLM alternative and for 4 agents at once (MTP, 4 sequences), but this one is faster for
the use case. Our Qwen3.8-27B configuration with each step (same tests as above):

| Our configuration ([`../vllm`](../vllm) image) | Context | Chat | Gen. 6.4k / 32k / 100k | Edit (short / after 32k) | Prompt 6.4k / 32k / 100k |
|---|---|---|---|---|---|
| Triton attention with the fix of [`../vllm/vllm.patch`](../vllm/vllm.patch) (before) | 147456 | 86.3 | 79.6 / 66.7 / 40.5 | 107.1 / 89.0 | 1143 / 867 / 544 |
| pth-flash-attn (copied from HyperQwen's image) | 147456 | 81.2 | 90.6 / 80.4 / 69.1 | 107.8 / 100.7 | 1212 / 1042 / 794 |
| pth-flash-attn (built from its source) + FP16 recurrent state (current) | 155648 | 82.8 | 83.4 / 79.4 / 66.3 | 112.2 / 102.9 | 1209 / 1054 / 812 |

The chat and the 6.4k depth test vary by up to ±10 % between runs. The pth-flash-attn backend worked with this
repository's CPU KV offloading in both images.

## Other notes

- **Tool calls** work with the `qwen3_coder` parser, as in [`../vllm-agx-z2e`](../vllm-agx-z2e); HyperQwen also sets
  `--tool-strict-level parameter`. HyperQwen's `prepare` makes the chat template accept tool-call arguments as JSON
  arrays (some clients send them) and the reasoning efforts `minimal`/`high`/`max` (mapped to the template's levels).
- **Killing the container** (`docker rm -f`) leaves a zombie process for a minute or two while the 32 GiB of pinned RAM
  is released; `docker compose stop` (graceful) does not.
- HyperQwen requires the API key on every path except `/health`, `/ping`, `/load` and `/version`, so `/metrics`
  and `/tokenize` need the key too (unlike stock vLLM).
- Speculative decoding (MTP and DFlash2) does not change the sampled distribution. The int8 KV cache, the
  pth-flash-attn kernel and the FP16 recurrent state are HyperQwen's defaults for this mode on an RTX 3090; HyperQwen
  measured GSM8K 0.96 and perplexity unchanged with them. Not measured here.
