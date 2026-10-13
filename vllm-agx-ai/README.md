# vLLM on agx-ai (RTX 3070)

vLLM on the RTX 3070 (8 GB) of the `agx-ai` LXC container, as an alternative to
[`../llama-cpp-agx-ai`](../llama-cpp-agx-ai) on the same GPU and port (9931). Only one of the two fits on the GPU at a
time. [`docker-compose.yml`](docker-compose.yml) uses the image of [`../vllm`](../vllm) (`vllm/vllm-openai:v0.31.0`,
CUDA 13.0, with the fixes of [`../vllm/vllm.patch`](../vllm/vllm.patch)) and the API key of the llama.cpp server.
The model is selected by commenting the config volume lines in or out.

| Configuration | Model | Context | Chat | Generation at 6.4k / 36.5k tokens | Prompt processing |
|---|---|---|---|---|---|
| [`config-gemma-4-26b-a4b-qat.yaml`](config-gemma-4-26b-a4b-qat.yaml) (default) | Gemma 4 26B-A4B QAT (Google's Q4_0, bit-exact), experts in RAM, MTP 2 | 98304 | 32 t/s | 30 / 27 t/s | 1100 t/s at 6.4k, 870 t/s at 36.5k, 580 t/s at 96k |
| [`config-qwen3.8-flash-next.yaml`](config-qwen3.8-flash-next.yaml) | Qwen3.8-Flash-Next W4A16 with int8 dense layers, experts and n-gram table in RAM, no MTP | 20480 | 15.3 t/s | 15.2 / 15.2 (at 16k) t/s | 396 t/s at 6.4k, 478 t/s at 19.5k |

**Compared with llama.cpp on this GPU** ([`../llama-cpp-agx-ai`](../llama-cpp-agx-ai/README.md)): **llama.cpp is
clearly better for Gemma 4 26B-A4B.** It generates 1.7 times as fast (chat ~55 vs 32 t/s, 45-49 vs 27 t/s at 36-70k
tokens), processes long prompts twice as fast (~1220 t/s at 119k vs 580 t/s at 96k) and fits 25 % more context
(122880 vs 98304). Neither fits the model in 8 GB, but they keep the experts in RAM differently: llama.cpp computes
them on the 16-core EPYC, while vLLM has no CPU kernels for them, so the GPU reads the selected experts from RAM over
PCIe for every token.

**Qwen3.8-Flash-Next: vLLM is faster, llama.cpp has 4 times the context.** vLLM generates 13 % faster (15.3 vs
13.6 t/s) and processes prompts ~20 % faster, as the GPU reads the experts over PCIe faster than the CPU decodes
llama.cpp's IQ quantization types. llama.cpp fits 81920 tokens of context instead of 20480, runs the vision encoder,
starts in 1.5 min instead of 13 and needs neither the re-quantized checkpoint nor the patched image, see
[Qwen3.8-Flash-Next](#qwen38-flash-next).

The rest of this file has the measurements behind the settings. They were made on 2026-10-12 and 2026-10-13 with the
tools in [`../benchmark`](../benchmark/README.md), on test servers
([`../benchmark/vllm_test_server.sh`](../benchmark/vllm_test_server.sh), port 9933) with the production containers
stopped. The results are in [`../benchmark/results/agx-ai.jsonl`](../benchmark/results/agx-ai.jsonl) (labels starting
with "agx-ai vLLM"). VRAM figures are those of the vLLM processes (`nvidia-smi --query-compute-apps`); the GPU has
no display, and 7839 MiB of its 8192 MiB are usable.

## Hardware

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS in an LXC container on Proxmox VE (kernel 7.0.14-14-pve), host `agx-h12` (Supermicro H12SSL-i) |
| CPU | AMD EPYC 7302 (Zen 2, 16 cores, 32 threads; AVX2, no AVX-512), of which the container gets 24 threads (see [`../llama-cpp-agx-ai`](../llama-cpp-agx-ai/README.md)) |
| RAM | Registered DDR4 in 8 channels (16 GB ranks according to EDAC), 128 GiB available to the container; the frequency needs root on the host to read (`dmidecode`). ~60 GB/s copy bandwidth in the container (PyTorch, 16 threads) |
| NPU | none |
| GPU | NVIDIA GeForce RTX 3070, headless; driver 610.57.04 (CUDA 13.3), core clock locked to 1600 MHz by [`../llama-cpp-agx-ai/gpu-power-limit`](../llama-cpp-agx-ai/gpu-power-limit), PCIe 4.0 x16 |
| GPU architecture | Ampere (GA104, sm_86) |
| VRAM | 8 GB GDDR6 (8192 MiB) |

**The 1600 MHz clock lock (or the 170 W power limit) is required**, as for llama.cpp: without it, long prompts can
crash the GPU (Xid 79), see [`../llama-cpp-agx-ai`](../llama-cpp-agx-ai/README.md#xid-79-crashes). During these
tests, the clock lock was active (1590 MHz at ~125 W under full load) and the power limit was the default 240 W,
which is now also the default of the script.

## Setup

```sh
# Image: ../vllm (built by ../.github/workflows/deploy-docker-vllm.yml)
docker pull ghcr.io/agenttix/agx-ai/vllm:latest
# Gemma 4 26B-A4B: the bit-exact QAT checkpoint with an fp16 embedding (~10 min on the CPU, ~30 GB of RAM; downloads
# Google's Q4_0 GGUF, 15 GB, which can be deleted afterwards)
docker run --rm --memory 40g -v "$PWD/../vllm-radeon-vii":/src:ro -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    -v ~/.cache/vllm-models:/out --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm:latest \
    /src/repack_gemma4_qat.py --out /out/gemma-4-26B-A4B-it-qat-q4_0-ct-embed16-head8 --embed-bits 16 --lm-head-bits 8
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub python:3-slim sh -c \
    "pip install -q 'huggingface_hub[hf_xet]' && hf download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
# Qwen3.8-Flash-Next (125 GB, must stay in the cache), then its dense layers and lm_head in int8 (~10 min, 65 GB)
docker run --rm -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub python:3-slim sh -c \
    "pip install -q 'huggingface_hub[hf_xet]' && hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
    --revision ef554143369a706525336f6b42a09094835dc077 --exclude 'docs/*' --exclude 'runtime/*'"
docker run --rm --memory 40g -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro -v ~/.cache/vllm-models:/out \
    -v "$PWD":/src:ro --entrypoint python3 ghcr.io/agenttix/agx-ai/vllm:latest /src/quantize_flash_next_dense.py \
    --out /out/Qwen3.8-Flash-Next-W4A16-FP8PLE-dense-int8 --bits 8 --lm-head-bits 8
# Stop llama.cpp on this GPU and start vLLM
docker compose -f ../llama-cpp-agx-ai/docker-compose.yml stop
docker compose up -d
```

The Qwen3.8-Flash-Next configuration needs the fixes of 2026-10-13 in [`../vllm/vllm.patch`](../vllm/vllm.patch)
(UVA weight offloading), which are in the image once the workflow has rebuilt it after the push. Until then, build
it locally with `docker build -t ghcr.io/agenttix/agx-ai/vllm:latest ../vllm` (done on agx-ai on 2026-10-13). The
Gemma configuration works with and without them.

The first start compiles kernels for a few minutes; later starts of Gemma take ~3 min (the caches are in the `vllm-cache`
volume), most of it for loading the 15 GiB checkpoint and pinning 12 GiB of it in RAM.

## Gemma 4 26B-A4B

The bit-exact checkpoint of [`../vllm-radeon-vii`](../vllm-radeon-vii/README.md#gemma-4-26b-a4b-qat) (all linear
weights from Google's Q4_0 GGUF), with Google's QAT MTP drafter, as in [`../vllm-agx-z2e`](../vllm-agx-z2e). It does
not fit in 8 GB: the experts alone take 11.2 GiB.

**Experts in RAM.** vLLM 0.31 has no CPU kernels for the experts of a GPU model, and no expert offloading in the
upstream code (only RFCs and forks, e.g. [vllm-project/vllm#38256](https://github.com/vllm-project/vllm/issues/38256)).
It does have generic weight offloading. With `offload-backend: uva` and `cpu-offload-params: [experts, embed_tokens]`,
the selected parameters are moved to pinned RAM and the GPU reads them directly over PCIe (unified virtual
addressing, zero-copy), without copying whole layers. The fused MoE kernel reads only the 8 experts per layer that
the router selects, ~765 MiB per token, so generation is bound by PCIe 4.0 x16 at ~20 GB/s. The embedding is only
looked up, so it stays in fp16 in RAM (on the 16 GB GPUs, the embedding has to be quantized to int4 to fit). On
the GPU remain the attention, the shared MLP, the routers, the int8 lm_head and the MTP drafter (4.12 GiB according
to vLLM). The `prefetch` backend would copy whole layers to the GPU instead, i.e. all 128 experts of a layer for
every token.

**Prompt processing** reads all experts of every layer over PCIe for every 2048-token chunk, so it is 2-2.5 times
slower than in llama.cpp, which also moves all expert weights over PCIe for every batch of a prompt (see
[its README](../llama-cpp-agx-ai/README.md#xid-79-crashes)), but with bulk copies instead of the zero-copy reads of
the MoE kernel.

| Variant (all with an int8 KV cache, one sequence) | Context | Chat | Gen. 6.4k | Gen. 36.5k | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|---|
| No MTP | 32768 | 26.6 | 25.2 | | 1135 t/s at 6.4k | 6118 MiB |
| MTP 2 | 40960 | 31.6 | 30.2 | | 1102 t/s at 6.4k | 6442 MiB |
| MTP 2 | 81920 | 31.6 | | 27.3 | 868 t/s at 36.5k, 649 t/s at 79k | 7502 MiB |
| MTP 2, 4096-token prefill chunks | 40960 | | | 26.7 | 934 t/s at 36.5k, 923 t/s at 39k | 7320 MiB |
| MTP 2, KV cache 2.85 GB | 98304 | 32.0 | | | 584 t/s at 96k | 7722 MiB |
| MTP 3, KV cache 2.7 GB, CPU KV offloading | 98304 | 31.7 | 29.9 | | 1073 t/s at 6.4k | 7622 MiB |

- **MTP** with 2 drafted tokens: +19 % (acceptance 65-70 %). The 3 verified tokens select more distinct experts than
  one token, so each step reads more from RAM; 3 drafted tokens were not faster (acceptance 57-61 %).
- **Context.** The KV cache takes ~27 KiB per token with int8 (including the sliding-window layers), and vLLM needs
  ~1.05 times the context. With 2.85 GB (108630 tokens), the peak was 7722 MiB with a 96000-token prompt, too close
  to the 7839 MiB, so the configuration uses 2.7 GB (102909 tokens) for the same 98304 tokens of context.
- **4096-token prefill chunks** were only 8 % faster at 36.5k tokens, but the sliding-window layers then need a
  larger KV cache per sequence, which halved the context (40352 tokens at most with 1.68 GiB).
- **Experts on the GPU.** Each layer of experts kept on the GPU (lowering `cpu-offload-gb`) would cost 408 MiB, i.e.
  ~15k tokens of context, and remove only 1/30 of the PCIe reads of a token. Not tested.
- **CPU KV offloading** (`kv-offloading-size: 16`, `kv-offloading-backend: native`) works for this model only without
  MTP. Test: a 25k-token conversation, then two 90k-token prompts that evict it from the GPU, then the conversation
  again. Without MTP, it came back in 0.3 s instead of 25 s (369 MB loaded from RAM). With the MTP drafter (2 or 3
  tokens, with 1 or 50 generated tokens per request), the connector stored the blocks but never restored them, and
  the conversation took 26.5-27.9 s again. vLLM logs that it excludes the trailing chunk of the drafter's KV group
  from offloading; the lookup then apparently fails for the whole request. As MTP speeds up every request and the
  GPU prefix cache already holds ~100k tokens, the configuration uses MTP without KV offloading.
  [`../vllm-agx-z2e/config-gemma-4-26b-a4b-qat.yaml`](../vllm-agx-z2e/config-gemma-4-26b-a4b-qat.yaml) uses the same
  combination, so it probably has the same problem.

Results of the configuration (`--tests generation prompt chat depth --prompt-tokens 96000 --depth 6400 36500`, with
the image including the fixes of 2026-10-13):

| Test | Result |
|---|---|
| Generation, 256 tokens | 36.7 t/s (MTP acceptance 98 % with the repetitive text of this test) |
| Chat, 6 requests | 32.0 t/s (28.4-35.4, acceptance 70 %) |
| Depth 6.4k / 36.5k | 30.0 / 27.6 t/s (prompt 1102 / 868 t/s) |
| Prompt of 96000 tokens | 584 t/s |
| Peak VRAM | 7582 MiB |

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a 125B MoE (512 experts, 10 active per token, 48 layers, 12 of them with Qwen Sparse Attention
and 36 with Gated DeltaNet) with a 51B-parameter n-gram embedding table. On agx-z2e it runs with the single-RTX-3090
runtime of DominikBucko/qwen38-flash-next-3090 (see [`../vllm-agx-z2e`](../vllm-agx-z2e/README.md#qwen38-flash-next)),
which needs 9.8 GB of VRAM for its BF16 dense weights alone and has no option for smaller GPUs. Stock vLLM 0.31 runs
the model with the experts read from RAM over UVA as for Gemma, but only with six fixes and a re-quantized
checkpoint. The vLLM CPU backend is no alternative: it would need the whole model in RAM, the model has no CPU
implementation in vLLM, and the fast paths need AVX-512.

**Checkpoint.** [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)
(the checkpoint of the agx-z2e runtime) is the only one that stock vLLM loads with an FP8 n-gram table (47.6 GiB
instead of 95 GiB in BF16): the routed experts in Intel's AutoRound int4 (58.4 GiB), everything else in BF16 (9.2 GiB).
The other int4 checkpoints keep the n-gram table in BF16, or need patched vLLM builds for their FP8/INT6 layers.
[`quantize_flash_next_dense.py`](quantize_flash_next_dense.py) quantizes the BF16 Gated DeltaNet, attention and
shared expert projections and the lm_head to int8 with round-to-nearest (group size 128, the GPTQ format of the
experts), which takes the weights on the GPU from 8.0 GiB (without the embedding) to 5.03 GiB. Its
relative error is 0.8 % (int4: 14 %). The output directory gets new shards for the quantized tensors (65 GB) and
symbolic links to the n-gram table and the other files of the source, which must stay in the Hugging Face cache.

**The fixes**, in [`../vllm/vllm.patch`](../vllm/vllm.patch) (details in [`../vllm/README.md`](../vllm/README.md)),
found one at a time:

| Problem | Symptom | Fix |
|---|---|---|
| PyTorch's pinned memory rounds every block up to the next power of two | the 47.6 GiB n-gram table took 64 GiB, the 838 MB expert tensors 1 GiB each: the RAM ran out (the engine was killed silently at 129.5 of 131 GB) | exactly sized pinned memory (`cudaHostRegister` of an anonymous mapping); now 113 GB of RAM in use, 14-17 GB free |
| The offloader skipped a module whose first parameter is already on the CPU | layer 1, which holds the pinned n-gram table, kept its experts (1.25 GiB) on the GPU | look for the device among all parameters |
| The embedding is created outside `make_layers` | `cpu-offload-params: embed_tokens` had no effect (1.2 GiB on the GPU) | Qwen4Exp offloads it |
| The lm_head was created without the quantization config | the int8 lm_head did not load | pass it (checkpoints without a quantized lm_head are unaffected) |
| The Marlin post-processing of GPTQ experts transposed them to GPU copies and back | 2.8 GB of temporary GPU memory per layer, out of memory | transpose views only, repack w13 and w2 one after the other, reading the offloaded weights directly |
| Triton has no FP8 e4m3 type before sm_89 | `type fp8e4nv not supported in this architecture` in the n-gram lookup | copy the FP8 rows as uint8 |

Two settings are needed as well: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (the first request ran out of GPU memory
with the default of 512 MiB, which the memory profiling does not account for) and `TORCHINDUCTOR_COMPILE_THREADS=1`
(no RAM for parallel compile workers), both in [`docker-compose.yml`](docker-compose.yml).

**Memory.** ~107 GiB of the container's 128 GiB of RAM are pinned (experts, n-gram table and embedding), and
5.03 GiB of weights are on the GPU. The remaining ~1.1 GB of VRAM holds a KV cache of 24041 tokens in BF16 (~45 KiB
per token; the FP8 KV cache of this model is dequantized with Triton's e4m3 type, which needs sm_89). With 1.45 GB,
a 23500-token prompt ran out of memory, so the context is 20480. MTP does not fit. Raising the memory limit of the
LXC container (the host has 256 GB) would not help the context, only the margin. Loading takes ~13 min (~10 min for
reading 125 GB and pinning the weights, as the page cache has no room).

| Variant (KV cache in BF16) | Context | Chat | Gen. 6.4k / 16k | Prompt processing | Peak VRAM |
|---|---|---|---|---|---|
| Eager mode, KV cache 0.8 GB | 12288 | 7.4 | 7.7 / - | 383 t/s at 6.4k | 7104 MiB |
| CUDA graphs for decoding, KV cache 0.8 GB | 12288 | 15.3 | 15.2 / - | 396 t/s at 6.4k | 7062 MiB |
| **CUDA graphs, KV cache 1.1 GB (current)** | 20480 | 15.3 | 15.2 / 15.2 | 396 / 421 t/s at 6.4k / 16k, 478 t/s at 19.5k | 7322 MiB |

- **Generation** reads ~1.2 GB of experts per token over PCIe (10 of 512 experts in each of the 48 layers) and does
  not slow down with the context (Qwen Sparse Attention).
- **Quality.** In the long-context retrieval test of
  [`../benchmark/llama_cpp_quality_http.py`](../benchmark/llama_cpp_quality_http.py) (44 questions in a 17360-token
  context), the int8 checkpoint answered 38/44 (single lookups 32/32, two-step 6/12) and llama.cpp's IQ3_S 37/44
  (32/32, 5/12), so the round-to-nearest int8 layers do not cost measurable quality.

**Compared with llama.cpp** ([`../llama-cpp-agx-ai`](../llama-cpp-agx-ai/README.md#qwen38-flash-next)): vLLM generates
13 % faster (15.3 vs 13.6 t/s in chat, 15.2 vs 12.1 t/s at 16-36.5k tokens) and processes prompts 20 % faster at
6.4k tokens (396 vs 328 t/s), as the GPU reads the experts over PCIe faster than the 16 cores decode llama.cpp's IQ
types. llama.cpp fits 4 times the context (81920 vs 20480), runs the vision encoder, starts in 1.5 min instead of 13,
and needs no re-quantized checkpoint or patched image.

## Commands

```sh
cd ../benchmark
# Serving speed (the configuration of docker-compose.yml)
./llama_cpp_bench_http.py --url http://localhost:9931 --env-file ../llama-cpp-agx-ai/llama-cpp.env --no-embedding \
    --tests generation prompt chat depth --prompt-tokens 96000 --depth 6400 36500 --repeat 1 \
    --server-config ../vllm-agx-ai/config-gemma-4-26b-a4b-qat.yaml --label "..."
# A configuration variant on a test server (port 9933; stop the production container first)
docker compose -f ../vllm-agx-ai/docker-compose.yml stop
CACHE_VOLUME=vllm-agx-ai_vllm-cache DOCKER_ARGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCHINDUCTOR_COMPILE_THREADS=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128" ./vllm_test_server.sh ghcr.io/agenttix/agx-ai/vllm:latest /tmp/config-variant.yaml
./llama_cpp_bench_http.py --url http://localhost:9933 --no-embedding --tests chat depth --repeat 1 --depth 6400 \
    --server-config /tmp/config-variant.yaml --label "..."
docker rm -f vllm-test && docker compose -f ../vllm-agx-ai/docker-compose.yml start
```
