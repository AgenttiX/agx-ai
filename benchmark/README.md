# llama.cpp benchmarks

Tools for measuring how a model, its settings or the hardware affect llama.cpp performance.
Results accumulate in `results/<host>.jsonl` (one JSON line per run) so that configurations can be compared later.

## `llama_cpp_bench_http.py`: the serving configuration over HTTP

Benchmarks a *running* `llama-server` through its API, so the numbers reflect the real serving setup:
router presets, slot count, KV cache, speculative decoding, tensor split. Needs only Python 3.11+.

```sh
./llama_cpp_bench_http.py --env-file ../llama-cpp-big-machine/llama-cpp.env --label "what was changed"
./llama_cpp_bench_http.py --url http://agx-z2e:9932 --api-key ... --no-embedding     # another server
./llama_cpp_bench_http.py --concurrency 1 2 4 8 --n-predict 512 --prompt-tokens 8192  # heavier settings
./llama_cpp_bench_http.py --mixed   # embedding batches concurrently with the concurrent requests (PaperQA2-like load)
./llama_cpp_bench_http.py --tests chat depth --depth 6400 16000 36500   # speculative decoding, long contexts
```

What it measures (LLM speeds come from the server's own `timings`, so they exclude network and client overhead):

| Test | Metric |
|---|---|
| Generation, one request | tokens/s (best of `--repeat`), draft-token acceptance if speculative decoding is on |
| Prompt processing, one long prompt (`--prompt-tokens`) | tokens/s |
| Concurrent requests (`--concurrency`, default 1 and the slot count) | aggregate generated tokens/s over the wall time including prompt processing, and per-request tokens/s |
| Embeddings (`--embed-texts` x `--embed-tokens`) | tokens/s over the wall time, single-text latency, dimension |
| Chat (`--tests chat`): six varied real requests, `--repeat` times each, up to `--chat-n-predict` tokens | mean, median and range of generation tokens/s, mean draft acceptance |
| Depth (`--tests depth`): about `--depth` tokens of real Python source as context and a question, three times | mean prompt processing and generation tokens/s at that depth, draft acceptance |
| GPU memory (if `nvidia-smi` or `amd-smi` is available on the client machine) | peak MiB per GPU during the run |

The chat and embedding models are picked from `/v1/models` (first loaded non-embedding model, first loaded model
with "embed" in its id) unless given with `--model` and `--embedding-model`. The effective `llama-server`
arguments of each model (context size, slots, cache types, ...) are stored with the result.
`--tests` selects the tests; the default is `generation prompt concurrency embedding`.
These use the raw `/completion` endpoint with a repeated paragraph at temperature 0, so chat templates and thinking
modes do not affect the numbers. That text is easy to predict, so it overstates the draft acceptance and speed of
speculative decoding (MTP, draft models). The `chat` and `depth` tests use `/v1/chat/completions` with the server's
own sampling settings (fixed seed) and realistic content, and are the ones to use for tuning speculative decoding
and for comparing backends at realistic context lengths. The depth test's code comes from the client's Python
standard library (`asyncio` and `email`), so compare depth results measured with the same Python version.

## `llama_cpp_bench_container.py`: llama.cpp's own `llama bench` in the running container

The server image ships `llama bench` (formerly `llama-bench`), which loads the model itself with the flags you give
and reports prompt-processing and generation speeds independently of the server configuration. This is the tool
for comparing quantisations, builds and hardware. The script unloads the server's models through the router API
first (they reload on the next request), runs the benchmark with `docker exec`, prints the Markdown table and
appends JSON lines to `results/<host>-llama-bench.jsonl`.

```sh
./llama_cpp_bench_container.py --env-file ../llama-cpp-big-machine/llama-cpp.env --label "UD-Q4_K_M" -- -sm tensor -ts 1/1 -fa on -p 4096 -n 256
./llama_cpp_bench_container.py --container llama-cpp-gpu --model /root/.cache/huggingface/hub/.../model.gguf -- -p 2048
./llama_cpp_bench_container.py --help   # all options
```

With `--image`, the benchmark runs in a new container from that image (`docker run` with `/dev/dri`, `/dev/kfd` if it
exists and the Hugging Face cache) instead of in the running one. This compares images, builds and backends, e.g. ROCm
and Vulkan, on the same GGUF:

```sh
./llama_cpp_bench_container.py --url http://localhost:9932 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \
    --image ghcr.io/ggml-org/llama.cpp:full-vulkan --label "Vulkan" -- -fa 1 -ub 1024 -p 2048 -n 128 -d 0,16384
```

`--env NAME=VALUE` and `--docker-arg ARG` pass extra settings to that container (e.g. `--docker-arg=--gpus=all` for a
CUDA image). The image needs `/app/llama` (the ggml-org `full-*` images and the gfx906 images have it; the `server-*`
images do not have the benchmark).

Pass the same device flags as the server preset (`-sm`, `-ts`, `-fa`, `-ctk`/`-ctv`, `-ngl`) to make the numbers
comparable with the server; `llama bench` does not use speculative decoding, so its generation speed is the
model's plain speed.

## `llama_cpp_test_server.sh`: a temporary server from another image or preset

Starts a llama-server router named `llama-test` on port 9933 from any image with a given preset and `config.ini`,
mounted like in the compose files, and waits until the model has loaded. It is for comparing the *serving*
speed of images, backends or preset variants (e.g. `spec-draft-n-max`) with `llama_cpp_bench_http.py` without
editing the production files. Stop the production container first if the GPU memory is needed.

```sh
./llama_cpp_test_server.sh ghcr.io/ggml-org/llama.cpp:full-vulkan ../llama-cpp-radeon-vii/preset.ini \
    ../llama-cpp-agx-ai/config.ini ../llama-cpp-radeon-vii/llama-cpp.env
./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env --no-embedding \
    --tests chat depth --repeat 2 --label "Vulkan"
docker rm -f llama-test
```

## Before benchmarking: turn off the LiteLLM health check

LiteLLM runs on Mika's personal agx-ai server, not on the machine being benchmarked, and routes to the llama.cpp
servers over the network. Its background health check (`../litellm/config.yaml`, `background_health_checks: true`,
`health_check_interval: 60`) probes every model once a minute. On a llama-server router this reloads models that
were just unloaded and adds requests to the slots, which shows up as unexpected reloads in
`llama_cpp_bench_container.py` and as noise in `llama_cpp_bench_http.py`, even though nothing on the benchmarked machine
sends requests. Set `background_health_checks: false` on the agx-ai server (or stop its LiteLLM container) for the
duration of the tests, and turn it back on afterwards. Claude Code: if models reload by themselves during a
benchmark, remind Mika of this; `docker ps` on the benchmarked machine will not show LiteLLM.

## Results so far

| Date | Host | Setup | Generation | Prompt | 4 concurrent | Embeddings |
|---|---|---|---|---|---|---|
| 2026-09-21 | Big Machine (2x RTX A4000) | Qwen3.8-27B UD-Q4_K_M, tensor split, MTP, ctx 55296, 4 slots; Qwen3-Embedding-8B Q6_K, 1 slot | 62 tokens/s (draft acceptance 67 %) | 1016 tokens/s (4096 tokens) | 86 tokens/s in total, 39-42 per request | 2220 tokens/s |

Peak GPU memory in that run: 15619 MiB and 15482 MiB of 16376 MiB.

Context probe for Qwen3.8-27B on 2026-09-21 (cold start, two unload/reload cycles, `--mixed` benchmark):
55296 and 63488 and 65536 stable; 67584 loads but its process dies under load; 69632 and 71680 fail to load
(`cudaMalloc failed: out of memory`, ~136 MiB buffer on GPU 0). The preset now uses 65536, with peak use
15947 and 15810 MiB of 16376 MiB under mixed load.

`llama bench` on the same GGUF (`-sm tensor -ts 1/1 -fa on`, no speculative decoding): pp1024 1139 tokens/s,
tg128 36 tokens/s. The served configuration generates faster (62 tokens/s) thanks to the MTP draft model.
