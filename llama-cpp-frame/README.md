# llama.cpp on the Steam Frame

llama.cpp router server for agx-frame, a Valve Steam Frame VR headset
(Snapdragon 8 Gen 3, Adreno 750 GPU, 16 GB of unified LPDDR5X).
It runs in rootless Podman from the official `ghcr.io/ggml-org/llama.cpp:server-vulkan` image (arm64), as a
Quadlet systemd user unit ([llama-cpp.container](llama-cpp.container)), with the models from the host's Hugging Face
cache. The models and their settings are in [preset.ini](preset.ini), and the server-wide settings in
[config.ini](config.ini). Measured on 2026-10-26 with build b11870 (74bca3950).

| Model | Context | VRAM = RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|
| Gemma 4 E4B QAT UD-Q4_K_XL (default) | 32768 | 4.5 GiB (model 3.9 GiB, vision/audio encoder, KV cache) | 9.0 t/s, chat 8.9 t/s; 8.7 t/s at 4k context | 53 t/s (2048 tokens); 42 t/s at 4k context | The CPU generates faster in short chats (12.6 t/s), but processes prompts at half the speed |
| Gemma 4 E2B QAT UD-Q4_K_XL | 32768 | 2.4 GiB | 13.7 t/s | 106 t/s (1024 tokens) | The CPU generates 26 t/s with a short context |
| Gemma 4 12B QAT UD-Q4_K_XL | 16384 | 6.8 GiB | 4.2 t/s | 19 t/s (1024 tokens) | `ubatch-size = 128` (512 hangs the GPU); leaves only 3.2 GiB free |
| Qwen3.5-9B Q4_0 | 16384 | 5.4 GiB | 6.1 t/s | 32 t/s (1024 tokens) | |

All on the GPU (Vulkan), measured with the server ([../benchmark/llama_cpp_bench_http.py](../benchmark/llama_cpp_bench_http.py)):
generation of 128-256 tokens, a 1024- or 2048-token prompt, and for E4B also the chat test; the 4k-context numbers
are from `llama bench`. The memory use is the drop of `MemAvailable` (10.0 GiB without llama.cpp) when the model is
loaded. All results are in `../benchmark/results/agx-frame*.jsonl`.

**Summary:**

- Everything runs on the GPU with Vulkan. The CPU generates 5-80 % faster with a short context (4 threads; the smaller the model, the bigger the difference), but
  it processes prompts slower, falls far behind with a longer context (E4B at 4k tokens: prompt processing
  10.7 vs 43 t/s, generation 7.6 vs 8.7 t/s), heats up to 95 °C within minutes and competes with the VR
  runtime for the CPU. The GPU stayed at ~41 °C.
- Podman works without drawbacks: the container (Ubuntu 26.04, Mesa 26.0.8) is as fast as a native build
  with SteamOS' own Mesa 26.3. Docker is not available on SteamOS, but rootless Podman 5.5 is preinstalled
  and keeps its images in `/home`, so the setup survives SteamOS upgrades.
- A local build (tuned for the Cortex-A720/X4, with or without KleidiAI) is not faster than the official
  build, which already selects the `armv8.6_2` CPU variant (dotprod, i8mm, fp16, SVE2) at runtime.
- Generation on the GPU is not limited by the memory bandwidth: the CPU generates faster from the same memory, and
  the GPU is about as fast with a 2.4 GiB model as with a 3.9 GiB one (~9 t/s). The Adreno 750 has no matrix
  cores or accelerated int8 dot products in Turnip, so llama.cpp uses its scalar shaders.

## Installation

```sh
mkdir -p ~/.config/containers/systemd
ln -s ~/Git/agx-ai/llama-cpp-frame/llama-cpp.container ~/.config/containers/systemd/
# Only if it does not exist yet (it is not in Git):
echo "LLAMA_API_KEY=$(openssl rand -base64 32)" > ~/Git/agx-ai/llama-cpp-frame/llama-cpp.env
systemctl --user daemon-reload
systemctl --user start llama-cpp
journalctl --user -u llama-cpp -f
```

The models are downloaded by llama-server into `~/.cache/huggingface/hub` on first use.
The unit starts with the user's systemd session (`WantedBy=default.target`); to start it at boot without a login,
enable lingering (`sudo loginctl enable-linger steamos`). Installing the unit is the only step that was not done
during the setup: the server was tested by running the same `podman run` command by hand.

Use `127.0.0.1` instead of `localhost` on the Frame itself: Podman's port forwarding (pasta) resets connections to
`[::1]:9931`, and `localhost` resolves to `::1` first. For example:

```sh
cd ../benchmark
./llama_cpp_bench_http.py --url http://127.0.0.1:9931 --env-file ../llama-cpp-frame/llama-cpp.env --no-embedding \
    --tests generation prompt chat --prompt-tokens 2048
./llama_cpp_bench_container.py --image ghcr.io/ggml-org/llama.cpp:full-vulkan \
    --model /root/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-qat-GGUF/snapshots/.../gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
    -- -ngl 99 -fa 1 -p 512 -n 128
```

`llama_cpp_bench_container.py` uses Podman automatically when Docker is not installed, and `--exe` runs a local
build instead of a container.

## Hardware and OS

| | |
|---|---|
| Computer | Valve Steam Frame (VR headset) |
| OS | SteamOS 0.4.5 "holo", VR variant (build 20261022.6150929), Linux 6.18.0, aarch64 |
| SoC | Qualcomm Snapdragon 8 Gen 3 (SM8650) |
| CPU | 8 cores, ARMv9.2-A: 1x Cortex-X4 @ 3.05 GHz (CPU 7), 3x Cortex-A720 @ 3.15 GHz (CPUs 2-4), 2x Cortex-A720 @ 2.96 GHz (CPUs 5-6), 2x Cortex-A520 @ 2.27 GHz (CPUs 0-1); 12 MiB L3 |
| CPU features | NEON, dotprod, fp16, i8mm, bf16, SVE, SVE2 (128-bit vector length); no SME |
| RAM | 16 GB (15.3 GiB visible to the OS), unified with the GPU |
| RAM type | LPDDR5X (the frequency is not visible from userspace; the SoC supports up to LPDDR5X-9600, 76.8 GB/s) |
| GPU | Qualcomm Adreno 750 (integrated), 231-903 MHz |
| GPU architecture | Adreno 7xx (A7xx), driver Mesa Turnip 26.3.0-devel (Vulkan 1.4) on the host, 26.0.8 in the container |
| VRAM | Unified with the RAM: Vulkan reports a single 14.34 GiB device-local heap (11.5 GiB in the container's Mesa) |
| NPU | Qualcomm Hexagon (CDSP), not usable by llama.cpp (see below) |
| Storage | `/home` 929 GB, root 10 GB |
| Swap | 7.6 GiB |

## Free memory

In desktop mode with Steam, the SteamVR services and Plasma running (no VR game):

| | Without llama.cpp | With E4B loaded (Vulkan, 32k context, vision/audio encoder) |
|---|---|---|
| `MemAvailable` | 10.0 GiB | 5.4 GiB |
| Vulkan heap budget | 9.38 GiB of 14.34 GiB | - |
| Swap | 7.6 GiB free | 7.3 GiB free |

The RAM and the "VRAM" are the same pool, so the ~10 GiB is shared by the model, the KV cache and everything else.
Only one model is loaded at a time (`models-max = 1`). Models of up to ~7 GiB (e.g. Gemma 4 12B at Q4) fit, but
then little is left for anything else; the 26B-A4B MoE (13.6 GiB at UD-Q4_K_XL) does not fit.
A VR game running at the same time would leave much less memory and GPU time.

## llama bench

`-fa 1`, `-ngl 99` for Vulkan, `-dev none -ngl 0` for the CPU, t/s:

| Model (size) | Vulkan pp512 | Vulkan tg128 | CPU 4 thr pp512 | CPU 4 thr tg128 | CPU 8 thr pp512 | CPU 8 thr tg128 |
|---|---|---|---|---|---|---|
| Gemma 4 E2B QAT UD-Q4_K_XL (2.4 GiB) | **119** | 14.1 | 50 | **25.6** | 67 | 19.0 |
| Gemma 4 E4B QAT UD-Q4_K_XL (3.9 GiB) | **58** | 9.1 | 34 | **13.3** | 46 | 10.5 |
| Gemma 4 E4B QAT Q4_0, Google (4.8 GiB) | **58** | 8.5 | 33 | **11.3** | 45 | 9.4 |
| Qwen3.5-4B Q4_0 (2.4 GiB) | **59** | 9.4 | 46 | **11.0** | 57 | 8.9 |
| Qwen3.5-9B Q4_0 (5.0 GiB) | 32 | 6.2 | 28 | **6.5** | **37** | 5.7 |
| Gemma 4 12B QAT UD-Q4_K_XL (6.2 GiB) | **21** (`-ub 128`) | 4.3 | 14 | **5.3** | 19 | 4.6 |

With a longer context (E4B, `-d 4096`), the CPU falls behind:

| E4B at 4096 tokens of context | pp512 | pp2048 | tg128 |
|---|---|---|---|
| Vulkan | **43** | **42** | **8.7** |
| CPU 8 threads | 10.7 | - | 7.6 |

- **Vulkan:** flash attention on or off makes no difference (E4B: pp512 57.8 vs 57.3, tg 9.1 vs 9.0).
  The generation speed barely depends on the model size (E4B 9.1, Qwen3.5-4B 9.4 t/s at 3.9 vs 2.4 GiB), so it is
  limited by the GPU's compute or per-kernel overhead rather than by the memory bandwidth.
- **GPU hang:** Gemma 4 12B with the default `ubatch-size` 512 crashed with `vk::DeviceLostError`, and the kernel
  reset the GPU (`a6xx_recover` in the kernel log). With `-ub 128`, it works. The kernel's hang check kills GPU
  jobs that run too long; that may also affect the VR compositor, so keep the ubatch size small for big models.
- **CPU threads:** generation is fastest with 4 threads and gets slower with more (the A520 cores and memory
  contention), prompt processing is fastest with all 8. llama-server can use both (`threads = 4`,
  `threads-batch = 8`). Pinning to the big cores (`-C 0xFC --cpu-strict 1`) did not help: E4B tg 13.7 t/s with
  4 threads, pp512 40 t/s with 6 threads.
- **Thermals:** after ~15 minutes of CPU benchmarks, the CPU reached 95 °C. The CPU numbers above are therefore
  partly throttled, and the repeat runs drifted by a few percent. The GPU stayed at ~41 °C under load.
- **Q4_0 repacking:** the QAT models are almost entirely Q4_0. On the CPU they are repacked at load time to
  `q4_0_4x8` for the dotprod/i8mm kernels, a 2.5 GiB copy for E4B in addition to the mmapped model.
- Google's own E4B QAT GGUF (`google/gemma-4-E4B-it-qat-q4_0-gguf`) is 0.9 GiB larger than Unsloth's and slightly
  slower, so Unsloth's UD-Q4_K_XL is used.

## llama-server

Measured with [llama_cpp_bench_http.py](../benchmark/llama_cpp_bench_http.py) (results in
`../benchmark/results/agx-frame.jsonl`), Gemma 4 E4B, 1 slot, 32k context:

| Configuration | Generation (256 tokens) | Prompt processing (2048 tokens) | Chat (6 prompts, up to 512 tokens) |
|---|---|---|---|
| Vulkan (preset.ini) | 9.0 t/s | 52.7 t/s | 8.9 t/s |
| Vulkan, MTP 2 (`spec-type = draft-mtp`) | 9.5 t/s (acceptance 56 %) | - | 9.5 t/s (8.7-11.4, acceptance 56 %) |
| CPU, `threads = 4`, `threads-batch = 8` | **12.8 t/s** | 26.1 t/s | **12.6 t/s** (12.5-12.8) |

MTP is only ~6 % faster on the GPU and costs memory, so it is not enabled.
The CPU is 40 % faster in short chats, but processes the 2048-token prompt at half the speed of the GPU (a 2k-token
prompt takes 78 s instead of 39 s), and with a longer context it falls behind in generation too (see above).
For short chats without a system prompt, the CPU configuration in the comment of [preset.ini](preset.ini) is faster.
While the CPU configuration was loaded, 2.6 GiB of swap were in use, since the repacked Q4_0 weights take another
2.5 GiB of RAM.

## Container vs. local build

To see whether the container or a local build loses performance, llama.cpp b11870 (the same commit as the image)
was built on the host with SteamOS' own GCC 15.1 and Mesa 26.3 (SPIRV-Headers had to be installed to `~/.local`
first, and the include path passed by hand):

```sh
cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/.local/spirv-headers \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+fp16 -DCMAKE_C_FLAGS=-mtune=cortex-a720 \
    -DCMAKE_CXX_FLAGS="-mtune=cortex-a720 -isystem $HOME/.local/spirv-headers/include" -DGGML_VULKAN=ON
```

| E4B UD-Q4_K_XL | Vulkan pp512 | Vulkan tg64 | CPU 4 thr pp512 | CPU 4 thr tg64 | CPU 6 thr pp512 | CPU 6 thr tg64 |
|---|---|---|---|---|---|---|
| Container (`full-vulkan`, Mesa 26.0.8, `armv8.6_2` CPU variant) | 57.8 | 9.1 | 30.5 | 13.7 | 40.4 | 13.3 |
| Local build (Mesa 26.3, `-march=armv9.2-a+fp16 -mtune=cortex-a720`) | 57.4 | 9.0 | 29.9 | 12.8 | 41.1 | 12.5 |
| Local build with KleidiAI (`-DGGML_CPU_KLEIDIAI=ON`) | - | - | 28.1 | 12.4 | 37.7 | 12.1 |

(CPU runs pinned to CPUs 2-7, `-C 0xFC --cpu-strict 1`.) The differences are within the run-to-run and thermal
variation, so there is no reason to build locally. GCC's `-march=native`/`-mcpu=native` would detect the CPU as
`cortex-a520` (a little core), so a local build should not use `GGML_NATIVE=ON`.
KleidiAI found I8MM kernels for Q4_0 and Q8_0 but none for F32 or Q4_K.

## Backends that do not work here

- **Hexagon NPU:** llama.cpp's Hexagon backend (and the `linux-arm64-snapdragon` release) needs Qualcomm's FastRPC
  (`/dev/fastrpc-*`, `libcdsprpc`). SteamOS has none of it. Instead, Valve's own driver (`dsp-sm8650`) exposes the
  CDSP as `/dev/cdsp` (group `cdsp`, of which the user is a member), and SteamVR uses it for eye tracking
  (`eyetracking -b CDSP`) and for `dsp_service`. That interface is undocumented, so the NPU is not usable for LLMs.
- **OpenCL (Adreno):** llama.cpp's Adreno OpenCL kernels need Qualcomm's proprietary OpenCL driver. SteamOS has only
  the OpenCL ICD loader, without any ICD (no Rusticl either).
- **vLLM:** CPU only, see [../vllm-frame](../vllm-frame).

## Vulkan properties

From `vulkaninfo` (Turnip on the host):

- fp16 arithmetic, 16-bit and 8-bit storage, int8: supported
- `VK_KHR_shader_integer_dot_product`: present, but `integerDotProduct8BitSignedAccelerated = false`
- `VK_KHR_cooperative_matrix`: not supported (llama.cpp: "matrix cores: none", "int dot: 0")
- Subgroup size 64-128, 32 KiB shared memory per workgroup
- `maxMemoryAllocationSize` = 4 GiB, `maxStorageBufferRange` = 128 MiB. This did not cause problems with any of the
  models above.
