# llama.cpp on the Steam Frame

Feasibility study of llama.cpp on agx-frame, a Valve Steam Frame VR headset (2026-10-26).
Nothing has been installed or benchmarked yet.

| Model | Context | VRAM | RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|---|
| - | - | - | - | - | - | Not set up yet. See the plan below. |

**Summary:** llama.cpp is feasible. The official prebuilt `ubuntu-vulkan-arm64` release should run as is
(same glibc 2.39 as Ubuntu 24.04, all the libraries it needs are in SteamOS) from a directory in `/home`, so neither
Docker nor any change to the read-only root partition is needed. The GPU is usable only through Vulkan (Mesa Turnip).
The Hexagon NPU and the Adreno OpenCL backend are not usable on this OS. A local build is cheap
(all the build tools are preinstalled) and may give a small CPU speedup, but is not required.

## Hardware and OS

| | |
|---|---|
| Computer | Valve Steam Frame (VR headset) |
| OS | SteamOS 0.4.5 "holo", VR variant (build 20261022.6150929), Linux 6.18.0, aarch64 |
| SoC | Qualcomm Snapdragon 8 Gen 3 (SM8650) |
| CPU | 8 cores, ARMv9.2-A: 1x Cortex-X4 @ 3.05 GHz, 3x Cortex-A720 @ 3.15 GHz, 2x Cortex-A720 @ 2.96 GHz, 2x Cortex-A520 @ 2.27 GHz; 12 MiB L3 |
| CPU features | NEON, dotprod, fp16, i8mm, bf16, SVE, SVE2 (128-bit vector length); no SME |
| RAM | 16 GB (15.3 GiB visible to the OS), unified with the GPU |
| RAM type | LPDDR5X (the frequency is not visible from userspace; the SoC supports up to LPDDR5X-9600, 76.8 GB/s) |
| GPU | Qualcomm Adreno 750 (integrated), up to 903 MHz |
| GPU architecture | Adreno 7xx (A7xx), driver Mesa Turnip 26.3.0-devel (Vulkan 1.4) |
| VRAM | Unified with the RAM: Vulkan reports a single 14.34 GiB device-local heap |
| NPU | Qualcomm Hexagon (in the SoC, but not exposed: no `/dev/fastrpc*`, no remoteproc) |
| Storage | `/home` 929 GB (762 GB free), root 10 GB |
| Swap | 7.6 GiB |

## Free memory

Measured in desktop mode with Steam, SteamVR services and Plasma running (no VR game):

| | |
|---|---|
| `MemTotal` | 15.3 GiB |
| `MemAvailable` | 10.0 GiB (5.9 GiB free, 5.8 GiB cache) |
| Vulkan heap budget | 9.38 GiB of 14.34 GiB |
| Swap | 7.6 GiB free |

Since the memory is unified, the RAM and the "VRAM" are the same ~10 GiB pool, and a model on the GPU uses it just like
a model on the CPU. Realistically ~8 GiB is usable for the model and KV cache, e.g. a 7-9B model at Q4 or a
12-14B model at Q3/IQ3. Running a VR game or SteamVR at the same time will leave a lot less.
The memory bandwidth (at most ~77 GB/s, shared with the display pipeline) limits token generation to roughly
15-20 t/s for a 4 GB model, and a MoE model with few active parameters (e.g. Gemma 4 26B-A4B or Qwen3 30B-A3B at a
small quant) would be much faster but does not fit in ~8 GiB unless quantized very aggressively.

## Existing builds

The llama.cpp GitHub releases (checked with b11889) contain these Linux aarch64 builds:

| Build | Backends | Useful here? |
|---|---|---|
| `llama-bXXXX-bin-ubuntu-arm64.tar.gz` | CPU | Yes, CPU only |
| `llama-bXXXX-bin-ubuntu-vulkan-arm64.tar.gz` | CPU + Vulkan | **Yes, the best candidate** |
| `llama-bXXXX-bin-linux-arm64-snapdragon.tar.gz` | CPU + Hexagon NPU + OpenCL (Adreno) | No, see below |
| `llama-bXXXX-bin-ubuntu-cuda-13.4-arm64.tar.gz` | CUDA | No (NVIDIA only) |

The Ubuntu builds are built on Ubuntu 24.04 (glibc 2.39), which matches SteamOS (glibc 2.39). SteamOS already has
`libvulkan.so.1`, `libgomp.so.1`, `libssl.so.3` and `libstdc++.so.6`, so the archive can just be extracted to
`/home` and run. The official Docker images are not an option since Docker is not installed and the root
partition is reset on upgrades. (Rootless Podman 5.5.2 is preinstalled with its storage in `/home`, so containers
would be possible, but they are not needed for llama.cpp.)

### Vulkan

The GPU is supported by Mesa's Turnip Vulkan driver, which is what SteamOS uses for games. Relevant properties
reported by `vulkaninfo`:

- fp16 arithmetic, 16-bit and 8-bit storage, int8: supported
- `VK_KHR_shader_integer_dot_product`: present, but `integerDotProduct8BitSignedAccelerated = false`
- `VK_KHR_cooperative_matrix`: **not supported**, so llama.cpp falls back to its scalar matrix multiplication
  shaders. Prompt processing will be much slower than on GPUs with matrix cores.
- Subgroup size 64-128 (with `subgroupSizeControl`), 32 KiB shared memory per workgroup
- `maxMemoryAllocationSize` = 4 GiB, `maxStorageBufferRange` = **128 MiB**. llama.cpp splits the model into
  allocations, but single tensors larger than 128 MiB (e.g. the token embedding or output tensor of a model with a
  large vocabulary) may fail or fall back to the CPU. This has to be tested.

The Adreno 750 is roughly comparable to a low-end discrete GPU (on the order of 4 TFLOPS FP32 and twice that in FP16,
theoretical, at the 903 MHz limit seen here). Since token generation is memory-bound and the memory is shared, Vulkan mainly helps prompt
processing and keeps the CPU cores free; token generation may be similar to the CPU. Both need benchmarking.

### Snapdragon build (Hexagon NPU + Adreno OpenCL)

The `linux-arm64-snapdragon` build targets Qualcomm's own Linux images and is not usable here:

- The Hexagon backend needs FastRPC (`/dev/fastrpc-*`) and a booted CDSP. The kernel has `CONFIG_QCOM_FASTRPC=y`,
  but there are no FastRPC devices and no remoteproc instances, so the DSP is not running and the NPU is unavailable.
  Enabling it would require Valve to ship the DSP firmware and device tree support.
- The OpenCL backend with Adreno kernels is written for Qualcomm's proprietary OpenCL driver and its extensions.
  SteamOS has only the OpenCL ICD loader (`libOpenCL.so.1`) without any ICD (no `/etc/OpenCL/vendors`, no Rusticl).
- Its CPU code is compiled for only `armv8.2a+fp16+dotprod`.

## Would a local build help?

Partially. The official arm64 builds use `GGML_CPU_ALL_VARIANTS=ON` with `GGML_BACKEND_DL=ON`, which builds these
Linux CPU variants and selects the best one at runtime:

| Variant | Features | Runs here |
|---|---|---|
| `armv8.0_1` ... `armv8.2_3` | up to dotprod, fp16, SVE | yes |
| `armv8.6_1` | dotprod, fp16, SVE, i8mm | yes |
| `armv8.6_2` | dotprod, fp16, SVE, i8mm, SVE2 | **yes, would be selected** |
| `armv9.2_1`, `armv9.2_2` | ... + SME | no (no SME) |

So the prebuilt CPU backend already uses all the important instructions of this CPU (dotprod, i8mm, fp16, SVE2).
A local build could still add:

- **Tuning** for the big cores (`-mcpu=cortex-x4` or `-mcpu=cortex-a720` via `GGML_CPU_ARM_ARCH` / `CMAKE_C_FLAGS`).
  Note: `-march=native` / `GGML_NATIVE=ON` is not a good choice here, since GCC detects the heterogeneous CPU as
  `cortex-a520` (the little core) and tunes for that. The instruction sets of all three core types are identical.
- **BF16** (`+bf16`), which none of the Linux variants enables. It matters only for BF16 models.
- **KleidiAI** (`-DGGML_CPU_KLEIDIAI=ON`), Arm's optimized i8mm/dotprod micro-kernels, which are not enabled in the
  release builds. They speed up prompt processing of Q4_0 and Q8_0 models on the CPU.
- A newer Vulkan shader build against the system's Vulkan headers 1.4.309 (no practical difference expected).

The expected gain is small (single-digit percent for token generation, possibly more for prompt processing with
KleidiAI), but building is easy: GCC 15.1, Clang 19.1, CMake 3.29, Ninja, `glslc` and the Vulkan headers are all
preinstalled in SteamOS. The build output can live in `/home`, so it survives SteamOS upgrades.
However, the preinstalled compilers and headers are on the root partition and may change or disappear on upgrades,
so the prebuilt release is the more robust default.

## Other considerations

- **Thread placement:** with 1+5+2 heterogeneous cores, using the A520 cores tends to slow llama.cpp down. Start with
  `-t 6` pinned to the X4 and A720 cores (CPUs 2-7, e.g. with `taskset -c 2-7`) and compare with other values.
- **Thermals and power:** the headset is designed for VR sessions and runs on battery.
  Sustained inference may be throttled; benchmarks should be run long enough to see the steady state.
- **Serving:** `llama-server` can run as a systemd user service from `/home`, which also survives SteamOS upgrades.

## Plan

1. Download the latest `ubuntu-vulkan-arm64` release to `/home/steamos/llama.cpp/` and check `llama-cli --list-devices`.
2. Benchmark a small model (e.g. a 4B model at Q4_0 and Q4_K_M) with the scripts in `../benchmark/`
   on the CPU only (`-ngl 0`) and on Vulkan (`-ngl 99`), with different thread counts.
3. Test whether larger models hit the 128 MiB `maxStorageBufferRange` limit.
4. If the CPU path wins, try a local build with `-mcpu=cortex-x4+bf16` and KleidiAI.
