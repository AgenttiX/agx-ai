# vLLM on the Steam Frame

Feasibility study of vLLM on agx-frame, a Valve Steam Frame VR headset (2026-10-26).
Nothing has been installed or benchmarked yet. For the hardware, the OS and the free memory, see
[`../llama-cpp-frame/README.md`](../llama-cpp-frame/README.md#hardware-and-os).

| Model | Context | VRAM | RAM | Generation | Prompt processing | Notes |
|---|---|---|---|---|---|---|
| - | - | - | - | - | - | Not set up yet. [`../llama-cpp-frame`](../llama-cpp-frame) is expected to be clearly better on this hardware. |

**Summary:** vLLM can run here only on the CPU. There is an official prebuilt aarch64 CPU wheel that should install
into a Python virtual environment in `/home` without Docker, but vLLM has no backend for the Adreno GPU (no Vulkan
or OpenCL support) or the Hexagon NPU. With ~10 GiB of free unified memory, the overhead of PyTorch and vLLM, and
limited quantization support on Arm CPUs, it is useful only for small models and mostly for testing. llama.cpp
(GGUF quantization and Vulkan) will be a much better fit.

## Hardware in brief

- Snapdragon 8 Gen 3 (SM8650): 1x Cortex-X4, 5x Cortex-A720, 2x Cortex-A520; ARMv9.2-A with NEON, dotprod, fp16,
  bf16, i8mm, SVE, SVE2 (128-bit), no SME
- 16 GB LPDDR5X unified memory, ~10.0 GiB available in desktop mode (`MemAvailable`), 7.6 GiB swap
- Adreno 750 GPU (Mesa Turnip Vulkan only), Hexagon NPU not exposed
- SteamOS 0.4.5, glibc 2.39, Python 3.12.3 on the read-only root partition

## Existing builds

The vLLM v0.32.0 release (2026-10-18) contains these aarch64 builds:

| Build | Useful here? |
|---|---|
| `vllm-0.32.0+cpu-cp38-abi3-manylinux_2_39_aarch64.whl` (GitHub release) | **Yes**, CPU backend. Requires glibc 2.39, which SteamOS has. |
| `vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl` (PyPI) | No, CUDA (for NVIDIA Grace/Jetson) |
| `vllm-0.32.0+cu129-...aarch64.whl` | No, CUDA |
| Docker image (`docker/Dockerfile.cpu`, `linux/arm64`) | Docker is not installed and the root partition is reset on upgrades. Rootless Podman 5.5.2 is preinstalled with its storage in `/home`, so the CPU image could be run with Podman if needed. |

The CPU wheel can be installed with `uv` into a virtual environment in `/home` (with the CPU-only PyTorch for
aarch64), so it survives SteamOS upgrades. The system Python is on the root partition, so `uv` should manage its own
Python interpreter instead of relying on it.

### GPU and NPU

- **GPU:** vLLM supports CUDA, ROCm, Intel XPU, TPU and a few accelerator plugins, but not Vulkan or OpenCL. PyTorch
  has no usable Vulkan backend for this either. There is no way to use the Adreno 750 with vLLM.
- **NPU:** the Hexagon NPU is not exposed by the kernel (no FastRPC devices, no remoteproc), and vLLM has no Hexagon
  backend anyway.

## Would a local build help?

Probably only slightly, and it is expensive. In `cmake/cpu_extension.cmake`, the Arm CPU backend detects the
features from `/proc/cpuinfo` of the build machine and compiles with at most
`-march=armv8.2-a+bf16+dotprod+fp16+i8mm`. The release wheel is built on Arm build servers (`arm64_cpu_queue_release`,
with `VLLM_BUILD_ACL=ON`) that most likely support BF16 and I8MM, so the wheel should already use the same flags
that a local build would get. (If the wheel lacks them, `VLLM_CPU_ARM_BF16=1` and `VLLM_CPU_ARM_I8MM=1` force them
on in a local build.) Neither vLLM's own kernels nor the bundled Arm Compute Library (`ARM_COMPUTE_ARCH=armv8.2-a`)
are compiled for SVE/SVE2, and a local build would not change that without patching the build scripts.
The heavy lifting is done by PyTorch, oneDNN and the Arm Compute Library, which have their own runtime dispatch.

A local build would also take a long time on this device (the C++ extension, oneDNN and ACL) and needs ~10 GB of
disk and a lot of RAM. The compilers (GCC 15.1, Clang 19.1, CMake 3.29) are preinstalled, but on the root partition.
Unless the prebuilt wheel turns out not to work, a local build is not worth it.

## Expected limitations

- **Memory:** PyTorch and vLLM take a couple of GiB on their own, so with ~10 GiB available only models of a few
  billion parameters fit in BF16 (e.g. 1-4B), with a modest KV cache (`VLLM_CPU_KVCACHE_SPACE`).
- **Quantization:** most of vLLM's quantized kernels (AWQ/GPTQ Marlin, FP8, etc.) are GPU-only. On Arm CPUs the
  options are limited, so llama.cpp's GGUF quantizations make much better use of the memory.
- **Speed:** token generation is bound by the ~77 GB/s memory bandwidth like llama.cpp, so BF16 weights make it
  roughly 4 times slower than Q4 GGUF models of the same size.
- **Threads:** the CPU backend binds OpenMP threads to cores (`VLLM_CPU_OMP_THREADS_BIND`). It should be limited to
  the X4 and A720 cores (CPUs 2-7), since the A520 cores would slow the whole step down.

## Plan

1. Create a virtual environment in `/home` with `uv` and install the CPU wheel from the GitHub release.
2. Run a small BF16 model (e.g. ~1B parameters) with `VLLM_CPU_OMP_THREADS_BIND=2-7` and benchmark it with the
   scripts in `../benchmark/`.
3. Compare with llama.cpp on the same model.
