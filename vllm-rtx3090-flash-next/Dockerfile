# The single-RTX-3090 runtime of Qwen3.8-Flash-Next (https://github.com/DominikBucko/qwen38-flash-next-3090, v0.1.0
# release image) with image and video input: the vision tower runs on the CPU (cpu_vision.py), its inputs stay on the
# CPU and videos get their own pixel budget (vllm-mm.patch), and the serve script enables image and video input and
# sets gpu-memory-utilization (flash-next.patch), and the CPU expert kernel has integer activation modes and less
# per-layer overhead (cpu-moe.patch). See README.md.
FROM ghcr.io/dominikbucko/qwen38-flash-next-3090@sha256:7f176605b59c462af21b1b62fd854f9f3cca96e3d11a295581297483d45490a3

COPY cpu_vision.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/cpu_vision.py
COPY flash-next.patch vllm-mm.patch cpu-moe.patch /tmp/
RUN patch -p1 -d / < /tmp/flash-next.patch && patch -p1 -d / < /tmp/vllm-mm.patch \
    && patch -p1 -d / < /tmp/cpu-moe.patch \
    && rm /tmp/flash-next.patch /tmp/vllm-mm.patch /tmp/cpu-moe.patch
