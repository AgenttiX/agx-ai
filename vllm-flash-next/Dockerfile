# The single-RTX-3090 runtime of Qwen3.8-Flash-Next (https://github.com/DominikBucko/qwen38-flash-next-3090, v0.1.0
# release image) with image input: the vision tower runs on the CPU (cpu_vision.py), and the serve script enables
# image input and sets gpu-memory-utilization (flash-next.patch). See README.md.
FROM ghcr.io/dominikbucko/qwen38-flash-next-3090@sha256:7f176605b59c462af21b1b62fd854f9f3cca96e3d11a295581297483d45490a3

COPY cpu_vision.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/cpu_vision.py
COPY flash-next.patch /tmp/flash-next.patch
RUN patch -p1 -d / < /tmp/flash-next.patch && rm /tmp/flash-next.patch
