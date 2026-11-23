# HyperQwen (https://github.com/syv-ai/HyperQwen, the image of its commit 3d6c954, vLLM 0.33.0) with image and video input: the
# vision tower of Qwen3.8-27B runs on the CPU (cpu_vision.py), and its inputs stay on the CPU
# (hyperqwen-vision.patch), so that it takes no VRAM. See README.md.
FROM ghcr.io/syv-ai/hyperqwen@sha256:68557954f5700e3bbdab37e0e310b83adc42e2cd20ddcd71e46b5690ca92f71e

COPY cpu_vision.py /app/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_vl_cpu_vision.py
COPY hyperqwen-vision.patch /tmp/
RUN patch -p1 -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/hyperqwen-vision.patch \
    && rm /tmp/hyperqwen-vision.patch
