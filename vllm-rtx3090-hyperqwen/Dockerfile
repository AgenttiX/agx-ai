# HyperQwen (https://github.com/syv-ai/HyperQwen, the image of its commit ce9615c) with image and video input: the
# vision tower of Qwen3.8-27B runs on the CPU (cpu_vision.py), and its inputs stay on the CPU
# (hyperqwen-vision.patch), so that it takes no VRAM. See README.md.
FROM ghcr.io/syv-ai/hyperqwen@sha256:0cf71b42d3017f9af0f7a46ef5817eedc5a39b6cc9c789ccd8a54138d9292ea0

COPY cpu_vision.py /app/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_vl_cpu_vision.py
COPY hyperqwen-vision.patch /tmp/
RUN patch -p1 -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/hyperqwen-vision.patch \
    && rm /tmp/hyperqwen-vision.patch
