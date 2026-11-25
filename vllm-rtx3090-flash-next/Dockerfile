# The single-RTX-3090 runtime of Qwen3.8-Flash-Next (https://github.com/DominikBucko/qwen38-flash-next-3090, v0.1.0
# release image) with image and video input: the vision tower runs on the CPU (cpu_vision.py), its inputs stay on the
# CPU and videos get their own pixel budget (vllm-mm.patch), and the serve script enables image and video input and
# sets gpu-memory-utilization (flash-next.patch), and the CPU expert kernel has integer activation modes and less
# per-layer overhead (cpu-moe.patch), and can hand some of the cold experts to a Radeon VII worker (rvii-moe.patch,
# rvii_experts/), and has a faster int8 kernel and a prefetch of the next layer's likely experts (cpu-moe-opt.patch);
# the sampler filters MTP's 4 rows with vLLM's Triton top-k/top-p and the lm_head can be int8 (flash-next.patch), and
# the Radeon VII worker can compute some of its experts in prefill (rvii_prefill.py) and re-rank its experts from
# their use (rvii-prefill.patch: publishes the hot experts and starts with the worker's ranking).
# See README.md.
FROM ghcr.io/dominikbucko/qwen38-flash-next-3090@sha256:7f176605b59c462af21b1b62fd854f9f3cca96e3d11a295581297483d45490a3

COPY cpu_vision.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/cpu_vision.py
# Optional refusal-direction removal (QWEN38_ABLITERATION, README.md): the module and the shipped directions
COPY abliteration.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/abliteration.py
COPY abliteration /opt/qwen38/abliteration
COPY rvii_experts/rvii_protocol.h rvii_prefill.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/
COPY flash-next.patch vllm-mm.patch cpu-moe.patch rvii-moe.patch cpu-moe-opt.patch rvii-prefill.patch /tmp/
RUN patch -p1 -d / < /tmp/flash-next.patch && patch -p1 -d / < /tmp/vllm-mm.patch \
    && patch -p1 -d / < /tmp/cpu-moe.patch && patch -p1 -d / < /tmp/rvii-moe.patch \
    && patch -p1 -d / < /tmp/cpu-moe-opt.patch && patch -p1 -d / < /tmp/rvii-prefill.patch \
    && rm /tmp/flash-next.patch /tmp/vllm-mm.patch /tmp/cpu-moe.patch /tmp/rvii-moe.patch /tmp/cpu-moe-opt.patch \
       /tmp/rvii-prefill.patch
