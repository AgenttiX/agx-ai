# vLLM for gfx906 (Radeon VII, MI50, MI60) with the fixes of vllm-gfx906.patch (see README.md):
# - exllama GEMM in column chunks for very large matrices (the lm_head), whose scratch buffer would otherwise take
#   1.4-2.4 GiB of the 16 GB,
# - quantized embeddings in the Qwen3.5/3.8 MTP module,
# - the Triton attention backend, which crashed on a missing CommonAttentionMetadata._seq_lens_cpu,
# - the video loader for checkpoints without a video preprocessor config (Gemma 4), which failed on the
#   {backend: class name} mapping of transformers 5.19.
FROM aiinfos/vllm-gfx906-mobydick:v0.30.0.x-rocm7.14-pytorch2.13.0-260726fedb27

COPY vllm-gfx906.patch /tmp/vllm-gfx906.patch
RUN patch -p1 -d /opt/vllm-venv/lib/python3.12/site-packages/vllm < /tmp/vllm-gfx906.patch \
    && rm /tmp/vllm-gfx906.patch

# The vision towers of Gemma 4 and Qwen3.8 on the CPU (GEMMA4_VISION_DEVICE=cpu, QWEN38_VISION_DEVICE=cpu), so that
# images and videos need no VRAM for them, see README.md.
COPY cpu_onednn.py /opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/models/agx_cpu_onednn.py
COPY cpu_vision_gemma4.py /opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/models/gemma4_cpu_vision.py
COPY cpu_vision_qwen.py /opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_vl_cpu_vision.py
COPY cpu-vision.patch /tmp/cpu-vision.patch
RUN patch -p1 -d /opt/vllm-venv/lib/python3.12/site-packages/vllm < /tmp/cpu-vision.patch \
    && rm /tmp/cpu-vision.patch
