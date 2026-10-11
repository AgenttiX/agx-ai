# vLLM v0.31.0 (CUDA 13.0) with the fixes of vllm.patch (see README.md):
# - Qwen3.5/3.8 passes its quantization config to the token embedding, so that checkpoints with a quantized embedding
#   load, e.g. JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB for a single 16 GB GPU,
# - the native CPU KV offloading is allowed with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
FROM vllm/vllm-openai:v0.31.0

COPY vllm.patch /tmp/vllm.patch
RUN patch -p1 -d /usr/local/lib/python3.12/dist-packages/vllm < /tmp/vllm.patch \
    && rm /tmp/vllm.patch
