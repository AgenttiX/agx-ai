# vLLM v0.31.0 (CUDA 13.0) with the fixes of vllm.patch (see README.md):
# - Qwen3.5/3.8 passes its quantization config to the token embedding, so that checkpoints with a quantized embedding
#   load, e.g. JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB for a single 16 GB GPU,
# - the native CPU KV offloading is allowed with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
# - Triton attention uses its split-KV kernel for speculative decoding too, and larger prefill blocks on sm_86,
# - UVA weight offloading works for MoE models with experts in RAM on small GPUs, e.g. Qwen3.8-Flash-Next on 8 GB,
# - native CPU KV offloading restores sliding-window models with an MTP drafter in the full-attention group (Gemma 4),
# - Qwen3.8-Flash-Next can run its vision tower on the CPU (QWEN38_VISION_DEVICE=cpu) for image and video input,
# - and use an FP8 KV cache below sm_89.
FROM vllm/vllm-openai:v0.31.0

COPY vllm.patch /tmp/vllm.patch
RUN patch -p1 -d /usr/local/lib/python3.12/dist-packages/vllm < /tmp/vllm.patch \
    && rm /tmp/vllm.patch
