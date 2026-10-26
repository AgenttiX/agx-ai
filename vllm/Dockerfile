# vLLM v0.31.0 (CUDA 13.0) with the fixes of vllm.patch (see README.md):
# - Qwen3.5/3.8 passes its quantization config to the token embedding, so that checkpoints with a quantized embedding
#   load, e.g. JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB for a single 16 GB GPU,
# - the native CPU KV offloading is allowed with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
# - Triton attention uses its split-KV kernel for speculative decoding too, and larger prefill blocks on sm_86,
# - UVA weight offloading works for MoE models with experts in RAM on small GPUs, e.g. Qwen3.8-Flash-Next on 8 GB,
# - native CPU KV offloading restores sliding-window models with an MTP drafter in the full-attention group (Gemma 4),
# - Qwen3.8-Flash-Next can run its vision tower on the CPU (QWEN38_VISION_DEVICE=cpu) for image and video input,
# - and use an FP8 KV cache below sm_89.
# It also includes the Gemma 4 checkpoint scripts (repack_gemma4_qat.py, quantize_gemma4_ct_embedding.py) as commands,
# and on x86-64 the pth-flash-attn kernel (an attention backend plugin, --attention-backend CUSTOM) for int8 KV caches
# on sm_80/86/89.
FROM vllm/vllm-openai:v0.31.0

COPY vllm.patch /tmp/vllm.patch
RUN patch -p1 -d /usr/local/lib/python3.12/dist-packages/vllm < /tmp/vllm.patch \
    && rm /tmp/vllm.patch
# pth-flash-attn v0.1.0 (https://github.com/noon-at-cgn/pth-flash-attn), the commit vendored by HyperQwen. The CUDA
# build takes ~1 min with 4 jobs. Not on ARM64: it would be compiled under QEMU, and it has no kernels for the
# GPUs of the ARM64 machines (sm_90+).
ARG TARGETARCH
ARG PTH_FLASH_ATTN_COMMIT=b0d4f7ee228448ecc3d85a2015db208ed8d21954
RUN if [ "$TARGETARCH" = amd64 ]; then \
        TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" MAX_JOBS=4 pip install --no-cache-dir --no-deps --no-build-isolation \
            "https://github.com/noon-at-cgn/pth-flash-attn/archive/${PTH_FLASH_ATTN_COMMIT}.tar.gz" \
        && python3 -c "import pth_flash_attn.ops"; \
    fi
# quantize_gemma4_ct_embedding.py imports repack_gemma4_qat from its own directory.
COPY --chmod=755 repack_gemma4_qat.py quantize_gemma4_ct_embedding.py /usr/local/bin/
