# vLLM for gfx906 (Radeon VII, MI50, MI60) with the fixes of vllm-gfx906.patch (see README.md):
# - exllama GEMM in column chunks for very large matrices (the lm_head), whose scratch buffer would otherwise take
#   1.4-2.4 GiB of the 16 GB,
# - quantized embeddings in the Qwen3.5/3.8 MTP module,
# - the Triton attention backend, which crashed on a missing CommonAttentionMetadata._seq_lens_cpu.
FROM aiinfos/vllm-gfx906-mobydick:v0.30.0.x-rocm7.14-pytorch2.13.0-260726fedb27

COPY vllm-gfx906.patch /tmp/vllm-gfx906.patch
RUN patch -p1 -d /opt/vllm-venv/lib/python3.12/site-packages/vllm < /tmp/vllm-gfx906.patch \
    && rm /tmp/vllm-gfx906.patch
