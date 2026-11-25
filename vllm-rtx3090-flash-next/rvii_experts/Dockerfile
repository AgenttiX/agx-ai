# The Radeon VII expert worker of the Qwen3.8-Flash-Next runtime, see README.md.
# The vLLM gfx906 image (../../vllm-radeon-vii) has ROCm with hipcc, and Python with numpy for prepare.py.
FROM ghcr.io/agenttix/agx-ai/vllm-gfx906:latest
WORKDIR /opt/rvii
COPY rvii_protocol.h rvii_worker.hip prepare.py bench_rvii.py bench_prefill.py rerank.py rankings.json \
     rankings-measured.json entrypoint.sh ./
COPY usage/general.bin usage/
RUN hipcc -O3 --offload-arch=gfx906 -std=c++17 -o /usr/local/bin/rvii_worker rvii_worker.hip && \
    chmod +x entrypoint.sh
ENTRYPOINT ["/opt/rvii/entrypoint.sh"]
