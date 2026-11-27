# HyperQwen (https://github.com/syv-ai/HyperQwen, the image of its commit 3d6c954, vLLM 0.33.0) with image and video input: the
# vision tower of Qwen3.8-27B runs on the Radeon VII (rvii_vision/server.py) or the CPU (cpu_vision.py), and its inputs
# stay on the CPU (hyperqwen-vision.patch), so that it takes no VRAM. Its pth-flash-attn kernel can also read vLLM's
# int4_per_token_head KV cache (pthfa-int4/), for twice the context: PTH_KV_DTYPE=int4_per_token_head (with PTH_FA=1)
# selects it in the launcher; unset, the KV cache stays int8_per_token_head. See README.md.
FROM ghcr.io/syv-ai/hyperqwen@sha256:371e1209cd3e717f47c1e94a47d7f20ca63133859c57964d4bd1b38278b7af7b

COPY cpu_vision.py /app/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_vl_cpu_vision.py
COPY hyperqwen-vision.patch /tmp/
RUN patch -p1 -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/hyperqwen-vision.patch \
    && rm /tmp/hyperqwen-vision.patch

# The int4 KV cache in pth-flash-attn (pthfa-int4/README.md). install.sh skips the build when its stamp matches the
# (unchanged) pinned sources, so the stamp is removed first; the int8 kernels are compiled unchanged.
COPY pthfa-int4/pthfa-int4.patch /tmp/
RUN patch -p1 -d pthfa < /tmp/pthfa-int4.patch && rm /tmp/pthfa-int4.patch \
    && rm -f venv/lib/python3.12/site-packages/pth_flash_attn/.pthfa-stamp \
    && PY=venv/bin/python bash pthfa/install.sh \
    && sed -i 's/--attention-backend CUSTOM --kv-cache-dtype int8_per_token_head"/--attention-backend CUSTOM --kv-cache-dtype ${PTH_KV_DTYPE:-int8_per_token_head}"/' \
        single-user/start_qwen.sh \
    && grep -q 'kv-cache-dtype ${PTH_KV_DTYPE:-int8_per_token_head}' single-user/start_qwen.sh
