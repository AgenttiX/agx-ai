#!/bin/sh
# Starts a HyperQwen test server of an int4-hyperqwen image (Dockerfile of this directory) on port 9935 with the
# settings of ../docker-compose.yml (SPEC=dflash2 CTX=long PTH_FA=1 KV_MEM=5.95 GB, up to 4 sequences).
#
#   KV=int4 MAX_LEN=262128 sh serve.sh   # int4_per_token_head on pth-flash-attn
#   KV=int8 MAX_LEN=139264 sh serve.sh   # int8_per_token_head on pth-flash-attn (the current configuration)
#
# Optional: MORE_ARGS (appended to EXTRA_ARGS), IMAGE (default int4-hyperqwen:dev), KV_OFFLOAD_GB (CPU KV cache, default none), VISION=1 (the vision
# tower on the CPU, with the limits of ../docker-compose.yml; that file runs it on the Radeon VII, which takes no
# VRAM of the RTX 3090 either). The container is int4-hq; stop it with `docker rm -f int4-hq`.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
KV=${KV:-int4}
case "$KV" in
  int4) DTYPE=int4_per_token_head ;;
  int8) DTYPE=int8_per_token_head ;;
  *) echo "KV must be int4 or int8" >&2; exit 1 ;;
esac
EXTRA="--served-model-name Qwen/Qwen3.8-27B${MORE_ARGS:+ $MORE_ARGS}"
set --
if [ -n "${KV_OFFLOAD_GB:-}" ]; then
  set -- "$@" -e KV_OFFLOAD_GB="$KV_OFFLOAD_GB"
fi
if [ "${VISION:-0}" = 1 ]; then
  set -- "$@" -e VISION=1 -e VISION_OFFLOAD=0 -e QWEN38_VISION_DEVICE=cpu -e QWEN38_VISION_THREADS=32 \
    -e VLLM_VIDEO_LOADER_BACKEND=qwen3_vl
  EXTRA="$EXTRA --limit-mm-per-prompt {\"image\":64,\"video\":1}"
  EXTRA="$EXTRA --mm-processor-kwargs {\"max_pixels\":1048576,\"videos_kwargs\":{\"max_pixels\":12582912}}"
  EXTRA="$EXTRA --media-io-kwargs {\"video\":{\"max_frames\":64}} --skip-mm-profiling"
fi
docker rm -f int4-hq >/dev/null 2>&1 || true
docker run -d --name int4-hq --gpus all --shm-size 36g \
  -v /home/mika/.cache/vllm-models/hyperqwen:/app/models \
  -v int4-hq-cache:/cache \
  --env-file "$HERE/../../llama-cpp/llama-cpp.env" \
  -e HOME=/cache -e PORT=8000 -e PREPARE=0 \
  -e SPEC=dflash2 -e CTX=long -e PTH_FA=1 -e PTH_KV_DTYPE="$DTYPE" \
  -e KV_MEM="${KV_MEM:-5950000000}" -e MAX_LEN="${MAX_LEN:-139264}" \
  -e EMBED_UVA=1 -e GPU_UTIL="${GPU_UTIL:-0.85}" \
  -e EXTRA_ARGS="$EXTRA" "$@" \
  -p 9935:8000 \
  --entrypoint /bin/sh "${IMAGE:-int4-hyperqwen:dev}" \
  -c "sed -i '/untested on this backend; refusing. Use PTH_FA=0 for a CPU KV tier./{n;s/exit 1 ;;/;;/}' single-user/start_qwen.sh && VLLM_API_KEY=\"\$LLAMA_API_KEY\" exec bash docker/entrypoint.sh single"
echo "waiting for http://localhost:9935/health ..."
i=0
until curl -sf http://localhost:9935/health >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 240 ] || ! docker ps -q -f name=int4-hq | grep -q .; then
    echo "server did not come up" >&2
    docker logs --tail 60 int4-hq >&2 || true
    exit 1
  fi
  sleep 5
done
docker logs int4-hq 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | tail -2
