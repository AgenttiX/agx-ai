#!/bin/sh
# Start a test server of Qwen3.8-Flash-Next as in docker-compose.yml, with another image and/or settings, on port 9933
# (container fn-test), and wait until it answers. Stop the production container first (the VRAM is needed):
#
#   docker stop vllm-flash-next
#   ./test_server.sh IMAGE [KEY=VALUE ...]   # environment overrides, e.g. QWEN38_INT8_SKINNY=0
#   cd .. && uv run -m agx_ai.benchmark.bench_http --url http://localhost:9933 --env-file llama-cpp/llama-cpp.env \
#       --no-embedding --tests chat depth edit --repeat 2 --depth 6400 32000 --edit-depth 0 --label "..."
#   docker rm -f fn-test && docker start vllm-flash-next
#
# Optional environment of this script: FN_VOLUMES (space-separated extra volumes, e.g. "/tmp/trace:/trace" for
# QWEN38_DECODE_TRACE=/trace/x.bin), FN_CAP_ADD (space-separated capabilities, e.g. SYS_NICE), FN_TIMEOUT (seconds to
# wait, default 2400). The Radeon VII worker (rvii-experts) keeps running from the production compose file.
set -eu
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }
IMG=$1
shift
cd "$(dirname "$0")"
OVERRIDE=$(mktemp --suffix=.yml)
trap 'rm -f "$OVERRIDE"' EXIT
{
  echo "services:"
  echo "  vllm-flash-next:"
  echo "    image: $IMG"
  echo "    container_name: fn-test"
  echo "    restart: 'no'"
  echo "    ports: !override"
  echo "      - \"9933:8000\""
  if [ $# -gt 0 ]; then
    echo "    environment:"
    for kv in "$@"; do echo "      ${kv%%=*}: \"${kv#*=}\""; done
  fi
  if [ -n "${FN_VOLUMES:-}" ]; then
    echo "    volumes:"
    for v in $FN_VOLUMES; do echo "      - $v"; done
  fi
  if [ -n "${FN_CAP_ADD:-}" ]; then
    echo "    cap_add:"
    for c in $FN_CAP_ADD; do echo "      - $c"; done
  fi
} > "$OVERRIDE"
docker rm -f fn-test >/dev/null 2>&1 || true
docker compose -p vllm-rtx3090-flash-next -f docker-compose.yml -f "$OVERRIDE" up -d --no-deps vllm-flash-next 2>&1 | tail -1
t0=$(date +%s)
until curl -sf localhost:9933/health >/dev/null; do
  sleep 10
  if ! docker ps --format '{{.Names}}' | grep -qx fn-test; then
    echo "The container exited:"; docker logs fn-test 2>&1 | tail -20; exit 1
  fi
  if [ $(($(date +%s) - t0)) -gt "${FN_TIMEOUT:-2400}" ]; then echo "Timeout"; exit 1; fi
done
echo "Up after $(($(date +%s) - t0)) s"
nvidia-smi --query-compute-apps=used_memory --format=csv,noheader
docker logs fn-test 2>&1 | grep -E "GPU KV cache size|Radeon VII expert worker ready|cpu_moe: could not" | tail -3
