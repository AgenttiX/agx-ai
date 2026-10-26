#!/usr/bin/env bash
# Start a temporary vLLM server from any image with a given vllm serve --config YAML file, for A/B tests of images
# or configuration variants without touching the production container (the vLLM counterpart of
# llama_cpp_test_server.sh). Waits until the server answers, then prints the GPU memory use and the KV cache size.
# Benchmark it with llama_cpp_bench_http.py --url, which detects vLLM.
#
# Usage: ./vllm_test_server.sh IMAGE CONFIG [ENV_FILE] [PORT] [-- EXTRA VLLM ARGS...]
#   CONFIG    YAML file for vllm serve --config
#   ENV_FILE  env file with LLAMA_API_KEY, used as the API key (default: none)
#   PORT      host port (default 9933)
# The Hugging Face cache, /home/$USER/.cache/vllm-models (as /models) and the compile caches (the volume
# CACHE_VOLUME, default that of ../vllm-radeon-vii) are mounted like in the compose files.
# Extra docker arguments: DOCKER_ARGS="-e VAR=..."
# On NVIDIA (nvidia-smi exists), the GPUs are given with --gpus "device=$GPUS" (default: all), e.g. GPUS=1.
# GPUS=none gives no GPUs, for the CPU image of vLLM. GPUS=amd gives the AMD GPUs even when nvidia-smi exists (a
# machine with both, e.g. agx-z2e with the Radeon VII and the RTX 3090).
# Container name: NAME=... (default vllm-test). Stop the production container first, and remove this one afterwards:
#   docker rm -f vllm-test
#
# Example:
#   ./vllm_test_server.sh aiinfos/vllm-gfx906-mobydick:v0.30.0.x-rocm7.14-pytorch2.13.0-260726fedb27 \
#       /tmp/config-variant.yaml ../llama-cpp/llama-cpp.env -- --max-model-len 32768
#   ./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp/llama-cpp.env \
#       --no-embedding --tests chat depth --repeat 2 --depth 6400 16000 --server-config /tmp/config-variant.yaml
set -euo pipefail

image=$1
config=$(realpath "$2")
shift 2
env_file=
port=9933
if [[ $# -gt 0 && $1 != -- ]]; then env_file=$1; shift; fi
if [[ $# -gt 0 && $1 != -- ]]; then port=$1; shift; fi
[[ $# -gt 0 && $1 == -- ]] && shift
name=${NAME:-vllm-test}

args=(--name "$name" -p "$port:8000" --ipc host
      -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub"
      -v "$HOME/.cache/vllm-models:/models:ro"
      -v "${CACHE_VOLUME:-vllm-radeon-vii_vllm-cache}:/root/.cache/vllm"
      -e HF_HUB_OFFLINE=1 -e TRITON_CACHE_DIR=/root/.cache/vllm/triton
      -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/inductor
      -v "$config:/etc/vllm/config.yaml:ro")
if [[ ${GPUS:-} == none ]]; then
    :  # a CPU image: with the GPUs (and libcuda), Triton would select its CUDA driver instead of the CPU one
elif [[ ${GPUS:-} != amd ]] && command -v nvidia-smi >/dev/null; then
    args+=(--gpus "\"device=${GPUS:-all}\"")
else
    args+=(--device /dev/dri --group-add video)
    [[ -e /dev/kfd ]] && args+=(--device /dev/kfd)
fi
key_arg=
if [[ -n $env_file ]]; then
    args+=(--env-file "$(realpath "$env_file")")
    key_arg='--api-key "$LLAMA_API_KEY"'
fi
# shellcheck disable=SC2206
args+=(${DOCKER_ARGS:-})

docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d "${args[@]}" --entrypoint /bin/sh "$image" \
    -c "exec vllm serve --config /etc/vllm/config.yaml $key_arg \"\$@\"" vllm "$@" >/dev/null
start=$(date +%s)
while sleep 5; do
    if curl -sf "http://localhost:$port/health" >/dev/null; then
        echo "ready on port $port after $(( $(date +%s) - start )) s"
        docker logs "$name" 2>&1 | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Graph capturing finished' || true
        if command -v amd-smi >/dev/null; then amd-smi metric -m | grep -E '^GPU|USED_VRAM:' | grep -v VISIBLE; fi
        if command -v nvidia-smi >/dev/null; then
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
        fi
        exit 0
    fi
    docker ps -q -f "name=^$name\$" | grep -q . || break
    (( $(date +%s) - start < 1800 )) || break
done
echo "FAILED to start:" >&2
docker logs "$name" 2>&1 | grep -iE 'error|out of memory|exception' | tail -8 >&2
exit 1
