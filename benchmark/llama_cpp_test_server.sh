#!/usr/bin/env bash
# Start a temporary llama-server router from any image with a given preset, for A/B tests of images, backends
# (ROCm vs Vulkan) or preset variants without touching the production container. Waits until the load-on-startup
# model has loaded, then prints the GPU memory use. Benchmark it with llama_cpp_bench_http.py --url.
#
# Usage: ./llama_cpp_test_server.sh IMAGE PRESET [CONFIG] [ENV_FILE] [PORT]
#   CONFIG    server config.ini (default ../llama-cpp-agx-ai/config.ini)
#   ENV_FILE  env file with LLAMA_API_KEY (default: none)
#   PORT      host port (default 9933)
# Extra environment for the container: DOCKER_ARGS="-e GGML_VK_VISIBLE_DEVICES=0 ..."
# Stop the production container first if the GPU memory is needed, and remove this one afterwards:
#   docker rm -f llama-test
#
# Example (Vulkan on the Radeon VII with a preset variant):
#   ./llama_cpp_test_server.sh ghcr.io/ggml-org/llama.cpp:full-vulkan /tmp/preset-variant.ini \
#       ../llama-cpp-agx-ai/config.ini ../llama-cpp-radeon-vii/llama-cpp.env
#   ./llama_cpp_bench_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \
#       --no-embedding --tests chat depth --repeat 2 --depth 6400 16000 36500 --label "Vulkan"
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
image=$1
preset=$(realpath "$2")
config=$(realpath "${3:-$here/../llama-cpp-agx-ai/config.ini}")
env_file=${4:-}
port=${5:-9933}

args=(--name llama-test -p "$port:9931" --device /dev/dri --group-add video
      -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub"
      -v "$config:/etc/llama.cpp/config.ini:ro" -v "$preset:/etc/llama.cpp/preset.ini:ro")
[[ -e /dev/kfd ]] && args+=(--device /dev/kfd)  # ROCm
[[ -n $env_file ]] && args+=(--env-file "$(realpath "$env_file")")
# shellcheck disable=SC2206
args+=(${DOCKER_ARGS:-})

docker rm -f llama-test >/dev/null 2>&1 || true
docker run -d "${args[@]}" --entrypoint /app/llama-server "$image" >/dev/null
for _ in $(seq 1 100); do
    sleep 3
    if docker logs llama-test 2>&1 | grep -q 'llama_server: model loaded'; then
        echo "loaded on port $port"
        if command -v amd-smi >/dev/null; then amd-smi metric -m | grep -E '^GPU|USED_VRAM:' | grep -v VISIBLE; fi
        if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; fi
        exit 0
    fi
    docker ps -q -f name=llama-test | grep -q . || break
done
echo "FAILED to load:" >&2
docker logs llama-test 2>&1 | grep -iE 'error|fail|out of memory|exception' | tail -5 >&2
exit 1
