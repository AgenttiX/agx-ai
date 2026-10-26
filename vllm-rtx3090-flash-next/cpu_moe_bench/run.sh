#!/usr/bin/env bash
# Benchmark the CPU expert kernel of the image offline: the RAM read bandwidth, then the decode path of the kernel in
# its three modes (FP32, INT16 and INT8 activations) with 8, 16 and 24 threads, and the error of the integer modes.
# Needs g++ with C++20, ~8 GB of free RAM and the CPU pool's cores idle (stop the server first). See README.md.
# Usage: ./run.sh [IMAGE]   (default ghcr.io/agenttix/agx-ai/vllm-flash-next:latest)
set -euo pipefail
cd "$(dirname "$0")"
image=${1:-ghcr.io/agenttix/agx-ai/vllm-flash-next:latest}
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
src=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/cpu_moe.cpp
id=$(docker create "$image")
docker cp -q "$id:$src" "$build/cpu_moe.cpp"
docker rm "$id" >/dev/null
flags=(-std=c++20 -O3 -mavx2 -mfma -mf16c -pthread)
g++ "${flags[@]}" bw.cpp -o "$build/bw"
g++ "${flags[@]}" -DPF_ROWS=6 -Istub -DKERNEL="\"$build/cpu_moe.cpp\"" bench.cpp -o "$build/bench"
# The pool's CPUs as in the runtime: 1, 2 or 3 cores of each 4-core CCX of the Threadripper 3970X
p8=1,5,9,13,17,21,25,29
p16=1-2,5-6,9-10,13-14,17-18,21-22,25-26,29-30
p24=1-3,5-7,9-11,13-15,17-19,21-23,25-27,29-31
for cpus in 1 $p8 $p16 $p24; do "$build/bw" 8 "$cpus"; done
"$build/bench" $p16 4 7 20 dump "$build/ref.bin" >/dev/null
for mode in FP32 INT16 INT8; do
  for cpus in $p8 $p16 $p24; do
    echo -n "$mode: "
    env QWEN38_CPU_EXPERTS_INT16=$([ $mode = INT16 ] && echo 1 || echo 0) \
        QWEN38_CPU_EXPERTS_INT8=$([ $mode = INT8 ] && echo 1 || echo 0) \
        "$build/bench" "$cpus" 4 7 20 compare "$build/ref.bin" | tr '\n' ' '
    echo
  done
done
