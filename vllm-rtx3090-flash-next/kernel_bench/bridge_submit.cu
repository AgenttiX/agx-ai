// Microbenchmark of the cold-expert bridge's submit kernel (bridge_cuda.cu of the runtime image): how fast the RTX 3090
// writes a decode request (4 tokens x 2560 FP32 = 40 KB) into pinned host memory and publishes it, in CUDA graphs.
// Variants: the runtime's kernel (one block, 4-byte stores), 16- and 8-byte loads with float4 stores, all loads issued
// first, several blocks with a ticket (the last one publishes), and a conversion kernel + copy-engine memcpy + publish.
// Result on agx-z2e (2026-11-24): ~10-11 us for every variant except the strided float4 stores (35-67 us), i.e. the
// write to host memory, not the kernel, sets the time. See README.md.
//
//   docker run --rm --gpus all -v "$PWD/kernel_bench":/w -w /w --entrypoint sh \
//       ghcr.io/agenttix/agx-ai/vllm-flash-next:latest -c "nvcc -O3 -arch=sm_86 -std=c++17 bridge_submit.cu -o /tmp/b && /tmp/b"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
constexpr int H = 2560, TOPK = 10, MAXT = 4;

__global__ void submit_old(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                           const float* __restrict__ w, int T, char* req, long long* counter, int layer, int bump,
                           const signed char* __restrict__ mask) {
  __shared__ long long seq;
  if (threadIdx.x == 0) {
    long long c = *counter;
    if (bump) { c += 1; *counter = c; }
    seq = c * 64 + layer;
    *reinterpret_cast<int*>(req + 8) = T;
    *reinterpret_cast<int*>(req + 12) = layer;
  }
  int* ido = reinterpret_cast<int*>(req + 64);
  float* wo = reinterpret_cast<float*>(req + 224);
  for (int i = threadIdx.x; i < MAXT * TOPK; i += blockDim.x) {
    const bool v = i < T * TOPK && !(mask && mask[i]);
    ido[i] = v ? ids[i] : -1;
    wo[i] = v ? w[i] : 0.f;
  }
  float* xo = reinterpret_cast<float*>(req + 448);
  for (int i = threadIdx.x; i < T * H; i += blockDim.x) xo[i] = __bfloat162float(x[i]);
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(seq) : "memory");
}

// New: 8 BF16 per 16-byte load, all of a thread's loads issued before its stores, 2 float4 stores per load.
template <int THREADS>
__global__ void __launch_bounds__(THREADS) submit_new(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                           const float* __restrict__ w, int T, char* req, long long* counter, int layer, int bump,
                           const signed char* __restrict__ mask) {
  __shared__ long long seq;
  if (threadIdx.x == 0) {
    long long c = *counter;
    if (bump) { c += 1; *counter = c; }
    seq = c * 64 + layer;
    *reinterpret_cast<int*>(req + 8) = T;
    *reinterpret_cast<int*>(req + 12) = layer;
  }
  if (threadIdx.x < MAXT * TOPK) {
    const int i = threadIdx.x;
    const bool v = i < T * TOPK && !(mask && mask[i]);
    reinterpret_cast<int*>(req + 64)[i] = v ? ids[i] : -1;
    reinterpret_cast<float*>(req + 224)[i] = v ? w[i] : 0.f;
  }
  constexpr int PER = (MAXT * H / 8 + THREADS - 1) / THREADS;
  const int n8 = T * H / 8;
  const uint4* x8 = reinterpret_cast<const uint4*>(x);
  float4* xo = reinterpret_cast<float4*>(req + 448);
  uint4 v[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = threadIdx.x + k * THREADS;
    if (i < n8) v[k] = __ldg(x8 + i);
  }
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = threadIdx.x + k * THREADS;
    if (i < n8) {
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[k]);
      const float2 a0 = __bfloat1622float2(b[0]), a1 = __bfloat1622float2(b[1]);
      const float2 a2 = __bfloat1622float2(b[2]), a3 = __bfloat1622float2(b[3]);
      xo[2 * i] = make_float4(a0.x, a0.y, a1.x, a1.y);
      xo[2 * i + 1] = make_float4(a2.x, a2.y, a3.x, a3.y);
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(seq) : "memory");
}

// New 2: 4 BF16 per 8-byte load, one float4 store per load at consecutive addresses within a warp (512 B per
// warp store, as the 4-byte stores of the old kernel give 128 B), all loads of a thread issued before its stores.
template <int THREADS>
__global__ void __launch_bounds__(THREADS) submit_new2(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                           const float* __restrict__ w, int T, char* req, long long* counter, int layer, int bump,
                           const signed char* __restrict__ mask) {
  __shared__ long long seq;
  if (threadIdx.x == 0) {
    long long c = *counter;
    if (bump) { c += 1; *counter = c; }
    seq = c * 64 + layer;
    *reinterpret_cast<int*>(req + 8) = T;
    *reinterpret_cast<int*>(req + 12) = layer;
  }
  if (threadIdx.x < MAXT * TOPK) {
    const int i = threadIdx.x;
    const bool v = i < T * TOPK && !(mask && mask[i]);
    reinterpret_cast<int*>(req + 64)[i] = v ? ids[i] : -1;
    reinterpret_cast<float*>(req + 224)[i] = v ? w[i] : 0.f;
  }
  constexpr int PER = (MAXT * H / 4 + THREADS - 1) / THREADS;
  const int n4 = T * H / 4;
  const uint2* x4 = reinterpret_cast<const uint2*>(x);
  float4* xo = reinterpret_cast<float4*>(req + 448);
  uint2 v[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = threadIdx.x + k * THREADS;
    if (i < n4) v[k] = __ldg(x4 + i);
  }
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = threadIdx.x + k * THREADS;
    if (i < n4) {
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[k]);
      const float2 a0 = __bfloat1622float2(b[0]), a1 = __bfloat1622float2(b[1]);
      xo[i] = make_float4(a0.x, a0.y, a1.x, a1.y);
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(seq) : "memory");
}

// Old with unrolled loads only (4-byte stores as before).
template <int THREADS>
__global__ void __launch_bounds__(THREADS) submit_old_unr(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                           const float* __restrict__ w, int T, char* req, long long* counter, int layer, int bump,
                           const signed char* __restrict__ mask) {
  __shared__ long long seq;
  if (threadIdx.x == 0) {
    long long c = *counter;
    if (bump) { c += 1; *counter = c; }
    seq = c * 64 + layer;
    *reinterpret_cast<int*>(req + 8) = T;
    *reinterpret_cast<int*>(req + 12) = layer;
  }
  if (threadIdx.x < MAXT * TOPK) {
    const int i = threadIdx.x;
    const bool v = i < T * TOPK && !(mask && mask[i]);
    reinterpret_cast<int*>(req + 64)[i] = v ? ids[i] : -1;
    reinterpret_cast<float*>(req + 224)[i] = v ? w[i] : 0.f;
  }
  constexpr int PER = (MAXT * H + THREADS - 1) / THREADS;
  const int n = T * H;
  float* xo = reinterpret_cast<float*>(req + 448);
  __nv_bfloat16 v[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k) { const int i = threadIdx.x + k * THREADS; if (i < n) v[k] = x[i]; }
#pragma unroll
  for (int k = 0; k < PER; ++k) { const int i = threadIdx.x + k * THREADS; if (i < n) xo[i] = __bfloat162float(v[k]); }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(seq) : "memory");
}

// Multi-block: each block converts and writes a slice; the last block to finish (a ticket in device memory, reset by
// it) publishes the seq after a system fence. ticket must start at 0.
__global__ void submit_mb(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                          const float* __restrict__ w, int T, char* req, long long* counter, int layer, int bump,
                          const signed char* __restrict__ mask, unsigned* ticket) {
  if (blockIdx.x == 0 && threadIdx.x < MAXT * TOPK) {
    const int i = threadIdx.x;
    const bool v = i < T * TOPK && !(mask && mask[i]);
    reinterpret_cast<int*>(req + 64)[i] = v ? ids[i] : -1;
    reinterpret_cast<float*>(req + 224)[i] = v ? w[i] : 0.f;
  }
  const int n4 = T * H / 4;
  const uint2* x4 = reinterpret_cast<const uint2*>(x);
  float4* xo = reinterpret_cast<float4*>(req + 448);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    const uint2 v = __ldg(x4 + i);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
    const float2 a0 = __bfloat1622float2(b[0]), a1 = __bfloat1622float2(b[1]);
    xo[i] = make_float4(a0.x, a0.y, a1.x, a1.y);
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(ticket, 1u) == gridDim.x - 1) {
      __threadfence();
      long long c = *reinterpret_cast<volatile long long*>(counter);
      if (bump) { c += 1; *counter = c; }
      *reinterpret_cast<int*>(req + 8) = T;
      *reinterpret_cast<int*>(req + 12) = layer;
      __threadfence_system();
      *ticket = 0;
      asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(c * 64 + layer) : "memory");
    }
  }
}

__global__ void convert_dev(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ w,
                            int T, char* stage, const signed char* __restrict__ mask) {
  if (threadIdx.x < MAXT * TOPK && blockIdx.x == 0) {
    const int i = threadIdx.x;
    const bool v = i < T * TOPK && !(mask && mask[i]);
    reinterpret_cast<int*>(stage + 64)[i] = v ? ids[i] : -1;
    reinterpret_cast<float*>(stage + 224)[i] = v ? w[i] : 0.f;
  }
  const int n4 = T * H / 4;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(x) + i);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
    const float2 a0 = __bfloat1622float2(b[0]), a1 = __bfloat1622float2(b[1]);
    reinterpret_cast<float4*>(stage + 448)[i] = make_float4(a0.x, a0.y, a1.x, a1.y);
  }
}
__global__ void publish_k(char* req, long long* counter, int T, int layer, int bump) {
  long long c = *counter;
  if (bump) { c += 1; *counter = c; }
  *reinterpret_cast<int*>(req + 8) = T;
  *reinterpret_cast<int*>(req + 12) = layer;
  __threadfence_system();
  asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(c * 64 + layer) : "memory");
}

int main() {
  const int T = 4;
  __nv_bfloat16* x; int* ids; float* w; long long* counter; char* req; char* reqd;
  cudaMalloc(&x, MAXT * H * 2); cudaMalloc(&ids, 160); cudaMalloc(&w, 160); cudaMalloc(&counter, 8);
  cudaMemset(counter, 0, 8);
  __nv_bfloat16 hx[MAXT * H];
  for (int i = 0; i < MAXT * H; ++i) hx[i] = __float2bfloat16((i % 97) * 0.01f - 0.3f);
  cudaMemcpy(x, hx, sizeof hx, cudaMemcpyHostToDevice);
  cudaHostAlloc(&req, 1 << 20, cudaHostAllocMapped);
  cudaHostGetDevicePointer((void**)&reqd, req, 0);
  cudaStream_t s; cudaStreamCreate(&s);
  auto bench = [&](const char* name, auto launch) {
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 100; ++i) launch();
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int r = 0; r < 20; ++r) cudaGraphLaunch(ge, s);
    cudaEventRecord(b, s); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    float xs[MAXT * H]; memcpy(xs, req + 448, sizeof xs);
    double err = 0; for (int i = 0; i < T * H; ++i) err += fabs(xs[i] - __bfloat162float(hx[i]));
    printf("%-14s %6.2f us per submit (err %g, %s)\n", name, ms * 1000 / 2000, err, cudaGetErrorString(cudaGetLastError()));
  };
  bench("old 256", [&] { submit_old<<<1, 256, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  unsigned* ticket; cudaMalloc(&ticket, 4); cudaMemset(ticket, 0, 4);
  for (int nb : {2, 4, 8, 10, 16, 20, 40})
    for (int nt : {128, 256}) {
      char nm[32]; snprintf(nm, 32, "mb %dx%d", nb, nt);
      bench(nm, [&] { submit_mb<<<nb, nt, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr, ticket); });
    }
  char* stage; cudaMalloc(&stage, 65536);
  bench("ce copy", [&] {
    convert_dev<<<10, 256, 0, s>>>(x, ids, w, T, stage, nullptr);
    cudaMemcpyAsync(req + 64, stage + 64, 448 - 64 + T * H * 4, cudaMemcpyDeviceToHost, s);
    publish_k<<<1, 1, 0, s>>>(reqd, counter, T, 3, 1);
  });
  bench("new2 128", [&] { submit_new2<128><<<1, 128, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("new2 256", [&] { submit_new2<256><<<1, 256, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("new2 512", [&] { submit_new2<512><<<1, 512, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("oldunr 256", [&] { submit_old_unr<256><<<1, 256, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("oldunr 1024", [&] { submit_old_unr<1024><<<1, 1024, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("new 256", [&] { submit_new<256><<<1, 256, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("new 512", [&] { submit_new<512><<<1, 512, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("new 1024", [&] { submit_new<1024><<<1, 1024, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  bench("old 256", [&] { submit_old<<<1, 256, 0, s>>>(x, ids, w, T, reqd, counter, 3, 1, nullptr); });
  return 0;
}
