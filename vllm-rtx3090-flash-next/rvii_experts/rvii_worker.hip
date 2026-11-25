// Radeon VII expert worker for the Qwen3.8-Flash-Next runtime: keeps some of the cold experts in the Radeon VII's
// VRAM and computes them on request of the CPU expert pool, over the shared-memory protocol of rvii_protocol.h.
// See README.md.
//
// Weights: prepare.py repacks them so that every wavefront reads contiguous 1 KiB per load instruction. Per expert
// (EXPERT_BYTES, all sections multiples of 256 B):
//   gate/up qweight: [cb 10][g 20][s 8][lane 64] x 16 B = {gate r, up r, gate r+1, up r+1} (int32 each) of column
//     c = 64 cb + lane, r = 16 g + 2 s (packed rows of 8 k, low nibble first)
//   gate/up scales:  [cb 10][g 20][lane 64] x {gate, up} fp16
//   down qweight:    [cb 40][g 5][s 4][lane 64] x 16 B = rows r..r+3 of column c = 64 cb + lane, r = 16 g + 4 s
//   down scales:     [cb 40][g 5][lane 64] fp16
// Math: per quantization group g (128 k, zero point 8), y = s_g * sum_k (q_k - 8) x_k. The activations (x, and
// SiLU(gate) * up) are quantized per 64 values to 24-bit integers (as exact as FP32 relative to the block's range,
// see quant64), so that the products with the int4 weights are exact integer dot products (v_dot8_u32_u4: 8 products
// per instruction, without converting the weights); the rest is FP32.
//
// A request is one kernel launch (k_req) whose workgroups have three roles, in the order of dispatch:
//   copy:    x (64 k per workgroup, a token per wave) from the shared memory (zero-copy, host-registered) to planes;
//   gate/up: + SiLU, 64 intermediate columns x 4 groups per workgroup (50 per job);
//   down:    64 hidden columns per workgroup, whose waves take (job, group) units (40 S workgroups).
// Each role waits (spinning on flags in VRAM, set to the request's generation number) only for the data that it
// needs, after it has issued the loads of its weights: so the weight loads overlap with the earlier roles' work. The
// waits cannot deadlock, as the workgroups that they wait for have lower indices and are dispatched first. The last
// down workgroups write y straight into the shared memory, and the very last one writes the response seq: the worker
// thread does not wait for the kernel.
#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "rvii_protocol.h"

#define CK(x)                                                                                    \
  do {                                                                                           \
    hipError_t e_ = (x);                                                                         \
    if (e_ != hipSuccess) {                                                                      \
      fprintf(stderr, "rvii: %s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));    \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

namespace {

constexpr int H = RVII_H, I = RVII_I, MAXT = RVII_MAXT, MAXJ = RVII_MAXJ;
constexpr size_t GU_W = size_t(10) * 20 * 8 * 64 * 16;  // 1638400
constexpr size_t GU_S = size_t(10) * 20 * 64 * 4;       // 51200
constexpr size_t DN_W = size_t(40) * 5 * 4 * 64 * 16;   // 819200
constexpr size_t DN_S = size_t(40) * 5 * 64 * 2;        // 25600
constexpr size_t EXPERT_BYTES = GU_W + GU_S + DN_W + DN_S;  // 2534400
constexpr int KA = 5;     // gate/up workgroups per job and 64 columns (4 groups each)
constexpr int MAXS = 64;  // max down workgroups per 64 columns

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));  // a 16-B load

struct Job {
  int slot, n;
  int tok[MAXT];
  float wt[MAXT];
};

// Kernel arguments (passed by value, read with scalar loads).
// Data that one workgroup writes and another reads in the same launch (the planes and infos) is in its own 64-B
// lines per flag, so that no (L1 or scalar) cache line can hold it before its flag is set.
struct Args {
  const uint8_t* w;   // expert slot 0
  const float* xh;    // x in the shared memory (device pointer of the host-registered mapping)
  float* yh;          // y in the shared memory
  int64_t* seqh;      // the response seq in the shared memory
  uint32_t* xq;       // [MAXT][40][8][8] x in 24-bit planes, per 64 k (see quant64)
  float* xi;          // [MAXT][40][16] their infos
  uint32_t* aq;       // [MAXJ][MAXT][10][8][8] SiLU(gate) * up of each job's tokens, in planes
  float* ai;          // [MAXJ][MAXT][10][16]
  float* gpart;       // [MAXJ][10][KA][2 MAXT][64] partial sums of the gate/up workgroups
  float* ypart;       // [MAXS][MAXT][H] partial outputs of the down workgroups
  uint32_t* xflag;    // [40] = gen when the planes of x of block hb (64 k) are written
  uint32_t* aflag;    // [MAXJ][10] = gen when the planes of act of job j, columns 64 cb.. are written
  int* cnt_gu;        // [MAXJ][10] arrival counters of the gate/up workgroups (0 between requests)
  int* cnt_dn;        // [40] of the down workgroups
  int* cnt_y;         // [1] of the down workgroups that wrote their part of y
  int64_t seq;        // the request's seq
  uint32_t gen;       // the request's generation number (1, 2, ...)
  unsigned long long* dbg;  // RVII_TRACE builds: [8 events][min, max] of s_memrealtime (per-phase timing)
  int T, J, S;
  Job jobs[MAXJ];
};
#ifdef RVII_TRACE
__device__ __forceinline__ void trace(const Args& a, int e) {
  if (threadIdx.x == 0) {
    const unsigned long long t = __builtin_amdgcn_s_memrealtime();
    atomicMin(a.dbg + 2 * e, t);
    atomicMax(a.dbg + 2 * e + 1, t);
  }
}
#define TR(e) trace(a, e)
#else
#define TR(e)
#endif

__device__ __forceinline__ int uniform(int v) { return __builtin_amdgcn_readfirstlane(v); }
__device__ __forceinline__ uint32_t dot8(uint32_t a, uint32_t b, uint32_t c) {
  return __builtin_amdgcn_udot8(a, b, c, false);  // v_dot8_u32_u4: c + the sum of the 8 nibble products
}
constexpr uint32_t ONES = 0x11111111u;

// Each wave stages the planes and infos of its 64-k block pair for all tokens of its job in LDS (st, written with
// vector loads after the flag), and reads them back as uniform ds_read_b128 broadcasts for v_dot8.
constexpr int STW = 136;  // dwords per token: 128 of planes (2 blocks), 2 x 4 of infos
__device__ __forceinline__ void stage(uint32_t* st, const uint32_t* q, const float* info, int lane) {
  if (lane < 32) reinterpret_cast<u32x4*>(st)[lane] = reinterpret_cast<const u32x4*>(q)[lane];
  else if (lane < 34) reinterpret_cast<u32x4*>(st + 128)[lane - 32] = reinterpret_cast<const u32x4*>(info)[4 * (lane - 32)];
}

// The activations as integers, for v_dot8_u32_u4 with the int4 weights: a block of 64 values (one per lane of the
// wave) is quantized to X = round((v - min) / xs) with 24 bits (xs = (max - min) / (2^24 - 1)), which is as exact
// as FP32 relative to the block's range. X is stored in 6 planes of 4 bits: qp[r][p] packs nibble p of X of the 8
// values 8 r .. 8 r + 7, in the weights' nibble order. With the weights q of a column:
//   sum_k q_k v_k = xs sum_p 16^p sum_k q_k X_k^(p) + min sum_k q_k,   info = {xs, min, sum_k v_k}.
// Each sum_k q_k X_k^(p) (<= 64 * 225) is exact in the dot8's int32.
template <int C> __device__ __forceinline__ uint32_t dpp(uint32_t v) {
  return __builtin_amdgcn_mov_dpp(int(v), C, 0xF, 0xF, false);
}
template <int C> __device__ __forceinline__ float dpp(float v) { return __uint_as_float(dpp<C>(__float_as_uint(v))); }
__device__ __forceinline__ float rl(float v, int l) { return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l)); }
constexpr int QP_1032 = 0xB1, QP_2301 = 0x4E, ROW_HALF_MIRROR = 0x141, ROW_MIRROR = 0x140;
__device__ __forceinline__ void quant64(float v, uint32_t* qp, float* info) {
  const int lane = threadIdx.x & 63;
  // min, max and sum over the wave: within rows of 16 lanes with DPP, then over the 4 rows (uniform values)
  float mn = v, mx = v, sm = v;
#define RED_STEP(C) mn = fminf(mn, dpp<C>(mn)); mx = fmaxf(mx, dpp<C>(mx)); sm += dpp<C>(sm);
  RED_STEP(QP_1032) RED_STEP(QP_2301) RED_STEP(ROW_HALF_MIRROR) RED_STEP(ROW_MIRROR)
#undef RED_STEP
  mn = fminf(fminf(rl(mn, 0), rl(mn, 16)), fminf(rl(mn, 32), rl(mn, 48)));
  mx = fmaxf(fmaxf(rl(mx, 0), rl(mx, 16)), fmaxf(rl(mx, 32), rl(mx, 48)));
  sm = (rl(sm, 0) + rl(sm, 16)) + (rl(sm, 32) + rl(sm, 48));
  const float xs = (mx - mn) * (1.f / 16777215.f), inv = xs > 0.f ? 1.f / xs : 0.f;
  const uint32_t X = min(uint32_t(rintf((v - mn) * inv)), 0xFFFFFFu);
#pragma unroll
  for (int p = 0; p < 6; ++p) {  // OR over each 8 lanes (both quads of a half row)
    uint32_t c = ((X >> (4 * p)) & 15u) << (4 * (lane & 7));
    c |= dpp<QP_1032>(c);
    c |= dpp<QP_2301>(c);
    c |= dpp<ROW_HALF_MIRROR>(c);
    if ((lane & 7) == p) qp[(lane >> 3) * 8 + p] = c;
  }
  if (lane == 0) { info[0] = xs; info[1] = mn; info[2] = sm; }
}
// sum_p 16^p d[p]
__device__ __forceinline__ float planes(const uint32_t* d) {
  float r = float(d[5]);
#pragma unroll
  for (int p = 4; p >= 0; --p) r = fmaf(r, 16.f, float(d[p]));
  return r;
}
// sum_k (q_k - 8) v_k of a 64-k block from the planes' dot products d, the weights' sum qs and the block's info.
__device__ __forceinline__ float block_dot(const uint32_t* d, uint32_t qs, const uint32_t* info) {
  const u32x4 in = *reinterpret_cast<const u32x4*>(info);
  return fmaf(__uint_as_float(in.x), planes(d), fmaf(__uint_as_float(in.y), float(qs), -8.f * __uint_as_float(in.z)));
}

// The last of n workgroups that call this (after writing their partial results) gets true, and then sees the
// others' results (read them with ld_agent). The counter is reset for the next request.
__device__ __forceinline__ bool last_block(int* cnt, int n) {
  __shared__ int old;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    old = atomicAdd(cnt, 1);
    if (old == n - 1) *cnt = 0;
  }
  __syncthreads();
  __threadfence();
  return old == n - 1;
}
__device__ __forceinline__ float ld_agent(const float* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
// Set a flag after the whole workgroup's writes.
__device__ __forceinline__ void set_flag(uint32_t* f, uint32_t gen) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) __hip_atomic_store(f, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
// Wait (the calling wave) for a flag. The acquire invalidates L1.
__device__ __forceinline__ void wait_flag(const uint32_t* f, uint32_t gen) {
  while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != gen) __builtin_amdgcn_s_sleep(2);
}

// Copy role: 64-k block hb of x, token w (wave w), to planes.
__device__ void copy_x(const Args& a, int hb) {
  const int t = threadIdx.x >> 6, lane = threadIdx.x & 63;
  if (t < a.T) quant64(a.xh[t * H + hb * 64 + lane], a.xq + (t * 40 + hb) * 64, a.xi + (t * 40 + hb) * 16);
  set_flag(a.xflag + hb, a.gen);
  TR(1);
}

// Gate/up role: job j, columns c = 64 cb + lane, group g = 4 ks + w for wave w. The 4 waves' sums are added in
// LDS, the KA workgroups' in gpart by the last of them, which applies SiLU.
template <int N>
__device__ __forceinline__ void gate_up(const Args& a, const Job& jb, int cb, int g, float (*red)[2 * MAXT][64],
                                        uint32_t* st) {
  const int w = threadIdx.x >> 6, lane = threadIdx.x & 63;
  const uint8_t* ew = a.w + size_t(jb.slot) * EXPERT_BYTES;
  const u32x4* qw = reinterpret_cast<const u32x4*>(ew) + ((cb * 20 + g) * 8) * 64 + lane;
  u32x4 q[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) q[s] = __builtin_nontemporal_load(qw + s * 64);
  const uint32_t s2 = reinterpret_cast<const uint32_t*>(ew + GU_W)[(cb * 20 + g) * 64 + lane];
  wait_flag(a.xflag + 2 * g, a.gen);
  wait_flag(a.xflag + 2 * g + 1, a.gen);
  TR(2);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int b0 = jb.tok[i] * 40 + 2 * g;
    stage(st + i * STW, a.xq + b0 * 64, a.xi + b0 * 16, lane);
  }
  float ag[N], au[N];
#pragma unroll
  for (int i = 0; i < N; ++i) ag[i] = au[i] = 0.f;
#pragma unroll
  for (int h = 0; h < 2; ++h) {  // the two 64-k blocks of the group: steps 4 h .. 4 h + 3
    uint32_t qg = 0, qu = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const u32x4 v = q[4 * h + s];
      qg = dot8(v.z, ONES, dot8(v.x, ONES, qg));
      qu = dot8(v.w, ONES, dot8(v.y, ONES, qu));
    }
#pragma unroll 1  // (unrolled, the compiler hoists all LDS reads and runs out of VGPRs)
    for (int i = 0; i < N; ++i) {
      const uint32_t* xs = st + i * STW + h * 64;
      uint32_t dg[6] = {0, 0, 0, 0, 0, 0}, du[6] = {0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const u32x4 v = q[4 * h + s];  // rows 2 s, 2 s + 1 of the block
        const u32x4* xr = reinterpret_cast<const u32x4*>(xs + 16 * s);
        const u32x4 a0 = xr[0], a1 = xr[1], b0 = xr[2], b1 = xr[3];
        const uint32_t pa[6] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y}, pb[6] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y};
#pragma unroll
        for (int p = 0; p < 6; ++p) {
          dg[p] = dot8(v.z, pb[p], dot8(v.x, pa[p], dg[p]));
          du[p] = dot8(v.w, pb[p], dot8(v.y, pa[p], du[p]));
        }
      }
      ag[i] += block_dot(dg, qg, st + i * STW + 128 + 4 * h);
      au[i] += block_dot(du, qu, st + i * STW + 128 + 4 * h);
    }
  }
  const float sg = float(__builtin_bit_cast(_Float16, uint16_t(s2))),
              su = float(__builtin_bit_cast(_Float16, uint16_t(s2 >> 16)));
#pragma unroll
  for (int i = 0; i < N; ++i) {
    red[w][2 * i][lane] = sg * ag[i];
    red[w][2 * i + 1][lane] = su * au[i];
  }
}

__device__ void gate_up_role(const Args& a, int j, int cb, int ks, uint32_t* lds) {
  float (*red)[2 * MAXT][64] = reinterpret_cast<float (*)[2 * MAXT][64]>(lds);
  const int w = uniform(threadIdx.x >> 6), lane = threadIdx.x & 63;
  uint32_t* st = lds + 4 * 2 * MAXT * 64 + w * MAXT * STW;
  const Job& jb = a.jobs[j];
  switch (jb.n) {
    case 1: gate_up<1>(a, jb, cb, ks * 4 + w, red, st); break;
    case 2: gate_up<2>(a, jb, cb, ks * 4 + w, red, st); break;
    case 3: gate_up<3>(a, jb, cb, ks * 4 + w, red, st); break;
    default: gate_up<4>(a, jb, cb, ks * 4 + w, red, st); break;
  }
  TR(3);
  __syncthreads();
  // Wave w: rows r = 2 i + {0: gate, 1: up} of the workgroup's sum to gpart[j][cb][ks][r][lane].
  float* gp = a.gpart + size_t(j * 10 + cb) * KA * 2 * MAXT * 64;
  for (int r = w; r < 2 * jb.n; r += 4)
    gp[(ks * 2 * MAXT + r) * 64 + lane] = red[0][r][lane] + red[1][r][lane] + red[2][r][lane] + red[3][r][lane];
  if (!last_block(a.cnt_gu + j * 10 + cb, KA)) return;
  // The last workgroup: wave i < n computes token i's SiLU(gate) * up, to planes.
  const int i = w;
  if (i < jb.n) {
    float gs = 0.f, us = 0.f;
#pragma unroll
    for (int k = 0; k < KA; ++k) {
      gs += ld_agent(gp + (k * 2 * MAXT + 2 * i) * 64 + lane);
      us += ld_agent(gp + (k * 2 * MAXT + 2 * i + 1) * 64 + lane);
    }
    const int blk = (j * MAXT + i) * 10 + cb;
    quant64(gs / (1.f + __expf(-gs)) * us, a.aq + blk * 64, a.ai + blk * 16);
  }
  set_flag(a.aflag + j * 10 + cb, a.gen);
  TR(4);
}

// Down role: columns c = 64 cb + lane; wave w of workgroup sp takes the (job, group) units u = 4 sp + w + 4 S m.
// Each unit adds wt * the expert's output to the job's token rows. The 4 waves' rows are added in LDS, the S
// workgroups' in ypart by the last of them, which writes y into the shared memory.
template <int N>
__device__ __forceinline__ void down_unit(const Args& a, int j, int g, const u32x4* q, float sd, float* y,
                                          uint32_t* st) {
  const Job& jb = a.jobs[j];
  const int lane = threadIdx.x & 63;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int b0 = (j * MAXT + i) * 10 + 2 * g;
    stage(st + i * STW, a.aq + b0 * 64, a.ai + b0 * 16, lane);
  }
  float val[N];
#pragma unroll
  for (int i = 0; i < N; ++i) val[i] = 0.f;
#pragma unroll
  for (int h = 0; h < 2; ++h) {  // the two 64-k blocks of the group: steps 2 h, 2 h + 1
    uint32_t qs = 0;
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
      for (int r = 0; r < 4; ++r) qs = dot8(q[2 * h + s][r], ONES, qs);
#pragma unroll 1
    for (int i = 0; i < N; ++i) {
      const uint32_t* as = st + i * STW + h * 64;
      uint32_t d[6] = {0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const u32x4* ar = reinterpret_cast<const u32x4*>(as + 8 * (4 * s + r));
          const u32x4 a0 = ar[0], a1 = ar[1];
          const uint32_t pa[6] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y};
#pragma unroll
          for (int p = 0; p < 6; ++p) d[p] = dot8(q[2 * h + s][r], pa[p], d[p]);
        }
      val[i] += block_dot(d, qs, st + i * STW + 128 + 4 * h);
    }
  }
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float v = jb.wt[i] * sd * val[i];
    // tok is wave-uniform: a uniform branch per row keeps y in registers.
#pragma unroll
    for (int t = 0; t < MAXT; ++t)
      if (jb.tok[i] == t) y[t] += v;
  }
}

__device__ void down_role(const Args& a, int cb, int sp, uint32_t* lds) {
  float (*red)[MAXT][64] = reinterpret_cast<float (*)[MAXT][64]>(lds);
  const int w = uniform(threadIdx.x >> 6), lane = threadIdx.x & 63;
  uint32_t* st = lds + 4 * 2 * MAXT * 64 + w * MAXT * STW;
  const int U = a.J * 5;
  float y[MAXT] = {0.f, 0.f, 0.f, 0.f};
  // Software pipeline: the next unit's weights are loaded before the current one is computed.
  int u = 4 * sp + w;
  u32x4 q[4], qn[4];
  float sd = 0.f, sdn = 0.f;
  auto load = [&](int uu, u32x4* qq, float& ss) {
    const int j = uu / 5, g = uu % 5;
    const uint8_t* ew = a.w + size_t(a.jobs[j].slot) * EXPERT_BYTES;
    const u32x4* p = reinterpret_cast<const u32x4*>(ew + GU_W + GU_S) + ((cb * 5 + g) * 4) * 64 + lane;
#pragma unroll
    for (int s = 0; s < 4; ++s) qq[s] = __builtin_nontemporal_load(p + s * 64);
    ss = float(reinterpret_cast<const _Float16*>(ew + GU_W + GU_S + DN_W)[(cb * 5 + g) * 64 + lane]);
  };
  if (u < U) load(u, q, sd);
  while (u < U) {
    const int un = u + 4 * a.S;
    if (un < U) load(un, qn, sdn);
    const int j = u / 5, g = u % 5;
    wait_flag(a.aflag + j * 10 + 2 * g, a.gen);
    wait_flag(a.aflag + j * 10 + 2 * g + 1, a.gen);
    TR(5);
    switch (a.jobs[j].n) {
      case 1: down_unit<1>(a, j, g, q, sd, y, st); break;
      case 2: down_unit<2>(a, j, g, q, sd, y, st); break;
      case 3: down_unit<3>(a, j, g, q, sd, y, st); break;
      default: down_unit<4>(a, j, g, q, sd, y, st); break;
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) q[s] = qn[s];
    sd = sdn;
    u = un;
  }
  TR(6);
#pragma unroll
  for (int t = 0; t < MAXT; ++t) red[w][t][lane] = y[t];
  __syncthreads();
  const int t = w;
  if (t < a.T)
    a.ypart[(sp * MAXT + t) * H + cb * 64 + lane] = red[0][t][lane] + red[1][t][lane] + red[2][t][lane] + red[3][t][lane];
  if (!last_block(a.cnt_dn + cb, a.S)) return;
  if (t < a.T) {
    float v = 0.f;
    for (int p = 0; p < a.S; ++p) v += ld_agent(a.ypart + (p * MAXT + t) * H + cb * 64 + lane);
    a.yh[t * H + cb * 64 + lane] = v;
  }
  // The last of the 40 column blocks answers: y is written (over PCIe, in order) before the seq.
  __threadfence_system();
  if (last_block(a.cnt_y, 40) && threadIdx.x == 0) {
    __threadfence_system();
    __hip_atomic_store(a.seqh, a.seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    TR(7);
  }
}

__global__ void __launch_bounds__(256) k_req(const Args a) {
  // LDS: the 4 waves' partial sums (gate/up: [4][2 MAXT][64], down: [4][MAXT][64]), then each wave's staging area.
  __shared__ __attribute__((aligned(16))) uint32_t lds[4 * 2 * MAXT * 64 + 4 * MAXT * STW];
  const int b = blockIdx.x;
  TR(0);
  if (b < 40) copy_x(a, b);
  else if (b < 40 + 50 * a.J) {
    const int r = b - 40;
    gate_up_role(a, r / 50, r % 50 / KA, r % KA, lds);
  } else {
    const int r = b - 40 - 50 * a.J;
    down_role(a, r % 40, r / 40, lds);
  }
}

// ---------------------------------------------------------------- prefill (RVII_PF_* of rvii_protocol.h)
// A chunk of up to RVII_PF_MAXT tokens; the (expert, token) pairs of the masked experts, grouped by expert. FP16
// activations with FP32 accumulation (v_dot2_f32_f16), the int4 weights converted to FP16 in registers (q - 8 is
// exact in FP16; the group's scale is applied to each group's FP32 sum):
//   x:   x (BF16, copied into y's memory first) to FP16 x16[T][H] in pair order: in each 8 k, the dwords hold the pairs
//        (k, k + 4) for k = 0..3, as the conversion of a packed int32 of weights yields them;
//   gate/up: one launch for all pairs of a batch: a wave per (expert, 64 columns, up to PF_TT tokens); lane = column,
//        with both its gate and up weights; the activations are wave-uniform (scalar loads, SGPR operands). Writes
//        SiLU(gate) * up in FP16 (clamped to its range) to act[pair][I] in pair order;
//   down: one launch per expert (its tokens are distinct, so the adds to y do not race): a wave per (64 hidden
//        columns, up to PF_TT tokens), adds wt * the output to y[tok] (FP32);
//   out: y to BF16 (round to nearest even, as the RTX 3090 rounds each expert's output) into the act buffer, which
//        is copied into the shared memory.
// (Tried: x and y over PCIe by the kernels instead of copies, 2 ms slower; the chunk in 2-4 parts of tokens, so that
// the transfers overlap with the computation, 2-11 ms slower per layer, as the experts' tiles get smaller.)
#ifndef PF_TT_GU
#define PF_TT_GU 8
#endif
#ifndef PF_TT_DN
#define PF_TT_DN 8
#endif
#ifndef PF_GU_ROWS
#define PF_GU_ROWS 4
#endif
#ifndef PF_DN_ROWS
#define PF_DN_ROWS 8
#endif
constexpr int PF_TT = PF_TT_GU;           // tokens per wave of gate/up
constexpr int PF_TD = PF_TT_DN;           // tokens per wave of down
constexpr int PF_PAIRS = 32768;           // pairs per batch: act[PF_PAIRS][I] FP16 = 40 MiB = the BF16 y of 8192 tokens
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

// The 8 int4 weights of a packed int32 (k = 0..7, low nibble first) as FP16 pairs (k, k + 4) of (q - 8).
__device__ __forceinline__ void cvt8(uint32_t q, half2_t* o) {
  const half2_t bias = {_Float16(-1032.f), _Float16(-1032.f)};
#pragma unroll
  for (int p = 0; p < 4; ++p) o[p] = __builtin_bit_cast(half2_t, ((q >> (4 * p)) & 0x000F000Fu) | 0x64006400u) + bias;
}
__device__ __forceinline__ float dot2(half2_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(a, __builtin_bit_cast(half2_t, b), c, false);
}
__device__ __forceinline__ int pair_pos(int k) { return (k & ~7) + 2 * (k & 3) + ((k >> 2) & 1); }

// xb (BF16 [T][H]) to x16 (FP16, pair order); a thread per 8 values.
__global__ void __launch_bounds__(256) k_pf_x16(const uint16_t* __restrict__ xb, int T, uint32_t* __restrict__ x16) {
  const int t = blockIdx.x * 256 + threadIdx.x;
  if (t >= T * (H / 8)) return;
  const u32x4 v = reinterpret_cast<const u32x4*>(xb)[t];
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  float f[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    f[2 * j] = __uint_as_float(w[j] << 16);
    f[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
  }
  u32x4 o;
  uint32_t* op = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const half2_t h = {_Float16(f[p]), _Float16(f[p + 4])};
    op[p] = __builtin_bit_cast(uint32_t, h);
  }
  reinterpret_cast<u32x4*>(x16)[t] = o;
}

struct PfItem {
  int slot, pair, n, cb, arow;  // expert slot, first pair, pairs (<= PF_TT), 64-column block, its row of act
};

__device__ __forceinline__ _Float16 to_half_clamped(float v, unsigned* clamped) {
  if (!(fabsf(v) <= 65504.f)) {
    atomicAdd(clamped, 1u);
    v = v != v ? 0.f : copysignf(65504.f, v);
  }
  return _Float16(v);
}

// The 8 int4 weights of a packed int32 scaled: FP16 pairs (k, k + 4) of (q - 8) * s (rounded to FP16).
__device__ __forceinline__ void cvt8s(uint32_t q, half2_t s2, half2_t* o) {
  cvt8(q, o);
#pragma unroll
  for (int p = 0; p < 4; ++p) o[p] *= s2;
}

template <int TT>
__global__ void __launch_bounds__(64) k_pf_gu(const uint8_t* __restrict__ w, const uint32_t* __restrict__ x16,
                                              const PfItem* __restrict__ items, const int* __restrict__ tok,
                                              _Float16* __restrict__ act, unsigned* clamped) {
  const PfItem it = items[blockIdx.x];
  const int lane = threadIdx.x;
  const uint8_t* ew = w + size_t(it.slot) * EXPERT_BYTES;
  const int* tk = tok + it.pair;
  float ag[TT], au[TT];
#pragma unroll
  for (int i = 0; i < TT; ++i) ag[i] = au[i] = 0.f;
  for (int g = 0; g < 20; ++g) {
    const u32x4* qw = reinterpret_cast<const u32x4*>(ew) + ((it.cb * 20 + g) * 8) * 64 + lane;
    const uint32_t sc = reinterpret_cast<const uint32_t*>(ew + GU_W)[(it.cb * 20 + g) * 64 + lane];
    const _Float16 sgh = __builtin_bit_cast(_Float16, uint16_t(sc)), suh = __builtin_bit_cast(_Float16, uint16_t(sc >> 16));
    const half2_t sg = {sgh, sgh}, su = {suh, suh};
#pragma unroll
    for (int h = 0; h < 16 / PF_GU_ROWS; ++h) {  // PF_GU_ROWS packed rows (8 k each) at a time
      half2_t wg[PF_GU_ROWS][4], wu[PF_GU_ROWS][4];
#pragma unroll
      for (int s = 0; s < PF_GU_ROWS / 2; ++s) {
        const u32x4 q = __builtin_nontemporal_load(qw + (PF_GU_ROWS / 2 * h + s) * 64);
        cvt8s(q.x, sg, wg[2 * s]); cvt8s(q.y, su, wu[2 * s]); cvt8s(q.z, sg, wg[2 * s + 1]); cvt8s(q.w, su, wu[2 * s + 1]);
      }
      const int k2 = (g * 128 + h * 8 * PF_GU_ROWS) / 2;  // dword offset of these k in a row of x16
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        if (i < it.n) {
          const uint32_t* xp = x16 + size_t(tk[i]) * (H / 2) + k2;
#pragma unroll
          for (int r = 0; r < PF_GU_ROWS; ++r)
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const uint32_t xv = xp[4 * r + p];
              ag[i] = dot2(wg[r][p], xv, ag[i]);
              au[i] = dot2(wu[r][p], xv, au[i]);
            }
        }
      }
    }
  }
  const int pos = pair_pos(it.cb * 64 + lane);
#pragma unroll
  for (int i = 0; i < TT; ++i)
    if (i < it.n) act[size_t(it.arow + i) * I + pos] = to_half_clamped(ag[i] / (1.f + __expf(-ag[i])) * au[i], clamped);
}

// One expert: a wave per (64 hidden columns, up to TT of its n tokens).
template <int TT>
__global__ void __launch_bounds__(64) k_pf_dn(const uint8_t* __restrict__ ew, const uint32_t* __restrict__ act,
                                              const int* __restrict__ tok, const float* __restrict__ wt, int n,
                                              float* __restrict__ y) {
  const int cb = blockIdx.x % 40, i0 = blockIdx.x / 40 * TT, lane = threadIdx.x;
  const int m = min(TT, n - i0);
  float acc[TT];
#pragma unroll
  for (int i = 0; i < TT; ++i) acc[i] = 0.f;
  for (int g = 0; g < 5; ++g) {
    const u32x4* qw = reinterpret_cast<const u32x4*>(ew + GU_W + GU_S) + ((cb * 5 + g) * 4) * 64 + lane;
    const _Float16 sdh = reinterpret_cast<const _Float16*>(ew + GU_W + GU_S + DN_W)[(cb * 5 + g) * 64 + lane];
    const half2_t sd = {sdh, sdh};
#pragma unroll
    for (int h = 0; h < 16 / PF_DN_ROWS; ++h) {  // PF_DN_ROWS packed rows at a time (a multiple of 4)
      half2_t wd[PF_DN_ROWS][4];
#pragma unroll
      for (int s = 0; s < PF_DN_ROWS / 4; ++s) {
        const u32x4 q = __builtin_nontemporal_load(qw + (PF_DN_ROWS / 4 * h + s) * 64);
        cvt8s(q.x, sd, wd[4 * s]); cvt8s(q.y, sd, wd[4 * s + 1]); cvt8s(q.z, sd, wd[4 * s + 2]); cvt8s(q.w, sd, wd[4 * s + 3]);
      }
      const int k2 = (g * 128 + h * 8 * PF_DN_ROWS) / 2;
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        if (i < m) {
          const uint32_t* ap = act + size_t(i0 + i) * (I / 2) + k2;
#pragma unroll
          for (int r = 0; r < PF_DN_ROWS; ++r)
#pragma unroll
            for (int p = 0; p < 4; ++p) acc[i] = dot2(wd[r][p], ap[4 * r + p], acc[i]);
        }
      }
    }
  }
  const int c = cb * 64 + lane;
#pragma unroll
  for (int i = 0; i < TT; ++i)
    if (i < m) y[size_t(tok[i0 + i]) * H + c] += wt[i0 + i] * acc[i];
}

// y (FP32) to BF16, round to nearest even; a thread per 4 values.
__global__ void __launch_bounds__(256) k_pf_out(const float* __restrict__ y, int n4, uint2* __restrict__ yb) {
  const int t = blockIdx.x * 256 + threadIdx.x;
  if (t >= n4) return;
  const float4 v = reinterpret_cast<const float4*>(y)[t];
  auto bf = [](float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
  };
  yb[t] = make_uint2(bf(v.x) | (bf(v.y) << 16), bf(v.z) | (bf(v.w) << 16));
}

std::atomic<bool> g_stop{false};
std::atomic<bool> g_done{false};  // the heartbeat runs until the requests are drained
void on_signal(int) { g_stop.store(true); }

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

long env_long(const char* name, long def) {
  const char* v = getenv(name);
  return v && *v ? strtol(v, nullptr, 10) : def;
}

// The prefill segment and its buffers (see the prefill kernels above).
struct Prefill {
  uint8_t* shm = nullptr;        // the segment (host-registered)
  const int16_t* map = nullptr;  // the main segment's map
  const uint8_t* w = nullptr;    // expert slot 0 in VRAM
  hipStream_t st{};
  uint32_t* x16 = nullptr;       // [MAXT][H] FP16 in pair order
  float* y = nullptr;            // [MAXT][H]; first the staging of x (BF16)
  _Float16* act = nullptr;       // [PF_PAIRS][I]; last the staging of y (BF16)
  int* tok = nullptr;            // [MAXT * TOPK] token of each pair, grouped by expert
  float* wt = nullptr;           // their routing weights
  PfItem* items = nullptr;       // gate/up work items
  int* tok_h = nullptr;          // pinned staging of tok, wt and items
  float* wt_h = nullptr;
  PfItem* items_h = nullptr;
  unsigned* clamped = nullptr;   // [1] in VRAM
  int64_t last = 0;
  long n_req = 0, n_bad = 0;
  double busy_s = 0, t_stat = 0;
  long stat_pairs = 0, stat_req = 0, stat_experts = 0, stat_tokens = 0;
  int stats_every = int(env_long("RVII_PF_STATS", 0));  // log the mean time per request every N requests
  static constexpr int MAX_ITEMS = 10 * (RVII_PF_MAXT * RVII_PF_TOPK / PF_TT + RVII_E);

  int64_t* req_seq() const { return reinterpret_cast<int64_t*>(shm + RVII_PF_REQ); }
  int64_t* resp_seq() const { return reinterpret_cast<int64_t*>(shm + RVII_PF_RESP); }

  void init(const uint8_t* weights, const int16_t* m) {
    w = weights;
    map = m;
    shm_unlink(RVII_PF_SHM_NAME);  // a segment left by a crashed worker: a new one (the GPU side remaps by pid)
    const int fd = shm_open(RVII_PF_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("rvii: shm_open prefill"); exit(1); }
    fchmod(fd, 0666);
    if (ftruncate(fd, RVII_PF_SHM_BYTES) != 0) { perror("rvii: ftruncate prefill"); exit(1); }
    shm = static_cast<uint8_t*>(mmap(nullptr, RVII_PF_SHM_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (shm == MAP_FAILED) { perror("rvii: mmap prefill"); exit(1); }
    ::close(fd);
    CK(hipHostRegister(shm, RVII_PF_SHM_BYTES, hipHostRegisterMapped));
    CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    constexpr size_t P = size_t(RVII_PF_MAXT) * RVII_PF_TOPK;
    static_assert(size_t(PF_PAIRS) * I >= size_t(RVII_PF_MAXT) * H, "act holds the BF16 y");
    CK(hipMalloc(&x16, sizeof(uint16_t) * RVII_PF_MAXT * H));
    CK(hipMalloc(&y, sizeof(float) * RVII_PF_MAXT * H));
    CK(hipMalloc(&act, sizeof(_Float16) * PF_PAIRS * I));
    CK(hipMalloc(&tok, sizeof(int) * P));
    CK(hipMalloc(&wt, sizeof(float) * P));
    CK(hipMalloc(&items, sizeof(PfItem) * MAX_ITEMS));
    CK(hipHostMalloc(&tok_h, sizeof(int) * P));
    CK(hipHostMalloc(&wt_h, sizeof(float) * P));
    CK(hipHostMalloc(&items_h, sizeof(PfItem) * MAX_ITEMS));
    CK(hipMalloc(&clamped, sizeof(unsigned)));
    CK(hipMemset(clamped, 0, sizeof(unsigned)));
    auto* hd = reinterpret_cast<RviiPfHeader*>(shm);
    hd->version = RVII_PF_VERSION;
    hd->maxt = RVII_PF_MAXT;
    hd->topk = RVII_PF_TOPK;
    std::atomic_thread_fence(std::memory_order_release);
    reinterpret_cast<std::atomic<uint64_t>*>(&hd->magic)->store(RVII_PF_MAGIC, std::memory_order_release);
  }

  // Serves a request if there is one; returns whether it did.
  bool poll() {
    const int64_t seq = __atomic_load_n(req_seq(), __ATOMIC_ACQUIRE);
    if (seq == last || seq == 0) return false;
    last = seq;
    const double t0 = now_s();
    const int T = *reinterpret_cast<const int32_t*>(shm + RVII_PF_REQ + 8);
    const int layer = *reinterpret_cast<const int32_t*>(shm + RVII_PF_REQ + 12);
    if (T < 1 || T > RVII_PF_MAXT || layer < 0 || layer >= RVII_L) {
      if (n_bad++ < 10) fprintf(stderr, "rvii: invalid prefill request seq %lld (T %d layer %d)\n", (long long)seq, T, layer);
      std::memset(shm + RVII_PF_Y, 0, sizeof(uint16_t) * H * (T >= 1 && T <= RVII_PF_MAXT ? T : 0));
      __atomic_store_n(resp_seq(), seq, __ATOMIC_RELEASE);
      return true;
    }
    // The pairs of the masked experts, grouped by expert (a counting sort).
    const int32_t* ids = reinterpret_cast<const int32_t*>(shm + RVII_PF_IDS);
    const float* wts = reinterpret_cast<const float*>(shm + RVII_PF_W);
    const uint8_t* mask = shm + RVII_PF_MASK + layer * RVII_E;
    const int16_t* lmap = map + layer * RVII_E;
    int start[RVII_E + 1] = {0};
    const int n = T * RVII_PF_TOPK;
    for (int i = 0; i < n; ++i) {
      const int e = ids[i];
      if (e >= 0 && e < RVII_E && mask[e] && lmap[e] >= 0) start[e + 1]++;
    }
    for (int e = 0; e < RVII_E; ++e) start[e + 1] += start[e];
    const int pairs = start[RVII_E];
    {
      int pos[RVII_E];
      std::memcpy(pos, start, sizeof(pos));
      for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e >= 0 && e < RVII_E && mask[e] && lmap[e] >= 0) {
          tok_h[pos[e]] = i / RVII_PF_TOPK;
          wt_h[pos[e]++] = wts[i];
        }
      }
    }
    // x (BF16) into y's memory, then to FP16 in pair order.
    if (pairs > 0) {
      CK(hipMemcpyAsync(y, shm + RVII_PF_X, sizeof(uint16_t) * T * H, hipMemcpyHostToDevice, st));
      CK(hipMemcpyAsync(tok, tok_h, sizeof(int) * pairs, hipMemcpyHostToDevice, st));
      CK(hipMemcpyAsync(wt, wt_h, sizeof(float) * pairs, hipMemcpyHostToDevice, st));
      k_pf_x16<<<(T * (H / 8) + 255) / 256, 256, 0, st>>>(reinterpret_cast<const uint16_t*>(y), T, x16);
    }
    CK(hipMemsetAsync(y, 0, sizeof(float) * T * H, st));
    // Batches of experts with at most PF_PAIRS pairs: gate/up of the batch, then down per expert.
    int experts = 0, nitems = 0;
    for (int e0 = 0; e0 < RVII_E;) {
      int e1 = e0;
      while (e1 < RVII_E && (e1 == e0 || start[e1 + 1] - start[e0] <= PF_PAIRS)) ++e1;
      const int pair0 = start[e0];
      const int it0 = nitems;
      for (int e = e0; e < e1; ++e)
        for (int p = start[e]; p < start[e + 1]; p += PF_TT)
          for (int cb = 0; cb < 10; ++cb)
            items_h[nitems++] = PfItem{lmap[e], p, std::min(PF_TT, start[e + 1] - p), cb, p - pair0};
      if (nitems > it0) {
        CK(hipMemcpyAsync(items + it0, items_h + it0, sizeof(PfItem) * (nitems - it0), hipMemcpyHostToDevice, st));
        k_pf_gu<PF_TT><<<nitems - it0, 64, 0, st>>>(w, x16, items + it0, tok, act, clamped);
        for (int e = e0; e < e1; ++e) {
          const int ne = start[e + 1] - start[e];
          if (ne == 0) continue;
          ++experts;
          k_pf_dn<PF_TD><<<40 * ((ne + PF_TD - 1) / PF_TD), 64, 0, st>>>(
              w + size_t(lmap[e]) * EXPERT_BYTES, reinterpret_cast<const uint32_t*>(act + size_t(start[e] - pair0) * I),
              tok + start[e], wt + start[e], ne, y);
        }
      }
      e0 = e1;
    }
    k_pf_out<<<(T * H / 4 + 255) / 256, 256, 0, st>>>(y, T * H / 4, reinterpret_cast<uint2*>(act));
    CK(hipMemcpyAsync(shm + RVII_PF_Y, act, sizeof(uint16_t) * T * H, hipMemcpyDeviceToHost, st));
    CK(hipStreamSynchronize(st));
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(resp_seq(), seq, __ATOMIC_RELEASE);
    const double dt = now_s() - t0;
    busy_s += dt;
    ++n_req;
    stat_req++; stat_pairs += pairs; stat_experts += experts; stat_tokens += T; t_stat += dt;
    if (stats_every > 0 && stat_req >= stats_every) {
      unsigned cl = 0;
      CK(hipMemcpy(&cl, clamped, sizeof(cl), hipMemcpyDeviceToHost));
      fprintf(stderr, "rvii prefill: %ld requests, per request %.0f tokens, %.0f experts, %.0f pairs, %.2f ms; "
                      "%u values clamped to FP16 (total)\n", stat_req, double(stat_tokens) / stat_req,
              double(stat_experts) / stat_req, double(stat_pairs) / stat_req, t_stat / stat_req * 1e3, cl);
      stat_req = stat_pairs = stat_experts = stat_tokens = 0;
      t_stat = 0;
    }
    return true;
  }

  void close() {
    if (!shm) return;
    (void)hipStreamSynchronize(st);
    (void)hipHostUnregister(shm);
    munmap(shm, RVII_PF_SHM_BYTES);
    shm_unlink(RVII_PF_SHM_NAME);
    shm = nullptr;
  }
};

}  // namespace

int main(int argc, char** argv) {
  // --max-per-layer: print the number of experts per layer that fit in VRAM (RVII_RESERVE_MIB stays free).
  if (argc >= 2 && !strcmp(argv[1], "--max-per-layer")) {
    CK(hipFree(nullptr));
    size_t fr = 0, tot = 0;
    CK(hipMemGetInfo(&fr, &tot));
    const size_t reserve = size_t(env_long("RVII_RESERVE_MIB", 640)) << 20;
    printf("%zu\n", fr > reserve ? (fr - reserve) / (size_t(RVII_L) * EXPERT_BYTES) : 0);
    return 0;
  }
  if (argc < 2) {
    fprintf(stderr, "usage: rvii_worker CACHE_DIR (with experts.bin and slots.bin from prepare.py)\n"
                    "       rvii_worker --max-per-layer\n");
    return 2;
  }
  const std::string dir = argv[1];
  // slots.bin: int32 (layer, expert) per slot, in the order of experts.bin.
  std::vector<int32_t> slots;
  {
    FILE* f = fopen((dir + "/slots.bin").c_str(), "rb");
    if (!f) { perror("rvii: slots.bin"); return 1; }
    int32_t le[2];
    while (fread(le, sizeof(le), 1, f) == 1) slots.push_back(le[0]), slots.push_back(le[1]);
    fclose(f);
  }
  const int nexp = int(slots.size() / 2);
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);

  // The shared memory: header and map first (ready = 0), the weights next, then ready = 1.
  const int fd = shm_open(RVII_SHM_NAME, O_CREAT | O_RDWR, 0666);
  if (fd < 0) { perror("rvii: shm_open"); return 1; }
  fchmod(fd, 0666);
  if (ftruncate(fd, RVII_SHM_BYTES) != 0) { perror("rvii: ftruncate"); return 1; }
  uint8_t* shm = static_cast<uint8_t*>(mmap(nullptr, RVII_SHM_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (shm == MAP_FAILED) { perror("rvii: mmap"); return 1; }
  close(fd);
  std::memset(shm, 0, RVII_SHM_BYTES);  // (a segment left by a crashed worker: no stale request)
  auto* hd = reinterpret_cast<RviiHeader*>(shm);
  auto* ready = reinterpret_cast<std::atomic<uint32_t>*>(&hd->ready);
  auto* heartbeat = reinterpret_cast<std::atomic<uint64_t>*>(&hd->heartbeat);
  ready->store(0, std::memory_order_release);
  hd->version = RVII_VERSION;
  hd->L = RVII_L; hd->E = RVII_E; hd->H = RVII_H; hd->I = RVII_I;
  hd->worker_pid = getpid();
  hd->n_experts = nexp;
  auto* map = reinterpret_cast<int16_t*>(shm + RVII_MAP_OFFSET);
  for (int i = 0; i < RVII_L * RVII_E; ++i) map[i] = -1;
  auto* rq = reinterpret_cast<RviiReqHead*>(shm + RVII_REQ_OFFSET);
  auto* rjobs = reinterpret_cast<const RviiJob*>(shm + RVII_REQ_OFFSET + RVII_REQ_JOBS);
  auto* resp_seq = reinterpret_cast<std::atomic<int64_t>*>(shm + RVII_RESP_OFFSET);
  auto* req_seq = reinterpret_cast<std::atomic<int64_t>*>(&rq->seq);
  float* yshm = reinterpret_cast<float*>(shm + RVII_RESP_OFFSET + RVII_RESP_Y);
  std::atomic_thread_fence(std::memory_order_release);
  reinterpret_cast<std::atomic<uint64_t>*>(&hd->magic)->store(RVII_MAGIC, std::memory_order_release);
  std::thread hb([&] {
    while (!g_done.load()) {
      heartbeat->fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });

  // The weights: experts.bin to VRAM through two pinned staging buffers.
  const double t0 = now_s();
  uint8_t* dw = nullptr;
  CK(hipMalloc(&dw, size_t(nexp) * EXPERT_BYTES));
  {
    const std::string path = dir + "/experts.bin";
    const int wf = open(path.c_str(), O_RDONLY);
    if (wf < 0) { perror("rvii: experts.bin"); return 1; }
    struct stat st;
    fstat(wf, &st);
    if (size_t(st.st_size) != size_t(nexp) * EXPERT_BYTES) {
      fprintf(stderr, "rvii: experts.bin has %lld bytes, expected %zu\n", (long long)st.st_size,
              size_t(nexp) * EXPERT_BYTES);
      return 1;
    }
    constexpr size_t CH = size_t(64) << 20;
    uint8_t* stage[2];
    hipEvent_t ev[2];
    hipStream_t cs;
    CK(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking));
    for (int b = 0; b < 2; ++b) { CK(hipHostMalloc(&stage[b], CH)); CK(hipEventCreate(&ev[b])); }
    const size_t total = size_t(st.st_size);
    int b = 0;
    for (size_t off = 0; off < total && !g_stop.load(); off += CH, b ^= 1) {
      const size_t n = std::min(CH, total - off);
      CK(hipEventSynchronize(ev[b]));  // the previous copy from this buffer is done
      for (size_t got = 0; got < n;) {
        const ssize_t r = pread(wf, stage[b] + got, n - got, off + got);
        if (r <= 0) { perror("rvii: read experts.bin"); return 1; }
        got += size_t(r);
      }
      CK(hipMemcpyAsync(dw + off, stage[b], n, hipMemcpyHostToDevice, cs));
      CK(hipEventRecord(ev[b], cs));
    }
    CK(hipStreamSynchronize(cs));
    for (int i = 0; i < 2; ++i) { CK(hipHostFree(stage[i])); CK(hipEventDestroy(ev[i])); }
    CK(hipStreamDestroy(cs));
    close(wf);
  }
  if (g_stop.load()) { g_done.store(true); hb.join(); munmap(shm, RVII_SHM_BYTES); shm_unlink(RVII_SHM_NAME); return 0; }

  // Buffers, and the shared memory mapped into the GPU's address space (x is read and y written over PCIe).
  Args a{};
  a.w = dw;
  CK(hipMalloc(&a.xq, sizeof(uint32_t) * MAXT * 40 * 64));
  CK(hipMalloc(&a.xi, sizeof(float) * MAXT * 40 * 16));
  CK(hipMalloc(&a.aq, sizeof(uint32_t) * MAXJ * MAXT * 10 * 64));
  CK(hipMalloc(&a.ai, sizeof(float) * MAXJ * MAXT * 10 * 16));
  CK(hipMalloc(&a.gpart, sizeof(float) * MAXJ * 10 * KA * 2 * MAXT * 64));
  CK(hipMalloc(&a.ypart, sizeof(float) * MAXS * MAXT * H));
  constexpr size_t NFLAG = 40 + MAXJ * 10 + MAXJ * 10 + 40 + 1;  // xflag, aflag, cnt_gu, cnt_dn, cnt_y
  uint32_t* flags = nullptr;
  CK(hipMalloc(&flags, sizeof(uint32_t) * NFLAG));
  CK(hipMemset(flags, 0, sizeof(uint32_t) * NFLAG));
  a.xflag = flags;
  a.aflag = flags + 40;
  a.cnt_gu = reinterpret_cast<int*>(flags + 40 + MAXJ * 10);
  a.cnt_dn = a.cnt_gu + MAXJ * 10;
  a.cnt_y = a.cnt_dn + 40;
#ifdef RVII_TRACE
  {
    unsigned long long h[16];
    for (int e = 0; e < 8; ++e) h[2 * e] = ~0ull, h[2 * e + 1] = 0;
    CK(hipMalloc(&a.dbg, sizeof(h)));
    CK(hipMemcpy(a.dbg, h, sizeof(h), hipMemcpyHostToDevice));
  }
#endif
  // (job, group) units per wave of the down role: fewer = more workgroups
  const int upw = std::max(1, int(env_long("RVII_DOWN_UNITS_PER_WAVE", 2)));
  const bool sync = env_long("RVII_SYNC", 0) != 0;  // wait for each kernel (debugging)
  CK(hipHostRegister(shm, RVII_SHM_BYTES, hipHostRegisterMapped));
  uint8_t* dshm = nullptr;
  CK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dshm), shm, 0));
  a.xh = reinterpret_cast<const float*>(dshm + RVII_REQ_OFFSET + RVII_REQ_X);
  a.yh = reinterpret_cast<float*>(dshm + RVII_RESP_OFFSET + RVII_RESP_Y);
  a.seqh = reinterpret_cast<int64_t*>(dshm + RVII_RESP_OFFSET);
  hipStream_t st;
  CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));

  for (int s = 0; s < nexp; ++s) map[slots[2 * s] * RVII_E + slots[2 * s + 1]] = int16_t(s);
  Prefill pf;
  pf.init(dw, map);
  int64_t last = 0;  // the last answered seq
  std::atomic_thread_fence(std::memory_order_release);
  ready->store(1, std::memory_order_release);
  fprintf(stderr, "rvii: ready, %d experts (%.2f GiB) loaded in %.1f s\n", nexp,
          double(nexp) * EXPERT_BYTES / (1 << 30), now_s() - t0);

  double t_last = now_s();
  long n_req = 0, n_bad = 0;
  // On SIGTERM: ready = 0 at once (no new decode requests, no new prefill chunks), but requests are served until
  // none has come for a second, so that a prefill chunk in progress on the RTX 3090 is finished.
  bool draining = false;
  for (;;) {
    if (g_stop.load(std::memory_order_relaxed)) {
      if (!draining) {
        draining = true;
        ready->store(0, std::memory_order_release);
        t_last = now_s();
      }
      if (now_s() - t_last > 1.0) break;
    }
    if (pf.poll()) { t_last = now_s(); continue; }
    const int64_t seq = req_seq->load(std::memory_order_acquire);
    if (seq == last || seq == 0) {
      // Spin while active; when idle for more than 100 ms, poll every 100 us so as not to burn a core, and after 2 s,
      // every 1 ms (~1000 instead of ~10000 wake-ups per second while the model is not used).
      for (int i = 0; i < 8; ++i) _mm_pause();
      const double idle = now_s() - t_last;
      if (idle > 0.1) {
        usleep(idle > 2.0 ? 1000 : 100);
        const hipError_t e = hipStreamQuery(st);  // report a failed kernel
        if (e != hipSuccess && e != hipErrorNotReady) CK(e);
      }
      continue;
    }
#ifdef RVII_TRACE
    const double t_seen = now_s();
#endif
    a.T = rq->T;
    const int layer = rq->layer, J = rq->J;
    bool bad = a.T < 1 || a.T > MAXT || layer < 0 || layer >= RVII_L || J < 0 || J > MAXJ;
    a.J = 0;
    for (int j = 0; j < J && !bad; ++j) {
      const RviiJob& r = rjobs[j];
      const int slot = r.e >= 0 && r.e < RVII_E ? map[layer * RVII_E + r.e] : -1;
      if (slot < 0 || r.n < 1 || r.n > a.T) { bad = true; break; }
      Job& o = a.jobs[a.J++];
      o.slot = slot;
      o.n = r.n;
      for (int i = 0; i < MAXT; ++i) {
        o.tok[i] = i < r.n ? r.tok[i] : 0;
        o.wt[i] = i < r.n ? r.wt[i] : 0.f;
        if (i < r.n && (r.tok[i] < 0 || r.tok[i] >= a.T)) bad = true;
      }
    }
    if (bad || a.J == 0) {
      // Zeros for a request without jobs, and for one that is not valid in the protocol (the CPU side checks the map
      // before it sends a job).
      if (bad && n_bad++ < 10)
        fprintf(stderr, "rvii: invalid request seq %lld (T %d layer %d J %d)\n", (long long)seq, a.T, layer, J);
      CK(hipStreamSynchronize(st));  // the previous request's kernel has finished
      std::memset(yshm, 0, sizeof(float) * MAXT * H);
      std::atomic_thread_fence(std::memory_order_seq_cst);
      resp_seq->store(seq, std::memory_order_release);
    } else {
      // The kernel writes y and the response seq itself.
      a.S = std::min(MAXS, (a.J * 5 + 4 * upw - 1) / (4 * upw));
      a.seq = seq;
      if (++a.gen == 0) ++a.gen;  // 0 = the flags' initial value
      k_req<<<40 + 50 * a.J + 40 * a.S, 256, 0, st>>>(a);
      if (sync) CK(hipStreamSynchronize(st));
#ifdef RVII_TRACE
      {  // per-phase times (us after the first workgroup's start; s_memrealtime runs at 25 MHz on the Radeon VII)
        static double acc[16], hl[3];
        static int cnt;
        unsigned long long h[16];
        const double tl = now_s();
        while (resp_seq->load(std::memory_order_acquire) != seq) {}
        const double tr = now_s();
        hl[0] += (tl - t_seen) * 1e6;
        hl[1] += (tr - t_seen) * 1e6;
        CK(hipStreamSynchronize(st));
        hl[2] += (now_s() - t_seen) * 1e6;
        CK(hipMemcpy(h, a.dbg, sizeof(h), hipMemcpyDeviceToHost));
        for (int e = 0; e < 16; ++e) acc[e] += (h[e] - h[0]) * 0.04;
        if (++cnt == 250) {
          fprintf(stderr, "trace T %d J %d: host: launch %.1f seq %.1f sync %.1f; gpu:", a.T, a.J, hl[0] / cnt,
                  hl[1] / cnt, hl[2] / cnt);
          hl[0] = hl[1] = hl[2] = 0;
          static const char* nm[8] = {"start", "copied", "gu-wait", "gu-comp", "act", "dn-wait", "dn-comp", "seq"};
          for (int e = 0; e < 8; ++e) fprintf(stderr, " %s %.1f/%.1f", nm[e], acc[2 * e] / cnt, acc[2 * e + 1] / cnt);
          fprintf(stderr, "\n");
          cnt = 0;
          for (double& v : acc) v = 0;
        }
        for (int e = 0; e < 8; ++e) h[2 * e] = ~0ull, h[2 * e + 1] = 0;
        CK(hipMemcpy(a.dbg, h, sizeof(h), hipMemcpyHostToDevice));
      }
#endif
    }
    last = seq;
    t_last = now_s();
    ++n_req;
  }
  ready->store(0, std::memory_order_release);
  (void)hipStreamSynchronize(st);
  g_done.store(true);
  hb.join();
  fprintf(stderr, "rvii: exiting after %ld requests (%ld invalid) and %ld prefill requests (%.1f s, %ld invalid)\n",
          n_req, n_bad, pf.n_req, pf.busy_s, pf.n_bad);
  pf.close();
  (void)hipHostUnregister(shm);
  munmap(shm, RVII_SHM_BYTES);
  shm_unlink(RVII_SHM_NAME);
  return 0;
}
