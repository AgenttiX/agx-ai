// Shared-memory protocol between the CPU expert pool of the Qwen3.8-Flash-Next runtime (cpu_moe.cpp, in the CUDA
// container) and the Radeon VII expert worker (rvii_worker, in a ROCm container). Both containers share the host's
// /dev/shm (ipc: host). See README.md.
//
// The worker keeps some of the cold experts (those that the RTX 3090 does not keep) in the Radeon VII's VRAM. For each
// decode request of a layer (1..4 tokens), the CPU pool sends the cold experts that the worker holds to the worker,
// computes the others itself, and adds the worker's partial sum to its own before it answers the RTX 3090.
//
// Expert math (as in cpu_moe.cpp): y = W_down (silu(W_gate x) * (W_up x)), with the checkpoint's AutoRound int4
// weights, symmetric (zero point 8), group size 128:
//   qweight int32 [K/8, N]: nibble j (low nibble first) of qweight[r][n] is the weight of row k = 8 r + j, column n;
//   scales fp16 [K/128, N]; w[k][n] = (nibble - 8) * scales[k / 128][n].
//   gate/up: K = H = 2560 -> N = I = 640; down: K = I = 640 -> N = H = 2560.
#pragma once
#include <stdint.h>

#define RVII_SHM_NAME "/qwen38-rvii"  // shm_open name, i.e. /dev/shm/qwen38-rvii
#define RVII_MAGIC 0x4949565238335651ULL  // "QV38RVII" in little endian
#define RVII_VERSION 1

#define RVII_L 48
#define RVII_E 512
#define RVII_H 2560
#define RVII_I 640
#define RVII_MAXT 4
#define RVII_MAXJ 40  // at most 4 tokens x 10 experts per layer

// Header at offset 0 (one page).
struct RviiHeader {
  uint64_t magic;      // RVII_MAGIC, written by the worker when it creates the segment
  uint32_t version;    // RVII_VERSION
  uint32_t ready;      // 1 after the map and the weights are loaded; 0 while loading and after the worker exits
  uint32_t L, E, H, I; // 48, 512, 2560, 640
  int64_t worker_pid;
  uint64_t heartbeat;  // incremented by the worker at least every 100 ms while it runs (also while busy)
  uint32_t n_experts;  // experts in the Radeon VII's VRAM
  uint32_t pad;
};

// Map at offset 4096: int16 map[L][E], the worker's slot of the expert (>= 0) if it is in the Radeon VII's VRAM,
// else -1. Written before ready = 1 and not changed while ready = 1.
#define RVII_MAP_OFFSET 4096

struct RviiJob {
  int32_t e;              // expert index within the layer (0..E-1); the worker looks up its slot in the map
  int32_t n;              // tokens of this expert (1..T)
  int32_t tok[RVII_MAXT]; // token rows (0..T-1)
  float wt[RVII_MAXT];    // routing weights
};

// Request at offset 65536: the CPU pool writes T, layer, J, the jobs and x, then seq (release, strictly increasing).
#define RVII_REQ_OFFSET 65536
#define RVII_REQ_JOBS 64     // RviiJob[RVII_MAXJ] at req + 64
#define RVII_REQ_X 2048      // float x[T][H] at req + 2048 (FP32 activations, as received from the RTX 3090)
struct RviiReqHead {
  int64_t seq;
  int32_t T, layer, J;
};

// Response at offset 131072: the worker writes y and then seq = the request's seq (release).
// y[t][h] = sum over jobs j, i < n_j with tok_j[i] = t of wt_j[i] * expert_{layer, e_j}(x[t])[h]; rows of tokens
// without jobs are zero.
#define RVII_RESP_OFFSET 131072
#define RVII_RESP_Y 64       // float y[T][H] at resp + 64
#define RVII_SHM_BYTES (RVII_RESP_OFFSET + RVII_RESP_Y + RVII_MAXT * RVII_H * 4 + 4096)

// ---- Prefill (2026-11-25): a second segment, between the RTX 3090 (stream_v2.py, rvii_prefill.py) and the worker.
// In a prefill chunk, the RTX 3090 streams the cold experts from RAM; the worker computes some of its experts instead
// (those set in the mask), and the RTX 3090 streams only the others. Per MoE layer, the RTX 3090 copies the routing
// and x into the segment (DMA) and publishes seq; the worker computes y and writes it and then the response seq; the
// RTX 3090 waits for it on the GPU and adds y to its FP32 accumulator. The worker is ready when the main segment's
// header says so (ready, heartbeat).
#define RVII_PF_SHM_NAME "/qwen38-rvii-pf"
#define RVII_PF_MAGIC 0x4650494956523351ULL  // "Q3RVIIPF" in little endian
#define RVII_PF_VERSION 1
#define RVII_PF_MAXT 8192    // tokens per prefill chunk (the runtime's max_num_batched_tokens)
#define RVII_PF_TOPK 10
struct RviiPfHeader {      // at offset 0
  uint64_t magic;          // RVII_PF_MAGIC, written by the worker when it creates the segment
  uint32_t version;        // RVII_PF_VERSION
  uint32_t maxt, topk;     // RVII_PF_MAXT, RVII_PF_TOPK
};
#define RVII_PF_REQ 64       // request: int64 seq, then int32 T, int32 layer (the RTX 3090 writes T and layer first)
#define RVII_PF_RESP 128     // response: int64 seq (the worker writes y first)
// uint8 mask[L][E] at 4096: 1 = the worker computes this expert in prefill (it must be in the map). Written by the
// RTX 3090 between prefill chunks.
#define RVII_PF_MASK 4096
#define RVII_PF_IDS 32768                                              // int32 ids[T][TOPK]
#define RVII_PF_W (RVII_PF_IDS + 4 * RVII_PF_MAXT * RVII_PF_TOPK)      // float w[T][TOPK]
#define RVII_PF_X (RVII_PF_W + 4 * RVII_PF_MAXT * RVII_PF_TOPK)        // bf16 x[T][H]
#define RVII_PF_Y (RVII_PF_X + 2 * RVII_PF_MAXT * RVII_H)              // bf16 y[T][H]: the masked experts' sum
#define RVII_PF_SHM_BYTES (RVII_PF_Y + 2 * RVII_PF_MAXT * RVII_H)
