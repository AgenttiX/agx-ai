// RAM read bandwidth: N threads pinned to the given CPUs stream through their part of a large buffer with AVX2 loads.
// Usage: bw GIB CPU,CPU,...   (prints GB/s, best of 5 passes)
#include <immintrin.h>
#include <pthread.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static std::vector<int> parse(const char* s) {
  std::vector<int> v;
  std::string str(s);
  size_t p = 0;
  while (p < str.size()) {
    size_t q = str.find(',', p);
    std::string tok = str.substr(p, q == std::string::npos ? std::string::npos : q - p);
    size_t d = tok.find('-');
    if (d == std::string::npos) v.push_back(std::stoi(tok));
    else for (int i = std::stoi(tok.substr(0, d)); i <= std::stoi(tok.substr(d + 1)); ++i) v.push_back(i);
    if (q == std::string::npos) break;
    p = q + 1;
  }
  return v;
}

int main(int argc, char** argv) {
  const size_t bytes = static_cast<size_t>(atof(argv[1]) * (1ull << 30));
  const std::vector<int> cpus = parse(argv[2]);
  const int n = static_cast<int>(cpus.size());
  char* buf = static_cast<char*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  madvise(buf, bytes, MADV_HUGEPAGE);
  std::vector<double> sums(n * 8);
  std::atomic<int> go{0}, done{0};
  const size_t per = (bytes / n) & ~size_t(4095);
  std::vector<std::thread> th;
  for (int i = 0; i < n; ++i)
    th.emplace_back([&, i] {
      cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[i], &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      char* p = buf + per * i;
      memset(p, 1, per);  // first touch on the thread's own CPU
      done.fetch_add(1);
      for (int pass = 1; pass <= 5; ++pass) {
        while (go.load() < pass) _mm_pause();
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t o = 0; o < per; o += 128) {
          a0 = _mm256_add_epi32(a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + o)));
          a1 = _mm256_add_epi32(a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + o + 32)));
          a2 = _mm256_add_epi32(a2, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + o + 64)));
          a3 = _mm256_add_epi32(a3, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + o + 96)));
        }
        a0 = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        sums[i * 8] += _mm256_extract_epi32(a0, 0);
        done.fetch_add(1);
      }
    });
  while (done.load() < n) _mm_pause();
  double best = 0;
  for (int pass = 1; pass <= 5; ++pass) {
    const int target = n * (pass + 1);
    auto t0 = std::chrono::steady_clock::now();
    go.store(pass);
    while (done.load() < target) _mm_pause();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::max(best, per * n / s / 1e9);
  }
  for (auto& t : th) t.join();
  printf("%d threads: %.1f GB/s\n", n, best);
  return 0;
}
