// Offline benchmark of the decode path of the runtime's CPU expert kernel (compute_layer of cpu_moe.cpp, compiled into
// this program with stub/torch/extension.h): synthetic int4 experts in a RAM arena of BENCH_L layers x BENCH_E
// experts (default 48 x 64, 7.7 GB), decode-like routing (each of T tokens picks `picks` distinct experts of a layer,
// the other of its 10 ids are hot experts) and the same thread pool. run.sh builds and runs it, see README.md.
// Usage: bench CPUS [T] [picks per token] [steps] [dump|compare FILE]
// The kernel's mode comes from its environment variables (QWEN38_CPU_EXPERTS_INT8 / _INT16); BENCH_SLICES sets the
// work slices (set_slices). `dump` writes the outputs, `compare` reports the relative error against such a file.
// BENCH_REPLAY=<trace> replays a decode trace of the server (QWEN38_DECODE_TRACE) instead: its routing on layers
// [0, BENCH_L) with BENCH_E=512, the experts of BENCH_SKIP (uint8 [48][512], 1 = on a GPU) skipped, and its GPU gaps
// between the layers spun out (BENCH_GAPS=0: none; BENCH_GAPMIN=ns: at least that long), for up to `steps` steps.
#include KERNEL
#ifndef SET_LAYER
#define SET_LAYER(l) (void)0
#define AFTER_LAYER(tab, E, l, L) (void)0
#define PF_REPORT() (void)0
#endif
#include <random>
#include <string>

static std::vector<int> parse_cpus(const std::string& str) {
  std::vector<int> v;
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

static uint16_t f16(float f) { return _cvtss_sh(f, 0); }

int main(int argc, char** argv) {
  const std::vector<int> cpus = parse_cpus(argv[1]);
  const int T = argc > 2 ? atoi(argv[2]) : 4, picks = argc > 3 ? atoi(argv[3]) : 7;
  const int steps = argc > 4 ? atoi(argv[4]) : 50;
  const std::string mode = argc > 5 ? argv[5] : "", file = argc > 6 ? argv[6] : "";
  const int L = std::getenv("BENCH_L") ? atoi(std::getenv("BENCH_L")) : 48;
  const int E = std::getenv("BENCH_E") ? atoi(std::getenv("BENCH_E")) : 64;
  constexpr size_t SLOT = 2534400;
  static const size_t offs[6] = {0, 2457600, 819200, 2483200, 1638400, 2508800};
  const size_t bytes = SLOT * L * E;
  char* arena = static_cast<char*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  madvise(arena, bytes, MADV_HUGEPAGE);
  std::mt19937_64 rng(1234);
  std::vector<int64_t> table(L * E * 6);
  std::uniform_real_distribution<float> us(0.001f, 0.004f);
  {  // synthetic experts: splitmix64 per slot, filled by 32 threads
    std::vector<std::thread> th;
    const int NT = 32;
    for (int w = 0; w < NT; ++w) th.emplace_back([&, w] {
      for (int s = w; s < L * E; s += NT) {
        char* b = arena + SLOT * s;
        uint64_t st = 0x9E3779B97F4A7C15ull * (s + 1);
        auto nx = [&] { uint64_t z = (st += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); };
        uint64_t* q = reinterpret_cast<uint64_t*>(b);
        for (size_t i = 0; i < 2457600 / 8; ++i) q[i] = nx();
        uint16_t* sc = reinterpret_cast<uint16_t*>(b + 2457600);
        for (size_t i = 0; i < 76800 / 2; ++i) { const uint64_t r = nx();
          sc[i] = f16((r & 1 ? 1.f : -1.f) * (0.001f + 0.003f * ((r >> 8) % 1000) / 1000.f)); }
        for (int i = 0; i < 6; ++i) table[s * 6 + i] = reinterpret_cast<int64_t>(b + offs[i]);
      }
    });
    for (auto& t : th) t.join();
  }
  std::vector<uint8_t> skip(E, 0);
  // Routing: each token picks `picks` distinct experts of the layer (the other ids are -1, e.g. hot experts).
  const int NL = steps * L;
  std::vector<int32_t> ids(static_cast<size_t>(NL) * T * 10, -1);
  std::vector<float> wts(static_cast<size_t>(NL) * T * 10, 0.f), xs(static_cast<size_t>(NL) * T * H);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (int r = 0; r < NL; ++r)
    for (int t = 0; t < T; ++t) {
      std::vector<int> e(E);
      for (int i = 0; i < E; ++i) e[i] = i;
      std::shuffle(e.begin(), e.end(), rng);
      for (int k = 0; k < picks; ++k) {
        ids[(static_cast<size_t>(r) * T + t) * 10 + k] = e[k];
        wts[(static_cast<size_t>(r) * T + t) * 10 + k] = 0.05f + 0.1f * (rng() % 100) / 100.f;
      }
      for (int h = 0; h < H; ++h) xs[(static_cast<size_t>(r) * T + t) * H + h] = nd(rng);
    }
  g_pool = new Pool(static_cast<int>(cpus.size()), cpus);
  if (const char* sl = std::getenv("BENCH_SLICES")) {
    std::vector<int64_t> b;
    for (const std::string& v : [&] { std::vector<std::string> r; std::string t(sl); size_t q;
           while ((q = t.find(',')) != std::string::npos) { r.push_back(t.substr(0, q)); t = t.substr(q + 1); }
           r.push_back(t); return r; }()) b.push_back(std::stoll(v));
    set_slices(b);
  }
  if (const char* rp = std::getenv("BENCH_REPLAY")) {
    // Replay a decode trace (QWEN38_DECODE_TRACE records) on layers [0, L): real routing, the hot and Radeon VII
    // experts skipped (BENCH_SKIP: a file of uint8 [48][512], 1 = not a CPU expert), and the traced GPU gaps
    // between the layers spun out (BENCH_GAPS=0: none). Needs BENCH_E=512.
    struct Rec { int64_t t_req, t_cpu, t_done; int32_t layer, T; int32_t ids[40]; };
    std::vector<Rec> recs;
    { FILE* f = std::fopen(rp, "rb"); Rec r; while (std::fread(&r, sizeof r, 1, f) == 1) recs.push_back(r); std::fclose(f); }
    std::vector<uint8_t> skip48(48 * 512, 0);
    { FILE* f = std::fopen(std::getenv("BENCH_SKIP"), "rb"); TORCH_CHECK(std::fread(skip48.data(), 1, skip48.size(), f) == skip48.size(), "skip"); std::fclose(f); }
    const bool gaps = !std::getenv("BENCH_GAPS") || std::atoi(std::getenv("BENCH_GAPS"));
    const int maxsteps = steps;
    std::vector<float> scratch(4 << 20), outb(MAXT * H), xin(MAXT * H);
    std::normal_distribution<float> nd2(0.f, 1.f);
    for (auto& v : xin) v = nd2(rng);
    std::vector<float> wv(40, 0.1f);
    auto spin_ns = [](int64_t ns) { const auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count() < ns) _mm_pause(); };
    int nsteps = 0; double cpu_s = 0; int64_t jobs0 = g_cold_jobs.load();
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i + 1 < recs.size() && nsteps < maxsteps; ++i) {
      const Rec& r = recs[i];
      if (r.layer == 0) ++nsteps;
      if (r.layer >= L || r.T < 1 || r.T > MAXT) continue;
      const auto c0 = std::chrono::steady_clock::now();
      SET_LAYER(r.layer);
      compute_layer(table.data() + static_cast<int64_t>(r.layer) * E * 6, E, skip48.data() + r.layer * 512, r.T, 10,
                    r.ids, wv.data(), xin.data(), outb.data(), scratch.data(), scratch.size());
      AFTER_LAYER(table.data(), E, r.layer, L);
      cpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
      const Rec& nx = recs[i + 1];
      const int64_t gap = nx.t_req - r.t_done;
      static const int64_t gapmin = std::getenv("BENCH_GAPMIN") ? std::atoll(std::getenv("BENCH_GAPMIN")) : 0;
      if (gaps && gap > 0 && gap < 20000000) {
        if (gap < gapmin) { spin_ns(gapmin); continue; }
        // the traced gap to the next request; for layers >= L the whole remaining layers' time of the step
        spin_ns(gap);
      }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double jobs = static_cast<double>(g_cold_jobs.load() - jobs0);
    std::printf("replay %d steps x %d layers: %.2f ms per step wall, CPU compute %.2f ms per step, %.1f experts per layer, %.1f GB/s during compute\n",
                nsteps, L, 1e3 * s / nsteps, 1e3 * cpu_s / nsteps, jobs / nsteps / L, jobs * 2457600.0 / cpu_s / 1e9);
    PF_REPORT();
    return 0;
  }
  std::vector<float> scratch(4 << 20), out(static_cast<size_t>(NL) * T * H);
  auto run = [&](int r) {
    const int layer = r % L;
    compute_layer(table.data() + static_cast<int64_t>(layer) * E * 6, E, skip.data(), T, 10,
                  ids.data() + static_cast<size_t>(r) * T * 10, wts.data() + static_cast<size_t>(r) * T * 10,
                  xs.data() + static_cast<size_t>(r) * T * H, out.data() + static_cast<size_t>(r) * T * H,
                  scratch.data(), scratch.size());
  };
  for (int r = 0; r < L; ++r) run(r);  // warm-up (page faults, thread start)
  g_cold_jobs.store(0);
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < NL; ++r) run(r);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double jobs = static_cast<double>(g_cold_jobs.load());
  std::printf("%zu threads, T=%d: %.2f ms per step (%d layers), %.1f experts per layer, %.1f GB/s\n", cpus.size(), T,
              1e3 * s / steps, L, jobs / NL, jobs * 2457600.0 / s / 1e9);
  if (mode == "dump") {
    FILE* f = std::fopen(file.c_str(), "wb");
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
  } else if (mode == "compare") {
    std::vector<float> ref(out.size());
    FILE* f = std::fopen(file.c_str(), "rb");
    TORCH_CHECK(std::fread(ref.data(), sizeof(float), ref.size(), f) == ref.size(), "short reference");
    std::fclose(f);
    double num = 0, den = 0, worst = 0;
    for (int r = 0; r < NL * T; ++r) {
      double a = 0, b = 0;
      for (int h = 0; h < H; ++h) {
        const double d = out[static_cast<size_t>(r) * H + h] - ref[static_cast<size_t>(r) * H + h];
        a += d * d; b += static_cast<double>(ref[static_cast<size_t>(r) * H + h]) * ref[static_cast<size_t>(r) * H + h];
      }
      num += a; den += b;
      if (b > 0) worst = std::max(worst, std::sqrt(a / b));
    }
    std::printf("relative error vs reference: %.2e (worst token %.2e)\n", std::sqrt(num / den), worst);
  }
  return 0;
}
