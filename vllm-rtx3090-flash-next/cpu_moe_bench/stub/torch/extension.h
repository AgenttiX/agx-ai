// Minimal stand-in for torch/extension.h, so that cpu_moe.cpp compiles into the offline benchmark.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define TORCH_CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "check failed: %s\n", #cond); std::abort(); } } while (0)
namespace torch {
enum ScalarType { kInt64 };
struct Tensor {
  void* p = nullptr;
  std::vector<int64_t> shape;
  template <class T> T* data_ptr() const { return static_cast<T*>(p); }
  int64_t size(int i) const { return shape[i]; }
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  int64_t dim() const { return static_cast<int64_t>(shape.size()); }
  Tensor contiguous() const { return *this; }
  ScalarType scalar_type() const { return kInt64; }
};
}
namespace py {
template <class T> struct call_guard {};
struct gil_scoped_release {};
}
struct StubModule { template <class... A> void def(A&&...) {} };
#define TORCH_EXTENSION_NAME stub
#define PYBIND11_MODULE(name, m) [[maybe_unused]] static void pybind_stub(StubModule& m)
