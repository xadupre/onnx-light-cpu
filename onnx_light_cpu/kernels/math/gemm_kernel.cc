// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::NodeKernelFn;
using rt_ns::RuntimeContext;
using rt_ns::Shape;
using rt_ns::Tensor;

// Immutable-plan cache backing the FP32/FP64 operator path (Roadmap PR06.4).
// The plan encodes the selected algorithm, blocking, thread count, and (unused
// here) constant-B packing derived from the dtype/shape/attributes; it is only
// rebuilt when the key changes so those selections are not re-derived per run.
struct GemmKernel::GemmPlanCache {
  bool has_key = false;
  int data_type = 0;
  bool trans_a = false;
  bool trans_b = false;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::unique_ptr<GemmPlan<float>> plan_f32;
  std::unique_ptr<GemmPlan<double>> plan_f64;

  template <typename T>
  const GemmPlan<T> &GetOrBuild(int dt, bool ta, bool tb, std::size_t rows, std::size_t cols,
                                std::size_t depth, T scale) {
    std::unique_ptr<GemmPlan<T>> &slot = Slot<T>();
    // Compare ``alpha`` against the cached plan's own ``T`` value so the
    // change-detection stays in the input precision instead of widening to
    // ``double`` (which could mask or fabricate differences for ``float``).
    const bool match = has_key && data_type == dt && trans_a == ta && trans_b == tb && m == rows &&
                       n == cols && k == depth && slot != nullptr && slot->alpha() == scale;
    if (!match) {
      slot = std::make_unique<GemmPlan<T>>(
          GemmPlanOptions<T>{ta, tb, rows, cols, depth, scale, T(0), {}});
      has_key = true;
      data_type = dt;
      trans_a = ta;
      trans_b = tb;
      m = rows;
      n = cols;
      k = depth;
    }
    return *slot;
  }

private:
  template <typename T> std::unique_ptr<GemmPlan<T>> &Slot() {
    if constexpr (std::is_same_v<T, float>) {
      return plan_f32;
    } else {
      static_assert(std::is_same_v<T, double>);
      return plan_f64;
    }
  }
};

GemmKernel::GemmKernel(const rt_ns::KernelContext &ctx)
    : rt_ns::KernelBase(ctx), plan_cache_(std::make_unique<GemmPlanCache>()) {}

GemmKernel::~GemmKernel() = default;

namespace {

constexpr std::int64_t kHalfConversionParallelGrainSize = 32 * rt_ns::kParallelForGrainSize;

template <typename Fn> void ParallelForHalfConversion(std::size_t count, Fn fn) {
  const auto total = static_cast<std::int64_t>(count);
  if (total < kHalfConversionParallelGrainSize) {
    fn(0, total);
    return;
  }
  rt_ns::ParallelFor(total, fn);
}

// Returns the two matrix dimensions of a 2-D ``Gemm`` operand, throwing when
// the tensor is not rank 2.
void Require2D(const Tensor &t, const char *name, std::size_t &rows, std::size_t &cols) {
  if (t.shape.size() != 2) {
    throw std::invalid_argument(std::string("onnx_light_cpu::GemmKernel: input ") + name +
                                " must be a 2-D matrix.");
  }
  rows = static_cast<std::size_t>(t.shape[0]);
  cols = static_cast<std::size_t>(t.shape[1]);
}

// Resolves ONNX unidirectional broadcasting without materializing C.
GemmBroadcast ResolveBiasLayout(const Tensor &c, std::size_t M, std::size_t N) {
  const std::size_t rank = c.shape.size();
  std::size_t c_rows = 1;
  std::size_t c_cols = 1;
  if (rank == 1) {
    c_cols = static_cast<std::size_t>(c.shape[0]);
  } else if (rank == 2) {
    c_rows = static_cast<std::size_t>(c.shape[0]);
    c_cols = static_cast<std::size_t>(c.shape[1]);
  } else if (rank != 0) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: bias C must have rank 0, 1 or 2.");
  }
  if (!(c_rows == M || c_rows == 1) || !(c_cols == N || c_cols == 1)) {
    throw std::invalid_argument(
        "onnx_light_cpu::GemmKernel: bias C is not broadcastable to the output shape.");
  }
  if (c_rows == 1 && c_cols == 1) {
    return GemmBroadcast::kScalar;
  }
  if (c_rows == 1) {
    return GemmBroadcast::kRow;
  }
  if (c_cols == 1) {
    return GemmBroadcast::kColumn;
  }
  return GemmBroadcast::kMatrix;
}

// Widens a FLOAT16 (``is_bfloat16 == false``) or BFLOAT16 (``is_bfloat16 ==
// true``) tensor's raw 16-bit elements into a fresh ``float32`` buffer,
// running the per-element bit decode across the shared thread pool for large
// tensors.
std::vector<float> WidenHalfLike(const Tensor &t, bool is_bfloat16) {
  const std::uint16_t *bits = reinterpret_cast<const std::uint16_t *>(t.bytes());
  const std::size_t n = static_cast<std::size_t>(t.element_count());
  std::vector<float> out(n);
  float *dst = out.data();
  ParallelForHalfConversion(n, [bits, dst, is_bfloat16](std::int64_t begin, std::int64_t end) {
    if (is_bfloat16) {
      for (std::int64_t i = begin; i < end; ++i) {
        dst[i] = rt_ns::Bfloat16BitsToFloat(bits[i]);
      }
    } else {
      for (std::int64_t i = begin; i < end; ++i) {
        dst[i] = rt_ns::Float16BitsToFloat(bits[i]);
      }
    }
  });
  return out;
}

} // namespace

// Shared implementation for the ``Run`` and both ``operator()`` overloads.
// ``c`` is null for the no-bias overload; otherwise it points at the bias
// tensor. When ``cache`` is non-null the FP32/FP64 paths reuse or rebuild a
// keyed immutable :cpp:class:`GemmPlan` instead of re-deriving the algorithm,
// blocking, and thread count on every run (Roadmap PR06.4).
Tensor GemmKernel::Compute(const Tensor &a, const Tensor &b, const Tensor *c, float alpha,
                           float beta, bool trans_a, bool trans_b, RuntimeContext *rt,
                           GemmPlanCache *cache) {
  if (a.data_type != b.data_type) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: A and B must share the same dtype.");
  }
  if (c != nullptr && c->data_type != a.data_type) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: bias C dtype must match A and B.");
  }

  std::size_t a_rows = 0, a_cols = 0, b_rows = 0, b_cols = 0;
  Require2D(a, "A", a_rows, a_cols);
  Require2D(b, "B", b_rows, b_cols);

  const std::size_t M = trans_a ? a_cols : a_rows;
  const std::size_t K = trans_a ? a_rows : a_cols;
  const std::size_t K_b = trans_b ? b_cols : b_rows;
  const std::size_t N = trans_b ? b_rows : b_cols;
  if (K != K_b) {
    throw std::invalid_argument(
        "onnx_light_cpu::GemmKernel: inner dimensions of A and B mismatch.");
  }

  const Shape out_shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)};
  const bool has_bias = c != nullptr && beta != 0.0f;

  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT: {
    std::optional<GemmPlan<float>> transient;
    const GemmPlan<float> *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuild<float>(a.data_type, trans_a, trans_b, M, N, K, alpha);
    } else {
      transient.emplace(GemmPlanOptions<float>{trans_a, trans_b, M, N, K, alpha, 0.0f, {}});
      plan = &*transient;
    }
    GemmEpilogue<float> epilogue;
    if (has_bias) {
      epilogue.bias = c->AsFloat();
      epilogue.bias_layout = ResolveBiasLayout(*c, M, N);
      epilogue.beta = beta;
    }
    const std::size_t n_bytes = M * N * sizeof(float);
    Tensor y = rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes,
                                       rt != nullptr ? rt->allocator() : nullptr);
    plan->Execute(a.AsFloat(), b.AsFloat(), epilogue, y.AsFloat());
    return y;
  }
  case DataType::DOUBLE: {
    std::optional<GemmPlan<double>> transient;
    const GemmPlan<double> *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuild<double>(a.data_type, trans_a, trans_b, M, N, K,
                                        static_cast<double>(alpha));
    } else {
      transient.emplace(
          GemmPlanOptions<double>{trans_a, trans_b, M, N, K, static_cast<double>(alpha), 0.0, {}});
      plan = &*transient;
    }
    GemmEpilogue<double> epilogue;
    if (has_bias) {
      epilogue.bias = c->AsDouble();
      epilogue.bias_layout = ResolveBiasLayout(*c, M, N);
      epilogue.beta = static_cast<double>(beta);
    }
    const std::size_t n_bytes = M * N * sizeof(double);
    Tensor y = rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes,
                                       rt != nullptr ? rt->allocator() : nullptr);
    plan->Execute(a.AsDouble(), b.AsDouble(), epilogue, y.AsDouble());
    return y;
  }
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    // No native half-precision micro-kernel: convert A and B to float32 while
    // packing them into the micro-kernel panels (no full-tensor widening),
    // accumulate the reduction in float32 and round the result back down. This
    // keeps the reduction in float32 precision, matching the common "compute in
    // fp32, store in fp16" convention used by most fp16/bf16 GEMM backends.
    const bool is_bfloat16 = static_cast<DataType>(a.data_type) == DataType::BFLOAT16;
    const auto *a_bits = reinterpret_cast<const std::uint16_t *>(a.bytes());
    const auto *b_bits = reinterpret_cast<const std::uint16_t *>(b.bytes());

    std::vector<float> c_f32;
    GemmEpilogue<float> epilogue;
    if (has_bias) {
      c_f32 = WidenHalfLike(*c, is_bfloat16);
      epilogue.bias = c_f32.data();
      epilogue.bias_layout = ResolveBiasLayout(*c, M, N);
      epilogue.beta = beta;
    }

    const std::size_t n_bytes = M * N * sizeof(std::uint16_t);
    Tensor y = rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes,
                                       rt != nullptr ? rt->allocator() : nullptr);
    epilogue.output_conversion =
        is_bfloat16 ? GemmOutputConversion::kBFloat16 : GemmOutputConversion::kFloat16;
    epilogue.converted_output = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
    std::vector<float> y_f32(M * N);
    GemmHalfWithEpilogue(is_bfloat16, trans_a, trans_b, M, N, K, alpha, a_bits, b_bits, epilogue,
                         y_f32.data());
    return y;
  }
  default:
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: unsupported data type " +
                                std::to_string(a.data_type) +
                                ", only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

Tensor GemmKernel::operator()(const Tensor &a, const Tensor &b, const Tensor &c, float alpha,
                              float beta, bool trans_a, bool trans_b, RuntimeContext *rt) const {
  return Compute(a, b, &c, alpha, beta, trans_a, trans_b, rt, nullptr);
}

Tensor GemmKernel::operator()(const Tensor &a, const Tensor &b, float alpha, bool trans_a,
                              bool trans_b, RuntimeContext *rt) const {
  return Compute(a, b, nullptr, alpha, 0.0f, trans_a, trans_b, rt, nullptr);
}

void GemmKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor *c = rt_ns::GetOptionalInput(node, 2, rt.tensors());
  const float alpha = rt_ns::GetAttributeFloatOrDefault(node, "alpha", 1.0f);
  const float beta = rt_ns::GetAttributeFloatOrDefault(node, "beta", 1.0f);
  const bool trans_a = rt_ns::GetAttributeIntOrDefault(node, "transA", 0) != 0;
  const bool trans_b = rt_ns::GetAttributeIntOrDefault(node, "transB", 0) != 0;
  Tensor y = Compute(a, b, c, alpha, beta, trans_a, trans_b, &rt, plan_cache_.get());
  rt_ns::SetOutput(node, 0, std::move(y), rt);
}

void RegisterGemmKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<GemmKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Gemm entry with the SIMD-accelerated kernel for the CPU device.
  rt_ns::RegisterKernelFn("", "Gemm", sym_ns::Device::kCPU, std::move(factory));
}

} // namespace onnx_light_cpu
