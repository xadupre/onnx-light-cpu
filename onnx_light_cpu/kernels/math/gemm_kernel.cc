// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
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

namespace {

constexpr const char *kBlockingMc = "blocking.mc";
constexpr const char *kBlockingNc = "blocking.nc";
constexpr const char *kBlockingKc = "blocking.kc";
constexpr const char *kCompactBlockingMc = "compact_blocking.mc";
constexpr const char *kCompactBlockingNc = "compact_blocking.nc";
constexpr const char *kCompactBlockingKc = "compact_blocking.kc";
constexpr const char *kMaximumParticipants = "parallel.maximum_threads";

constexpr std::array<int32_t, 4> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16)};

std::atomic<std::int64_t> &ActiveGemmInstances() {
  static std::atomic<std::int64_t> count{0};
  return count;
}

rt_ns::KernelTuningKey MakeTuningKey(int32_t element_type) {
  return {"onnx_light_cpu",      "Gemm", "simd_dispatch", element_type, sym_ns::Device::kCPU,
          GemmKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name : {kBlockingMc, kBlockingNc, kBlockingKc, kCompactBlockingMc,
                           kCompactBlockingNc, kCompactBlockingKc, kMaximumParticipants}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("Gemm ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(int32_t element_type) {
  return {MakeTuningKey(element_type),
          {{kBlockingMc, int64_t{0}},
           {kBlockingNc, int64_t{0}},
           {kBlockingKc, int64_t{0}},
           {kCompactBlockingMc, int64_t{0}},
           {kCompactBlockingNc, int64_t{0}},
           {kCompactBlockingKc, int64_t{0}},
           {kMaximumParticipants, int64_t{0}}}};
}

bool IsSupportedElementType(int32_t element_type) {
  return std::find(kSupportedElementTypes.begin(), kSupportedElementTypes.end(), element_type) !=
         kSupportedElementTypes.end();
}

} // namespace

// Immutable-plan cache backing the FP32/FP64/FP16/BF16 operator path (Roadmap
// PR06.4 and PR07.2). The plan encodes the selected algorithm, blocking, thread
// count, and (unused here) constant-B packing derived from the dtype/shape/
// attributes; it is only rebuilt when the key changes so those selections are
// not re-derived per run. FP16/BF16 share the same key but use a dedicated
// float32-accumulating :cpp:class:`GemmHalfPlan`.
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
  std::unique_ptr<GemmHalfPlan> plan_half;

  template <typename T>
  const GemmPlan<T> &GetOrBuild(int dt, bool ta, bool tb, std::size_t rows, std::size_t cols,
                                std::size_t depth, T scale, const Tuning &tuning) {
    std::unique_ptr<GemmPlan<T>> &slot = Slot<T>();
    // Compare ``alpha`` against the cached plan's own ``T`` value so the
    // change-detection stays in the input precision instead of widening to
    // ``double`` (which could mask or fabricate differences for ``float``).
    const bool match = has_key && data_type == dt && trans_a == ta && trans_b == tb && m == rows &&
                       n == cols && k == depth && slot != nullptr && slot->alpha() == scale;
    if (!match) {
      slot =
          std::make_unique<GemmPlan<T>>(GemmPlanOptions<T>{ta,
                                                           tb,
                                                           rows,
                                                           cols,
                                                           depth,
                                                           scale,
                                                           T(0),
                                                           {},
                                                           {tuning.mc, tuning.nc, tuning.kc, 0, 0},
                                                           tuning.maximum_participants});
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

  // FP16/BF16 counterpart of :cpp:func:`GetOrBuild`. ``is_bfloat16`` selects the
  // decode; ``dt`` (FLOAT16 vs BFLOAT16) keeps the key distinct across types.
  const GemmHalfPlan &GetOrBuildHalf(bool is_bfloat16, int dt, bool ta, bool tb, std::size_t rows,
                                     std::size_t cols, std::size_t depth, float scale,
                                     const Tuning &tuning) {
    const bool match = has_key && data_type == dt && trans_a == ta && trans_b == tb && m == rows &&
                       n == cols && k == depth && plan_half != nullptr &&
                       plan_half->alpha() == scale;
    if (!match) {
      plan_half = std::make_unique<GemmHalfPlan>(
          GemmHalfPlanOptions{is_bfloat16,
                              ta,
                              tb,
                              rows,
                              cols,
                              depth,
                              scale,
                              {tuning.mc, tuning.nc, tuning.kc, 0, 0},
                              {tuning.compact_mc, tuning.compact_nc, tuning.compact_kc, 0, 0},
                              tuning.maximum_participants});
      has_key = true;
      data_type = dt;
      trans_a = ta;
      trans_b = tb;
      m = rows;
      n = cols;
      k = depth;
    }
    return *plan_half;
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
    : rt_ns::KernelBase(ctx), plan_cache_(std::make_unique<GemmPlanCache>()) {
  ActiveGemmInstances().fetch_add(1, std::memory_order_relaxed);
}

GemmKernel::~GemmKernel() { ActiveGemmInstances().fetch_sub(1, std::memory_order_relaxed); }

std::int64_t GemmKernel::ActiveInstanceCountForTesting() noexcept {
  return ActiveGemmInstances().load(std::memory_order_relaxed);
}

void GemmKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (int32_t element_type : kSupportedElementTypes) {
      rt_ns::RegisterKernelTuningSchema(
          rt_ns::KernelTuningSchema(MakeTuningDefaults(element_type), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey GemmKernel::TuningKey(int32_t element_type) const {
  return IsSupportedElementType(element_type) ? MakeTuningKey(element_type)
                                              : rt_ns::KernelTuningKey{};
}

void GemmKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!IsSupportedElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey(parameters.key.element_type)) {
    throw std::invalid_argument("Gemm tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {static_cast<std::size_t>(parameters.Get<int64_t>(kBlockingMc)),
             static_cast<std::size_t>(parameters.Get<int64_t>(kBlockingNc)),
             static_cast<std::size_t>(parameters.Get<int64_t>(kBlockingKc)),
             static_cast<std::size_t>(parameters.Get<int64_t>(kCompactBlockingMc)),
             static_cast<std::size_t>(parameters.Get<int64_t>(kCompactBlockingNc)),
             static_cast<std::size_t>(parameters.Get<int64_t>(kCompactBlockingKc)),
             static_cast<std::size_t>(parameters.Get<int64_t>(kMaximumParticipants))};
  plan_cache_ = std::make_unique<GemmPlanCache>();
}

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
// running the per-element bit decode through the session executor for large
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
                           GemmPlanCache *cache, const Tuning &tuning) {
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
      plan = &cache->GetOrBuild<float>(a.data_type, trans_a, trans_b, M, N, K, alpha, tuning);
    } else {
      transient.emplace(GemmPlanOptions<float>{trans_a,
                                               trans_b,
                                               M,
                                               N,
                                               K,
                                               alpha,
                                               0.0f,
                                               {},
                                               {tuning.mc, tuning.nc, tuning.kc, 0, 0},
                                               tuning.maximum_participants});
      plan = &*transient;
    }
    GemmEpilogue<float> epilogue;
    if (has_bias) {
      epilogue.bias = c->AsFloat();
      epilogue.bias_layout = ResolveBiasLayout(*c, M, N);
      epilogue.beta = beta;
    }
    const std::size_t n_bytes = M * N * sizeof(float);
    Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, out_shape, n_bytes)
                             : rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes, nullptr);
    plan->Execute(a.AsFloat(), b.AsFloat(), epilogue, y.AsFloat());
    return y;
  }
  case DataType::DOUBLE: {
    std::optional<GemmPlan<double>> transient;
    const GemmPlan<double> *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuild<double>(a.data_type, trans_a, trans_b, M, N, K,
                                        static_cast<double>(alpha), tuning);
    } else {
      transient.emplace(GemmPlanOptions<double>{trans_a,
                                                trans_b,
                                                M,
                                                N,
                                                K,
                                                static_cast<double>(alpha),
                                                0.0,
                                                {},
                                                {tuning.mc, tuning.nc, tuning.kc, 0, 0},
                                                tuning.maximum_participants});
      plan = &*transient;
    }
    GemmEpilogue<double> epilogue;
    if (has_bias) {
      epilogue.bias = c->AsDouble();
      epilogue.bias_layout = ResolveBiasLayout(*c, M, N);
      epilogue.beta = static_cast<double>(beta);
    }
    const std::size_t n_bytes = M * N * sizeof(double);
    Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, out_shape, n_bytes)
                             : rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes, nullptr);
    plan->Execute(a.AsDouble(), b.AsDouble(), epilogue, y.AsDouble());
    return y;
  }
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    // No native half-precision micro-kernel: convert A and B to float32 while
    // packing them into the micro-kernel panels (no full-tensor widening),
    // accumulate the reduction in float32 and round the result back down. This
    // keeps the reduction in float32 precision, matching the common "compute in
    // fp32, store in fp16" convention used by most fp16/bf16 GEMM backends. The
    // immutable plan caches the algorithm, blocking, and thread count so they
    // are not re-derived per run (Roadmap PR07.2).
    const bool is_bfloat16 = static_cast<DataType>(a.data_type) == DataType::BFLOAT16;
    const auto *a_bits = reinterpret_cast<const std::uint16_t *>(a.bytes());
    const auto *b_bits = reinterpret_cast<const std::uint16_t *>(b.bytes());

    std::optional<GemmHalfPlan> transient;
    const GemmHalfPlan *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuildHalf(is_bfloat16, a.data_type, trans_a, trans_b, M, N, K, alpha,
                                    tuning);
    } else {
      transient.emplace(
          GemmHalfPlanOptions{is_bfloat16,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              alpha,
                              {tuning.mc, tuning.nc, tuning.kc, 0, 0},
                              {tuning.compact_mc, tuning.compact_nc, tuning.compact_kc, 0, 0},
                              tuning.maximum_participants});
      plan = &*transient;
    }

    std::vector<float> c_f32;
    GemmEpilogue<float> epilogue;
    if (has_bias) {
      c_f32 = WidenHalfLike(*c, is_bfloat16);
      epilogue.bias = c_f32.data();
      epilogue.bias_layout = ResolveBiasLayout(*c, M, N);
      epilogue.beta = beta;
    }

    const std::size_t n_bytes = M * N * sizeof(std::uint16_t);
    Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, out_shape, n_bytes)
                             : rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes, nullptr);
    epilogue.output_conversion =
        is_bfloat16 ? GemmOutputConversion::kBFloat16 : GemmOutputConversion::kFloat16;
    epilogue.converted_output = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
    rt_ns::detail::TemporaryTypedBuffer<float> y_f32(
        M * N, rt != nullptr ? rt->execution_allocator() : nullptr, "Gemm FP32 workspace");
    plan->Execute(a_bits, b_bits, epilogue, y_f32.data());
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
  return Compute(a, b, &c, alpha, beta, trans_a, trans_b, rt, nullptr, tuning_);
}

Tensor GemmKernel::operator()(const Tensor &a, const Tensor &b, float alpha, bool trans_a,
                              bool trans_b, RuntimeContext *rt) const {
  return Compute(a, b, nullptr, alpha, 0.0f, trans_a, trans_b, rt, nullptr, tuning_);
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
  Tensor y = Compute(a, b, c, alpha, beta, trans_a, trans_b, &rt, plan_cache_.get(), tuning_);
  rt_ns::SetOutput(node, 0, std::move(y), rt);
}

void RegisterGemmKernel() {
  GemmKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<GemmKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Gemm entry with the SIMD-accelerated kernel for the CPU device.
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Gemm";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = GemmKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
