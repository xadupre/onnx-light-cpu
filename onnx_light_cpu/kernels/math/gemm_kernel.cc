// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/math/gemm/gemm_plan.h"
#include "ml_cpu_compute/math/math.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
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
  std::unique_ptr<ml_cpu_compute::GemmPlan<float>> plan_f32;
  std::unique_ptr<ml_cpu_compute::GemmPlan<double>> plan_f64;
  std::unique_ptr<ml_cpu_compute::GemmHalfPlan> plan_half;

  template <typename T>
  const ml_cpu_compute::GemmPlan<T> &GetOrBuild(int dt, bool ta, bool tb, std::size_t rows,
                                                std::size_t cols, std::size_t depth, T scale,
                                                const Tuning &tuning) {
    std::unique_ptr<ml_cpu_compute::GemmPlan<T>> &slot = Slot<T>();
    // Compare ``alpha`` against the cached plan's own ``T`` value so the
    // change-detection stays in the input precision instead of widening to
    // ``double`` (which could mask or fabricate differences for ``float``).
    const bool match = has_key && data_type == dt && trans_a == ta && trans_b == tb && m == rows &&
                       n == cols && k == depth && slot != nullptr && slot->alpha() == scale;
    if (!match) {
      slot = std::make_unique<ml_cpu_compute::GemmPlan<T>>(
          ml_cpu_compute::GemmPlanOptions<T>{ta,
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
  const ml_cpu_compute::GemmHalfPlan &GetOrBuildHalf(bool is_bfloat16, int dt, bool ta, bool tb,
                                                     std::size_t rows, std::size_t cols,
                                                     std::size_t depth, float scale,
                                                     const Tuning &tuning) {
    const bool match = has_key && data_type == dt && trans_a == ta && trans_b == tb && m == rows &&
                       n == cols && k == depth && plan_half != nullptr &&
                       plan_half->alpha() == scale;
    if (!match) {
      plan_half =
          std::make_unique<ml_cpu_compute::GemmHalfPlan>(ml_cpu_compute::GemmHalfPlanOptions{
              is_bfloat16,
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
  template <typename T> std::unique_ptr<ml_cpu_compute::GemmPlan<T>> &Slot() {
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

// Returns the two matrix dimensions of a 2-D ``Gemm`` operand, throwing when
// the tensor is not rank 2.
void Require2D(const Tensor &t, const char *name, std::size_t &rows, std::size_t &cols) {
  if (t.shape.size() != 2) {
    throw std::invalid_argument(std::string("onnx_light_cpu::GemmKernel: input ") + name +
                                " must be a 2-D matrix.");
  }
  rows = ml_cpu_compute::CheckedDimension(t.shape[0], "GemmKernel", name);
  cols = ml_cpu_compute::CheckedDimension(t.shape[1], "GemmKernel", name);
}

// Resolves the bias layout within each output matrix without materializing C.
ml_cpu_compute::GemmBroadcast ResolveBiasLayout(const Tensor &c, const Shape &output_shape) {
  const std::size_t rank = c.shape.size();
  if (rank > output_shape.size()) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: bias C rank exceeds output rank.");
  }
  const std::size_t first_axis = output_shape.size() - rank;
  for (std::size_t i = 0; i < rank; ++i) {
    ml_cpu_compute::CheckedDimension(c.shape[i], "GemmKernel", "bias C");
    if (c.shape[i] != 1 && c.shape[i] != output_shape[first_axis + i]) {
      throw std::invalid_argument(
          "onnx_light_cpu::GemmKernel: bias C is not broadcastable to the output shape.");
    }
  }
  std::size_t c_rows = 1;
  std::size_t c_cols = 1;
  if (rank >= 1) {
    c_cols = static_cast<std::size_t>(c.shape[rank - 1]);
  }
  if (rank >= 2) {
    c_rows = static_cast<std::size_t>(c.shape[rank - 2]);
  }
  if (c_rows == 1 && c_cols == 1) {
    return ml_cpu_compute::GemmBroadcast::kScalar;
  }
  if (c_rows == 1) {
    return ml_cpu_compute::GemmBroadcast::kRow;
  }
  if (c_cols == 1) {
    return ml_cpu_compute::GemmBroadcast::kColumn;
  }
  return ml_cpu_compute::GemmBroadcast::kMatrix;
}

std::size_t BiasBatchOffset(const Tensor &c, const Shape &output_shape, std::size_t batch) {
  const std::size_t batch_rank = output_shape.size() > 1 ? output_shape.size() - 2 : 0;
  if (batch_rank == 0 || c.shape.size() <= 2) {
    return 0;
  }
  const std::size_t first_axis = output_shape.size() - c.shape.size();
  std::size_t stride = 1;
  for (std::size_t i = c.shape.size(); i > 0; --i) {
    if (first_axis + i - 1 < batch_rank) {
      break;
    }
    stride = ml_cpu_compute::CheckedMultiply(stride, static_cast<std::size_t>(c.shape[i - 1]),
                                             "GemmKernel", "bias stride");
  }
  std::size_t offset = 0;
  for (std::size_t axis = batch_rank; axis-- > 0;) {
    const std::size_t dim = static_cast<std::size_t>(output_shape[axis]);
    const std::size_t coordinate = batch % dim;
    batch /= dim;
    if (axis >= first_axis) {
      const std::size_t bias_dim = static_cast<std::size_t>(c.shape[axis - first_axis]);
      if (bias_dim != 1) {
        offset = ml_cpu_compute::CheckedAdd(
            offset,
            ml_cpu_compute::CheckedMultiply(coordinate, stride, "GemmKernel", "bias offset"),
            "GemmKernel", "bias offset");
      }
      stride = ml_cpu_compute::CheckedMultiply(stride, bias_dim, "GemmKernel", "bias stride");
    }
  }
  return offset;
}

void ValidateGemmBuffer(const Tensor &tensor, std::size_t element_size, const char *name) {
  const auto count = ml_cpu_compute::CheckedShapeIndexProduct(tensor.shape, "GemmKernel", name);
  const auto elements = ml_cpu_compute::CheckedDimension(count, "GemmKernel", name);
  const auto bytes = ml_cpu_compute::CheckedByteSize(elements, element_size, "GemmKernel", name);
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    throw std::invalid_argument(std::string("onnx_light_cpu::GemmKernel: input ") + name +
                                " buffer size does not match its shape.");
  }
}

template <typename T> void FillZeroDepth(T *output, std::size_t count, T alpha) {
  const T value = alpha * T(0);
  ml_cpu_compute::ExecuteRanges(static_cast<std::int64_t>(count), 1.0,
                                [=](std::int64_t begin, std::int64_t end) {
                                  std::fill(output + begin, output + end, value);
                                });
}

template <typename T>
void ExecuteGemmBatches(const ml_cpu_compute::GemmPlan<T> &plan, const Tensor &a, const Tensor &b,
                        const Tensor *c, const Shape &output_shape, std::size_t batches,
                        std::size_t a_stride, std::size_t y_stride,
                        ml_cpu_compute::GemmBroadcast bias_layout, T beta, T *output) {
  if (y_stride == 0) {
    return;
  }
  const T *a_data = a.As<T>();
  const T *b_data = b.As<T>();
  const T *bias = c != nullptr && beta != T(0) ? c->As<T>() : nullptr;
  ml_cpu_compute::GemmEpilogue<T> epilogue;
  epilogue.bias_layout = bias_layout;
  epilogue.beta = beta;
  for (std::size_t batch = 0; batch < batches; ++batch) {
    const T *a_batch = a_stride == 0 ? a_data : a_data + batch * a_stride;
    T *y_batch = output + batch * y_stride;
    epilogue.bias = bias == nullptr ? nullptr : bias + BiasBatchOffset(*c, output_shape, batch);
    if (output_shape.size() != 2 && plan.k() == 0) {
      // The GEMM zero-depth shortcut omits alpha, but Gemm-29 evaluates alpha * 0.
      ml_cpu_compute::ValidateGemmEpilogue(plan.m(), plan.n(), epilogue);
      FillZeroDepth(y_batch, y_stride, plan.alpha());
      ml_cpu_compute::ApplyGemmEpilogue(plan.m(), plan.n(), epilogue, y_batch);
    } else if (output_shape.size() == 2 || bias == nullptr) {
      plan.Execute(a_batch, b_data, epilogue, y_batch);
    } else {
      // Avoid materializing an expanded broadcast bias for each batch.
      ml_cpu_compute::ValidateGemmEpilogue(plan.m(), plan.n(), epilogue);
      plan.Execute(a_batch, b_data, nullptr, y_batch);
      ml_cpu_compute::ApplyGemmEpilogue(plan.m(), plan.n(), epilogue, y_batch);
    }
  }
}

} // namespace

// Shared implementation for the ``Run`` and both ``operator()`` overloads.
// ``c`` is null for the no-bias overload; otherwise it points at the bias
// tensor. When ``cache`` is non-null the FP32/FP64 paths reuse or rebuild a
// keyed immutable :cpp:class:`GemmPlan` instead of re-deriving the algorithm,
// blocking, and thread count on every run (Roadmap PR06.4).
Tensor GemmKernel::Compute(const Tensor &a, const Tensor &b, const Tensor *c, float alpha,
                           float beta, bool trans_a, bool trans_b, RuntimeContext *rt,
                           GemmPlanCache *cache, const Tuning &tuning) const {
  if (a.data_type != b.data_type) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: A and B must share the same dtype.");
  }
  if (c != nullptr && c->data_type != a.data_type) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: bias C dtype must match A and B.");
  }

  const std::size_t rank = a.shape.size();
  if (rank == 0 || (rank != 2 && ctx_.opset.version < 29)) {
    throw std::invalid_argument(
        "onnx_light_cpu::GemmKernel: input A must be rank 2 before opset 29 and rank >= 1 after.");
  }
  std::size_t a_rows =
      rank > 1 ? ml_cpu_compute::CheckedDimension(a.shape[rank - 2], "GemmKernel", "input A") : 1;
  const std::size_t a_cols =
      ml_cpu_compute::CheckedDimension(a.shape[rank - 1], "GemmKernel", "input A");
  std::size_t b_rows = 0, b_cols = 0;
  Require2D(b, "B", b_rows, b_cols);

  trans_a = trans_a && rank > 1;
  const std::size_t M = trans_a ? a_cols : a_rows;
  const std::size_t K = trans_a ? a_rows : a_cols;
  const std::size_t K_b = trans_b ? b_cols : b_rows;
  const std::size_t N = trans_b ? b_rows : b_cols;
  if (K != K_b) {
    throw std::invalid_argument(
        "onnx_light_cpu::GemmKernel: inner dimensions of A and B mismatch.");
  }

  Shape out_shape = a.shape;
  if (trans_a) {
    std::swap(out_shape[rank - 2], out_shape[rank - 1]);
  }
  out_shape[rank - 1] = static_cast<std::int64_t>(N);
  const auto output_count =
      ml_cpu_compute::CheckedShapeIndexProduct(out_shape, "GemmKernel", "output shape");
  const std::size_t output_elements =
      ml_cpu_compute::CheckedDimension(output_count, "GemmKernel", "output shape");
  Shape batch_shape;
  for (std::size_t i = 0; i < (rank > 1 ? rank - 2 : 0); ++i) {
    batch_shape.push_back(a.shape[i]);
  }
  const auto batch_count =
      ml_cpu_compute::CheckedShapeIndexProduct(batch_shape, "GemmKernel", "batch shape");
  const std::size_t batches =
      ml_cpu_compute::CheckedDimension(batch_count, "GemmKernel", "batch shape");
  const std::size_t a_stride =
      output_elements == 0
          ? 0
          : ml_cpu_compute::CheckedMultiply(a_rows, a_cols, "GemmKernel", "input matrix size");
  const std::size_t y_stride =
      output_elements == 0
          ? 0
          : ml_cpu_compute::CheckedMultiply(M, N, "GemmKernel", "output matrix size");
  const bool has_bias = c != nullptr && beta != 0.0f;
  const auto bias_layout = c != nullptr && (has_bias || ctx_.opset.version >= 29)
                               ? ResolveBiasLayout(*c, out_shape)
                               : ml_cpu_compute::GemmBroadcast::kNone;
  if (rank != 2) {
    if (!IsSupportedElementType(a.data_type)) {
      throw std::invalid_argument("onnx_light_cpu::GemmKernel: unsupported data type " +
                                  std::to_string(a.data_type) +
                                  ", only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
    }
    const std::size_t element_size = a.data_type == DataType::FLOAT    ? sizeof(float)
                                     : a.data_type == DataType::DOUBLE ? sizeof(double)
                                                                       : sizeof(std::uint16_t);
    ValidateGemmBuffer(a, element_size, "A");
    ValidateGemmBuffer(b, element_size, "B");
    if (c != nullptr) {
      ValidateGemmBuffer(*c, element_size, "C");
    }
  }

  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT: {
    std::optional<ml_cpu_compute::GemmPlan<float>> transient;
    const ml_cpu_compute::GemmPlan<float> *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuild<float>(a.data_type, trans_a, trans_b, M, N, K, alpha, tuning);
    } else {
      transient.emplace(
          ml_cpu_compute::GemmPlanOptions<float>{trans_a,
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
    const std::size_t n_bytes = ml_cpu_compute::CheckedByteSize(output_elements, sizeof(float),
                                                                "GemmKernel", "output byte size");
    Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, out_shape, n_bytes)
                             : rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes, nullptr);
    ExecuteGemmBatches(*plan, a, b, c, out_shape, batches, a_stride, y_stride, bias_layout, beta,
                       y.AsFloat());
    return y;
  }
  case DataType::DOUBLE: {
    std::optional<ml_cpu_compute::GemmPlan<double>> transient;
    const ml_cpu_compute::GemmPlan<double> *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuild<double>(a.data_type, trans_a, trans_b, M, N, K,
                                        static_cast<double>(alpha), tuning);
    } else {
      transient.emplace(
          ml_cpu_compute::GemmPlanOptions<double>{trans_a,
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
    const std::size_t n_bytes = ml_cpu_compute::CheckedByteSize(output_elements, sizeof(double),
                                                                "GemmKernel", "output byte size");
    Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, out_shape, n_bytes)
                             : rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes, nullptr);
    ExecuteGemmBatches(*plan, a, b, c, out_shape, batches, a_stride, y_stride, bias_layout,
                       static_cast<double>(beta), y.AsDouble());
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

    std::optional<ml_cpu_compute::GemmHalfPlan> transient;
    const ml_cpu_compute::GemmHalfPlan *plan = nullptr;
    if (cache != nullptr) {
      plan = &cache->GetOrBuildHalf(is_bfloat16, a.data_type, trans_a, trans_b, M, N, K, alpha,
                                    tuning);
    } else {
      transient.emplace(ml_cpu_compute::GemmHalfPlanOptions{
          is_bfloat16,
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

    const std::size_t n_bytes = ml_cpu_compute::CheckedByteSize(
        output_elements, sizeof(std::uint16_t), "GemmKernel", "output byte size");
    Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, out_shape, n_bytes)
                             : rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes, nullptr);
    if (output_elements == 0) {
      return y;
    }
    ml_cpu_compute::GemmEpilogue<float> epilogue;
    epilogue.output_conversion =
        has_bias ? ml_cpu_compute::GemmOutputConversion::kNone
                 : (is_bfloat16 ? ml_cpu_compute::GemmOutputConversion::kBFloat16
                                : ml_cpu_compute::GemmOutputConversion::kFloat16);
    rt_ns::detail::TemporaryTypedBuffer<float> y_f32(
        y_stride, rt != nullptr ? rt->execution_allocator() : nullptr, "Gemm FP32 workspace");
    auto *y_bits = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
    for (std::size_t batch = 0; batch < batches; ++batch) {
      const auto *a_batch = a_stride == 0 ? a_bits : a_bits + batch * a_stride;
      auto *y_batch = y_bits + batch * y_stride;
      epilogue.converted_output = y_batch;
      if (rank != 2 && K == 0) {
        FillZeroDepth(y_f32.data(), y_stride, alpha);
        ml_cpu_compute::ApplyGemmEpilogue(M, N, epilogue, y_f32.data());
      } else {
        plan->Execute(a_batch, b_bits, epilogue, y_f32.data());
      }
      if (has_bias) {
        const auto *bias_bits = reinterpret_cast<const std::uint16_t *>(c->bytes());
        bias_bits += BiasBatchOffset(*c, out_shape, batch);
        ml_cpu_compute::GemmAddHalfBiasConvertFloat32(y_f32.data(), bias_bits, y_batch, M, N, beta,
                                                      bias_layout, is_bfloat16);
      }
    }
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
  const RuntimeExecutionScope execution_scope(rt);
  return Compute(a, b, &c, alpha, beta, trans_a, trans_b, rt, nullptr, tuning_);
}

Tensor GemmKernel::operator()(const Tensor &a, const Tensor &b, float alpha, bool trans_a,
                              bool trans_b, RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope(rt);
  return Compute(a, b, nullptr, alpha, 0.0f, trans_a, trans_b, rt, nullptr, tuning_);
}

void GemmKernel::Run(RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope(&rt);
  rt.RecordKernelUsage(kName);
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
  NodeKernelFn factory =
      [](const NodeProto &node, RuntimeContext &,
         const rt_ns::KernelContext &kernel_context) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<GemmKernel>(kernel_context);
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
