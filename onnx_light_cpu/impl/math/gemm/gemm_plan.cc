// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/parallel_for.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace onnx_light_cpu {

namespace detail {

struct MatMulDimensions {
  std::vector<std::size_t> output_shape;
  std::vector<std::size_t> batch_shape;
  std::vector<std::size_t> a_batch_strides;
  std::vector<std::size_t> b_batch_strides;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::size_t batch_count = 0;
  std::size_t b_element_count = 0;
};

} // namespace detail

namespace {

std::size_t CheckedProduct(std::size_t left, std::size_t right, const char *name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string("onnx_light_cpu::GemmPlan: ") + name +
                              " element count overflows size_t.");
  }
  return left * right;
}

std::size_t ElementCount(std::span<const std::size_t> shape, const char *name) {
  std::size_t count = 1;
  for (std::size_t dimension : shape) {
    count = CheckedProduct(count, dimension, name);
  }
  return count;
}

template <typename T> std::size_t VectorLanes() {
  switch (DetectSimdLevel()) {
  case SimdLevel::kAVX512:
    return 64 / sizeof(T);
  case SimdLevel::kAVX2:
  case SimdLevel::kAVX:
    return 32 / sizeof(T);
  case SimdLevel::kSSE2:
    return 16 / sizeof(T);
  case SimdLevel::kNone:
  default:
    return 1;
  }
}

template <typename T, GemmAlgorithm Algorithm>
void ExecuteFloatKernel(bool trans_a, bool trans_b, std::size_t m, std::size_t n, std::size_t k,
                        T alpha, const T *a, const T *b, T beta, const T *c, T *y) {
  if constexpr (std::is_same_v<T, float>) {
    detail::GemmFloat32Planned<Algorithm>(trans_a, trans_b, m, n, k, alpha, a, b, beta, c, y);
  } else {
    static_assert(std::is_same_v<T, double>);
    detail::GemmFloat64Planned<Algorithm>(trans_a, trans_b, m, n, k, alpha, a, b, beta, c, y);
  }
}

template <typename T>
auto SelectKernel(GemmAlgorithm algorithm)
    -> void (*)(bool, bool, std::size_t, std::size_t, std::size_t, T, const T *, const T *, T,
                const T *, T *) {
  switch (algorithm) {
  case GemmAlgorithm::kDirect:
    return &ExecuteFloatKernel<T, GemmAlgorithm::kDirect>;
  case GemmAlgorithm::kSkinnyM:
    return &ExecuteFloatKernel<T, GemmAlgorithm::kSkinnyM>;
  case GemmAlgorithm::kSkinnyN:
    return &ExecuteFloatKernel<T, GemmAlgorithm::kSkinnyN>;
  case GemmAlgorithm::kSplitK:
    return &ExecuteFloatKernel<T, GemmAlgorithm::kSplitK>;
  case GemmAlgorithm::kGeneral:
    return &ExecuteFloatKernel<T, GemmAlgorithm::kGeneral>;
  }
  throw std::logic_error("onnx_light_cpu::GemmPlan: unsupported Gemm algorithm.");
}

std::size_t CeilDiv(std::size_t value, std::size_t divisor) {
  return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

std::size_t UsefulThreads(std::size_t m, std::size_t n) {
  const std::size_t row_tasks = CeilDiv(m, kGemmTileM);
  const std::size_t column_tasks = CeilDiv(n, kGemmTileN);
  const std::size_t max_threads = static_cast<std::size_t>(ParallelForThreadCount());
  if (row_tasks == 0 || column_tasks == 0) {
    return 1;
  }
  if (column_tasks > max_threads / row_tasks) {
    return max_threads;
  }
  return std::max<std::size_t>(1, std::min(max_threads, row_tasks * column_tasks));
}

detail::MatMulDimensions PrepareMatMulDimensions(std::span<const std::size_t> a_shape,
                                                 std::span<const std::size_t> b_shape, bool trans_a,
                                                 bool trans_b) {
  if (a_shape.empty() || b_shape.empty()) {
    throw std::invalid_argument("onnx_light_cpu::MatMulPlan: inputs must have rank at least one.");
  }
  if ((a_shape.size() == 1 && trans_a) || (b_shape.size() == 1 && trans_b)) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulPlan: transpose is not supported for rank-1 inputs.");
  }

  const std::size_t a_rows = a_shape.size() == 1 ? 1 : a_shape[a_shape.size() - 2];
  const std::size_t a_cols = a_shape.back();
  const std::size_t b_rows = b_shape.size() == 1 ? b_shape.front() : b_shape[b_shape.size() - 2];
  const std::size_t b_cols = b_shape.size() == 1 ? 1 : b_shape.back();

  detail::MatMulDimensions dimensions;
  dimensions.m = trans_a ? a_cols : a_rows;
  dimensions.k = trans_a ? a_rows : a_cols;
  const std::size_t b_k = trans_b ? b_cols : b_rows;
  dimensions.n = trans_b ? b_rows : b_cols;
  if (dimensions.k != b_k) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulPlan: inner dimensions of A and B do not match.");
  }

  const std::size_t a_batch_rank = a_shape.size() > 2 ? a_shape.size() - 2 : 0;
  const std::size_t b_batch_rank = b_shape.size() > 2 ? b_shape.size() - 2 : 0;
  const std::size_t batch_rank = std::max(a_batch_rank, b_batch_rank);
  dimensions.batch_shape.resize(batch_rank);
  dimensions.a_batch_strides.assign(batch_rank, 0);
  dimensions.b_batch_strides.assign(batch_rank, 0);

  for (std::size_t axis = 0; axis < batch_rank; ++axis) {
    const std::size_t a_leading = batch_rank - a_batch_rank;
    const std::size_t b_leading = batch_rank - b_batch_rank;
    const std::size_t a_dimension = axis < a_leading ? 1 : a_shape[axis - a_leading];
    const std::size_t b_dimension = axis < b_leading ? 1 : b_shape[axis - b_leading];
    if (a_dimension != b_dimension && a_dimension != 1 && b_dimension != 1) {
      throw std::invalid_argument(
          "onnx_light_cpu::MatMulPlan: batch dimensions of A and B are not broadcastable.");
    }
    dimensions.batch_shape[axis] = a_dimension == 1 ? b_dimension : a_dimension;
  }

  const auto prepare_strides = [&](std::span<const std::size_t> shape, std::size_t input_batch_rank,
                                   std::size_t matrix_elements, std::vector<std::size_t> &strides) {
    std::size_t stride = matrix_elements;
    const std::size_t leading = batch_rank - input_batch_rank;
    for (std::size_t index = input_batch_rank; index-- > 0;) {
      const std::size_t axis = leading + index;
      const std::size_t dimension = shape[index];
      strides[axis] = dimension == 1 && dimensions.batch_shape[axis] != 1 ? 0 : stride;
      stride = CheckedProduct(stride, dimension, "MatMul input");
    }
  };

  const std::size_t a_matrix_elements = CheckedProduct(a_rows, a_cols, "MatMul A");
  const std::size_t b_matrix_elements = CheckedProduct(b_rows, b_cols, "MatMul B");
  prepare_strides(a_shape, a_batch_rank, a_matrix_elements, dimensions.a_batch_strides);
  prepare_strides(b_shape, b_batch_rank, b_matrix_elements, dimensions.b_batch_strides);

  dimensions.batch_count = ElementCount(dimensions.batch_shape, "MatMul batch");
  dimensions.b_element_count = ElementCount(b_shape, "MatMul B");
  dimensions.output_shape = dimensions.batch_shape;
  if (a_shape.size() != 1) {
    dimensions.output_shape.push_back(dimensions.m);
  }
  if (b_shape.size() != 1) {
    dimensions.output_shape.push_back(dimensions.n);
  }
  ElementCount(dimensions.output_shape, "MatMul output");
  return dimensions;
}

detail::MatMulDimensions PrepareRankTwoMatMulDimensions(std::size_t m, std::size_t n, std::size_t k,
                                                        bool trans_a, bool trans_b) {
  const std::array<std::size_t, 2> a_shape =
      trans_a ? std::array<std::size_t, 2>{k, m} : std::array<std::size_t, 2>{m, k};
  const std::array<std::size_t, 2> b_shape =
      trans_b ? std::array<std::size_t, 2>{n, k} : std::array<std::size_t, 2>{k, n};
  return PrepareMatMulDimensions(a_shape, b_shape, trans_a, trans_b);
}

} // namespace

namespace detail {

GemmAlgorithm SelectGemmAlgorithm(bool trans_a, bool trans_b, std::size_t m, std::size_t n,
                                  std::size_t k, std::size_t vector_lanes) {
  if (k >= 4096 && m != 0 && m <= 64 && n <= 64 / m) {
    return GemmAlgorithm::kSplitK;
  }
  if (!trans_a && !trans_b && k <= 32) {
    return GemmAlgorithm::kDirect;
  }
  if (m <= kGemmMR) {
    return GemmAlgorithm::kSkinnyM;
  }
  if (n <= vector_lanes) {
    return GemmAlgorithm::kSkinnyN;
  }
  return GemmAlgorithm::kGeneral;
}

} // namespace detail

template <typename T>
GemmPlan<T>::GemmPlan(const GemmPlanOptions<T> &options)
    : trans_a_(options.trans_a), trans_b_(options.trans_b), m_(options.m), n_(options.n),
      k_(options.k), alpha_(options.alpha), beta_(options.beta),
      algorithm_(detail::SelectGemmAlgorithm(options.trans_a, options.trans_b, options.m, options.n,
                                             options.k, VectorLanes<T>())),
      blocking_{kGemmTileM, kGemmTileN, kGemmTileK, kGemmMR, 2 * VectorLanes<T>()},
      useful_threads_(UsefulThreads(options.m, options.n)),
      has_constant_b_(!options.constant_b.empty()),
      constant_b_(options.constant_b.begin(), options.constant_b.end()),
      kernel_(SelectKernel<T>(algorithm_)) {
  CheckedProduct(m_, k_, "A");
  const std::size_t b_count = CheckedProduct(k_, n_, "B");
  CheckedProduct(m_, n_, "Y");
  if (has_constant_b_ && constant_b_.size() != b_count) {
    throw std::invalid_argument("onnx_light_cpu::GemmPlan: constant B size does not match K * N.");
  }
}

template <typename T> void GemmPlan<T>::Execute(const T *a, const T *b, const T *c, T *y) const {
  if (m_ == 0 || n_ == 0) {
    return;
  }
  if (y == nullptr) {
    throw std::invalid_argument("onnx_light_cpu::GemmPlan: Y must not be null.");
  }
  if (k_ != 0 && a == nullptr) {
    throw std::invalid_argument("onnx_light_cpu::GemmPlan: A must not be null when K is nonzero.");
  }
  const T *resolved_b = has_constant_b_ ? constant_b_.data() : b;
  if (k_ != 0 && resolved_b == nullptr) {
    throw std::invalid_argument("onnx_light_cpu::GemmPlan: B must not be null when K is nonzero.");
  }
  kernel_(trans_a_, trans_b_, m_, n_, k_, alpha_, a, resolved_b, beta_, c, y);
}

template <typename T> void GemmPlan<T>::Execute(const T *a, const T *c, T *y) const {
  if (!has_constant_b_) {
    throw std::logic_error(
        "onnx_light_cpu::GemmPlan: Execute without B requires a constant-B plan.");
  }
  Execute(a, nullptr, c, y);
}

template <typename T>
MatMulPlan<T>::MatMulPlan(std::span<const std::size_t> a_shape,
                          std::span<const std::size_t> b_shape, bool trans_a, bool trans_b,
                          std::span<const T> constant_b)
    : MatMulPlan(PrepareMatMulDimensions(a_shape, b_shape, trans_a, trans_b), trans_a, trans_b,
                 constant_b) {}

template <typename T>
MatMulPlan<T>::MatMulPlan(std::size_t m, std::size_t n, std::size_t k, bool trans_a, bool trans_b,
                          std::span<const T> constant_b)
    : MatMulPlan(PrepareRankTwoMatMulDimensions(m, n, k, trans_a, trans_b), trans_a, trans_b,
                 constant_b) {}

template <typename T>
MatMulPlan<T>::MatMulPlan(detail::MatMulDimensions dimensions, bool trans_a, bool trans_b,
                          std::span<const T> constant_b)
    : output_shape_(std::move(dimensions.output_shape)),
      batch_shape_(std::move(dimensions.batch_shape)),
      a_batch_strides_(std::move(dimensions.a_batch_strides)),
      b_batch_strides_(std::move(dimensions.b_batch_strides)), batch_count_(dimensions.batch_count),
      output_matrix_elements_(CheckedProduct(dimensions.m, dimensions.n, "MatMul output matrix")),
      has_constant_b_(!constant_b.empty()), constant_b_(constant_b.begin(), constant_b.end()),
      gemm_plan_(GemmPlanOptions<T>{trans_a, trans_b, dimensions.m, dimensions.n, dimensions.k}) {
  if (has_constant_b_ && constant_b_.size() != dimensions.b_element_count) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulPlan: constant B size does not match its shape.");
  }
}

template <typename T>
std::size_t MatMulPlan<T>::BatchOffset(std::size_t batch,
                                       std::span<const std::size_t> strides) const {
  std::size_t offset = 0;
  for (std::size_t axis = batch_shape_.size(); axis-- > 0;) {
    const std::size_t coordinate = batch % batch_shape_[axis];
    batch /= batch_shape_[axis];
    offset += coordinate * strides[axis];
  }
  return offset;
}

template <typename T> void MatMulPlan<T>::Execute(const T *a, const T *b, T *y) const {
  if (batch_count_ == 0 || output_matrix_elements_ == 0) {
    return;
  }
  if (y == nullptr) {
    throw std::invalid_argument("onnx_light_cpu::MatMulPlan: Y must not be null.");
  }
  if (gemm_plan_.k() != 0 && a == nullptr) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulPlan: A must not be null when K is nonzero.");
  }
  const T *resolved_b = has_constant_b_ ? constant_b_.data() : b;
  if (gemm_plan_.k() != 0 && resolved_b == nullptr) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulPlan: B must not be null when K is nonzero.");
  }

  for (std::size_t batch = 0; batch < batch_count_; ++batch) {
    const T *batch_a = a == nullptr ? nullptr : a + BatchOffset(batch, a_batch_strides_);
    const T *batch_b =
        resolved_b == nullptr ? nullptr : resolved_b + BatchOffset(batch, b_batch_strides_);
    gemm_plan_.Execute(batch_a, batch_b, nullptr, y + batch * output_matrix_elements_);
  }
}

template <typename T> void MatMulPlan<T>::Execute(const T *a, T *y) const {
  if (!has_constant_b_ && gemm_plan_.k() != 0 && batch_count_ != 0 &&
      output_matrix_elements_ != 0) {
    throw std::logic_error(
        "onnx_light_cpu::MatMulPlan: Execute without B requires a constant-B plan.");
  }
  Execute(a, nullptr, y);
}

template <typename T>
void StridedBatchedGemm(const GemmPlan<T> &plan, std::size_t batch_count, const T *a,
                        std::ptrdiff_t stride_a, const T *b, std::ptrdiff_t stride_b, const T *c,
                        std::ptrdiff_t stride_c, T *y, std::ptrdiff_t stride_y) {
  if (batch_count != 0 && (a == nullptr || y == nullptr)) {
    throw std::invalid_argument("onnx_light_cpu::StridedBatchedGemm: A and Y must not be null.");
  }
  if (batch_count != 0 && !plan.has_constant_b() && b == nullptr) {
    throw std::invalid_argument(
        "onnx_light_cpu::StridedBatchedGemm: B must not be null for a dynamic-B plan.");
  }
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const auto offset = static_cast<std::ptrdiff_t>(batch);
    plan.Execute(a + offset * stride_a, plan.has_constant_b() ? nullptr : b + offset * stride_b,
                 c == nullptr ? nullptr : c + offset * stride_c, y + offset * stride_y);
  }
}

template <typename T> void GroupedGemm(std::span<const GroupedGemmProblem<T>> problems) {
  for (const GroupedGemmProblem<T> &problem : problems) {
    if (problem.plan == nullptr) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupedGemm: every problem must reference a plan.");
    }
    problem.plan->Execute(problem.a, problem.b, problem.c, problem.y);
  }
}

template class GemmPlan<float>;
template class GemmPlan<double>;
template class MatMulPlan<float>;
template class MatMulPlan<double>;

template void StridedBatchedGemm<float>(const GemmPlan<float> &, std::size_t, const float *,
                                        std::ptrdiff_t, const float *, std::ptrdiff_t,
                                        const float *, std::ptrdiff_t, float *, std::ptrdiff_t);
template void StridedBatchedGemm<double>(const GemmPlan<double> &, std::size_t, const double *,
                                         std::ptrdiff_t, const double *, std::ptrdiff_t,
                                         const double *, std::ptrdiff_t, double *, std::ptrdiff_t);
template void GroupedGemm<float>(std::span<const GroupedGemmProblem<float>>);
template void GroupedGemm<double>(std::span<const GroupedGemmProblem<double>>);

} // namespace onnx_light_cpu
