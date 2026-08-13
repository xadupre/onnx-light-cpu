// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/parallel_for.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace onnx_light_cpu {

namespace {

std::size_t CheckedProduct(std::size_t left, std::size_t right, const char *name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string("onnx_light_cpu::GemmPlan: ") + name +
                              " element count overflows size_t.");
  }
  return left * right;
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
auto SelectKernel(GemmAlgorithm algorithm) -> void (*)(bool, bool, std::size_t, std::size_t,
                                                       std::size_t, T, const T *, const T *, T,
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
MatMulPlan<T>::MatMulPlan(std::size_t m, std::size_t n, std::size_t k, bool trans_a, bool trans_b,
                          std::span<const T> constant_b)
    : gemm_plan_(GemmPlanOptions<T>{trans_a, trans_b, m, n, k, T(1), T(0), constant_b}) {}

template <typename T> void MatMulPlan<T>::Execute(const T *a, const T *b, T *y) const {
  gemm_plan_.Execute(a, b, nullptr, y);
}

template <typename T> void MatMulPlan<T>::Execute(const T *a, T *y) const {
  gemm_plan_.Execute(a, nullptr, y);
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
