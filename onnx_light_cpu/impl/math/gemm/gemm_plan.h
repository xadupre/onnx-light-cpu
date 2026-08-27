// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace onnx_light_cpu {

namespace detail {

struct MatMulDimensions;

} // namespace detail

/// Immutable inputs used to prepare a typed :cpp:class:`GemmPlan`.
template <typename T> struct GemmPlanOptions {
  bool trans_a = false;
  bool trans_b = false;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  T alpha = T(1);
  T beta = T(0);

  /// Optional physical B matrix copied into plan-owned storage.
  ///
  /// Its size must be ``k * n`` elements, regardless of ``trans_b``. An empty
  /// span selects a dynamic B supplied to :cpp:func:`GemmPlan::Execute`.
  std::span<const T> constant_b;
  GemmBlocking blocking;
  std::size_t maximum_participants = 0;
};

/// Reusable prepared rank-2 general matrix multiplication.
///
/// The plan owns constant B when supplied, records the selected blocking and
/// typed algorithm entry point, and can be reused for multiple A/C tensors.
template <typename T> class GemmPlan {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "GemmPlan supports float and double.");

  explicit GemmPlan(const GemmPlanOptions<T> &options);

  void Execute(const T *a, const T *b, const T *c, T *y) const;
  void Execute(const T *a, const T *c, T *y) const;

  /// Executes the prepared product and applies ``epilogue`` (broadcast bias,
  /// residual, activation, and optional FP16/BF16 narrowing) in place, reusing
  /// the plan's cached algorithm, blocking, and thread count. ``b`` supplies the
  /// dynamic right-hand matrix; a constant-B plan ignores it.
  void Execute(const T *a, const T *b, const GemmEpilogue<T> &epilogue, T *y) const;

  /// Constant-B overload of the epilogue-aware :cpp:func:`Execute`.
  void Execute(const T *a, const GemmEpilogue<T> &epilogue, T *y) const;

  bool trans_a() const noexcept { return trans_a_; }
  bool trans_b() const noexcept { return trans_b_; }
  std::size_t m() const noexcept { return m_; }
  std::size_t n() const noexcept { return n_; }
  std::size_t k() const noexcept { return k_; }
  T alpha() const noexcept { return alpha_; }
  T beta() const noexcept { return beta_; }
  GemmAlgorithm algorithm() const noexcept { return algorithm_; }
  const GemmBlocking &blocking() const noexcept { return blocking_; }
  std::size_t useful_threads() const noexcept { return useful_threads_; }
  bool has_constant_b() const noexcept { return has_constant_b_; }

private:
  using KernelFn = void (*)(bool, bool, std::size_t, std::size_t, std::size_t, T, const T *,
                            const T *, T, const T *, T *, const GemmBlocking *);

  bool trans_a_;
  bool trans_b_;
  std::size_t m_;
  std::size_t n_;
  std::size_t k_;
  T alpha_;
  T beta_;
  GemmAlgorithm algorithm_;
  std::size_t participant_limit_;
  GemmBlocking blocking_;
  std::size_t useful_threads_;
  bool has_constant_b_;
  std::vector<T> constant_b_;
  KernelFn kernel_;
};

/// Immutable inputs used to prepare a :cpp:class:`GemmHalfPlan`.
struct GemmHalfPlanOptions {
  bool is_bfloat16 = false;
  bool trans_a = false;
  bool trans_b = false;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  float alpha = 1.0f;
  GemmBlocking blocking;
  GemmBlocking compact_blocking;
  std::size_t maximum_participants = 0;
};

/// Reusable prepared rank-2 FP16/BF16 general matrix multiplication.
///
/// Native FP16/BF16 kernels keep compact panels in the source format; fallback
/// paths convert to float32 while packing or reducing. The plan records both
/// blocking profiles and the selected algorithm once. B is always dynamic.
class GemmHalfPlan {
public:
  explicit GemmHalfPlan(const GemmHalfPlanOptions &options);

  /// Executes ``alpha * op(A) @ op(B)`` into the float32 workspace ``y`` and
  /// applies ``epilogue`` (broadcast bias, residual, activation, and the
  /// FP16/BF16 output narrowing) in place, reusing the plan's cached algorithm,
  /// blocking, and thread count. ``a`` and ``b`` are raw 16-bit patterns.
  void Execute(const std::uint16_t *a, const std::uint16_t *b, const GemmEpilogue<float> &epilogue,
               float *y) const;

  bool is_bfloat16() const noexcept { return is_bfloat16_; }
  bool trans_a() const noexcept { return trans_a_; }
  bool trans_b() const noexcept { return trans_b_; }
  std::size_t m() const noexcept { return m_; }
  std::size_t n() const noexcept { return n_; }
  std::size_t k() const noexcept { return k_; }
  float alpha() const noexcept { return alpha_; }
  GemmAlgorithm algorithm() const noexcept { return algorithm_; }
  const GemmBlocking &blocking() const noexcept { return blocking_; }
  const GemmBlocking &compact_blocking() const noexcept { return compact_blocking_; }
  std::size_t useful_threads() const noexcept { return useful_threads_; }

private:
  using KernelFn = void (*)(bool, bool, bool, std::size_t, std::size_t, std::size_t, float,
                            const std::uint16_t *, const std::uint16_t *, float *,
                            const GemmBlocking *, const GemmBlocking *);

  bool is_bfloat16_;
  bool trans_a_;
  bool trans_b_;
  std::size_t m_;
  std::size_t n_;
  std::size_t k_;
  float alpha_;
  GemmAlgorithm algorithm_;
  std::size_t participant_limit_;
  GemmBlocking blocking_;
  GemmBlocking compact_blocking_;
  std::size_t useful_threads_;
  KernelFn kernel_;
};

/// Prepared ONNX MatMul shape and broadcast adapter.
template <typename T> class MatMulPlan {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MatMulPlan supports float and double.");

  MatMulPlan(std::span<const std::size_t> a_shape, std::span<const std::size_t> b_shape,
             bool trans_a = false, bool trans_b = false, std::span<const T> constant_b = {});
  MatMulPlan(std::size_t m, std::size_t n, std::size_t k, bool trans_a = false,
             bool trans_b = false, std::span<const T> constant_b = {});

  void Execute(const T *a, const T *b, T *y) const;
  void Execute(const T *a, T *y) const;
  void ExecuteHalf(const GemmHalfPlan &half_plan, const std::uint16_t *a, const std::uint16_t *b,
                   const GemmEpilogue<float> &epilogue, float *y) const;

  std::span<const std::size_t> output_shape() const noexcept { return output_shape_; }
  std::span<const std::size_t> batch_shape() const noexcept { return batch_shape_; }
  std::size_t batch_count() const noexcept { return batch_count_; }
  bool has_constant_b() const noexcept { return has_constant_b_; }
  const GemmPlan<T> &gemm_plan() const noexcept { return gemm_plan_; }

private:
  MatMulPlan(detail::MatMulDimensions dimensions, bool trans_a, bool trans_b,
             std::span<const T> constant_b);

  std::size_t BatchOffset(std::size_t batch, std::span<const std::size_t> strides) const;

  std::vector<std::size_t> output_shape_;
  std::vector<std::size_t> batch_shape_;
  std::vector<std::size_t> a_batch_strides_;
  std::vector<std::size_t> b_batch_strides_;
  std::size_t batch_count_;
  std::size_t output_matrix_elements_;
  bool has_constant_b_;
  std::vector<T> constant_b_;
  GemmPlan<T> gemm_plan_;
};

/// Executes equally-shaped products whose matrix starts are separated by
/// element strides. C may be null when the plan has no bias.
template <typename T>
void StridedBatchedGemm(const GemmPlan<T> &plan, std::size_t batch_count, const T *a,
                        std::ptrdiff_t stride_a, const T *b, std::ptrdiff_t stride_b, const T *c,
                        std::ptrdiff_t stride_c, T *y, std::ptrdiff_t stride_y);

/// One independently planned product in a grouped Gemm invocation.
template <typename T> struct GroupedGemmProblem {
  const GemmPlan<T> *plan = nullptr;
  const T *a = nullptr;
  const T *b = nullptr;
  const T *c = nullptr;
  T *y = nullptr;
};

/// Executes a heterogeneous collection of independently planned products.
template <typename T> void GroupedGemm(std::span<const GroupedGemmProblem<T>> problems);

extern template class GemmPlan<float>;
extern template class GemmPlan<double>;
extern template class MatMulPlan<float>;
extern template class MatMulPlan<double>;

extern template void StridedBatchedGemm<float>(const GemmPlan<float> &, std::size_t, const float *,
                                               std::ptrdiff_t, const float *, std::ptrdiff_t,
                                               const float *, std::ptrdiff_t, float *,
                                               std::ptrdiff_t);
extern template void StridedBatchedGemm<double>(const GemmPlan<double> &, std::size_t,
                                                const double *, std::ptrdiff_t, const double *,
                                                std::ptrdiff_t, const double *, std::ptrdiff_t,
                                                double *, std::ptrdiff_t);
extern template void GroupedGemm<float>(std::span<const GroupedGemmProblem<float>>);
extern template void GroupedGemm<double>(std::span<const GroupedGemmProblem<double>>);

} // namespace onnx_light_cpu
