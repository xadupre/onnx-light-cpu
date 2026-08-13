// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace onnx_light_cpu {

/// Computational path selected for a matrix multiplication plan.
enum class GemmAlgorithm {
  kGeneral,
  kDirect,
  kSkinnyM,
  kSkinnyN,
  kSplitK,
};

/// Cache and register blocking selected for a matrix multiplication plan.
struct GemmBlocking {
  std::size_t mc = 0;
  std::size_t nc = 0;
  std::size_t kc = 0;
  std::size_t mr = 0;
  std::size_t nr = 0;
};

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
};

/// Reusable prepared rank-2 general matrix multiplication.
///
/// The plan owns constant B when supplied, records the selected blocking and
/// typed execution entry point, and can be reused for multiple A/C tensors.
/// Step 1 deliberately delegates arithmetic to the existing Gemm kernel; later
/// roadmap steps can replace the selected function without changing callers.
template <typename T> class GemmPlan {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "GemmPlan supports float and double.");

  explicit GemmPlan(const GemmPlanOptions<T> &options);

  void Execute(const T *a, const T *b, const T *c, T *y) const;
  void Execute(const T *a, const T *c, T *y) const;

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
                            const T *, T, const T *, T *);

  bool trans_a_;
  bool trans_b_;
  std::size_t m_;
  std::size_t n_;
  std::size_t k_;
  T alpha_;
  T beta_;
  GemmAlgorithm algorithm_;
  GemmBlocking blocking_;
  std::size_t useful_threads_;
  bool has_constant_b_;
  std::vector<T> constant_b_;
  KernelFn kernel_;
};

/// Prepared rank-2 MatMul foundation.
///
/// Rank promotion and batch broadcasting are intentionally deferred to the
/// complete shape adapter in roadmap step 2.
template <typename T> class MatMulPlan {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MatMulPlan supports float and double.");

  MatMulPlan(std::size_t m, std::size_t n, std::size_t k, bool trans_a = false,
             bool trans_b = false, std::span<const T> constant_b = {});

  void Execute(const T *a, const T *b, T *y) const;
  void Execute(const T *a, T *y) const;

  const GemmPlan<T> &gemm_plan() const noexcept { return gemm_plan_; }

private:
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
