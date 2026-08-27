// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Distance metrics accepted by the ``com.microsoft`` ``CDist`` operator's
/// ``metric`` attribute (see the ONNX Runtime contrib op contract). Only the
/// two metrics onnx-light-cpu implements are represented here; every other
/// string accepted by the ``scipy.spatial.distance.cdist`` metric list is
/// rejected by the kernel before it ever reaches these primitives.
enum class CDistMetric {
  kSqEuclidean,
  kEuclidean,
};

/// Row-parallel execution tuning for :cpp:func:`CDistFloat32WithTuning` /
/// :cpp:func:`CDistFloat64WithTuning`, following the same conventions as
/// ``UnaryExecutionTuning`` (onnx_light_cpu/impl/math/unary_execution_tuning.h):
/// thresholds and target block sizes are expressed in bytes and converted to a
/// row count via the byte size of a single ``A`` row (``n`` elements), so
/// blocking scales with how much of ``A`` (and, transitively, ``B``) one
/// dispatched block actually touches.
struct CDistExecutionTuning {
  // Zero disables executor dispatch: every row runs on the calling thread.
  std::size_t parallel_threshold_bytes = 0;
  std::size_t target_block_bytes = 1;
  // Zero uses every participant made available by the session executor.
  std::size_t max_participants = 0;
  // Prefer the session executor's shared cost model when it is available.
  bool use_cost_model = true;
  // Zero leaves the participant count to the executor cost model.
  std::size_t preferred_participants = 0;

  bool operator==(const CDistExecutionTuning &) const = default;
};

inline constexpr CDistExecutionTuning kDefaultCDistFloat32ExecutionTuning{128 * 1024, 16 * 1024, 0,
                                                                          true, 0};
inline constexpr CDistExecutionTuning kDefaultCDistFloat64ExecutionTuning{256 * 1024, 32 * 1024, 0,
                                                                          true, 0};

/// Computes the pairwise distance matrix between the rows of ``a`` (``m x n``)
/// and ``b`` (``k x n``), both row-major, writing the ``m x k`` row-major
/// result into ``c``. For every output element ``c[i, j]`` the sum of squared
/// differences ``sum((a[i, :] - b[j, :]) ** 2)`` is accumulated directly (no
/// ``|a|^2 + |b|^2 - 2 a.b`` expansion) for numerical stability, then square
/// rooted when ``metric`` is :cpp:enumerator:`CDistMetric::kEuclidean`. Uses
/// :cpp:var:`kDefaultCDistFloat32ExecutionTuning`.
void CDistFloat32(const float *a, const float *b, float *c, std::size_t m, std::size_t k,
                  std::size_t n, CDistMetric metric);
void CDistFloat32WithTuning(const float *a, const float *b, float *c, std::size_t m, std::size_t k,
                            std::size_t n, CDistMetric metric, const CDistExecutionTuning &tuning);

/// ``float64`` counterpart of :cpp:func:`CDistFloat32`. Uses
/// :cpp:var:`kDefaultCDistFloat64ExecutionTuning`.
void CDistFloat64(const double *a, const double *b, double *c, std::size_t m, std::size_t k,
                  std::size_t n, CDistMetric metric);
void CDistFloat64WithTuning(const double *a, const double *b, double *c, std::size_t m,
                            std::size_t k, std::size_t n, CDistMetric metric,
                            const CDistExecutionTuning &tuning);

} // namespace onnx_light_cpu
