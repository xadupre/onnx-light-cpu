// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Row-parallel execution tuning for the ``BiasGelu`` primitives below,
/// following the same conventions as ``UnaryExecutionTuning``
/// (onnx_light_cpu/impl/math/unary_execution_tuning.h): thresholds and target
/// block sizes are expressed in bytes of one broadcast row (``inner``
/// elements, i.e. the bias length) and converted to a row count so blocking
/// stays aligned with the bias broadcast boundary.
struct BiasGeluExecutionTuning {
  // Zero disables executor dispatch: every row runs on the calling thread.
  std::size_t parallel_threshold_bytes = 0;
  std::size_t target_block_bytes = 1;
  // Zero uses every participant made available by the session executor.
  std::size_t max_participants = 0;
  // Prefer the session executor's shared cost model when it is available.
  bool use_cost_model = true;
  // Zero leaves the participant count to the executor cost model.
  std::size_t preferred_participants = 0;

  bool operator==(const BiasGeluExecutionTuning &) const = default;
};

inline constexpr BiasGeluExecutionTuning kDefaultBiasGeluFloat32ExecutionTuning{
    256 * 1024, 256 * 1024, 0, false, 0};
inline constexpr BiasGeluExecutionTuning kDefaultBiasGeluFloat64ExecutionTuning{
    512 * 1024, 64 * 1024, 0, true, 0};
inline constexpr BiasGeluExecutionTuning kDefaultBiasGeluHalfExecutionTuning{128 * 1024, 32 * 1024,
                                                                             0, true, 0};

/// Computes ``output[row, col] = Gelu(a[row, col] + bias[col])`` for
/// ``outer`` broadcast rows of ``inner`` elements each (``a``/``output`` are
/// ``outer x inner`` row-major, ``bias`` has ``inner`` elements), using the
/// exact erf-based Gaussian Error Linear Unit
/// ``Gelu(z) = 0.5 * z * (1 + erf(z / sqrt(2)))``. Uses
/// :cpp:var:`kDefaultBiasGeluFloat32ExecutionTuning`.
void BiasGeluFloat32(const float *a, const float *bias, float *output, std::size_t outer,
                     std::size_t inner);
void BiasGeluFloat32WithTuning(const float *a, const float *bias, float *output, std::size_t outer,
                               std::size_t inner, const BiasGeluExecutionTuning &tuning);

/// ``float64`` counterpart of :cpp:func:`BiasGeluFloat32`. Uses
/// :cpp:var:`kDefaultBiasGeluFloat64ExecutionTuning`.
void BiasGeluFloat64(const double *a, const double *bias, double *output, std::size_t outer,
                     std::size_t inner);
void BiasGeluFloat64WithTuning(const double *a, const double *bias, double *output,
                               std::size_t outer, std::size_t inner,
                               const BiasGeluExecutionTuning &tuning);

/// ``float16`` counterpart of :cpp:func:`BiasGeluFloat32`. ``a``/``bias``/
/// ``output`` are the raw IEEE 754 half-precision bit patterns (as
/// ``uint16_t``); each row is decoded to ``float32``, evaluated, and encoded
/// back once. Uses :cpp:var:`kDefaultBiasGeluHalfExecutionTuning`.
void BiasGeluFloat16(const std::uint16_t *a, const std::uint16_t *bias, std::uint16_t *output,
                     std::size_t outer, std::size_t inner);
void BiasGeluFloat16WithTuning(const std::uint16_t *a, const std::uint16_t *bias,
                               std::uint16_t *output, std::size_t outer, std::size_t inner,
                               const BiasGeluExecutionTuning &tuning);

/// ``bfloat16`` counterpart of :cpp:func:`BiasGeluFloat16`. Uses
/// :cpp:var:`kDefaultBiasGeluHalfExecutionTuning`.
void BiasGeluBFloat16(const std::uint16_t *a, const std::uint16_t *bias, std::uint16_t *output,
                      std::size_t outer, std::size_t inner);
void BiasGeluBFloat16WithTuning(const std::uint16_t *a, const std::uint16_t *bias,
                                std::uint16_t *output, std::size_t outer, std::size_t inner,
                                const BiasGeluExecutionTuning &tuning);

} // namespace onnx_light_cpu
