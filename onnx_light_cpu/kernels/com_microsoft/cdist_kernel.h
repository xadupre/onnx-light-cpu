// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <string>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// Scalar onnx-light-cpu kernel for the ``com.microsoft`` ``CDist`` contrib
/// operator.
///
/// ``CDistKernel`` derives from onnx-light's
/// :cpp:class:`onnx_light::core::runtime::KernelBase` so it plugs into the
/// runtime exactly like a built-in kernel: the dispatch table constructs it
/// once per node and calls :cpp:func:`Run` on every execution. Given a rank-2
/// ``A`` (``M x N``) and a rank-2 ``B`` (``K x N``), it computes the ``M x K``
/// pairwise distance matrix using a direct sum of squared differences (see
/// :cpp:func:`CDistFloat32`) for either the ``sqeuclidean`` (default) or
/// ``euclidean`` ``metric`` attribute; every other ``scipy.spatial.distance``
/// metric name accepted by the ONNX Runtime contract is rejected since
/// onnx-light-cpu only implements these two. The computation is delegated to
/// the scalar, cache-friendly ``CDist*`` routines declared in
/// ``onnx_light_cpu/impl/com_microsoft/cdist.h``, scheduled row-by-row through
/// ``ExecuteRanges``/``ExecuteCostedRanges`` (no explicit SIMD).
class CDistKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::CDist";
  static constexpr std::uint32_t kTuningAbi = 1;

  CDistKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
              const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static void RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningKey
  TuningKey(int32_t element_type) const override;
  void
  Configure(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningParameters &parameters) override;

  /// Reads the node's ``A`` and ``B`` inputs together with the ``metric``
  /// attribute, computes the pairwise distance matrix and stores the single
  /// output back into ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh ``M x K`` output tensor and writes the pairwise
  /// distance matrix between ``a``'s and ``b``'s rows into it. ``metric`` must
  /// be ``"sqeuclidean"`` (the default) or ``"euclidean"``. When ``rt`` is
  /// non-null its allocator backs the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const std::string &metric = "sqeuclidean",
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Writes the pairwise distance matrix into the caller-supplied ``output``
  /// tensor, whose ``data_type`` and ``shape`` must already match ``(M, K)``.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b, const std::string &metric,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  CDistExecutionTuning tuning_ = kDefaultCDistFloat32ExecutionTuning;
  bool tuning_configured_ = false;
};

/// Registers the onnx-light-cpu ``CDist`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device, under the ``com.microsoft``
/// domain (opset version 1).
void RegisterCDistKernel();

} // namespace onnx_light_cpu
