// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// SIMD-accelerated onnx-light kernel for the ONNX ``Exp`` operator.
///
/// ``ExpKernel`` derives from onnx-light's
/// :cpp:class:`onnx_light::core::runtime::KernelBase` so it plugs into the
/// runtime exactly like a built-in kernel: the dispatch table constructs it
/// once per node and calls :cpp:func:`Run` on every execution. The computation
/// is delegated to the SIMD ``ExpFloat*`` routines declared in
/// ``onnx_light_cpu/impl/math/math_kernels.h`` (runtime AVX2/SSE2 dispatch) for
/// ``float32``/``float64`` directly; ``float16`` and ``bfloat16`` use vector
/// conversion blocks around the same float32 approximation.
class ExpKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr std::uint32_t kTuningAbi = 2;

  ExpKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
            const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static void RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningKey
  TuningKey(int32_t element_type) const override;
  void
  Configure(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningParameters &parameters) override;

  /// Library-qualified name identifying this kernel, recorded through
  /// :cpp:func:`RecordKernelUsage` on every :cpp:func:`Run` so callers can
  /// tell the onnx-light-cpu kernel apart from onnx-light's built-in ``Exp``.
  static constexpr const char *kName = "onnx_light_cpu::Exp";

  /// Reads the node's single input, computes the elementwise exponential and
  /// stores the single output back into ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh output tensor and writes ``exp(x)`` into it. When ``rt``
  /// is non-null its allocator backs the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Writes ``exp(x)`` into the caller-supplied ``output`` tensor, whose
  /// ``data_type``, ``shape`` and buffer size must already match ``x``.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  UnaryExecutionTuning tuning_ = kDefaultExpLogExecutionTuning;
  bool tuning_configured_ = false;
};

/// SIMD-accelerated onnx-light kernel for the ONNX ``Log`` operator.
///
/// ``LogKernel`` mirrors :cpp:class:`ExpKernel`, delegating to the SIMD
/// ``LogFloat*`` routines for ``float32``/``float64``, a scalar exact FP16
/// path, and vector conversion blocks for ``bfloat16``.
class LogKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr std::uint32_t kTuningAbi = 2;

  LogKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
            const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static void RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningKey
  TuningKey(int32_t element_type) const override;
  void
  Configure(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningParameters &parameters) override;

  /// Library-qualified name identifying this kernel, recorded through
  /// :cpp:func:`RecordKernelUsage` on every :cpp:func:`Run` so callers can
  /// tell the onnx-light-cpu kernel apart from onnx-light's built-in ``Log``.
  static constexpr const char *kName = "onnx_light_cpu::Log";

  /// Reads the node's single input, computes the elementwise natural logarithm
  /// and stores the single output back into ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh output tensor and writes ``log(x)`` into it. When ``rt``
  /// is non-null its allocator backs the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Writes ``log(x)`` into the caller-supplied ``output`` tensor, whose
  /// ``data_type``, ``shape`` and buffer size must already match ``x``.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  UnaryExecutionTuning tuning_ = kDefaultExpLogExecutionTuning;
  bool tuning_configured_ = false;
};

/// Registers the onnx-light-cpu ``Exp`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// After this call, every ``Exp`` node dispatched by onnx-light's runtime
/// (``RunNode`` / ``RuntimeSession``, and therefore any model executed through
/// ``ReferenceEvaluator``) resolves to :cpp:class:`ExpKernel`. Registering
/// under the default ONNX domain replaces the built-in ``Exp`` entry.
void RegisterExpKernel();

/// Registers the onnx-light-cpu ``Log`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device, replacing the built-in ``Log``
/// entry with :cpp:class:`LogKernel`.
void RegisterLogKernel();

} // namespace onnx_light_cpu
