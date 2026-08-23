// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace ONNX_LIGHT_NAMESPACE {
class NodeProto;
} // namespace ONNX_LIGHT_NAMESPACE

namespace onnx_light_cpu {

/// CPU kernel for the ``ai.onnx::Attention`` operator (opset 23 and 24).
///
/// Roadmap PR11 scope: the stateless FP32 materialized baseline for rank-3
/// and rank-4 MHA/GQA/MQA, with ``scale``, causal, boolean and additive
/// ``attn_mask`` support. Tensor ``past``/``present`` cache, ``softcap``,
/// the observable ``qk_matmul_output``, ``nonpad_kv_seqlen``, and any type
/// other than FLOAT are not yet advertised; ``Run`` throws
/// ``std::invalid_argument`` if a node still reaches this kernel with one of
/// those features wired.
class AttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit AttentionKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Attention";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Direct invocation used by backend test cases: computes stateless FP32
  /// Attention for ``q``/``k``/``v`` (and optional ``mask``, ``nullptr`` when
  /// there is none), honoring ``is_causal``/``scale``/head-count attributes
  /// carried on ``node``. Throws ``std::invalid_argument`` for any feature
  /// outside the materialized baseline (see class documentation).
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &q,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &k,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &v,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *mask,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterAttentionKernel();

} // namespace onnx_light_cpu
