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
/// Supports rank-3/rank-4 FP32, FP16, and BF16 MHA/GQA/MQA. Streaming
/// invocations may consume tensor ``past`` inputs and v24
/// ``nonpad_kv_seqlen`` directly; observable ``present`` and
/// ``qk_matmul_output`` outputs use the materialized FP32 path.
class AttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit AttentionKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Attention";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Direct invocation used by backend test cases.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &q,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &k,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &v,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *mask,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_k = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_v = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *nonpad_kv_seqlen = nullptr) const;
};

void RegisterAttentionKernel();

} // namespace onnx_light_cpu
