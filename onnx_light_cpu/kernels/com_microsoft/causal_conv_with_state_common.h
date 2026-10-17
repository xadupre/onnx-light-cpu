// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu::detail {

struct CausalConvWithStatePlan {
  std::size_t batch, channels, length, width, pad;
  ONNX_LIGHT_NAMESPACE::core::runtime::Shape state_shape;
  std::size_t output_bytes, state_bytes;
};

CausalConvWithStatePlan
PlanCausalConvWithState(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
                        const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &weight,
                        const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias,
                        const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_state,
                        const std::string &activation, bool channels_last, std::int64_t dilation);

template <typename Kernel>
void RunCausalConvWithState(Kernel &kernel, const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                            ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  rt.RecordKernelUsage(Kernel::kName);
  if (node.input_size() < 2 || node.input_size() > 4 || node.input(0).empty() ||
      node.input(1).empty()) {
    throw std::invalid_argument("CausalConvWithState: expected two to four inputs.");
  }
  rt_ns::RequireOutputCount(node, 2);
  if (node.output(0).empty() || node.output(1).empty() ||
      rt_ns::GetAttributeIntOrDefault(node, "ndim", 1) != 1 ||
      rt_ns::GetAttributeIntOrDefault(node, "state_window", 0) != 0 ||
      (rt_ns::GetAttributeIntOrDefault(node, "channels_last", 0) != 0 &&
       rt_ns::GetAttributeIntOrDefault(node, "channels_last", 0) != 1)) {
    throw std::invalid_argument(
        "CausalConvWithState: requires two outputs, ndim=1 and state_window=0.");
  }
  auto [output, state] =
      kernel(rt_ns::GetInput(node, 0, rt.tensors()), rt_ns::GetInput(node, 1, rt.tensors()),
             rt_ns::GetOptionalInput(node, 2, rt.tensors()),
             rt_ns::GetOptionalInput(node, 3, rt.tensors()),
             rt_ns::GetAttributeStringOrDefault(node, "activation", "none"),
             rt_ns::GetAttributeIntOrDefault(node, "channels_last", 0) == 1,
             rt_ns::GetAttributeIntOrDefault(node, "dilation", 1), &rt);
  rt_ns::SetOutput(node, 0, std::move(output), rt);
  rt_ns::SetOutput(node, 1, std::move(state), rt);
}

} // namespace onnx_light_cpu::detail
