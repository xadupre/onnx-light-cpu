// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/moe_kernel.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using rt_ns::DataType;
using rt_ns::Tensor;

void RequireShape(const Tensor &tensor, std::initializer_list<std::int64_t> dimensions,
                  const char *label) {
  if (tensor.shape.size() != dimensions.size() ||
      !std::equal(tensor.shape.begin(), tensor.shape.end(), dimensions.begin())) {
    throw std::invalid_argument(std::string("MoE: invalid shape for ") + label + ".");
  }
}

} // namespace

MoEKernel::MoEKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node, const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
  const auto k = rt_ns::GetAttributeIntOrDefault(node, "k", 1);
  if (k < 1) {
    throw std::invalid_argument("MoE: k must be positive.");
  }
  attributes_.k = CheckedDimension(k, "MoE", "k");
  attributes_.activation = rt_ns::GetAttributeStringOrDefault(node, "activation_type", "relu");
  if (attributes_.activation != "relu" && attributes_.activation != "gelu" &&
      attributes_.activation != "silu" && attributes_.activation != "identity" &&
      attributes_.activation != "swiglu") {
    throw std::invalid_argument("MoE: unsupported activation_type.");
  }
  const auto fusion = rt_ns::GetAttributeIntOrDefault(node, "swiglu_fusion", 0);
  if ((attributes_.activation == "swiglu" && fusion != 1) ||
      (attributes_.activation != "swiglu" && fusion != 0) ||
      rt_ns::GetAttributeIntOrDefault(node, "use_sparse_mixer", 0) != 0) {
    throw std::invalid_argument("MoE: unsupported fusion or sparse mixer.");
  }
  const auto normalize = rt_ns::GetAttributeIntOrDefault(node, "normalize_routing_weights", 0);
  if (normalize != 0 && normalize != 1) {
    throw std::invalid_argument("MoE: normalize_routing_weights must be 0 or 1.");
  }
  attributes_.normalize = normalize == 1;
  attributes_.alpha = rt_ns::GetAttributeFloatOrDefault(node, "activation_alpha", 1.0f);
  attributes_.beta = rt_ns::GetAttributeFloatOrDefault(node, "activation_beta", 0.0f);
  attributes_.limit = rt_ns::GetAttributeFloatOrDefault(node, "swiglu_limit",
                                                        std::numeric_limits<float>::infinity());
  if (!std::isfinite(attributes_.alpha) || !std::isfinite(attributes_.beta) ||
      std::isnan(attributes_.limit) || attributes_.limit < 0) {
    throw std::invalid_argument("MoE: invalid activation parameters.");
  }
}

Tensor MoEKernel::operator()(const Tensor &input, const Tensor &router, const Tensor &fc1,
                             const Tensor *bias1, const Tensor &fc2, const Tensor *bias2,
                             rt_ns::RuntimeContext *rt) const {
  const auto dtype = static_cast<DataType>(input.data_type);
  if (dtype != DataType::FLOAT && dtype != DataType::FLOAT16) {
    throw std::invalid_argument("MoE: only FLOAT and FLOAT16 are supported by the CPU kernel.");
  }
  for (const Tensor *tensor : {&router, &fc1, bias1, &fc2, bias2}) {
    if (tensor != nullptr && tensor->data_type != input.data_type) {
      throw std::invalid_argument("MoE: all input types must match.");
    }
  }
  if (input.shape.size() != 2 && input.shape.size() != 3) {
    throw std::invalid_argument("MoE: input must have rank 2 or 3.");
  }
  const std::size_t hidden = CheckedDimension(input.shape[input.shape.size() - 1], "MoE", "hidden");
  const std::size_t tokens = CheckedDimension(input.shape[0], "MoE", "tokens");
  const std::size_t rows =
      input.shape.size() == 3
          ? CheckedMultiply(tokens, CheckedDimension(input.shape[1], "MoE", "sequence"), "MoE",
                            "token count")
          : tokens;
  if (hidden == 0 || router.shape.size() != 2) {
    throw std::invalid_argument("MoE: hidden must be positive and router must have rank 2.");
  }
  const std::size_t experts = CheckedDimension(router.shape[1], "MoE", "experts");
  if (experts == 0 || attributes_.k > experts ||
      router.shape[0] != static_cast<std::int64_t>(rows) || fc2.shape.size() != 3) {
    throw std::invalid_argument("MoE: invalid router, k, or fc2 dimensions.");
  }
  const std::size_t intermediate = CheckedDimension(fc2.shape[2], "MoE", "intermediate");
  if (intermediate == 0) {
    throw std::invalid_argument("MoE: intermediate must be positive.");
  }
  const std::size_t fc1_width =
      CheckedMultiply(intermediate, attributes_.activation == "swiglu" ? 2 : 1, "MoE", "fc1 width");
  RequireShape(fc1,
               {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(fc1_width),
                static_cast<std::int64_t>(hidden)},
               "fc1_experts_weights");
  RequireShape(fc2,
               {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(hidden),
                static_cast<std::int64_t>(intermediate)},
               "fc2_experts_weights");
  if (bias1 != nullptr) {
    RequireShape(*bias1, {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(fc1_width)},
                 "fc1_experts_bias");
  }
  if (bias2 != nullptr) {
    RequireShape(*bias2, {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(hidden)},
                 "fc2_experts_bias");
  }
  CheckedProduct({experts, fc1_width, hidden}, "MoE", "fc1 elements");
  CheckedProduct({experts, hidden, intermediate}, "MoE", "fc2 elements");
  CheckedMultiply(rows, experts, "MoE", "router elements");
  const auto bytes = CheckedByteSize(CheckedMultiply(rows, hidden, "MoE", "output elements"),
                                     input.element_size(), "MoE", "output bytes");
  Tensor output = rt != nullptr
                      ? rt->MakeOutputTensor(0, input.data_type, input.shape, bytes)
                      : rt_ns::MakeOutputTensor(input.data_type, input.shape, bytes, nullptr);
  ComputeMoE(input.bytes(), router.bytes(), fc1.bytes(), bias1 ? bias1->bytes() : nullptr,
             fc2.bytes(), bias2 ? bias2->bytes() : nullptr, output.mutable_bytes(), rows, hidden,
             experts, intermediate, dtype == DataType::FLOAT16, attributes_);
  return output;
}

void MoEKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  if (node.input_size() < 5 || node.input_size() > 8 ||
      (node.input_size() > 6 && !node.input(6).empty()) ||
      (node.input_size() > 7 && !node.input(7).empty())) {
    throw std::invalid_argument("MoE: FC3 inputs are not supported on CPU.");
  }
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::SetOutput(
      node, 0,
      (*this)(rt_ns::GetInput(node, 0, rt), rt_ns::GetInput(node, 1, rt),
              rt_ns::GetInput(node, 2, rt), rt_ns::GetOptionalInput(node, 3, rt.tensors()),
              rt_ns::GetInput(node, 4, rt), rt_ns::GetOptionalInput(node, 5, rt.tensors()), &rt),
      rt);
}

void RegisterMoEKernel() {
  rt_ns::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                   rt_ns::RuntimeContext &, const rt_ns::KernelContext &ctx) {
    return std::make_unique<MoEKernel>(node, ctx);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "MoE";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = MoEKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16};
  info.since_version = 1;
  info.until_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
