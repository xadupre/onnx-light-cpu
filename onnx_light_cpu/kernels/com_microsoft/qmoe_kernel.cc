// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/qmoe_kernel.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

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
using rt_ns::DataType;
using rt_ns::Tensor;

MoEDataType FloatType(const Tensor &tensor) {
  switch (static_cast<DataType>(tensor.data_type)) {
  case DataType::FLOAT:
    return MoEDataType::kFloat32;
  case DataType::FLOAT16:
    return MoEDataType::kFloat16;
  case DataType::BFLOAT16:
    return MoEDataType::kBFloat16;
  default:
    throw std::invalid_argument("QMoE: floating inputs must be FLOAT, FLOAT16 or BFLOAT16.");
  }
}

void RequireShape(const Tensor &tensor, std::initializer_list<std::int64_t> shape,
                  const char *label) {
  if (tensor.shape.size() != shape.size() ||
      !std::equal(tensor.shape.begin(), tensor.shape.end(), shape.begin())) {
    throw std::invalid_argument(std::string("QMoE: invalid shape for ") + label + ".");
  }
  CheckedShapeIndexProduct(tensor.shape, "QMoE", label);
}

QMoEWeight Weight(const Tensor &tensor, const Tensor &scales, std::size_t experts,
                  std::size_t inputs, std::size_t outputs, bool output_major,
                  const QMoEAttributes &attributes, const char *label, bool rowwise = false) {
  if (static_cast<DataType>(tensor.data_type) != DataType::UINT8) {
    throw std::invalid_argument("QMoE: expert weights must have type UINT8.");
  }
  const auto rows = output_major ? outputs : inputs;
  const auto columns = output_major ? inputs : outputs;
  const auto factor = 8 / attributes.expert_weight_bits;
  const auto packed = columns / factor + (columns % factor != 0);
  RequireShape(tensor,
               {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(rows),
                static_cast<std::int64_t>(packed)},
               label);
  if (attributes.block_size == 0) {
    RequireShape(
        scales,
        {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(rowwise ? rows : columns)},
        label);
  } else {
    const auto blocks = columns / attributes.block_size + (columns % attributes.block_size != 0);
    RequireShape(scales,
                 {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(rows),
                  static_cast<std::int64_t>(blocks)},
                 label);
  }
  return {reinterpret_cast<const std::uint8_t *>(tensor.bytes()),
          scales.bytes(),
          FloatType(scales),
          output_major,
          rowwise,
          rowwise};
}

bool Fits(const Tensor &weights, const Tensor &scales, std::size_t experts, std::size_t inputs,
          std::size_t outputs, bool output_major, bool rowwise, const QMoEAttributes &attributes) {
  const auto rows = output_major ? outputs : inputs;
  const auto columns = output_major ? inputs : outputs;
  const auto factor = 8 / attributes.expert_weight_bits;
  if (weights.shape.size() != 3 || weights.shape[0] != static_cast<std::int64_t>(experts) ||
      weights.shape[1] != static_cast<std::int64_t>(rows) ||
      weights.shape[2] != static_cast<std::int64_t>(columns / factor + (columns % factor != 0))) {
    return false;
  }
  if (attributes.block_size == 0) {
    return scales.shape.size() == 2 && scales.shape[0] == static_cast<std::int64_t>(experts) &&
           scales.shape[1] == static_cast<std::int64_t>(rowwise ? rows : columns);
  }
  const auto blocks = columns / attributes.block_size + (columns % attributes.block_size != 0);
  return scales.shape.size() == 3 && scales.shape[0] == static_cast<std::int64_t>(experts) &&
         scales.shape[1] == static_cast<std::int64_t>(rows) &&
         scales.shape[2] == static_cast<std::int64_t>(blocks);
}

} // namespace

QMoEKernel::QMoEKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node, const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
  const auto k = rt_ns::GetAttributeIntOrDefault(node, "k", 1);
  const auto bits = rt_ns::GetAttributeIntOrDefault(node, "expert_weight_bits", 4);
  const auto block = rt_ns::GetAttributeIntOrDefault(node, "block_size", 0);
  if (k <= 0 || (bits != 4 && bits != 8) || block < 0) {
    throw std::invalid_argument("QMoE: invalid k, expert_weight_bits or block_size.");
  }
  if (block != 0 && (block < 16 || (block & (block - 1)) != 0)) {
    throw std::invalid_argument("QMoE: block_size must be zero or a power of two at least 16.");
  }
  attributes_.k = CheckedDimension(k, "QMoE", "k");
  attributes_.expert_weight_bits = static_cast<std::size_t>(bits);
  attributes_.block_size = CheckedDimension(block, "QMoE", "block_size");
  attributes_.activation = rt_ns::GetAttributeStringOrDefault(node, "activation_type", "relu");
  const auto fusion = rt_ns::GetAttributeIntOrDefault(node, "swiglu_fusion", 0);
  if ((attributes_.activation == "swiglu" && fusion != 1) ||
      (attributes_.activation != "swiglu" && fusion != 0) ||
      rt_ns::GetAttributeIntOrDefault(node, "use_sparse_mixer", 0) != 0) {
    throw std::invalid_argument("QMoE: unsupported activation fusion or sparse mixer.");
  }
  if (rt_ns::GetAttributeIntOrDefault(node, "weights_prepacked", 0) != 0) {
    throw std::invalid_argument("QMoE: CUDA weights_prepacked are not supported on CPU.");
  }
  const auto normalize = rt_ns::GetAttributeIntOrDefault(node, "normalize_routing_weights", 0);
  if (normalize != 0 && normalize != 1) {
    throw std::invalid_argument("QMoE: normalize_routing_weights must be 0 or 1.");
  }
  attributes_.normalize = normalize == 1;
  attributes_.alpha = rt_ns::GetAttributeFloatOrDefault(node, "activation_alpha", 1.0f);
  attributes_.beta = rt_ns::GetAttributeFloatOrDefault(node, "activation_beta", 0.0f);
  attributes_.limit = rt_ns::GetAttributeFloatOrDefault(node, "swiglu_limit",
                                                        std::numeric_limits<float>::infinity());
  if ((attributes_.activation != "relu" && attributes_.activation != "gelu" &&
       attributes_.activation != "silu" && attributes_.activation != "identity" &&
       attributes_.activation != "swiglu") ||
      !std::isfinite(attributes_.alpha) || !std::isfinite(attributes_.beta) ||
      std::isnan(attributes_.limit) || attributes_.limit < 0) {
    throw std::invalid_argument("QMoE: invalid activation_type or activation parameters.");
  }
}

Tensor QMoEKernel::operator()(const Tensor &input, const Tensor &router, const Tensor &fc1,
                              const Tensor &scales1, const Tensor *bias1, const Tensor &fc2,
                              const Tensor &scales2, const Tensor *bias2, const Tensor *fc3,
                              const Tensor *scales3, const Tensor *bias3,
                              rt_ns::RuntimeContext *rt) const {
  const auto type = FloatType(input);
  for (const auto *tensor : {&router, bias1, bias2, bias3}) {
    if (tensor && tensor->data_type != input.data_type) {
      throw std::invalid_argument("QMoE: input, router and biases must have matching types.");
    }
  }
  if ((fc3 == nullptr) != (scales3 == nullptr) || (fc3 == nullptr && bias3 != nullptr)) {
    throw std::invalid_argument("QMoE: FC3 weights and scales must be supplied together.");
  }
  if (input.shape.size() != 2 && input.shape.size() != 3) {
    throw std::invalid_argument("QMoE: input must have rank 2 or 3.");
  }
  CheckedShapeIndexProduct(input.shape, "QMoE", "input");
  CheckedShapeIndexProduct(router.shape, "QMoE", "router");
  for (const auto *tensor : {&fc1, &scales1, &fc2, &scales2, fc3, scales3}) {
    if (tensor) {
      CheckedShapeIndexProduct(tensor->shape, "QMoE", "weights and scales");
    }
  }
  const auto hidden = CheckedDimension(input.shape.back(), "QMoE", "hidden");
  const auto tokens =
      input.shape.size() == 2
          ? CheckedDimension(input.shape[0], "QMoE", "tokens")
          : CheckedDimension(CheckedIndexMultiply(input.shape[0], input.shape[1], "QMoE", "tokens"),
                             "QMoE", "tokens");
  if (hidden == 0 || router.shape.size() != 2 || fc2.shape.size() != 3 ||
      router.shape[0] != static_cast<std::int64_t>(tokens)) {
    throw std::invalid_argument("QMoE: invalid input, router or FC2 shape.");
  }
  const auto experts = CheckedDimension(router.shape[1], "QMoE", "experts");
  const bool fused = attributes_.activation == "swiglu";
  bool output_major = fused;
  bool rowwise = false;
  std::size_t intermediate = 0;
  std::size_t width = 0;
  const auto fits = [&](bool major, bool row_scales) {
    if (fc1.shape.size() != 3 || (major && fused && fc1.shape[1] % 2 != 0)) {
      return false;
    }
    const auto candidate = CheckedDimension(major ? fc1.shape[1] / (fused ? 2 : 1) : fc2.shape[1],
                                            "QMoE", "intermediate");
    const auto candidate_width = CheckedMultiply(candidate, fused ? 2 : 1, "QMoE", "FC1 width");
    if (!Fits(fc1, scales1, experts, hidden, candidate_width, major, row_scales, attributes_) ||
        !Fits(fc2, scales2, experts, candidate, hidden, major, row_scales, attributes_) ||
        (fc3 &&
         !Fits(*fc3, *scales3, experts, hidden, candidate, major, row_scales, attributes_))) {
      return false;
    }
    output_major = major;
    rowwise = row_scales;
    intermediate = candidate;
    width = candidate_width;
    return true;
  };
  // Prefer the exact producer formats: separate linears are transposed whereas
  // fused interleaved weights are output-major. Also accept CPU row-scaled tensors.
  if (!fits(fused, false) && !fits(!fused, false) &&
      !(attributes_.block_size == 0 && fits(true, true))) {
    throw std::invalid_argument("QMoE: incompatible expert weight and scale layouts.");
  }
  auto w1 = Weight(fc1, scales1, experts, hidden, width, output_major, attributes_,
                   "FC1 weights/scales", rowwise);
  auto w2 = Weight(fc2, scales2, experts, intermediate, hidden, output_major, attributes_,
                   "FC2 weights/scales", rowwise);
  QMoEWeight w3;
  if (fc3) {
    w3 = Weight(*fc3, *scales3, experts, hidden, intermediate, output_major, attributes_,
                "FC3 weights/scales", rowwise);
  }
  if (bias1) {
    RequireShape(*bias1, {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(width)},
                 "FC1 bias");
  }
  if (bias2) {
    RequireShape(*bias2, {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(hidden)},
                 "FC2 bias");
  }
  if (bias3) {
    RequireShape(*bias3,
                 {static_cast<std::int64_t>(experts), static_cast<std::int64_t>(intermediate)},
                 "FC3 bias");
  }
  const auto size =
      ComputeQMoEScratchSize(tokens, hidden, experts, intermediate, attributes_, fc3 != nullptr);
  const auto bytes = CheckedByteSize(CheckedMultiply(tokens, hidden, "QMoE", "output"),
                                     input.element_size(), "QMoE", "output bytes");
  auto *allocator = rt ? rt->execution_allocator() : ctx_.allocator;
  Tensor output = rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, bytes)
                     : rt_ns::MakeOutputTensor(input.data_type, input.shape, bytes, allocator);
  if (tokens == 0) {
    return output;
  }
  rt_ns::detail::TemporaryTypedBuffer<float> values(size.value_count, allocator,
                                                    "QMoE dequantization");
  rt_ns::detail::TemporaryTypedBuffer<float> workspace(size.moe.value_count, allocator,
                                                       "QMoE execution");
  rt_ns::detail::TemporaryTypedBuffer<MoERoute> routes(size.moe.route_count, allocator,
                                                       "QMoE routes");
  ComputeQMoE(input.bytes(), router.bytes(), w1, bias1 ? bias1->bytes() : nullptr, w2,
              bias2 ? bias2->bytes() : nullptr, fc3 ? &w3 : nullptr,
              bias3 ? bias3->bytes() : nullptr, output.mutable_bytes(), tokens, hidden, experts,
              intermediate, type, attributes_, values.data(), size.value_count,
              {routes.data(), size.moe.route_count, workspace.data(), size.moe.value_count,
               size.moe.token_blocks});
  return output;
}

void QMoEKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  if (node.input_size() < 7 || node.input_size() > 14) {
    throw std::invalid_argument("QMoE: expected seven to fourteen input slots.");
  }
  for (int index = 11; index < node.input_size(); ++index) {
    if (!node.input(index).empty()) {
      throw std::invalid_argument("QMoE: zero points are not supported by the CPU contract.");
    }
  }
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt), rt_ns::GetInput(node, 1, rt),
                           rt_ns::GetInput(node, 2, rt), rt_ns::GetInput(node, 3, rt),
                           rt_ns::GetOptionalInput(node, 4, rt.tensors()),
                           rt_ns::GetInput(node, 5, rt), rt_ns::GetInput(node, 6, rt),
                           rt_ns::GetOptionalInput(node, 7, rt.tensors()),
                           rt_ns::GetOptionalInput(node, 8, rt.tensors()),
                           rt_ns::GetOptionalInput(node, 9, rt.tensors()),
                           rt_ns::GetOptionalInput(node, 10, rt.tensors()), &rt),
                   rt);
}

void RegisterQMoEKernel() {
  rt_ns::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                   rt_ns::RuntimeContext &, const rt_ns::KernelContext &ctx) {
    return std::make_unique<QMoEKernel>(node, ctx);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "QMoE";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = QMoEKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  info.until_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
