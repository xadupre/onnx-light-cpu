// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/elementwise/variadic_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

VariadicOperator ParseOperator(std::string_view op_type) {
  if (op_type == "Sum") {
    return VariadicOperator::kSum;
  }
  if (op_type == "Mean") {
    return VariadicOperator::kMean;
  }
  if (op_type == "Min") {
    return VariadicOperator::kMin;
  }
  if (op_type == "Max") {
    return VariadicOperator::kMax;
  }
  throw std::invalid_argument("onnx_light_cpu::VariadicElementwiseKernel: unsupported operator.");
}

VariadicElementwisePlan MakePlan(VariadicOperator op, const rt_ns::Tensors &inputs) {
  std::vector<std::int32_t> types;
  std::vector<std::vector<std::int64_t>> shapes;
  types.reserve(inputs.size());
  shapes.reserve(inputs.size());
  for (const rt_ns::Tensor &input : inputs) {
    types.push_back(input.data_type);
    shapes.emplace_back(input.shape.begin(), input.shape.end());
  }
  return VariadicElementwisePlan(op, types, shapes);
}

std::vector<const void *> InputPointers(const rt_ns::Tensors &inputs) {
  std::vector<const void *> pointers;
  pointers.reserve(inputs.size());
  for (const rt_ns::Tensor &input : inputs) {
    pointers.push_back(input.bytes());
  }
  return pointers;
}

std::vector<rt_ns::DataType> MinMaxTypes() {
  return {rt_ns::DataType::FLOAT,  rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
          rt_ns::DataType::INT8,   rt_ns::DataType::INT16,  rt_ns::DataType::INT32,
          rt_ns::DataType::INT64,  rt_ns::DataType::UINT8,  rt_ns::DataType::UINT16,
          rt_ns::DataType::UINT32, rt_ns::DataType::UINT64};
}

} // namespace

VariadicElementwiseKernel::VariadicElementwiseKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                                     const rt_ns::KernelContext &ctx)
    : rt_ns::KernelBase(ctx), op_(ParseOperator(node.op_type())) {
  set_node(node);
}

rt_ns::Tensor VariadicElementwiseKernel::operator()(const rt_ns::Tensors &inputs,
                                                    rt_ns::RuntimeContext *rt) const {
  const VariadicElementwisePlan plan = MakePlan(op_, inputs);
  const rt_ns::Shape output_shape(
      std::vector<std::int64_t>(plan.output_shape().begin(), plan.output_shape().end()));
  const auto output_type = static_cast<rt_ns::DataType>(plan.data_type());
  const std::size_t output_bytes = plan.element_count() * plan.element_size();
  rt_ns::Tensor output =
      rt != nullptr ? rt->MakeOutputTensor(0, output_type, output_shape, output_bytes)
                    : rt_ns::MakeOutputTensor(output_type, output_shape, output_bytes, nullptr);
  const std::vector<const void *> pointers = InputPointers(inputs);
  plan.Execute(pointers, output.mutable_bytes());
  return output;
}

void VariadicElementwiseKernel::operator()(const rt_ns::Tensors &inputs,
                                           rt_ns::Tensor &output) const {
  const VariadicElementwisePlan plan = MakePlan(op_, inputs);
  const rt_ns::Shape output_shape(
      std::vector<std::int64_t>(plan.output_shape().begin(), plan.output_shape().end()));
  if (output.data_type != static_cast<std::int32_t>(plan.data_type()) ||
      output.shape != output_shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::VariadicElementwiseKernel: output tensor metadata mismatch.");
  }
  const std::vector<const void *> pointers = InputPointers(inputs);
  plan.Execute(pointers, output.mutable_bytes());
}

void VariadicElementwiseKernel::Run(rt_ns::RuntimeContext &rt) {
  const auto &node = *node_;
  RecordKernelUsage(std::string("onnx_light_cpu::") + ToString(op_));
  rt_ns::RequireMinInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::Tensors inputs;
  inputs.reserve(node.input_size());
  for (int input = 0; input < node.input_size(); ++input) {
    inputs.push_back(rt_ns::GetInput(node, input, rt.tensors()));
  }
  rt_ns::SetOutput(node, 0, (*this)(inputs, &rt), rt);
}

void RegisterVariadicElementwiseKernels() {
  for (const VariadicOperator op : {VariadicOperator::kSum, VariadicOperator::kMean,
                                    VariadicOperator::kMin, VariadicOperator::kMax}) {
    const std::string op_type = ToString(op);
    rt_ns::NodeKernelFn factory =
        [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
           rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
      return std::make_unique<VariadicElementwiseKernel>(node, rt.kernel_ctx());
    };
    KernelRegistration info;
    info.domain = "";
    info.op_type = op_type;
    info.device = sym_ns::Device::kCPU;
    info.kernel_name = std::string("onnx_light_cpu::") + op_type;
    info.types = op == VariadicOperator::kSum || op == VariadicOperator::kMean
                     ? std::vector<rt_ns::DataType>{rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE}
                     : MinMaxTypes();
    info.since_version = 13;
    RegisterKernel(std::move(info), std::move(factory));
  }
}

} // namespace onnx_light_cpu
