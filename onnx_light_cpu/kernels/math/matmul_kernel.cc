// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using DataType = rt_ns::DataType;
using RuntimeContext = rt_ns::RuntimeContext;
using Shape = rt_ns::Shape;
using Tensor = rt_ns::Tensor;

namespace {

std::vector<std::size_t> ShapeAsSize(const Shape &shape) {
  std::vector<std::size_t> result;
  result.reserve(shape.size());
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument(
          "onnx_light_cpu::MatMulKernel: dynamic dimensions are unsupported.");
    }
    result.push_back(static_cast<std::size_t>(dimension));
  }
  return result;
}

Shape OutputShape(std::span<const std::size_t> shape) {
  Shape result;
  result.reserve(shape.size());
  for (const std::size_t dimension : shape) {
    result.push_back(static_cast<std::int64_t>(dimension));
  }
  return result;
}

std::size_t ElementCount(std::span<const std::size_t> shape) {
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

template <typename T> Tensor ComputeFloating(const Tensor &a, const Tensor &b, RuntimeContext *rt) {
  const auto a_shape = ShapeAsSize(a.shape);
  const auto b_shape = ShapeAsSize(b.shape);
  const MatMulPlan<T> plan(a_shape, b_shape);
  const Shape output_shape = OutputShape(plan.output_shape());
  const std::size_t output_elements = ElementCount(plan.output_shape());
  const std::size_t bytes = output_elements * sizeof(T);
  Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, output_shape, bytes)
                           : rt_ns::MakeOutputTensor(a.data_type, output_shape, bytes, nullptr);
  const auto *a_data = reinterpret_cast<const T *>(a.bytes());
  const auto *b_data = reinterpret_cast<const T *>(b.bytes());
  auto *y_data = reinterpret_cast<T *>(y.mutable_bytes());
  plan.Execute(a_data, b_data, y_data);
  return y;
}

Tensor ComputeHalf(const Tensor &a, const Tensor &b, RuntimeContext *rt, bool bfloat16) {
  const auto a_shape = ShapeAsSize(a.shape);
  const auto b_shape = ShapeAsSize(b.shape);
  const MatMulPlan<float> plan(a_shape, b_shape);
  std::vector<float> a_f32(a.element_count());
  std::vector<float> b_f32(b.element_count());
  const auto *a_bits = reinterpret_cast<const std::uint16_t *>(a.bytes());
  const auto *b_bits = reinterpret_cast<const std::uint16_t *>(b.bytes());
  for (std::size_t i = 0; i < a_f32.size(); ++i) {
    a_f32[i] =
        bfloat16 ? rt_ns::Bfloat16BitsToFloat(a_bits[i]) : rt_ns::Float16BitsToFloat(a_bits[i]);
  }
  for (std::size_t i = 0; i < b_f32.size(); ++i) {
    b_f32[i] =
        bfloat16 ? rt_ns::Bfloat16BitsToFloat(b_bits[i]) : rt_ns::Float16BitsToFloat(b_bits[i]);
  }
  const Shape output_shape = OutputShape(plan.output_shape());
  const std::size_t output_elements = ElementCount(plan.output_shape());
  Tensor y = rt != nullptr
                 ? rt->MakeOutputTensor(0, a.data_type, output_shape,
                                        output_elements * sizeof(std::uint16_t))
                 : rt_ns::MakeOutputTensor(a.data_type, output_shape,
                                           output_elements * sizeof(std::uint16_t), nullptr);
  std::vector<float> output(output_elements);
  plan.Execute(a_f32.data(), b_f32.data(), output.data());
  auto *y_bits = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
  for (std::size_t i = 0; i < output.size(); ++i) {
    y_bits[i] =
        bfloat16 ? rt_ns::FloatToBfloat16Bits(output[i]) : rt_ns::FloatToFloat16Bits(output[i]);
  }
  return y;
}

} // namespace

MatMulKernel::MatMulKernel(const rt_ns::KernelContext &ctx) : KernelBase(ctx) {}

Tensor MatMulKernel::operator()(const Tensor &a, const Tensor &b, RuntimeContext *rt) const {
  if (a.data_type != b.data_type) {
    throw std::invalid_argument("onnx_light_cpu::MatMulKernel: inputs must share the same dtype.");
  }
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT:
    return ComputeFloating<float>(a, b, rt);
  case DataType::DOUBLE:
    return ComputeFloating<double>(a, b, rt);
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
    return ComputeHalf(a, b, rt, static_cast<DataType>(a.data_type) == DataType::BFLOAT16);
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulKernel: unsupported data type; expected FLOAT, DOUBLE, FLOAT16 or "
        "BFLOAT16.");
  }
}

void MatMulKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(a, b, &rt), rt);
}

void RegisterMatMulKernel() {
  rt_ns::NodeKernelFn matmul = [](const NodeProto &node,
                                  RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<MatMulKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "MatMul";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = MatMulKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(matmul));
}

} // namespace onnx_light_cpu
