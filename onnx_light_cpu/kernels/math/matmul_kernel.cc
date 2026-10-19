// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "cpu_compute/math/gemm/gemm_plan.h"
#include "cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
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

template <typename T>
Tensor ComputeFloating(const Tensor &a, const Tensor &b, RuntimeContext *rt,
                       const cpu_compute::MatMulPlan<T> &plan) {
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

Tensor ComputeHalf(const Tensor &a, const Tensor &b, RuntimeContext *rt, bool bfloat16,
                   const cpu_compute::MatMulPlan<float> &plan) {
  const auto *a_bits = reinterpret_cast<const std::uint16_t *>(a.bytes());
  const auto *b_bits = reinterpret_cast<const std::uint16_t *>(b.bytes());
  const Shape output_shape = OutputShape(plan.output_shape());
  const std::size_t output_elements = ElementCount(plan.output_shape());
  Tensor y = rt != nullptr
                 ? rt->MakeOutputTensor(0, a.data_type, output_shape,
                                        output_elements * sizeof(std::uint16_t))
                 : rt_ns::MakeOutputTensor(a.data_type, output_shape,
                                           output_elements * sizeof(std::uint16_t), nullptr);
  const cpu_compute::GemmPlan<float> &float_plan = plan.gemm_plan();
  const cpu_compute::GemmHalfPlan half_plan(cpu_compute::GemmHalfPlanOptions{
      bfloat16, float_plan.trans_a(), float_plan.trans_b(), float_plan.m(), float_plan.n(),
      float_plan.k(), float_plan.alpha()});
  cpu_compute::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = bfloat16 ? cpu_compute::GemmOutputConversion::kBFloat16
                                        : cpu_compute::GemmOutputConversion::kFloat16;
  epilogue.converted_output = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
  rt_ns::detail::TemporaryTypedBuffer<float> output(
      output_elements, rt != nullptr ? rt->execution_allocator() : nullptr,
      "MatMul FP32 workspace");
  plan.ExecuteHalf(half_plan, a_bits, b_bits, epilogue, output.data());
  return y;
}

} // namespace

struct MatMulKernel::MatMulPlanCache {
  int data_type = 0;
  std::vector<std::size_t> a_shape;
  std::vector<std::size_t> b_shape;
  std::unique_ptr<cpu_compute::MatMulPlan<float>> plan_f32;
  std::unique_ptr<cpu_compute::MatMulPlan<double>> plan_f64;

  template <typename T>
  const cpu_compute::MatMulPlan<T> &GetOrBuild(int type,
                                               std::span<const std::size_t> current_a_shape,
                                               std::span<const std::size_t> current_b_shape) {
    std::unique_ptr<cpu_compute::MatMulPlan<T>> &slot = Slot<T>();
    const bool shapes_match = a_shape.size() == current_a_shape.size() &&
                              b_shape.size() == current_b_shape.size() &&
                              std::equal(a_shape.begin(), a_shape.end(), current_a_shape.begin()) &&
                              std::equal(b_shape.begin(), b_shape.end(), current_b_shape.begin());
    if (slot == nullptr || data_type != type || !shapes_match) {
      slot = std::make_unique<cpu_compute::MatMulPlan<T>>(current_a_shape, current_b_shape);
      data_type = type;
      a_shape.assign(current_a_shape.begin(), current_a_shape.end());
      b_shape.assign(current_b_shape.begin(), current_b_shape.end());
    }
    return *slot;
  }

private:
  template <typename T> std::unique_ptr<cpu_compute::MatMulPlan<T>> &Slot() {
    if constexpr (std::is_same_v<T, float>) {
      return plan_f32;
    } else {
      static_assert(std::is_same_v<T, double>);
      return plan_f64;
    }
  }
};

MatMulKernel::MatMulKernel(const rt_ns::KernelContext &ctx)
    : KernelBase(ctx), plan_cache_(std::make_unique<MatMulPlanCache>()) {}

MatMulKernel::~MatMulKernel() = default;

Tensor MatMulKernel::Compute(const Tensor &a, const Tensor &b, RuntimeContext *rt,
                             MatMulPlanCache *cache) {
  if (a.data_type != b.data_type) {
    throw std::invalid_argument("onnx_light_cpu::MatMulKernel: inputs must share the same dtype.");
  }
  const auto a_shape = ShapeAsSize(a.shape);
  const auto b_shape = ShapeAsSize(b.shape);
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT: {
    std::optional<cpu_compute::MatMulPlan<float>> transient;
    const cpu_compute::MatMulPlan<float> *plan;
    if (cache == nullptr) {
      transient.emplace(a_shape, b_shape);
      plan = &*transient;
    } else {
      plan = &cache->GetOrBuild<float>(a.data_type, a_shape, b_shape);
    }
    return ComputeFloating<float>(a, b, rt, *plan);
  }
  case DataType::DOUBLE: {
    std::optional<cpu_compute::MatMulPlan<double>> transient;
    const cpu_compute::MatMulPlan<double> *plan;
    if (cache == nullptr) {
      transient.emplace(a_shape, b_shape);
      plan = &*transient;
    } else {
      plan = &cache->GetOrBuild<double>(a.data_type, a_shape, b_shape);
    }
    return ComputeFloating<double>(a, b, rt, *plan);
  }
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    std::optional<cpu_compute::MatMulPlan<float>> transient;
    const cpu_compute::MatMulPlan<float> *plan;
    if (cache == nullptr) {
      transient.emplace(a_shape, b_shape);
      plan = &*transient;
    } else {
      plan = &cache->GetOrBuild<float>(a.data_type, a_shape, b_shape);
    }
    return ComputeHalf(a, b, rt, static_cast<DataType>(a.data_type) == DataType::BFLOAT16, *plan);
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulKernel: unsupported data type; expected FLOAT, DOUBLE, FLOAT16 or "
        "BFLOAT16.");
  }
}

Tensor MatMulKernel::operator()(const Tensor &a, const Tensor &b, RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  return Compute(a, b, rt, nullptr);
}

void MatMulKernel::Run(RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, Compute(a, b, &rt, plan_cache_.get()), rt);
}

void RegisterMatMulKernel() {
  rt_ns::NodeKernelFn matmul =
      [](const NodeProto &node, RuntimeContext &,
         const rt_ns::KernelContext &kernel_context) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<MatMulKernel>(kernel_context);
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
