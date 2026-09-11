// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/concat_kernel.h"

#include "onnx_light_cpu/impl/tensor/concat_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_lib/common/safe_math.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

namespace {

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::Concat: ") + message);
}

int64_t Multiply(int64_t left, int64_t right) {
  int64_t result;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(left, right, &result)) {
    Invalid("shape or byte arithmetic overflow.");
  }
  return result;
}

int64_t Add(int64_t left, int64_t right) {
  int64_t result;
  if (ONNX_LIGHT_NAMESPACE::checked_add_overflow(left, right, &result)) {
    Invalid("concatenated axis overflow.");
  }
  return result;
}

std::size_t Size(int64_t value) { return ONNX_LIGHT_NAMESPACE::safe_cast_to_size(value, Invalid); }

int64_t Count(const Shape &shape) {
  bool empty = false;
  for (int64_t dim : shape) {
    if (dim < 0) {
      Invalid("negative tensor dimension.");
    }
    empty |= dim == 0;
  }
  return empty ? 0 : shape.product(0, shape.size(), "Concat");
}

void ValidateBuffer(const Tensor &tensor, std::size_t bytes) {
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    Invalid("tensor buffer size does not match shape.");
  }
}

struct ConcatPlan {
  Shape shape;
  int32_t dtype = 0;
  int64_t outer = 0;
  std::size_t row_bytes = 0;
  std::size_t bytes = 0;
  std::vector<ConcatInput> inputs;
};

template <typename GetInput> ConcatPlan Prepare(std::size_t count, GetInput input, int64_t axis) {
  if (count == 0) {
    Invalid("at least one input is required.");
  }
  const Tensor &first = input(0);
  const int64_t rank = static_cast<int64_t>(first.shape.size());
  if (rank == 0 || axis < -rank || axis >= rank) {
    Invalid("inputs must not be scalars and axis must be in [-rank, rank).");
  }
  if (axis < 0) {
    axis += rank;
  }
  const auto a = static_cast<std::size_t>(axis);
  const auto width = static_cast<int64_t>(first.element_size());
  ConcatPlan plan;
  plan.shape = first.shape;
  plan.shape[a] = 0;
  plan.dtype = first.data_type;
  for (std::size_t i = 0; i < count; ++i) {
    const Tensor &tensor = input(i);
    if (tensor.data_type != plan.dtype || tensor.shape.size() != first.shape.size()) {
      Invalid("all inputs must have the same dtype and rank.");
    }
    const int64_t elements = Count(tensor.shape);
    for (std::size_t dim = 0; dim < tensor.shape.size(); ++dim) {
      if (dim != a && tensor.shape[dim] != first.shape[dim]) {
        Invalid("non-axis input dimensions must match.");
      }
    }
    plan.shape[a] = Add(plan.shape[a], tensor.shape[a]);
    ValidateBuffer(tensor, Size(Multiply(elements, width)));
  }
  plan.bytes = Size(Multiply(Count(plan.shape), width));
  if (plan.bytes == 0) {
    return plan;
  }
  plan.outer = plan.shape.product(0, a, "Concat outer");
  const int64_t inner = plan.shape.product(a + 1, plan.shape.size(), "Concat inner");
  const int64_t inner_bytes = Multiply(inner, width);
  plan.row_bytes = Size(Multiply(plan.shape[a], inner_bytes));
  plan.inputs.reserve(count);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const Tensor &tensor = input(i);
    const std::size_t row_bytes = Size(Multiply(tensor.shape[a], inner_bytes));
    if (row_bytes != 0) {
      plan.inputs.push_back({tensor.bytes(), row_bytes, offset});
      offset += row_bytes;
    }
  }
  return plan;
}

void Copy(const ConcatPlan &plan, Tensor &output) {
  if (output.data_type != plan.dtype || output.shape != plan.shape) {
    Invalid("output dtype or shape mismatch.");
  }
  ValidateBuffer(output, plan.bytes);
  const auto destination = reinterpret_cast<std::uintptr_t>(output.bytes());
  for (const auto &input : plan.inputs) {
    const auto source = reinterpret_cast<std::uintptr_t>(input.data);
    const std::size_t bytes = static_cast<std::size_t>(plan.outer) * input.row_bytes;
    if (source <= destination ? destination - source < bytes : source - destination < plan.bytes) {
      Invalid("output must not overlap any input.");
    }
  }
  ConcatCopy(plan.inputs, output.mutable_bytes(), plan.outer, plan.row_bytes);
}

Tensor Allocate(const ConcatPlan &plan, rt_ns::RuntimeContext *rt,
                rt_ns::RawBufferAllocator *allocator) {
  return rt ? rt->MakeOutputTensor(0, plan.dtype, plan.shape, plan.bytes)
            : rt_ns::MakeOutputTensor(plan.dtype, plan.shape, plan.bytes, allocator);
}

} // namespace

ConcatKernel::ConcatKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                           const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

Tensor ConcatKernel::operator()(std::span<const Tensor> inputs, int64_t axis,
                                rt_ns::RuntimeContext *rt) const {
  const ConcatPlan plan =
      Prepare(inputs.size(), [&](std::size_t i) -> const Tensor & { return inputs[i]; }, axis);
  Tensor output = Allocate(plan, rt, ctx_.allocator);
  Copy(plan, output);
  return output;
}

void ConcatKernel::operator()(std::span<const Tensor> inputs, int64_t axis, Tensor &output) const {
  const ConcatPlan plan =
      Prepare(inputs.size(), [&](std::size_t i) -> const Tensor & { return inputs[i]; }, axis);
  Copy(plan, output);
}

void ConcatKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  rt_ns::RequireMinInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  if (!rt_ns::FindAttribute(node, "axis")) {
    Invalid("required axis attribute is missing.");
  }
  const int64_t axis = rt_ns::GetAttributeIntOrDefault(node, "axis", 0);
  const ConcatPlan plan = Prepare(
      static_cast<std::size_t>(node.input_size()),
      [&](std::size_t i) -> const Tensor & {
        return rt_ns::GetInput(node, static_cast<int>(i), rt.tensors());
      },
      axis);
  Tensor output = Allocate(plan, &rt, ctx_.allocator);
  Copy(plan, output);
  rt_ns::SetOutput(node, 0, std::move(output), rt);
}

void RegisterConcatKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<ConcatKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Concat";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = ConcatKernel::kName;
  info.types = {DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
                DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
                DataType::INT16,      DataType::UINT16,         DataType::INT32,
                DataType::UINT32,     DataType::INT64,          DataType::UINT64,
                DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
                DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
