// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/nonzero_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_lib/common/safe_math.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

namespace {

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::NonZero: ") + message);
}

} // namespace

NonZeroKernel::NonZeroKernel(const rt_ns::KernelContext &ctx) : KernelBase(ctx) {}

NonZeroKernel::NonZeroKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                             const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

Tensor NonZeroKernel::operator()(const Tensor &x, rt_ns::RuntimeContext *rt) const {
  if (x.data_type != DataType::BOOL) {
    const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::NonZero reference{ctx_};
    return reference(x, rt);
  }
  const int64_t total = x.shape.product(0, x.shape.size(), "NonZero input");
  const std::size_t input_bytes = ONNX_LIGHT_NAMESPACE::safe_cast_to_size(total, Invalid);
  if (x.size_bytes() != input_bytes || (input_bytes != 0 && x.bytes() == nullptr)) {
    Invalid("tensor buffer size does not match shape.");
  }

  const uint8_t *input = x.bytes();
  int64_t count = 0;
  for (int64_t i = 0; i < total; ++i) {
    count += input[i] != 0;
  }

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  const rt_ns::Shape output_shape{rank, count};
  int64_t elements;
  int64_t bytes;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(rank, count, &elements) ||
      ONNX_LIGHT_NAMESPACE::checked_mul_overflow(elements, int64_t{sizeof(int64_t)}, &bytes)) {
    Invalid("output byte size overflow.");
  }
  const std::size_t output_bytes = ONNX_LIGHT_NAMESPACE::safe_cast_to_size(bytes, Invalid);
  Tensor output =
      rt ? rt->MakeOutputTensor(0, DataType::INT64, output_shape, output_bytes)
         : rt_ns::MakeOutputTensor(DataType::INT64, output_shape, output_bytes, ctx_.allocator);
  // Scalars have no coordinates: the ONNX output shape is [0, count].
  if (output_bytes == 0) {
    return output;
  }
  int64_t *indices = output.AsInt64();
  int64_t column = 0;
  for (int64_t i = 0; i < total; ++i) {
    if (input[i] == 0) {
      continue;
    }
    int64_t flat = i;
    for (int64_t axis = rank; axis > 0; --axis) {
      const int64_t dim = x.shape[axis - 1];
      indices[(axis - 1) * count + column] = flat % dim;
      flat /= dim;
    }
    ++column;
  }
  return output;
}

void NonZeroKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

void RegisterNonZeroKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<NonZeroKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "NonZero";
  info.kernel_name = NonZeroKernel::kName;
  info.types = {DataType::BOOL};
  info.since_version = 9;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
