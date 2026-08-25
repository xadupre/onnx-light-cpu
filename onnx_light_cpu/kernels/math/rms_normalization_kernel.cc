// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

template <typename Storage> struct FloatCodec;

template <> struct FloatCodec<float> {
  static float Load(float value) noexcept { return value; }
  static float Store(float value) noexcept { return value; }
};

template <> struct FloatCodec<double> {
  static float Load(double value) noexcept { return static_cast<float>(value); }
  static double Store(float value) noexcept { return static_cast<double>(value); }
};

struct Float16Codec {
  using Storage = std::uint16_t;
  static float Load(Storage value) noexcept { return rt_ns::Float16BitsToFloat(value); }
  static Storage Store(float value) noexcept { return rt_ns::FloatToFloat16Bits(value); }
};

struct BFloat16Codec {
  using Storage = std::uint16_t;
  static float Load(Storage value) noexcept { return rt_ns::Bfloat16BitsToFloat(value); }
  static Storage Store(float value) noexcept { return rt_ns::FloatToBfloat16Bits(value); }
};

template <typename T> struct NativeCodec {
  using Storage = T;
  static float Load(Storage value) noexcept { return FloatCodec<Storage>::Load(value); }
  static Storage Store(float value) noexcept { return FloatCodec<Storage>::Store(value); }
};

template <typename Codec>
void NormalizeRows(const Tensor &x, const Tensor &scale, Tensor &output, std::size_t outer,
                   std::size_t inner, float epsilon) {
  using Storage = typename Codec::Storage;
  const auto *x_data = reinterpret_cast<const Storage *>(x.bytes());
  const auto *scale_data = reinterpret_cast<const Storage *>(scale.bytes());
  auto *output_data = reinterpret_cast<Storage *>(output.mutable_bytes());
  rt_ns::ParallelFor(static_cast<std::int64_t>(outer), [x_data, scale_data, output_data, inner,
                                                        epsilon](std::int64_t begin,
                                                                 std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * inner;
      float sum_squares = 0.0f;
      for (std::size_t column = 0; column < inner; ++column) {
        const float value = Codec::Load(x_data[offset + column]);
        sum_squares += value * value;
      }
      const float inverse_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(inner) + epsilon);
      for (std::size_t column = 0; column < inner; ++column) {
        const float normalized =
            Codec::Load(x_data[offset + column]) * inverse_rms * Codec::Load(scale_data[column]);
        output_data[offset + column] = Codec::Store(normalized);
      }
    }
  });
}

void DispatchNormalize(const Tensor &x, const Tensor &scale, Tensor &output, std::size_t outer,
                       std::size_t inner, float epsilon) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    NormalizeRows<NativeCodec<float>>(x, scale, output, outer, inner, epsilon);
    return;
  case DataType::DOUBLE:
    NormalizeRows<NativeCodec<double>>(x, scale, output, outer, inner, epsilon);
    return;
  case DataType::FLOAT16:
    NormalizeRows<Float16Codec>(x, scale, output, outer, inner, epsilon);
    return;
  case DataType::BFLOAT16:
    NormalizeRows<BFloat16Codec>(x, scale, output, outer, inner, epsilon);
    return;
  default:
    throw std::invalid_argument("onnx_light_cpu::RMSNormalization: unsupported input data type.");
  }
}

} // namespace

Tensor RmsNormalizationKernel::operator()(const Tensor &x, const Tensor &scale, std::int64_t axis,
                                          float epsilon, std::int64_t stash_type,
                                          RuntimeContext *rt) const {
  if (x.data_type != scale.data_type) {
    throw std::invalid_argument(
        "onnx_light_cpu::RMSNormalization: X and scale data types must match.");
  }
  if (stash_type != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::RMSNormalization: only stash_type=1 is supported.");
  }
  if (epsilon < 0.0f) {
    throw std::invalid_argument("onnx_light_cpu::RMSNormalization: epsilon must be non-negative.");
  }
  const std::int64_t rank = static_cast<std::int64_t>(x.shape.size());
  if (rank == 0 || axis < -rank || axis >= rank) {
    throw std::invalid_argument("onnx_light_cpu::RMSNormalization: axis is out of range.");
  }
  const std::size_t normalized_axis = static_cast<std::size_t>(axis < 0 ? axis + rank : axis);
  const std::vector<std::int64_t> expected_scale_dims(
      x.shape.begin() + static_cast<std::ptrdiff_t>(normalized_axis), x.shape.end());
  const rt_ns::Shape expected_scale_shape(expected_scale_dims);
  if (scale.shape != expected_scale_shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::RMSNormalization: scale shape must match the normalized dimensions.");
  }

  std::size_t outer = 1;
  std::size_t inner = 1;
  for (std::size_t i = 0; i < x.shape.size(); ++i) {
    if (x.shape[i] <= 0) {
      throw std::invalid_argument("onnx_light_cpu::RMSNormalization: dimensions must be positive.");
    }
    if (i < normalized_axis) {
      outer *= static_cast<std::size_t>(x.shape[i]);
    } else {
      inner *= static_cast<std::size_t>(x.shape[i]);
    }
  }
  const std::size_t bytes = static_cast<std::size_t>(x.element_count()) * x.element_size();
  Tensor output = rt != nullptr ? rt->MakeOutputTensor(0, x.data_type, x.shape, bytes)
                                : rt_ns::MakeOutputTensor(x.data_type, x.shape, bytes, nullptr);
  DispatchNormalize(x, scale, output, outer, inner, epsilon);
  return output;
}

void RmsNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &scale = rt_ns::GetInput(node, 1, rt.tensors());
  const std::int64_t axis = rt_ns::GetAttributeIntOrDefault(node, "axis", -1);
  const float epsilon = rt_ns::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-5f);
  const std::int64_t stash_type = rt_ns::GetAttributeIntOrDefault(node, "stash_type", 1);
  rt_ns::SetOutput(node, 0, (*this)(x, scale, axis, epsilon, stash_type, &rt), rt);
}

void RegisterRmsNormalizationKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<RmsNormalizationKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "RMSNormalization";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = RmsNormalizationKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 23;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
