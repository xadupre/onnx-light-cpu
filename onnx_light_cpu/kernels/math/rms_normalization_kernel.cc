// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/rms_normalization.h"
#include "onnx_light_cpu/kernels/math/normalization_helpers.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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

template <typename Fn> void ExecuteRows(std::size_t rows, std::size_t inner, Fn &&fn) {
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("onnx_light_cpu::RMSNormalization: row count exceeds int64_t.");
  }
  ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(inner) * 0.625,
                [&](std::int64_t begin, std::int64_t end) {
                  fn(static_cast<std::size_t>(begin), static_cast<std::size_t>(end));
                });
}

template <DataType XType, DataType VType>
void NormalizeTyped(const Tensor &x, const Tensor &scale, Tensor &output,
                    const normalization::BroadcastIndexer &scale_index, std::size_t outer,
                    std::size_t inner, float epsilon, bool scale_by_inner) {
  using XTraits = normalization::TypeTraits<XType>;
  using VTraits = normalization::TypeTraits<VType>;
  const auto *input = normalization::Data<XType>(x);
  const auto *scale_data = normalization::Data<VType>(scale);
  auto *result = normalization::MutableData<VType>(output);
  ExecuteRows(outer, inner, [&](std::size_t begin, std::size_t end) {
    for (std::size_t row = begin; row < end; ++row) {
      const std::size_t base = row * inner;
      const float mean_square =
          normalization::ComputeContiguousFloatMeanSquare<XType>(input + base, inner);
      const float inverse_rms = 1.0F / std::sqrt(mean_square + epsilon);
      if (scale_by_inner) {
        for (std::size_t i = 0; i < inner; ++i) {
          const float normalized = static_cast<float>(XTraits::Load(input, base + i)) * inverse_rms;
          const auto rounded_x = normalization::RoundFloatToType<XType>(normalized);
          const auto normalized_v =
              normalization::RoundFloatToType<VType>(static_cast<float>(rounded_x));
          if constexpr (VType == DataType::DOUBLE) {
            VTraits::Store(result, base + i, normalized_v * VTraits::Load(scale_data, i));
          } else {
            VTraits::Store(result, base + i,
                           normalized_v * static_cast<float>(VTraits::Load(scale_data, i)));
          }
        }
      } else if (scale_index.identity()) {
        for (std::size_t i = 0; i < inner; ++i) {
          const float normalized = static_cast<float>(XTraits::Load(input, base + i)) * inverse_rms;
          const auto rounded_x = normalization::RoundFloatToType<XType>(normalized);
          const auto normalized_v =
              normalization::RoundFloatToType<VType>(static_cast<float>(rounded_x));
          if constexpr (VType == DataType::DOUBLE) {
            VTraits::Store(result, base + i, normalized_v * VTraits::Load(scale_data, base + i));
          } else {
            VTraits::Store(result, base + i,
                           normalized_v * static_cast<float>(VTraits::Load(scale_data, base + i)));
          }
        }
      } else {
        for (std::size_t i = 0; i < inner; ++i) {
          const float normalized = static_cast<float>(XTraits::Load(input, base + i)) * inverse_rms;
          const auto rounded_x = normalization::RoundFloatToType<XType>(normalized);
          const auto normalized_v =
              normalization::RoundFloatToType<VType>(static_cast<float>(rounded_x));
          const std::size_t scale_position = scale_index.Index(base + i);
          if constexpr (VType == DataType::DOUBLE) {
            VTraits::Store(result, base + i,
                           normalized_v * VTraits::Load(scale_data, scale_position));
          } else {
            VTraits::Store(result, base + i,
                           normalized_v *
                               static_cast<float>(VTraits::Load(scale_data, scale_position)));
          }
        }
      }
    }
  });
}

void Normalize(const Tensor &x, const Tensor &scale, Tensor &output,
               const normalization::BroadcastIndexer &scale_index, std::size_t outer,
               std::size_t inner, float epsilon, bool scale_by_inner) {
  if (static_cast<DataType>(x.data_type) == DataType::FLOAT &&
      static_cast<DataType>(scale.data_type) == DataType::FLOAT && scale_by_inner) {
    RmsNormalizationFloat32(x.AsFloat(), scale.AsFloat(),
                            reinterpret_cast<float *>(output.mutable_bytes()), outer, inner,
                            epsilon);
    return;
  }
  if (static_cast<DataType>(x.data_type) == DataType::FLOAT16 &&
      static_cast<DataType>(scale.data_type) == DataType::FLOAT16 && scale_by_inner) {
    const auto *input = reinterpret_cast<const std::uint16_t *>(x.bytes());
    const auto *scale_data = reinterpret_cast<const std::uint16_t *>(scale.bytes());
    auto *result = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    RmsNormalizationFloat16(input, scale_data, result, outer, inner, epsilon);
    return;
  }
  if (static_cast<DataType>(x.data_type) == DataType::BFLOAT16 &&
      static_cast<DataType>(scale.data_type) == DataType::BFLOAT16 && scale_by_inner) {
    const auto *input = reinterpret_cast<const std::uint16_t *>(x.bytes());
    const auto *scale_data = reinterpret_cast<const std::uint16_t *>(scale.bytes());
    auto *result = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    RmsNormalizationBFloat16(input, scale_data, result, outer, inner, epsilon);
    return;
  }
  normalization::DispatchFloatType(x.data_type, [&]<DataType XType>() {
    normalization::DispatchFloatType(scale.data_type, [&]<DataType VType>() {
      NormalizeTyped<XType, VType>(x, scale, output, scale_index, outer, inner, epsilon,
                                   scale_by_inner);
    });
  });
}

} // namespace

Tensor RmsNormalizationKernel::operator()(const Tensor &x, const Tensor &scale, std::int64_t axis,
                                          float epsilon, std::int64_t stash_type,
                                          RuntimeContext *rt) const {
  normalization::RequireSupportedFloatType(x, kName, "X");
  normalization::RequireSupportedFloatType(scale, kName, "scale");
  if (stash_type != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::RMSNormalization: only stash_type=1 is supported.");
  }
  if (!std::isfinite(epsilon) || epsilon < 0.0f) {
    throw std::invalid_argument(
        "onnx_light_cpu::RMSNormalization: epsilon must be finite and non-negative.");
  }
  const std::int64_t rank = static_cast<std::int64_t>(x.shape.size());
  if (rank == 0) {
    throw std::invalid_argument("onnx_light_cpu::RMSNormalization: X must have rank at least 1.");
  }
  normalization::Product(x.shape, 0, x.shape.size(), kName);
  const std::size_t normalized_axis = static_cast<std::size_t>(
      normalization::NormalizeAxis(axis, x.shape.size(), RmsNormalizationKernel::kName));
  const normalization::BroadcastIndexer scale_index(x.shape, scale.shape, kName, "scale");
  const bool scale_by_inner =
      scale.shape.size() == x.shape.size() - normalized_axis &&
      std::equal(scale.shape.begin(), scale.shape.end(),
                 x.shape.begin() + static_cast<std::ptrdiff_t>(normalized_axis));
  const std::size_t outer = normalization::Product(x.shape, 0, normalized_axis, kName);
  const std::size_t inner = normalization::Product(x.shape, normalized_axis, x.shape.size(), kName);
  if (inner == 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::RMSNormalization: normalized dimensions must contain elements.");
  }
  Tensor output = normalization::AllocateOutput(scale.data_type, x.shape, 0, rt);
  Normalize(x, scale, output, scale_index, outer, inner, epsilon, scale_by_inner);
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
