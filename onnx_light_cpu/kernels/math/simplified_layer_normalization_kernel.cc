// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/simplified_layer_normalization_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"
#include "onnx_light_cpu/impl/math/rms_normalization.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/math/normalization_helpers.h"

#include "onnx_core/runtime/kernels/node_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace norm = normalization;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

void ValidateInput(const Tensor &tensor, const char *role) {
  norm::RequireSupportedFloatType(tensor, SimplifiedLayerNormalizationKernel::kName, role);
  const auto count = norm::Product(tensor.shape, 0, tensor.shape.size(),
                                   SimplifiedLayerNormalizationKernel::kName);
  const auto width = norm::ElementSize(tensor.data_type);
  if (count > std::numeric_limits<std::size_t>::max() / width ||
      tensor.size_bytes() != count * width || (count != 0 && tensor.bytes() == nullptr)) {
    throw std::invalid_argument(std::string(SimplifiedLayerNormalizationKernel::kName) +
                                ": input buffer size does not match shape.");
  }
}

template <DataType Type> auto Load(const Tensor &tensor, std::size_t index) {
  norm::StorageType<Type> value;
  std::memcpy(&value, tensor.bytes() + index * sizeof(value), sizeof(value));
  return norm::TypeTraits<Type>::Load(&value, 0);
}

bool FloatAligned(const Tensor &tensor) {
  return reinterpret_cast<std::uintptr_t>(tensor.bytes()) % alignof(float) == 0;
}

template <DataType XType, DataType VType, typename Acc>
void NormalizeRows(const Tensor &x, const Tensor &scale, SimplifiedLayerNormalizationResult &result,
                   const norm::BroadcastIndexer &scale_index, std::size_t width, float epsilon,
                   bool scale_by_inner, bool output_inverse, std::int64_t begin, std::int64_t end) {
  auto *output = norm::MutableData<VType>(result.y);
  for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
       ++row) {
    const std::size_t base = row * width;
    Acc mean_square;
    if constexpr (XType == DataType::FLOAT && std::is_same_v<Acc, float>) {
      if (FloatAligned(x)) {
        mean_square = ComputeNormalizationMeanSquareFloat32(x.AsFloat() + base, width);
      } else {
        float sum = 0.0F;
        for (std::size_t i = 0; i < width; ++i) {
          const float value = Load<XType>(x, base + i);
          sum += value * value;
        }
        mean_square = sum / static_cast<float>(width);
      }
    } else {
      Acc sums[4] = {};
      for (std::size_t i = 0; i < width; ++i) {
        const Acc value = static_cast<Acc>(Load<XType>(x, base + i));
        sums[i % 4] += value * value;
      }
      mean_square = ((sums[0] + sums[1]) + (sums[2] + sums[3])) / static_cast<Acc>(width);
    }
    const Acc inverse = Acc{1} / std::sqrt(mean_square + static_cast<Acc>(epsilon));
    if (output_inverse) {
      norm::TensorWriter(result.inv_std_var).StoreDouble(row, static_cast<double>(inverse));
    }
    if constexpr (XType == DataType::FLOAT && VType == DataType::FLOAT &&
                  std::is_same_v<Acc, float>) {
      if (scale_by_inner && FloatAligned(x) && FloatAligned(scale)) {
        ApplyNormalizationAffineFloat32(x.AsFloat() + base, scale.AsFloat(), nullptr, output + base,
                                        width, 0.0F, inverse);
        continue;
      }
    }
    for (std::size_t i = 0; i < width; ++i) {
      const std::size_t scale_position = scale_by_inner ? i : scale_index.Index(base + i);
      // Unlike RMSNormalization, the experimental operator rounds only after scaling.
      const Acc value = static_cast<Acc>(Load<XType>(x, base + i)) * inverse *
                        static_cast<Acc>(Load<VType>(scale, scale_position));
      norm::TypeTraits<VType>::Store(output, base + i,
                                     static_cast<norm::AccumulatorType<VType>>(value));
    }
  }
}

} // namespace

SimplifiedLayerNormalizationResult SimplifiedLayerNormalizationKernel::operator()(
    const Tensor &x, const Tensor &scale, std::int64_t axis, float epsilon, std::int64_t stash_type,
    bool output_inv_std_var, RuntimeContext *rt) const {
  ValidateInput(x, "X");
  ValidateInput(scale, "scale");
  if (stash_type != 1 && stash_type != 11) {
    throw std::invalid_argument(std::string(kName) + ": stash_type must be FLOAT or DOUBLE.");
  }
  const auto normalized_axis =
      static_cast<std::size_t>(norm::NormalizeAxis(axis, x.shape.size(), kName));
  const std::size_t rows = norm::Product(x.shape, 0, normalized_axis, kName);
  const std::size_t width = norm::Product(x.shape, normalized_axis, x.shape.size(), kName);
  if (width == 0 || rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(std::string(kName) +
                                ": normalized dimensions must be nonempty and rows fit int64_t.");
  }
  const norm::BroadcastIndexer scale_index(x.shape, scale.shape, kName, "scale");
  const bool scale_by_inner =
      scale.shape.size() == x.shape.size() - normalized_axis &&
      std::equal(scale.shape.begin(), scale.shape.end(),
                 x.shape.begin() + static_cast<std::ptrdiff_t>(normalized_axis));
  SimplifiedLayerNormalizationResult result;
  result.y = norm::AllocateOutput(scale.data_type, x.shape, 0, rt);
  if (output_inv_std_var) {
    auto stats_shape = x.shape;
    std::fill(stats_shape.begin() + static_cast<std::ptrdiff_t>(normalized_axis), stats_shape.end(),
              1);
    result.inv_std_var =
        norm::AllocateOutput(static_cast<std::int32_t>(stash_type), stats_shape, 1, rt);
  }
  if (x.data_type == DataType::FLOAT && scale.data_type == DataType::FLOAT && scale_by_inner &&
      FloatAligned(x) && FloatAligned(scale) && (!output_inv_std_var || stash_type == 1)) {
    auto *inverse =
        output_inv_std_var ? norm::MutableData<DataType::FLOAT>(result.inv_std_var) : nullptr;
    RmsNormalizationFloat32(x.AsFloat(), scale.AsFloat(),
                            norm::MutableData<DataType::FLOAT>(result.y), rows, width, epsilon,
                            inverse);
    return result;
  }
  norm::DispatchFloatType(x.data_type, [&]<DataType XType>() {
    norm::DispatchFloatType(scale.data_type, [&]<DataType VType>() {
      auto run = [&](std::int64_t begin, std::int64_t end) {
        if constexpr (XType == DataType::DOUBLE || VType == DataType::DOUBLE) {
          NormalizeRows<XType, VType, double>(x, scale, result, scale_index, width, epsilon,
                                              scale_by_inner, output_inv_std_var, begin, end);
        } else if (scale_by_inner) {
          NormalizeRows<XType, VType, float>(x, scale, result, scale_index, width, epsilon, true,
                                             output_inv_std_var, begin, end);
        } else {
          NormalizeRows<XType, VType, double>(x, scale, result, scale_index, width, epsilon, false,
                                              output_inv_std_var, begin, end);
        }
      };
      ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(width), run);
    });
  });
  return result;
}

void SimplifiedLayerNormalizationKernel::Run(RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  if (node.output_size() < 1 || node.output_size() > 2 || node.output(0).empty()) {
    throw std::invalid_argument(std::string(kName) + ": expected Y and optional inv_std_var.");
  }
  const bool output_inverse = node.output_size() == 2 && !node.output(1).empty();
  auto result =
      (*this)(rt_ns::GetInput(node, 0, rt.tensors()), rt_ns::GetInput(node, 1, rt.tensors()),
              rt_ns::GetAttributeIntOrDefault(node, "axis", -1),
              rt_ns::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-5F),
              rt_ns::GetAttributeIntOrDefault(node, "stash_type", 1), output_inverse, &rt);
  rt_ns::SetOutput(node, 0, std::move(result.y), rt);
  if (output_inverse) {
    rt_ns::SetOutput(node, 1, std::move(result.inv_std_var), rt);
  }
}

void RegisterSimplifiedLayerNormalizationKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node, RuntimeContext &rt) {
    auto kernel = std::make_unique<SimplifiedLayerNormalizationKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "SimplifiedLayerNormalization";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = SimplifiedLayerNormalizationKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
