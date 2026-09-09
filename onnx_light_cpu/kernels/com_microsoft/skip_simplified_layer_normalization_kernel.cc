// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"

#include "onnx_light_cpu/impl/com_microsoft/skip_simplified_layer_normalization.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_helpers.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace helpers = skip_simplified_layer_normalization;

bool FloatAligned(const runtime::Tensor &tensor) {
  return reinterpret_cast<std::uintptr_t>(tensor.bytes()) % alignof(float) == 0;
}

template <runtime::DataType Type>
void Normalize(const runtime::Tensor &input, const runtime::Tensor &skip,
               const runtime::Tensor &gamma, const runtime::Tensor *bias,
               SkipSimplifiedLayerNormalizationResult &result, const helpers::Dimensions &dims,
               float epsilon, bool output_sum, bool output_mean, bool output_inv_std_var) {
  using Traits = normalization::TypeTraits<Type>;
  auto *output = normalization::MutableData<Type>(result.output);
  auto *sum = output_sum ? normalization::MutableData<Type>(result.input_skip_bias_sum) : nullptr;
  ExecuteRanges(static_cast<std::int64_t>(dims.rows), static_cast<double>(dims.width),
                [&](std::int64_t begin, std::int64_t end) {
                  for (std::size_t row = static_cast<std::size_t>(begin);
                       row < static_cast<std::size_t>(end); ++row) {
                    const std::size_t base = row * dims.width;
                    const std::size_t skip_base = (row % dims.skip_rows) * dims.width;
                    float squares[4] = {};
                    for (std::size_t i = 0; i < dims.width; ++i) {
                      float value = helpers::Load<Type>(input, base + i) +
                                    helpers::Load<Type>(skip, skip_base + i);
                      if (bias != nullptr) {
                        value += helpers::Load<Type>(*bias, i);
                      }
                      squares[i % 4] += value * value;
                      if (sum != nullptr) {
                        Traits::Store(sum, base + i, value);
                      }
                    }
                    const float mean_square =
                        ((squares[0] + squares[1]) + (squares[2] + squares[3])) /
                        static_cast<float>(dims.width);
                    const float denominator = std::sqrt(mean_square + epsilon);
                    if (output_mean) {
                      result.mean.AsFloat()[row] = 0.0F;
                    }
                    if (output_inv_std_var) {
                      result.inv_std_var.AsFloat()[row] = 1.0F / denominator;
                    }
                    // Recomputes the FP32 residual rather than normalizing a narrowed sum.
                    for (std::size_t i = 0; i < dims.width; ++i) {
                      float residual = helpers::Load<Type>(input, base + i) +
                                       helpers::Load<Type>(skip, skip_base + i);
                      if (bias != nullptr) {
                        residual += helpers::Load<Type>(*bias, i);
                      }
                      const float value = (residual / denominator) * helpers::Load<Type>(gamma, i);
                      Traits::Store(output, base + i, value);
                    }
                  }
                });
}

} // namespace

SkipSimplifiedLayerNormalizationKernel::SkipSimplifiedLayerNormalizationKernel(
    const ONNX_LIGHT_NAMESPACE::NodeProto &node, const runtime::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

SkipSimplifiedLayerNormalizationResult SkipSimplifiedLayerNormalizationKernel::operator()(
    const runtime::Tensor &input, const runtime::Tensor &skip, const runtime::Tensor &gamma,
    const runtime::Tensor *bias, float epsilon, bool output_sum, bool output_mean,
    bool output_inv_std_var, runtime::RuntimeContext *rt) const {
  const auto dims = helpers::Validate(input, skip, gamma, bias, epsilon);
  auto result = helpers::Allocate(input, output_sum, output_mean, output_inv_std_var, rt);
  if (input.data_type == runtime::DataType::FLOAT && FloatAligned(input) && FloatAligned(skip) &&
      FloatAligned(gamma) && (bias == nullptr || FloatAligned(*bias))) {
    SkipSimplifiedLayerNormalizationFloat32(
        input.AsFloat(), skip.AsFloat(), gamma.AsFloat(),
        bias == nullptr ? nullptr : bias->AsFloat(), result.output.AsFloat(),
        output_sum ? result.input_skip_bias_sum.AsFloat() : nullptr,
        output_mean ? result.mean.AsFloat() : nullptr,
        output_inv_std_var ? result.inv_std_var.AsFloat() : nullptr, dims.rows, dims.width,
        dims.skip_rows, epsilon);
  } else if (input.data_type == runtime::DataType::FLOAT) {
    Normalize<runtime::DataType::FLOAT>(input, skip, gamma, bias, result, dims, epsilon, output_sum,
                                        output_mean, output_inv_std_var);
  } else if (input.data_type == runtime::DataType::FLOAT16) {
    Normalize<runtime::DataType::FLOAT16>(input, skip, gamma, bias, result, dims, epsilon,
                                          output_sum, output_mean, output_inv_std_var);
  } else {
    Normalize<runtime::DataType::BFLOAT16>(input, skip, gamma, bias, result, dims, epsilon,
                                           output_sum, output_mean, output_inv_std_var);
  }
  return result;
}

void SkipSimplifiedLayerNormalizationKernel::Run(runtime::RuntimeContext &rt) {
  RecordKernelUsage(kName);
  helpers::Run(*this, *node_, rt);
}

void RegisterSkipSimplifiedLayerNormalizationKernel() {
  runtime::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     runtime::RuntimeContext &rt) {
    return std::make_unique<SkipSimplifiedLayerNormalizationKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "SkipSimplifiedLayerNormalization";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = SkipSimplifiedLayerNormalizationKernel::kName;
  info.types = {runtime::DataType::FLOAT, runtime::DataType::FLOAT16, runtime::DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
