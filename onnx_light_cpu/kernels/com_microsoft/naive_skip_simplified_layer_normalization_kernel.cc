// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_skip_simplified_layer_normalization_kernel.h"

#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_helpers.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <cmath>
#include <memory>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace helpers = skip_simplified_layer_normalization;

template <runtime::DataType Type>
void Normalize(const runtime::Tensor &input, const runtime::Tensor &skip,
               const runtime::Tensor &gamma, const runtime::Tensor *bias,
               SkipSimplifiedLayerNormalizationResult &result, const helpers::Dimensions &dims,
               float epsilon, bool output_sum, bool output_mean, bool output_inv_std_var) {
  using Traits = normalization::TypeTraits<Type>;
  auto *output = normalization::MutableData<Type>(result.output);
  auto *sum = output_sum ? normalization::MutableData<Type>(result.input_skip_bias_sum) : nullptr;
  for (std::size_t row = 0; row < dims.rows; ++row) {
    const std::size_t base = row * dims.width;
    const std::size_t skip_base = (row % dims.skip_rows) * dims.width;
    float square_sum = 0.0F;
    for (std::size_t i = 0; i < dims.width; ++i) {
      float residual =
          helpers::Load<Type>(input, base + i) + helpers::Load<Type>(skip, skip_base + i);
      if (bias != nullptr) {
        residual += helpers::Load<Type>(*bias, i);
      }
      square_sum += residual * residual;
      if (sum != nullptr) {
        Traits::Store(sum, base + i, residual);
      }
    }
    const float denominator = std::sqrt(square_sum / static_cast<float>(dims.width) + epsilon);
    if (output_mean) {
      result.mean.AsFloat()[row] = 0.0F;
    }
    if (output_inv_std_var) {
      result.inv_std_var.AsFloat()[row] = 1.0F / denominator;
    }
    for (std::size_t i = 0; i < dims.width; ++i) {
      float residual =
          helpers::Load<Type>(input, base + i) + helpers::Load<Type>(skip, skip_base + i);
      if (bias != nullptr) {
        residual += helpers::Load<Type>(*bias, i);
      }
      const float value = (residual / denominator) * helpers::Load<Type>(gamma, i);
      Traits::Store(output, base + i, value);
    }
  }
}

} // namespace

NaiveSkipSimplifiedLayerNormalizationKernel::NaiveSkipSimplifiedLayerNormalizationKernel(
    const ONNX_LIGHT_NAMESPACE::NodeProto &node, const runtime::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

SkipSimplifiedLayerNormalizationResult NaiveSkipSimplifiedLayerNormalizationKernel::operator()(
    const runtime::Tensor &input, const runtime::Tensor &skip, const runtime::Tensor &gamma,
    const runtime::Tensor *bias, float epsilon, bool output_sum, bool output_mean,
    bool output_inv_std_var, runtime::RuntimeContext *rt) const {
  const auto dims = helpers::Validate(input, skip, gamma, bias, epsilon);
  auto result = helpers::Allocate(input, output_sum, output_mean, output_inv_std_var, rt);
  if (input.data_type == runtime::DataType::FLOAT) {
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

void NaiveSkipSimplifiedLayerNormalizationKernel::Run(runtime::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  helpers::Run(*this, *node_, rt);
}

void RegisterNaiveSkipSimplifiedLayerNormalizationKernel() {
  runtime::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     runtime::RuntimeContext &rt) {
    return std::make_unique<NaiveSkipSimplifiedLayerNormalizationKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "SkipSimplifiedLayerNormalization";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = NaiveSkipSimplifiedLayerNormalizationKernel::kName;
  info.types = {runtime::DataType::FLOAT, runtime::DataType::FLOAT16, runtime::DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
