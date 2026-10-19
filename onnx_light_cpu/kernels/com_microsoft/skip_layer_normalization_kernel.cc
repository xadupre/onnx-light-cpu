// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/skip_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "cpu_compute/execution.h"
#include "cpu_compute/math/normalization.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_helpers.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/node_helpers.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace helpers = skip_simplified_layer_normalization;

void ValidateVector(const runtime::Tensor &input, const runtime::Tensor *tensor, std::size_t width,
                    const char *name) {
  if (tensor == nullptr) {
    return;
  }
  helpers::ValidateBuffer(*tensor, name, "SkipLayerNormalization");
  normalization::RequireSameType(input, *tensor, "SkipLayerNormalization", name);
  normalization::RequireVector(*tensor, width, "SkipLayerNormalization", name);
}

template <runtime::DataType Type>
SkipLayerNormalizationResult
Normalize(const runtime::Tensor &input, const runtime::Tensor &skip, const runtime::Tensor &gamma,
          const runtime::Tensor *beta, const runtime::Tensor *bias, float epsilon, bool output_sum,
          bool output_mean, bool output_inv_std_var, runtime::RuntimeContext *rt) {
  using Traits = normalization::TypeTraits<Type>;
  const auto dims = helpers::Validate(input, skip, gamma, bias, epsilon, "SkipLayerNormalization");
  ValidateVector(input, beta, dims.width, "beta");
  SkipLayerNormalizationResult result;
  result.output = normalization::AllocateOutput(input.data_type, input.shape, 0, rt);
  if (output_mean) {
    auto shape = input.shape;
    shape.back() = 1;
    result.mean = normalization::AllocateOutput(runtime::DataType::FLOAT, shape, 1, rt);
  }
  if (output_inv_std_var) {
    auto shape = input.shape;
    shape.back() = 1;
    result.inv_std_var = normalization::AllocateOutput(runtime::DataType::FLOAT, shape, 2, rt);
  }
  if (output_sum) {
    result.input_skip_bias_sum = normalization::AllocateOutput(input.data_type, input.shape, 3, rt);
  }

  auto *output = normalization::MutableData<Type>(result.output);
  auto *sum = output_sum ? normalization::MutableData<Type>(result.input_skip_bias_sum) : nullptr;
  cpu_compute::ExecuteRanges(
      static_cast<std::int64_t>(dims.rows), static_cast<double>(dims.width),
      [&](std::int64_t begin, std::int64_t end) {
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          const std::size_t base = row * dims.width;
          const std::size_t skip_base = (row % dims.skip_rows) * dims.width;
          float mean = 0.0F;
          float m2 = 0.0F;
          for (std::size_t i = 0; i < dims.width; ++i) {
            float residual =
                helpers::Load<Type>(input, base + i) + helpers::Load<Type>(skip, skip_base + i);
            if (bias != nullptr) {
              residual += helpers::Load<Type>(*bias, i);
            }
            if (sum != nullptr) {
              Traits::Store(sum, base + i, residual);
            }
            const float delta = residual - mean;
            mean += delta / static_cast<float>(i + 1);
            m2 += delta * (residual - mean);
          }
          const float inverse = 1.0F / std::sqrt(m2 / static_cast<float>(dims.width) + epsilon);
          if (output_mean) {
            result.mean.AsFloat()[row] = mean;
          }
          if (output_inv_std_var) {
            result.inv_std_var.AsFloat()[row] = inverse;
          }
          for (std::size_t i = 0; i < dims.width; ++i) {
            float residual =
                helpers::Load<Type>(input, base + i) + helpers::Load<Type>(skip, skip_base + i);
            if (bias != nullptr) {
              residual += helpers::Load<Type>(*bias, i);
            }
            float value = (residual - mean) * inverse * helpers::Load<Type>(gamma, i);
            if (beta != nullptr) {
              value += helpers::Load<Type>(*beta, i);
            }
            Traits::Store(output, base + i, value);
          }
        }
      });
  return result;
}

SkipLayerNormalizationResult
NormalizeFloat32(const runtime::Tensor &input, const runtime::Tensor &skip,
                 const runtime::Tensor &gamma, const runtime::Tensor *beta,
                 const runtime::Tensor *bias, float epsilon, bool output_sum, bool output_mean,
                 bool output_inv_std_var, runtime::RuntimeContext *rt) {
  const auto dims = helpers::Validate(input, skip, gamma, bias, epsilon, "SkipLayerNormalization");
  ValidateVector(input, beta, dims.width, "beta");
  SkipLayerNormalizationResult result;
  result.output = normalization::AllocateOutput(input.data_type, input.shape, 0, rt);
  if (output_mean) {
    auto shape = input.shape;
    shape.back() = 1;
    result.mean = normalization::AllocateOutput(runtime::DataType::FLOAT, shape, 1, rt);
  }
  if (output_inv_std_var) {
    auto shape = input.shape;
    shape.back() = 1;
    result.inv_std_var = normalization::AllocateOutput(runtime::DataType::FLOAT, shape, 2, rt);
  }
  if (output_sum) {
    result.input_skip_bias_sum = normalization::AllocateOutput(input.data_type, input.shape, 3, rt);
  }

  const auto *input_data = input.AsFloat();
  const auto *skip_data = skip.AsFloat();
  const auto *gamma_data = gamma.AsFloat();
  const auto *beta_data = beta == nullptr ? nullptr : beta->AsFloat();
  const auto *bias_data = bias == nullptr ? nullptr : bias->AsFloat();
  auto *output = normalization::MutableData<runtime::DataType::FLOAT>(result.output);
  auto *sum = output_sum
                  ? normalization::MutableData<runtime::DataType::FLOAT>(result.input_skip_bias_sum)
                  : nullptr;
  cpu_compute::ExecuteRanges(
      static_cast<std::int64_t>(dims.rows), static_cast<double>(dims.width),
      [&](std::int64_t begin, std::int64_t end) {
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          const std::size_t base = row * dims.width;
          const std::size_t skip_base = (row % dims.skip_rows) * dims.width;
          cpu_compute::AddNormalizationResidualFloat32(input_data + base, skip_data + skip_base,
                                                       bias_data, output + base, dims.width);
          const auto moments =
              cpu_compute::ComputeNormalizationMomentsFloat32(output + base, dims.width);
          const float inverse = 1.0F / std::sqrt(moments.variance + epsilon);
          if (sum != nullptr) {
            std::memcpy(sum + base, output + base, dims.width * sizeof(float));
          }
          if (output_mean) {
            result.mean.AsFloat()[row] = moments.mean;
          }
          if (output_inv_std_var) {
            result.inv_std_var.AsFloat()[row] = inverse;
          }
          cpu_compute::ApplyNormalizationAffineFloat32(output + base, gamma_data, beta_data,
                                                       output + base, dims.width, moments.mean,
                                                       inverse);
        }
      });
  return result;
}

} // namespace

SkipLayerNormalizationKernel::SkipLayerNormalizationKernel(
    const ONNX_LIGHT_NAMESPACE::NodeProto &node, const runtime::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

SkipLayerNormalizationResult SkipLayerNormalizationKernel::operator()(
    const runtime::Tensor &input, const runtime::Tensor &skip, const runtime::Tensor &gamma,
    const runtime::Tensor *beta, const runtime::Tensor *bias, float epsilon, bool output_sum,
    bool output_mean, bool output_inv_std_var, runtime::RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  if (input.data_type == runtime::DataType::FLOAT) {
    return NormalizeFloat32(input, skip, gamma, beta, bias, epsilon, output_sum, output_mean,
                            output_inv_std_var, rt);
  }
  if (input.data_type == runtime::DataType::FLOAT16) {
    return Normalize<runtime::DataType::FLOAT16>(input, skip, gamma, beta, bias, epsilon,
                                                 output_sum, output_mean, output_inv_std_var, rt);
  }
  return Normalize<runtime::DataType::BFLOAT16>(input, skip, gamma, beta, bias, epsilon, output_sum,
                                                output_mean, output_inv_std_var, rt);
}

void SkipLayerNormalizationKernel::Run(runtime::RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  if (node.input_size() < 3 || node.input_size() > 5 || node.input(0).empty() ||
      node.input(1).empty() || node.input(2).empty()) {
    throw std::invalid_argument(
        "SkipLayerNormalization: expected input, skip, gamma, optional beta, and optional bias.");
  }
  if (node.output_size() < 1 || node.output_size() > 4 || node.output(0).empty()) {
    throw std::invalid_argument(
        "SkipLayerNormalization: expected output and up to three optional outputs.");
  }
  const runtime::Tensor *beta =
      node.input_size() > 3 && !node.input(3).empty() ? &runtime::GetInput(node, 3, rt) : nullptr;
  const runtime::Tensor *bias =
      node.input_size() > 4 && !node.input(4).empty() ? &runtime::GetInput(node, 4, rt) : nullptr;
  const bool output_mean = node.output_size() > 1 && !node.output(1).empty();
  const bool output_inv_std_var = node.output_size() > 2 && !node.output(2).empty();
  const bool output_sum = node.output_size() > 3 && !node.output(3).empty();
  auto result = (*this)(runtime::GetInput(node, 0, rt), runtime::GetInput(node, 1, rt),
                        runtime::GetInput(node, 2, rt), beta, bias,
                        runtime::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-12F), output_sum,
                        output_mean, output_inv_std_var, &rt);
  runtime::SetOutput(node, 0, std::move(result.output), rt);
  if (output_mean) {
    runtime::SetOutput(node, 1, std::move(result.mean), rt);
  }
  if (output_inv_std_var) {
    runtime::SetOutput(node, 2, std::move(result.inv_std_var), rt);
  }
  if (output_sum) {
    runtime::SetOutput(node, 3, std::move(result.input_skip_bias_sum), rt);
  }
}

void RegisterSkipLayerNormalizationKernel() {
  runtime::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     runtime::RuntimeContext &,
                                     const runtime::KernelContext &kernel_context) {
    return std::make_unique<SkipLayerNormalizationKernel>(node, kernel_context);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "SkipLayerNormalization";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = SkipLayerNormalizationKernel::kName;
  info.types = {runtime::DataType::FLOAT, runtime::DataType::FLOAT16, runtime::DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
