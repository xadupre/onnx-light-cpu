// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ml_cpu_compute/com_microsoft/matmul_nbits.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

namespace onnx_light_cpu {

struct MatMulNBitsAttributes {
  std::int64_t k = 0;
  std::int64_t n = 0;
  std::int64_t bits = 4;
  std::int64_t block_size = 0;
  std::int64_t accuracy_level = 0;
  std::int64_t weight_prepacked = 0;
};

class MatMulNBitsKernel final : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  MatMulNBitsKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  ~MatMulNBitsKernel() override;

  static constexpr const char *kName = "onnx_light_cpu::MatMulNBits";
  static constexpr std::uint32_t kTuningAbi = 2;

  static void RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningKey
  TuningKey(std::int32_t element_type) const override;
  void
  Configure(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningParameters &parameters) override;
  bool HasPreparations(const std::unordered_set<std::string> &immutable_inputs) const override;
  void Prepare(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt,
               const std::unordered_set<std::string> &immutable_inputs);
#if defined(ONNX_LIGHT_CPU_HAS_DECLARATIVE_PREPARATIONS)
  void DeclarePreparations(
      const std::unordered_set<std::string> &immutable_inputs,
      std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::KernelPreparation> &preparations) override;
#endif

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scales,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *zero_points = nullptr) const;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::EncodedValueProto &packed_weights,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             const ONNX_LIGHT_NAMESPACE::StructTypeCatalogue &catalogue = {},
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

private:
  struct PreparedInt4Plan;
  struct PreparedInt4State;
  struct PreparedInt8Plan;
  struct PreparedEncodedState;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  Compute(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scales,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias,
          ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt, bool allow_prepacking,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *zero_points = nullptr) const;
  ONNX_LIGHT_NAMESPACE::core::runtime::MatMulNBitsInputs
  ValidateEncodedWeights(const ONNX_LIGHT_NAMESPACE::EncodedValueProto &packed_weights,
                         const ONNX_LIGHT_NAMESPACE::StructTypeCatalogue &catalogue) const;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  ComputeInt3(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
              const ONNX_LIGHT_NAMESPACE::EncodedValueProto &packed_weights,
              const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias,
              ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt) const;

  MatMulNBitsAttributes attributes_;
  ml_cpu_compute::MatMulNBitsExecutionTuning tuning_ =
      ml_cpu_compute::kDefaultMatMulNBitsExecutionTuning;
  std::unique_ptr<const PreparedInt4State> prepared_int4_;
  std::unique_ptr<const PreparedInt4State> prepared_int8_;
  std::unique_ptr<const PreparedEncodedState> prepared_encoded_;
};

void RegisterMatMulNBitsKernel();

} // namespace onnx_light_cpu
