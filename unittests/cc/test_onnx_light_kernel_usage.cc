// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/cdist_kernel.h"
#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"
#include "onnx_light_cpu/kernels/traditionalml/tree_ensemble_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

// Every onnx-light-cpu kernel exposes a unique, library-qualified name so the
// kernel that a model dispatches to can be told apart from onnx-light's
// built-in one.
TEST(OnnxLightKernelUsage, KernelNamesAreLibraryQualified) {
  EXPECT_STREQ(onnx_light_cpu::AbsKernel::kName, "onnx_light_cpu::Abs");
  EXPECT_STREQ(onnx_light_cpu::AttentionKernel::kName, "onnx_light_cpu::Attention");
  EXPECT_STREQ(onnx_light_cpu::BatchNormalizationKernel::kName,
               "onnx_light_cpu::BatchNormalization");
  EXPECT_STREQ(onnx_light_cpu::BiasGeluKernel::kName, "onnx_light_cpu::BiasGelu");
  EXPECT_STREQ(onnx_light_cpu::CDistKernel::kName, "onnx_light_cpu::CDist");
  EXPECT_STREQ(onnx_light_cpu::ExpKernel::kName, "onnx_light_cpu::Exp");
  EXPECT_STREQ(onnx_light_cpu::LogKernel::kName, "onnx_light_cpu::Log");
  EXPECT_STREQ(onnx_light_cpu::GemmKernel::kName, "onnx_light_cpu::Gemm");
  EXPECT_STREQ(onnx_light_cpu::GroupNormalizationKernel::kName,
               "onnx_light_cpu::GroupNormalization");
  EXPECT_STREQ(onnx_light_cpu::InstanceNormalizationKernel::kName,
               "onnx_light_cpu::InstanceNormalization");
  EXPECT_STREQ(onnx_light_cpu::LayerNormalizationKernel::kName,
               "onnx_light_cpu::LayerNormalization");
  EXPECT_STREQ(onnx_light_cpu::LpNormalizationKernel::kName, "onnx_light_cpu::LpNormalization");
  EXPECT_STREQ(onnx_light_cpu::MatMulKernel::kName, "onnx_light_cpu::MatMul");
  EXPECT_STREQ(onnx_light_cpu::MatMulIntegerKernel::kName, "onnx_light_cpu::MatMulInteger");
  EXPECT_STREQ(onnx_light_cpu::QLinearMatMulKernel::kName, "onnx_light_cpu::QLinearMatMul");
  EXPECT_STREQ(onnx_light_cpu::NotKernel::kName, "onnx_light_cpu::Not");
  EXPECT_STREQ(onnx_light_cpu::MeanVarianceNormalizationKernel::kName,
               "onnx_light_cpu::MeanVarianceNormalization");
  EXPECT_STREQ(onnx_light_cpu::RmsNormalizationKernel::kName, "onnx_light_cpu::RMSNormalization");
  EXPECT_STREQ(onnx_light_cpu::SwiGLUKernel::kName, "onnx_light_cpu::SwiGLU");
  EXPECT_STREQ(onnx_light_cpu::TreeEnsembleKernel::kName, "onnx_light_cpu::TreeEnsemble");
}

// ``RegisteredKernelNames`` maps every overridden ONNX op_type to the
// library-qualified name of the accelerated kernel installed for it. It is
// derived from ``CollectRegisteredKernels()``'s structured inventory, so
// entries are ordered by ``domain`` and then ``op_type`` for the public name map.
TEST(OnnxLightKernelUsage, RegisteredKernelNames) {
  std::vector<std::pair<std::string, std::string>> expected = {
      {"Abs", "onnx_light_cpu::Abs"},
      {"Attention", "onnx_light_cpu::Attention"},
      {"BatchNormalization", "onnx_light_cpu::BatchNormalization"},
      {"Exp", "onnx_light_cpu::Exp"},
      {"Gemm", "onnx_light_cpu::Gemm"},
      {"GroupNormalization", "onnx_light_cpu::GroupNormalization"},
      {"InstanceNormalization", "onnx_light_cpu::InstanceNormalization"},
      {"LayerNormalization", "onnx_light_cpu::LayerNormalization"},
      {"Log", "onnx_light_cpu::Log"},
      {"LpNormalization", "onnx_light_cpu::LpNormalization"},
      {"MatMul", "onnx_light_cpu::MatMul"},
      {"MatMulInteger", "onnx_light_cpu::MatMulInteger"},
      {"Max", "onnx_light_cpu::Max"},
      {"Mean", "onnx_light_cpu::Mean"},
      {"MeanVarianceNormalization", "onnx_light_cpu::MeanVarianceNormalization"},
      {"Min", "onnx_light_cpu::Min"},
      {"Not", "onnx_light_cpu::Not"},
      {"QLinearMatMul", "onnx_light_cpu::QLinearMatMul"},
      {"RMSNormalization", "onnx_light_cpu::RMSNormalization"},
      {"Sum", "onnx_light_cpu::Sum"},
      {"SwiGLU", "onnx_light_cpu::SwiGLU"},
  };
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    expected.emplace_back(std::string(entry.op_type),
                          std::string("onnx_light_cpu::") + std::string(entry.op_type));
  }
  std::sort(expected.begin(), expected.end());
  expected.emplace_back("TreeEnsemble", "onnx_light_cpu::TreeEnsemble");
  expected.emplace_back("BiasGelu", "onnx_light_cpu::BiasGelu");
  expected.emplace_back("CDist", "onnx_light_cpu::CDist");
  EXPECT_EQ(onnx_light_cpu::RegisteredKernelNames(), expected);
}

// The usage log records names in invocation order and can be cleared.
TEST(OnnxLightKernelUsage, RecordAndClear) {
  onnx_light_cpu::ClearUsedKernelNames();
  EXPECT_TRUE(onnx_light_cpu::UsedKernelNames().empty());

  onnx_light_cpu::RecordKernelUsage(onnx_light_cpu::AbsKernel::kName);
  onnx_light_cpu::RecordKernelUsage(onnx_light_cpu::ExpKernel::kName);
  const std::vector<std::string> expected = {"onnx_light_cpu::Abs", "onnx_light_cpu::Exp"};
  EXPECT_EQ(onnx_light_cpu::UsedKernelNames(), expected);

  onnx_light_cpu::ClearUsedKernelNames();
  EXPECT_TRUE(onnx_light_cpu::UsedKernelNames().empty());
}

TEST(OnnxLightKernelUsage, RecordingCanBeDisabled) {
  onnx_light_cpu::ClearUsedKernelNames();
  onnx_light_cpu::SetKernelUsageRecording(false);
  onnx_light_cpu::RecordKernelUsage(onnx_light_cpu::AbsKernel::kName);
  EXPECT_TRUE(onnx_light_cpu::UsedKernelNames().empty());

  onnx_light_cpu::SetKernelUsageRecording(true);
  onnx_light_cpu::RecordKernelUsage(onnx_light_cpu::AbsKernel::kName);
  EXPECT_EQ(onnx_light_cpu::UsedKernelNames(), std::vector<std::string>({"onnx_light_cpu::Abs"}));
}

} // namespace
