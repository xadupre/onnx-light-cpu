// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

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
  EXPECT_STREQ(onnx_light_cpu::ExpKernel::kName, "onnx_light_cpu::Exp");
  EXPECT_STREQ(onnx_light_cpu::LogKernel::kName, "onnx_light_cpu::Log");
  EXPECT_STREQ(onnx_light_cpu::GemmKernel::kName, "onnx_light_cpu::Gemm");
  EXPECT_STREQ(onnx_light_cpu::MatMulKernel::kName, "onnx_light_cpu::MatMul");
  EXPECT_STREQ(onnx_light_cpu::MatMulIntegerKernel::kName, "onnx_light_cpu::MatMulInteger");
  EXPECT_STREQ(onnx_light_cpu::QLinearMatMulKernel::kName, "onnx_light_cpu::QLinearMatMul");
  EXPECT_STREQ(onnx_light_cpu::NotKernel::kName, "onnx_light_cpu::Not");
}

// ``RegisteredKernelNames`` maps every overridden ONNX op_type to the
// library-qualified name of the accelerated kernel installed for it. It is
// derived from ``CollectRegisteredKernels()``'s structured inventory, so
// entries are ordered by ``(domain, op_type, device, kernel_name)`` --
// alphabetically by ``op_type`` here, since every registration shares the
// same domain and device.
TEST(OnnxLightKernelUsage, RegisteredKernelNames) {
  std::vector<std::pair<std::string, std::string>> expected = {
      {"Abs", "onnx_light_cpu::Abs"},
      {"Attention", "onnx_light_cpu::Attention"},
      {"Exp", "onnx_light_cpu::Exp"},
      {"Gemm", "onnx_light_cpu::Gemm"},
      {"Log", "onnx_light_cpu::Log"},
      {"MatMul", "onnx_light_cpu::MatMul"},
      {"MatMulInteger", "onnx_light_cpu::MatMulInteger"},
      {"Not", "onnx_light_cpu::Not"},
      {"QLinearMatMul", "onnx_light_cpu::QLinearMatMul"},
  };
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    expected.emplace_back(std::string(entry.op_type),
                          std::string("onnx_light_cpu::") + std::string(entry.op_type));
  }
  std::sort(expected.begin(), expected.end());
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
