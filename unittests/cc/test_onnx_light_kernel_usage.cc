// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

// Every onnx-light-cpu kernel exposes a unique, library-qualified name so the
// kernel that a model dispatches to can be told apart from onnx-light's
// built-in one.
TEST(OnnxLightKernelUsage, KernelNamesAreLibraryQualified) {
  EXPECT_STREQ(onnx_light_cpu::AbsKernel::kName, "onnx_light_cpu::Abs");
  EXPECT_STREQ(onnx_light_cpu::ExpKernel::kName, "onnx_light_cpu::Exp");
  EXPECT_STREQ(onnx_light_cpu::LogKernel::kName, "onnx_light_cpu::Log");
  EXPECT_STREQ(onnx_light_cpu::GemmKernel::kName, "onnx_light_cpu::Gemm");
  EXPECT_STREQ(onnx_light_cpu::NotKernel::kName, "onnx_light_cpu::Not");
}

// ``RegisteredKernelNames`` maps every overridden ONNX op_type to the
// library-qualified name of the accelerated kernel installed for it.
TEST(OnnxLightKernelUsage, RegisteredKernelNames) {
  const std::vector<std::pair<std::string, std::string>> expected = {
      {"Abs", "onnx_light_cpu::Abs"}, {"Exp", "onnx_light_cpu::Exp"},
      {"Log", "onnx_light_cpu::Log"}, {"Gemm", "onnx_light_cpu::Gemm"},
      {"Not", "onnx_light_cpu::Not"},
  };
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

} // namespace
