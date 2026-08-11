// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Backend-test-driven checks for the onnx-light-cpu kernels.
//
// onnx-light ships its ONNX backend test cases as a C++-registered registry
// (``onnx_light/onnx_extensions/backend_test/cases/``) exposed through
// :cpp:func:`onnx_light::core::backend_test::CollectTestCases`. Each entry is a
// ``TestCase`` bundling a single-node ``ModelProto`` with one or more
// input/output ``DataSet`` computed by onnx-light itself.
//
// This test mirrors onnx-light's own
// ``unittests/cc/onnx_extensions/backend_test/test_backend_kernels.cc``: it
// collects the backend test cases registered for ``Abs``, ``Exp``, ``Log``,
// ``Gemm`` and ``Not`` (covering every element-type combination onnx-light
// registers for those ops), runs each case's input through the corresponding
// SIMD-accelerated onnx-light-cpu kernel, and checks the output matches the
// reference output shipped with the backend test case (using the case's own
// ``rtol``/``atol``).

#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;

using bt_ns::CollectTestCases;
using bt_ns::DataSet;
using bt_ns::TestCase;
using rt_ns::DataType;
using rt_ns::Tensor;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

// Compares a kernel output tensor against a backend test case's reference
// output. Integer/bool outputs must match exactly; floating-point outputs
// (including the 16-bit formats decoded through their bit patterns) are
// compared with the case's ``rtol``/``atol`` using the same
// ``|actual - expected| <= atol + rtol * |expected|`` rule as numpy's
// ``assert_allclose``.
void ExpectTensorClose(const Tensor &actual, const Tensor &expected, double rtol, double atol) {
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  const std::int64_t n = expected.element_count();
  ASSERT_EQ(actual.element_count(), n);

  switch (static_cast<DataType>(expected.data_type)) {
  case DataType::FLOAT: {
    const float *a = actual.AsFloat();
    const float *e = expected.AsFloat();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_NEAR(a[i], e[i], atol + rtol * std::fabs(static_cast<double>(e[i]))) << "i=" << i;
    }
    return;
  }
  case DataType::DOUBLE: {
    const double *a = actual.AsDouble();
    const double *e = expected.AsDouble();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_NEAR(a[i], e[i], atol + rtol * std::fabs(e[i])) << "i=" << i;
    }
    return;
  }
  case DataType::FLOAT16: {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.bytes());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.bytes());
    for (std::int64_t i = 0; i < n; ++i) {
      const double av = rt_ns::Float16BitsToFloat(a[i]);
      const double ev = rt_ns::Float16BitsToFloat(e[i]);
      EXPECT_NEAR(av, ev, atol + rtol * std::fabs(ev)) << "i=" << i;
    }
    return;
  }
  case DataType::BFLOAT16: {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.bytes());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.bytes());
    for (std::int64_t i = 0; i < n; ++i) {
      const double av = rt_ns::Bfloat16BitsToFloat(a[i]);
      const double ev = rt_ns::Bfloat16BitsToFloat(e[i]);
      EXPECT_NEAR(av, ev, atol + rtol * std::fabs(ev)) << "i=" << i;
    }
    return;
  }
  case DataType::INT8: {
    const std::int8_t *a = actual.AsInt8();
    const std::int8_t *e = expected.AsInt8();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_EQ(a[i], e[i]) << "i=" << i;
    }
    return;
  }
  case DataType::INT16: {
    const std::int16_t *a = actual.AsInt16();
    const std::int16_t *e = expected.AsInt16();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_EQ(a[i], e[i]) << "i=" << i;
    }
    return;
  }
  case DataType::INT32: {
    const std::int32_t *a = actual.AsInt32();
    const std::int32_t *e = expected.AsInt32();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_EQ(a[i], e[i]) << "i=" << i;
    }
    return;
  }
  case DataType::INT64: {
    const std::int64_t *a = actual.AsInt64();
    const std::int64_t *e = expected.AsInt64();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_EQ(a[i], e[i]) << "i=" << i;
    }
    return;
  }
  case DataType::BOOL: {
    const std::uint8_t *a = actual.AsBool();
    const std::uint8_t *e = expected.AsBool();
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_EQ(a[i], e[i]) << "i=" << i;
    }
    return;
  }
  default:
    FAIL() << "unsupported reference data type " << expected.data_type;
  }
}

// Runs every backend test case of a unary op (single input, single output)
// through ``kernel`` and compares against the case's reference output.
template <typename Kernel>
void RunUnaryBackendCases(const std::string &op_type, const Kernel &kernel) {
  const std::vector<TestCase> cases = CollectTestCases(op_type);
  ASSERT_FALSE(cases.empty()) << "no backend test cases registered for " << op_type;

  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    for (const DataSet &ds : tc.data_sets()) {
      ASSERT_EQ(ds.inputs.size(), 1u);
      ASSERT_EQ(ds.outputs.size(), 1u);
      const Tensor y = kernel(ds.inputs[0]);
      ExpectTensorClose(y, ds.outputs[0], tc.rtol, tc.atol);
    }
  }
}

TEST(OnnxLightBackendKernels, AbsRunsOnBackendTestCases) {
  const onnx_light_cpu::AbsKernel kernel(MakeCtx());
  RunUnaryBackendCases("Abs", [&](const Tensor &x) { return kernel(x); });
}

TEST(OnnxLightBackendKernels, ExpRunsOnBackendTestCases) {
  const onnx_light_cpu::ExpKernel kernel(MakeCtx());
  RunUnaryBackendCases("Exp", [&](const Tensor &x) { return kernel(x); });
}

TEST(OnnxLightBackendKernels, LogRunsOnBackendTestCases) {
  const onnx_light_cpu::LogKernel kernel(MakeCtx());
  RunUnaryBackendCases("Log", [&](const Tensor &x) { return kernel(x); });
}

TEST(OnnxLightBackendKernels, NotRunsOnBackendTestCases) {
  const onnx_light_cpu::NotKernel kernel(MakeCtx());
  RunUnaryBackendCases("Not", [&](const Tensor &x) { return kernel(x); });
}

TEST(OnnxLightBackendKernels, GemmRunsOnBackendTestCases) {
  const std::vector<TestCase> cases = CollectTestCases("Gemm");
  ASSERT_FALSE(cases.empty()) << "no backend test cases registered for Gemm";

  const onnx_light_cpu::GemmKernel kernel(MakeCtx());
  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_EQ(tc.model().ref_graph().ref_node().size(), 1u);
    const auto &node = tc.model().ref_graph().ref_node()[0];
    const float alpha = rt_ns::GetAttributeFloatOrDefault(node, "alpha", 1.0f);
    const float beta = rt_ns::GetAttributeFloatOrDefault(node, "beta", 1.0f);
    const bool trans_a = rt_ns::GetAttributeIntOrDefault(node, "transA", 0) != 0;
    const bool trans_b = rt_ns::GetAttributeIntOrDefault(node, "transB", 0) != 0;

    for (const DataSet &ds : tc.data_sets()) {
      ASSERT_GE(ds.inputs.size(), 2u);
      ASSERT_EQ(ds.outputs.size(), 1u);
      const Tensor y =
          ds.inputs.size() >= 3
              ? kernel(ds.inputs[0], ds.inputs[1], ds.inputs[2], alpha, beta, trans_a, trans_b)
              : kernel(ds.inputs[0], ds.inputs[1], alpha, trans_a, trans_b);
      ExpectTensorClose(y, ds.outputs[0], tc.rtol, tc.atol);
    }
  }
}

} // namespace
