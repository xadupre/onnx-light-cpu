// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/kernel_backend_test.h"

#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;

using bt_ns::CollectTestCases;
using bt_ns::DataSet;
using bt_ns::TestCase;
using rt_ns::DataType;
using rt_ns::Tensor;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

// Appends a failure description for a single mismatched element.
void ReportMismatch(std::vector<std::string> &failures, const std::string &case_name,
                    std::int64_t index, double actual, double expected, double tol) {
  failures.push_back(case_name + ": element " + std::to_string(index) +
                     " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected) +
                     " tol=" + std::to_string(tol));
}

// Compares a kernel output tensor against a backend test case's reference
// output. Integer/bool outputs must match exactly; floating-point outputs
// (including the 16-bit formats decoded through their bit patterns) are
// compared with the case's ``rtol``/``atol`` using the same
// ``|actual - expected| <= atol + rtol * |expected|`` rule as numpy's
// ``assert_allclose``. Any mismatch is appended to ``failures``.
void CompareTensor(const std::string &case_name, const Tensor &actual, const Tensor &expected,
                   double rtol, double atol, std::vector<std::string> &failures) {
  if (actual.data_type != expected.data_type) {
    failures.push_back(case_name + ": output data type " + std::to_string(actual.data_type) +
                       " != expected " + std::to_string(expected.data_type));
    return;
  }
  if (actual.shape != expected.shape) {
    failures.push_back(case_name + ": output shape mismatch");
    return;
  }
  const std::int64_t n = expected.element_count();
  if (actual.element_count() != n) {
    failures.push_back(case_name + ": output element count " +
                       std::to_string(actual.element_count()) + " != expected " +
                       std::to_string(n));
    return;
  }

  switch (static_cast<DataType>(expected.data_type)) {
  case DataType::FLOAT: {
    const float *a = actual.AsFloat();
    const float *e = expected.AsFloat();
    for (std::int64_t i = 0; i < n; ++i) {
      const double tol = atol + rtol * std::fabs(static_cast<double>(e[i]));
      if (!(std::fabs(static_cast<double>(a[i]) - static_cast<double>(e[i])) <= tol)) {
        ReportMismatch(failures, case_name, i, a[i], e[i], tol);
      }
    }
    return;
  }
  case DataType::DOUBLE: {
    const double *a = actual.AsDouble();
    const double *e = expected.AsDouble();
    for (std::int64_t i = 0; i < n; ++i) {
      const double tol = atol + rtol * std::fabs(e[i]);
      if (!(std::fabs(a[i] - e[i]) <= tol)) {
        ReportMismatch(failures, case_name, i, a[i], e[i], tol);
      }
    }
    return;
  }
  case DataType::FLOAT16: {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.bytes());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.bytes());
    for (std::int64_t i = 0; i < n; ++i) {
      const double av = rt_ns::Float16BitsToFloat(a[i]);
      const double ev = rt_ns::Float16BitsToFloat(e[i]);
      const double tol = atol + rtol * std::fabs(ev);
      if (!(std::fabs(av - ev) <= tol)) {
        ReportMismatch(failures, case_name, i, av, ev, tol);
      }
    }
    return;
  }
  case DataType::BFLOAT16: {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.bytes());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.bytes());
    for (std::int64_t i = 0; i < n; ++i) {
      const double av = rt_ns::Bfloat16BitsToFloat(a[i]);
      const double ev = rt_ns::Bfloat16BitsToFloat(e[i]);
      const double tol = atol + rtol * std::fabs(ev);
      if (!(std::fabs(av - ev) <= tol)) {
        ReportMismatch(failures, case_name, i, av, ev, tol);
      }
    }
    return;
  }
  case DataType::INT8: {
    const std::int8_t *a = actual.AsInt8();
    const std::int8_t *e = expected.AsInt8();
    for (std::int64_t i = 0; i < n; ++i) {
      if (a[i] != e[i]) {
        ReportMismatch(failures, case_name, i, a[i], e[i], 0.0);
      }
    }
    return;
  }
  case DataType::INT16: {
    const std::int16_t *a = actual.AsInt16();
    const std::int16_t *e = expected.AsInt16();
    for (std::int64_t i = 0; i < n; ++i) {
      if (a[i] != e[i]) {
        ReportMismatch(failures, case_name, i, a[i], e[i], 0.0);
      }
    }
    return;
  }
  case DataType::INT32: {
    const std::int32_t *a = actual.AsInt32();
    const std::int32_t *e = expected.AsInt32();
    for (std::int64_t i = 0; i < n; ++i) {
      if (a[i] != e[i]) {
        ReportMismatch(failures, case_name, i, a[i], e[i], 0.0);
      }
    }
    return;
  }
  case DataType::INT64: {
    const std::int64_t *a = actual.AsInt64();
    const std::int64_t *e = expected.AsInt64();
    for (std::int64_t i = 0; i < n; ++i) {
      if (a[i] != e[i]) {
        ReportMismatch(failures, case_name, i, static_cast<double>(a[i]), static_cast<double>(e[i]),
                       0.0);
      }
    }
    return;
  }
  case DataType::BOOL: {
    const std::uint8_t *a = actual.AsBool();
    const std::uint8_t *e = expected.AsBool();
    for (std::int64_t i = 0; i < n; ++i) {
      if (a[i] != e[i]) {
        ReportMismatch(failures, case_name, i, a[i], e[i], 0.0);
      }
    }
    return;
  }
  default:
    failures.push_back(case_name + ": unsupported reference data type " +
                       std::to_string(expected.data_type));
  }
}

// Runs every backend test case of a unary op (single input, single output)
// through ``kernel`` and collects any mismatches against the case's reference
// output.
std::vector<std::string> RunUnaryBackendCases(const std::string &op_type,
                                              const std::function<Tensor(const Tensor &)> &kernel) {
  std::vector<std::string> failures;
  const std::vector<TestCase> cases = CollectTestCases(op_type);
  if (cases.empty()) {
    failures.push_back("no backend test cases registered for " + op_type);
    return failures;
  }

  for (const TestCase &tc : cases) {
    for (const DataSet &ds : tc.data_sets()) {
      if (ds.inputs.size() != 1u || ds.outputs.size() != 1u) {
        failures.push_back(tc.name + ": expected a single input and output");
        continue;
      }
      const Tensor y = kernel(ds.inputs[0]);
      CompareTensor(tc.name, y, ds.outputs[0], tc.rtol, tc.atol, failures);
    }
  }
  return failures;
}

} // namespace

std::vector<std::string> RunAbsBackendCases() {
  const onnx_light_cpu::AbsKernel kernel(MakeCtx());
  return RunUnaryBackendCases("Abs", [&](const Tensor &x) { return kernel(x); });
}

std::vector<std::string> RunExpBackendCases() {
  const onnx_light_cpu::ExpKernel kernel(MakeCtx());
  return RunUnaryBackendCases("Exp", [&](const Tensor &x) { return kernel(x); });
}

std::vector<std::string> RunLogBackendCases() {
  const onnx_light_cpu::LogKernel kernel(MakeCtx());
  return RunUnaryBackendCases("Log", [&](const Tensor &x) { return kernel(x); });
}

std::vector<std::string> RunNotBackendCases() {
  const onnx_light_cpu::NotKernel kernel(MakeCtx());
  return RunUnaryBackendCases("Not", [&](const Tensor &x) { return kernel(x); });
}

std::vector<std::string> RunGemmBackendCases() {
  std::vector<std::string> failures;
  const std::vector<TestCase> cases = CollectTestCases("Gemm");
  if (cases.empty()) {
    failures.push_back("no backend test cases registered for Gemm");
    return failures;
  }

  const onnx_light_cpu::GemmKernel kernel(MakeCtx());
  for (const TestCase &tc : cases) {
    if (tc.model().ref_graph().ref_node().size() != 1u) {
      failures.push_back(tc.name + ": expected a single-node model");
      continue;
    }
    const auto &node = tc.model().ref_graph().ref_node()[0];
    const float alpha = rt_ns::GetAttributeFloatOrDefault(node, "alpha", 1.0f);
    const float beta = rt_ns::GetAttributeFloatOrDefault(node, "beta", 1.0f);
    const bool trans_a = rt_ns::GetAttributeIntOrDefault(node, "transA", 0) != 0;
    const bool trans_b = rt_ns::GetAttributeIntOrDefault(node, "transB", 0) != 0;

    for (const DataSet &ds : tc.data_sets()) {
      if (ds.inputs.size() < 2u || ds.outputs.size() != 1u) {
        failures.push_back(tc.name + ": expected at least two inputs and one output");
        continue;
      }
      const Tensor y =
          ds.inputs.size() >= 3
              ? kernel(ds.inputs[0], ds.inputs[1], ds.inputs[2], alpha, beta, trans_a, trans_b)
              : kernel(ds.inputs[0], ds.inputs[1], alpha, trans_a, trans_b);
      CompareTensor(tc.name, y, ds.outputs[0], tc.rtol, tc.atol, failures);
    }
  }
  return failures;
}

} // namespace onnx_light_cpu::backend_test
