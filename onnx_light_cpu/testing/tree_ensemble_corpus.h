// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/traditionalml/tree_ensemble.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::testing {

/// Reusable differential case for the ai.onnx.ml TreeEnsemble-5 schema.
struct TreeEnsembleCorpusCase {
  std::string name;
  std::string domain = "ai.onnx.ml";
  std::string op_type = "TreeEnsemble";
  std::int64_t opset = 5;
  TreeEnsembleAttributes attributes;
  std::vector<double> input;
  std::size_t rows = 0;
  std::vector<double> expected;
};

/// Generates the deterministic correctness corpus used by this project and by
/// later optimized TreeEnsemble kernels. Expected values always come from the
/// independent scalar oracle.
std::vector<TreeEnsembleCorpusCase> GenerateTreeEnsembleV5Corpus();

} // namespace onnx_light_cpu::testing
