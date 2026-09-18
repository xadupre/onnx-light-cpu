// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/nonzero_kernel.h"

namespace onnx_light_cpu {

int64_t CountNonZeroBool(const uint8_t *input, int64_t total) {
  int64_t count = 0;
  for (int64_t i = 0; i < total; ++i) {
    count += input[i] != 0;
  }
  return count;
}

void WriteNonZeroBoolIndices(const uint8_t *input, int64_t total, const int64_t *dimensions,
                             int64_t rank, int64_t count, int64_t *output) {
  if (rank == 0 || count == 0) {
    return;
  }
  int64_t column = 0;
  for (int64_t i = 0; i < total; ++i) {
    if (input[i] == 0) {
      continue;
    }
    int64_t flat = i;
    for (int64_t axis = rank; axis > 0; --axis) {
      const int64_t dim = dimensions[axis - 1];
      output[(axis - 1) * count + column] = flat % dim;
      flat /= dim;
    }
    ++column;
  }
}

} // namespace onnx_light_cpu
