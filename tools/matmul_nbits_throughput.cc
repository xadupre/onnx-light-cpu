// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: matmul_nbits_throughput M K N repeats\n";
    return 1;
  }
  const auto parse = [](const char *text) {
    std::size_t consumed = 0;
    const auto value = std::stoll(text, &consumed);
    if (value <= 0 || text[consumed] != '\0') {
      throw std::invalid_argument("Arguments must be positive integers.");
    }
    return static_cast<std::size_t>(value);
  };
  const auto m = parse(argv[1]), k = parse(argv[2]), n = parse(argv[3]);
  const auto repeats = parse(argv[4]);
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto blocks = onnx_light_cpu::CheckedAdd(k, 31, "benchmark", "blocks") / 32;
  const auto multiply = [](std::size_t a, std::size_t b) {
    return onnx_light_cpu::CheckedMultiply(a, b, "benchmark", "elements");
  };
  std::vector<float> a(multiply(m, k)), scales(multiply(n, blocks)), y(multiply(m, n));
  std::vector<std::uint8_t> b(multiply(scales.size(), 16));
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16;
  }
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = static_cast<float>(i % 7 + 1) / 128;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<std::uint8_t>((i * 37 + i / 17) & 255);
  }
  const double preparation = std::chrono::duration<double>(Clock::now() - start).count();
  const auto run = [&] {
    onnx_light_cpu::MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), m, k,
                                       n, 32);
  };
  run();
  std::vector<double> times;
  for (std::size_t i = 0; i < repeats; ++i) {
    const auto begin = Clock::now();
    run();
    times.push_back(std::chrono::duration<double>(Clock::now() - begin).count());
  }
  std::sort(times.begin(), times.end());
  std::cout << "M,K,N,dtype,threads,input_preparation_s,median_s,packed_bytes,weight_copied_bytes,"
               "workspace_bytes,implementation\n"
            << m << ',' << k << ',' << n << ",float32,1," << preparation << ','
            << times[times.size() / 2] << ',' << b.size() << ",0,"
            << onnx_light_cpu::kMatMulNBitsInt4WorkspaceBytes << ','
            << onnx_light_cpu::MatMulNBitsInt4Implementation() << '\n';
}
