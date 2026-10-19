// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/com_microsoft/matmul_nbits.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 5 || argc > 7 || (argc == 7 && std::string_view(argv[6]) != "--generic")) {
    std::cerr << "Usage: matmul_nbits_throughput M K N repeats [bits=4|8] [--generic]\n";
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
  const bool generic = argc >= 6 && std::string_view(argv[argc - 1]) == "--generic";
  const int positional_argc = argc - (generic ? 1 : 0);
  const auto bits = positional_argc == 6 ? parse(argv[5]) : 4;
  if (bits != 4 && bits != 8) {
    throw std::invalid_argument("bits must be 4 or 8.");
  }
  if (bits == 8 && k % 32 != 0) {
    throw std::invalid_argument("INT8 preparation requires K divisible by 32.");
  }
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto blocks = ml_cpu_compute::CheckedAdd(k, 31, "benchmark", "blocks") / 32;
  const auto multiply = [](std::size_t a, std::size_t b) {
    return ml_cpu_compute::CheckedMultiply(a, b, "benchmark", "elements");
  };
  std::vector<float> a(multiply(m, k)), scales(multiply(n, blocks)), y(multiply(m, n));
  std::vector<std::uint8_t> b(multiply(scales.size(), bits == 8 ? 32 : 16));
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
  const std::size_t slots =
      bits == 8 && !generic ? multiply(n / 8 + (n % 8 != 0), multiply(blocks, 8)) : 0;
  std::vector<std::uint8_t> weights(multiply(slots, 32));
  std::vector<std::int32_t> sums(slots);
  std::vector<float> packed_scales(slots);
  const auto first = Clock::now();
  if (bits == 8 && !generic) {
    ml_cpu_compute::PackMatMulNBitsInt8(b.data(), scales.data(), weights.data(), sums.data(),
                                        packed_scales.data(), k, n);
  }
  const double first_use = std::chrono::duration<double>(Clock::now() - first).count();
  const auto run = [&] {
    if (bits == 8 && !generic) {
      ml_cpu_compute::MatMulNBitsInt8Float32(a.data(), weights.data(), sums.data(),
                                             packed_scales.data(), nullptr, y.data(), m, k, n,
                                             {256, 64, 1});
    } else {
      ml_cpu_compute::MatMulNBits(a.data(), b.data(), scales.data(), nullptr, y.data(),
                                  ml_cpu_compute::DataType::FLOAT, m, k, n, bits, 32, {256, 64, 1});
    }
  };
  run();
  std::vector<double> times;
  for (std::size_t i = 0; i < repeats; ++i) {
    const auto begin = Clock::now();
    run();
    times.push_back(std::chrono::duration<double>(Clock::now() - begin).count());
  }
  std::sort(times.begin(), times.end());
  std::cout << "M,K,N,bits,dtype,threads,input_preparation_s,first_use_pack_s,median_s,"
               "packed_bytes,prepared_bytes,"
               "workspace_bytes,implementation\n"
            << m << ',' << k << ',' << n << ',' << bits << ",float32,1," << preparation << ','
            << first_use << ',' << times[times.size() / 2] << ',' << b.size() << ','
            << (generic ? 0
                        : weights.size() + sums.size() * sizeof(std::int32_t) +
                              packed_scales.size() * sizeof(float))
            << ','
            << (bits == 8 && !generic
                    ? ml_cpu_compute::MatMulNBitsInt8WorkspaceElements(m, k, n, {256, 64, 1}) *
                          sizeof(double)
                    : ml_cpu_compute::kMatMulNBitsInt4WorkspaceBytes)
            << ','
            << (generic     ? "generic"
                : bits == 8 ? ml_cpu_compute::MatMulNBitsInt8Implementation()
                            : ml_cpu_compute::MatMulNBitsInt4Implementation())
            << '\n';
}
