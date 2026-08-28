// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// C++ port of the former Python ``test_parallel_kernels.py``. The kernels are
// invoked from several threads at once on disjoint chunks of an array (and, in
// one case, on the same input from many threads) and the concatenated results
// are checked against the single-threaded reference. This guards against data
// races when kernels run concurrently.

#include "onnx_light_cpu/impl/logical/logical_kernels.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace {

// Split ``count`` indices into ``parts`` contiguous chunks and run ``kernel`` on
// each chunk from its own thread, writing into the shared output buffer.
template <typename Kernel, typename T>
void RunChunksConcurrently(Kernel kernel, const std::vector<T> &input, std::vector<T> &output,
                           int parts) {
  const std::size_t count = input.size();
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(parts));
  for (int p = 0; p < parts; ++p) {
    const std::size_t begin = count * static_cast<std::size_t>(p) / static_cast<std::size_t>(parts);
    const std::size_t end =
        count * static_cast<std::size_t>(p + 1) / static_cast<std::size_t>(parts);
    threads.emplace_back([kernel, &input, &output, begin, end]() {
      kernel(input.data() + begin, output.data() + begin, end - begin);
    });
  }
  for (auto &t : threads) {
    t.join();
  }
}

TEST(KernelsConcurrency, AbsChunksMatchSingleThread) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> x(1'000'003);
  for (auto &v : x) {
    v = dist(rng);
  }

  std::vector<float> got(x.size());
  RunChunksConcurrently(&onnx_light_cpu::AbsFloat32, x, got, 4);

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], std::fabs(x[i])) << "at index " << i;
  }
}

TEST(KernelsConcurrency, ExpChunksMatchSingleThread) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  std::vector<float> x(500'003);
  for (auto &v : x) {
    v = dist(rng);
  }

  std::vector<float> got(x.size());
  RunChunksConcurrently(&onnx_light_cpu::ExpFloat32, x, got, 4);

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(got[i], std::exp(x[i]), std::fabs(std::exp(x[i])) * 1e-4f + 1e-5f)
        << "at index " << i;
  }
}

TEST(KernelsConcurrency, LogChunksMatchSingleThread) {
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> dist(0.1f, 100.0f);
  std::vector<float> x(500'003);
  for (auto &v : x) {
    v = dist(rng);
  }

  std::vector<float> got(x.size());
  RunChunksConcurrently(&onnx_light_cpu::LogFloat32, x, got, 4);

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(got[i], std::log(x[i]), std::fabs(std::log(x[i])) * 1e-4f + 1e-5f)
        << "at index " << i;
  }
}

TEST(KernelsConcurrency, NotChunksMatchSingleThread) {
  std::mt19937 rng(3);
  std::bernoulli_distribution dist(0.5);
  std::vector<std::uint8_t> x(1'000'003);
  for (auto &v : x) {
    v = dist(rng) ? 1u : 0u;
  }

  std::vector<std::uint8_t> got(x.size());
  RunChunksConcurrently(&onnx_light_cpu::NotBool, x, got, 4);

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(got[i], x[i] ? 0u : 1u) << "at index " << i;
  }
}

TEST(KernelsConcurrency, ManyThreadsSameInput) {
  // Running the same input from many threads must not corrupt any result.
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> x(200'003);
  for (auto &v : x) {
    v = dist(rng);
  }

  constexpr int kThreads = 16;
  std::vector<std::vector<float>> outputs(kThreads, std::vector<float>(x.size()));
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(
        [&x, &out = outputs[t]]() { onnx_light_cpu::AbsFloat32(x.data(), out.data(), x.size()); });
  }
  for (auto &t : threads) {
    t.join();
  }

  for (const auto &out : outputs) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(out[i], std::fabs(x[i])) << "at index " << i;
    }
  }
}

} // namespace
