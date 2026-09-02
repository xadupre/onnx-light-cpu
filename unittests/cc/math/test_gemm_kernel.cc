// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"
#include "onnx_light_cpu/impl/math/gemm/float8/float8_conversion.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/simd_level.h"

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
#include "onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// Heap-allocation tracker used by the FP16/BF16 "no expanded operand" tests
// below. While ``g_alloc_recording`` is set, every global allocation updates
// ``g_alloc_peak`` with the largest single request seen (including worker-thread
// allocations), so a test can assert the half-precision path never allocates a
// buffer large enough to widen a full ``M*K`` or ``K*N`` operand (Roadmap PR07.2).
namespace {
std::atomic<bool> g_alloc_recording{false};
std::atomic<std::size_t> g_alloc_peak{0};

void RecordAllocation(std::size_t bytes) {
  if (!g_alloc_recording.load(std::memory_order_relaxed)) {
    return;
  }
  std::size_t previous = g_alloc_peak.load(std::memory_order_relaxed);
  while (bytes > previous &&
         !g_alloc_peak.compare_exchange_weak(previous, bytes, std::memory_order_relaxed)) {
  }
}

// Portable aligned allocation helpers: MSVC does not provide std::aligned_alloc,
// so fall back to _aligned_malloc / _aligned_free on that toolchain.
void *AlignedAlloc(std::size_t alignment, std::size_t size) {
#if defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

void AlignedFree(void *pointer) noexcept {
#if defined(_MSC_VER)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}
} // namespace

void *operator new(std::size_t bytes) {
  RecordAllocation(bytes);
  void *pointer = std::malloc(bytes != 0 ? bytes : 1);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void *operator new[](std::size_t bytes) { return ::operator new(bytes); }

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }

void *operator new(std::size_t bytes, std::align_val_t alignment) {
  RecordAllocation(bytes);
  const std::size_t align = static_cast<std::size_t>(alignment);
  const std::size_t size = (bytes + align - 1) / align * align;
  void *pointer = AlignedAlloc(align, size != 0 ? size : align);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void *operator new[](std::size_t bytes, std::align_val_t alignment) {
  return ::operator new(bytes, alignment);
}

void operator delete(void *pointer, std::align_val_t) noexcept { AlignedFree(pointer); }
void operator delete[](void *pointer, std::align_val_t) noexcept { AlignedFree(pointer); }
void operator delete(void *pointer, std::size_t, std::align_val_t) noexcept {
  AlignedFree(pointer);
}
void operator delete[](void *pointer, std::size_t, std::align_val_t) noexcept {
  AlignedFree(pointer);
}

namespace {

// Straightforward reference Gemm: Y = alpha * op(A) @ op(B) + beta * C.
template <typename T>
std::vector<T> ReferenceGemm(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                             std::size_t K, T alpha, const std::vector<T> &A,
                             const std::vector<T> &B, T beta, const std::vector<T> *C) {
  std::vector<T> Y(M * N, T(0));
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      T acc = T(0);
      for (std::size_t k = 0; k < K; ++k) {
        const T a = trans_a ? A[k * M + m] : A[m * K + k];
        const T b = trans_b ? B[n * K + k] : B[k * N + n];
        acc += a * b;
      }
      T value = alpha * acc;
      if (C != nullptr && beta != T(0)) {
        value += beta * (*C)[m * N + n];
      }
      Y[m * N + n] = value;
    }
  }
  return Y;
}

std::vector<float> RandomVector(std::size_t size, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> v(size);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

std::vector<double> RandomVectorD(std::size_t size, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> v(size);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

} // namespace

TEST(GemmFloat32, MatmulMatchesReference) {
  const std::size_t M = 5, N = 7, K = 3;
  const auto A = RandomVector(M * K, 1);
  const auto B = RandomVector(K * N, 2);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);

  std::vector<float> Y(M * N, -123.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-4f) << "i=" << i;
  }
}

TEST(GemmFloat32, AlphaBetaAndBias) {
  const std::size_t M = 4, N = 6, K = 8;
  const auto A = RandomVector(M * K, 3);
  const auto B = RandomVector(K * N, 4);
  const auto C = RandomVector(M * N, 5);
  const float alpha = 0.5f, beta = 2.0f;
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, alpha, A, B, beta, &C);

  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-3f) << "i=" << i;
  }
}

TEST(GemmFloat32, TypedEpilogueCombinesBroadcastResidualAndRelu) {
  const std::size_t M = 2, N = 3, K = 2;
  const std::vector<float> A = {1, -2, -3, 4};
  const std::vector<float> B = {1, 2, 3, 4, 5, 6};
  const std::vector<float> bias = {1, 2, 3};
  const std::vector<float> residual = {-1, 2};
  std::vector<float> Y(M * N);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.bias = bias.data();
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kRow;
  epilogue.beta = 2.0f;
  epilogue.residual = residual.data();
  epilogue.residual_layout = onnx_light_cpu::GemmBroadcast::kColumn;
  epilogue.residual_scale = 3.0f;
  epilogue.activation = onnx_light_cpu::GemmActivation::kRelu;

  onnx_light_cpu::GemmFloat32WithEpilogue(false, false, M, N, K, 1.0f, A.data(), B.data(), epilogue,
                                          Y.data());

  const std::vector<float> expected = {0, 0, 0, 21, 24, 27};
  EXPECT_EQ(Y, expected);
}

TEST(GemmFloat32, TypedEpilogueCombinesPostOpsAndFloat16Conversion) {
  const std::size_t M = 2, N = 3, K = 2;
  const std::vector<float> A = {1, -2, -3, 4};
  const std::vector<float> B = {1, 2, 3, 4, 5, 6};
  const std::vector<float> bias = {1, 2, 3};
  const std::vector<float> residual = {-1, 2};
  std::vector<float> workspace(M * N);
  std::vector<std::uint16_t> output(M * N);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.bias = bias.data();
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kRow;
  epilogue.beta = 2.0f;
  epilogue.residual = residual.data();
  epilogue.residual_layout = onnx_light_cpu::GemmBroadcast::kColumn;
  epilogue.residual_scale = 3.0f;
  epilogue.activation = onnx_light_cpu::GemmActivation::kRelu;
  epilogue.output_conversion = onnx_light_cpu::GemmOutputConversion::kFloat16;
  epilogue.converted_output = output.data();

  onnx_light_cpu::GemmFloat32WithEpilogue(false, false, M, N, K, 1.0f, A.data(), B.data(), epilogue,
                                          workspace.data());

  const std::vector<std::uint16_t> expected = {0x0000, 0x0000, 0x0000, 0x4d40, 0x4e00, 0x4ec0};
  EXPECT_EQ(output, expected);
}

TEST(GemmFloat64, TypedEpilogueSupportsScalarBias) {
  const std::vector<double> A = {1, 2, 3, 4};
  const std::vector<double> B = {1, 0, 0, 1};
  const double bias = 2.5;
  std::vector<double> Y(4);
  onnx_light_cpu::GemmEpilogue<double> epilogue;
  epilogue.bias = &bias;
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kScalar;
  epilogue.beta = 2.0;

  onnx_light_cpu::GemmFloat64WithEpilogue(false, false, 2, 2, 2, 1.0, A.data(), B.data(), epilogue,
                                          Y.data());

  const std::vector<double> expected = {6, 7, 8, 9};
  EXPECT_EQ(Y, expected);
}

TEST(GemmFloat32, UnitScalesWithBiasAcrossChunks) {
  const std::size_t M = 7, N = 19, K = 300;
  const auto A = RandomVector(M * K, 91);
  const auto B = RandomVector(K * N, 92);
  const auto C = RandomVector(M * N, 93);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 1.0f, &C);
  std::vector<float> Y(M * N, 0.0f);

  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 1.0f, C.data(),
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, TransposeVariants) {
  // These dimensions force the general path even with 16-lane AVX-512.
  const std::size_t M = 8, N = 17, K = 40;
  for (int ta = 0; ta < 2; ++ta) {
    for (int tb = 0; tb < 2; ++tb) {
      const bool trans_a = ta != 0;
      const bool trans_b = tb != 0;
      const auto A = RandomVector((trans_a ? K * M : M * K), 10 + ta * 2 + tb);
      const auto B = RandomVector((trans_b ? N * K : K * N), 20 + ta * 2 + tb);
      const auto expected =
          ReferenceGemm<float>(trans_a, trans_b, M, N, K, 1.0f, A, B, 0.0f, nullptr);
      std::vector<float> Y(M * N, 0.0f);
      onnx_light_cpu::GemmFloat32(trans_a, trans_b, M, N, K, 1.0f, A.data(), B.data(), 0.0f,
                                  nullptr, Y.data());
      for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(Y[i], expected[i], 2e-3f)
            << "trans_a=" << trans_a << " trans_b=" << trans_b << " i=" << i;
      }
    }
  }
}

TEST(GemmFloat32, SpecializedRowVariantsMatchReference) {
  const std::size_t N = 19, K = 40;
  for (std::size_t M = 1; M <= 4; ++M) {
    const auto A = RandomVector(M * K, static_cast<unsigned>(80 + M));
    const auto B = RandomVector(K * N, static_cast<unsigned>(90 + M));
    const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
    std::vector<float> Y(M * N, 0.0f);

    onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                                Y.data());

    for (std::size_t i = 0; i < M * N; ++i) {
      EXPECT_NEAR(Y[i], expected[i], 2e-3f) << "M=" << M << " i=" << i;
    }
  }
}

TEST(GemmFloat32, BetaZeroIgnoresBias) {
  const std::size_t M = 3, N = 3, K = 3;
  const auto A = RandomVector(M * K, 6);
  const auto B = RandomVector(K * N, 7);
  // C is deliberately garbage; beta == 0 must ignore it entirely.
  std::vector<float> C(M * N, std::numeric_limits<float>::quiet_NaN());
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-4f) << "i=" << i;
  }
}

TEST(GemmFloat32, LargeMatrixParallel) {
  const std::size_t M = 128, N = 96, K = 64;
  const auto A = RandomVector(M * K, 11);
  const auto B = RandomVector(K * N, 12);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

// Dimensions larger than the internal cache tiles (kGemmTileN=256, kGemmTileM=64)
// so the output is split across several column panels and row blocks. This is a
// regression test for a bug where the micro-kernels ignored the column-panel
// offset ``n0`` when writing ``Y``, corrupting every panel past the first.
TEST(GemmFloat32, MultiPanelExceedsTiles) {
  const std::size_t M = 130, N = 300, K = 40;
  const auto A = RandomVector(M * K, 21);
  const auto B = RandomVector(K * N, 22);
  const auto C = RandomVector(M * N, 23);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 1.5f, &C);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 0.75f, A.data(), B.data(), 1.5f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, PlannedGridCoversRowAndColumnPanels) {
  const std::size_t M = 70, N = 100, K = 130;
  const auto A = RandomVector(M * K, 31);
  const auto B = RandomVector(K * N, 32);
  const auto C = RandomVector(M * N, 33);
  const auto expected = ReferenceGemm<float>(false, true, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{32, 48, 64, 4, 16};

  onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, true, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, FusedPackingMatchesTransposedBAndBias) {
  constexpr std::size_t M = 12;
  constexpr std::size_t N = 2048;
  constexpr std::size_t K = 64;
  const auto A = RandomVector(M * K, 34);
  const auto B = RandomVector(N * K, 35);
  const auto C = RandomVector(M * N, 36);
  const auto expected = ReferenceGemm<float>(false, true, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{6, 1024, 64, 6, 16};
  onnx_light_cpu::ExecutionExecutorView executor{
      nullptr, 32,
      [](void *, int64_t num_blocks, void *task_context, onnx_light_cpu::ExecutionBlockFn task) {
        for (int64_t block = 0; block < num_blocks; ++block) {
          task(task_context, block);
        }
      }};
  onnx_light_cpu::ExecutionExecutorScope scope(&executor);

  onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, true, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

// Forces the six-row register tile that the AMD Zen AVX2+FMA profile selects
// (Roadmap PR06.5) so the mr == 6 micro-kernel path is exercised for
// correctness, including a row tail (M not a multiple of six) and a column tail.
TEST(GemmFloat32, PlannedSixRowRegisterTileMatchesReference) {
  const std::size_t M = 41, N = 53, K = 130;
  const auto A = RandomVector(M * K, 61);
  const auto B = RandomVector(K * N, 62);
  const auto C = RandomVector(M * N, 63);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{24, 48, 64, 6, 16};

  onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, false, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, PlannedTwentyFourRowRegisterTileMatchesReference) {
  const std::size_t M = 40, N = 37, K = 45;
  const auto A = RandomVector(M * K, 64);
  const auto B = RandomVector(K * N, 65);
  const auto C = RandomVector(M * N, 66);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{48, 48, 64, 24, 16};

  onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, false, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

// The tile loops walk one packed column micro-panel at a time, so a column
// panel wider than that micro-panel is split into several contiguous slices
// with a narrower trailing one. This covers those slices together with a
// trailing column panel, a row tail, several k chunks, and a transposed B.
TEST(GemmFloat32, PlannedColumnMicroPanelsCoverPartialSlices) {
  const std::size_t M = 41, N = 100, K = 70;
  const auto A = RandomVector(M * K, 71);
  const auto B = RandomVector(K * N, 72);
  const auto C = RandomVector(M * N, 73);
  const auto expected = ReferenceGemm<float>(false, true, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{24, 96, 32, 6, 16};

  onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, true, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

// K larger than the internal K-blocking chunk (kGemmTileK=256) so the k
// reduction is split across several chunks that must be accumulated together
// (GemmAccumMode::kAccumulate) instead of each overwriting Y. Also exercises
// transpose_a/transpose_b together with a k-chunk boundary that does not align
// with kMR-sized row groups.
TEST(GemmFloat32, LargeKSpansMultipleChunks) {
  const std::size_t M = 5, N = 7, K = 600;
  const auto A = RandomVector(M * K, 41);
  const auto B = RandomVector(K * N, 42);
  const auto C = RandomVector(M * N, 43);
  const float alpha = 1.25f, beta = 0.5f;
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, alpha, A, B, beta, &C);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, SplitKMatchesReference) {
  const std::size_t M = 2, N = 2, K = 4096;
  const auto A = RandomVector(M * K, 51);
  const auto B = RandomVector(K * N, 52);
  const auto C = RandomVector(M * N, 53);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 0.5f, &C);
  std::vector<float> Y(M * N, 0.0f);

  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 0.75f, A.data(), B.data(), 0.5f, C.data(),
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 5e-3f) << "i=" << i;
  }
}

TEST(GemmFloat32, SplitKSupportsTransposedInputs) {
  const std::size_t M = 2, N = 2, K = 4096;
  const auto A = RandomVector(K * M, 54);
  const auto B = RandomVector(N * K, 55);
  const auto expected = ReferenceGemm<float>(true, true, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N);

  onnx_light_cpu::GemmFloat32(true, true, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

// M much smaller than N: with the flattened (row block, column panel) task
// list this shape still spans multiple column panels, exercising the case
// that motivated splitting the parallel work across N and not only M.
TEST(GemmFloat32, SkinnyMWideNMultipleColumnPanels) {
  const std::size_t M = 2, N = 1000, K = 33;
  const auto A = RandomVector(M * K, 44);
  const auto B = RandomVector(K * N, 45);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, SkinnyMSingleRowRegisterPanelsMatchReference) {
  constexpr std::size_t M = 1;
  constexpr std::size_t N = 512;
  constexpr std::size_t K = 73;
  const auto A = RandomVector(M * K, 46);
  const auto B = RandomVector(K * N, 47);
  const auto C = RandomVector(M * N, 48);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

// A single output column (N == 1) with many rows and a long reduction hits the
// skinny-N path, whose unit-stride dot product carries sixteen partial sums for
// full-width AVX. K == 1000 is not a multiple of that unroll, so the scalar tail
// must finish the reduction exactly for every transpose layout and with bias.
TEST(GemmFloat32, SkinnyNColumnVectorLongKTail) {
  const std::size_t M = 200, N = 1, K = 1000;
  for (const bool trans_a : {false, true}) {
    for (const bool trans_b : {false, true}) {
      const auto A = RandomVector(M * K, 61);
      const auto B = RandomVector(K * N, 62);
      const auto C = RandomVector(M * N, 63);
      const auto expected = ReferenceGemm<float>(trans_a, trans_b, M, N, K, 0.75f, A, B, 1.5f, &C);
      std::vector<float> Y(M * N, 0.0f);
      onnx_light_cpu::GemmFloat32(trans_a, trans_b, M, N, K, 0.75f, A.data(), B.data(), 1.5f,
                                  C.data(), Y.data());
      for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(Y[i], expected[i], 1e-2f)
            << "trans_a=" << trans_a << " trans_b=" << trans_b << " i=" << i;
      }
    }
  }
}

TEST(GemmFloat32, EmptyKGivesBiasOnly) {
  const std::size_t M = 3, N = 4, K = 0;
  const auto C = RandomVector(M * N, 8);
  std::vector<float> Y(M * N, 7.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, nullptr, nullptr, 3.0f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], 3.0f * C[i], 1e-5f) << "i=" << i;
  }
}

TEST(GemmFloat32, EmptyMOrNReturnsWithoutWriting) {
  // M == 0 and N == 0 both hit the early return in GemmImpl: the output buffer
  // must be left untouched (here it is empty, so the call must simply not crash).
  std::vector<float> Y;
  onnx_light_cpu::GemmFloat32(false, false, /*M=*/0, /*N=*/4, /*K=*/3, 1.0f, nullptr, nullptr, 0.0f,
                              nullptr, Y.data());
  onnx_light_cpu::GemmFloat32(false, false, /*M=*/4, /*N=*/0, /*K=*/3, 1.0f, nullptr, nullptr, 0.0f,
                              nullptr, Y.data());
  SUCCEED();
}

TEST(GemmFloat32, SingleRowColumnTail) {
  // M == 1 exercises the mr == 1 register block and N == 5 leaves a column tail
  // (5 % 8 != 0 for float32) that the scalar micro-kernel finishes, with no bias.
  const std::size_t M = 1, N = 5, K = 4;
  const auto A = RandomVector(M * K, 31);
  const auto B = RandomVector(K * N, 32);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-4f) << "i=" << i;
  }
}

TEST(GemmFloat64, MatmulMatchesReference) {
  const std::size_t M = 5, N = 7, K = 9;
  const auto A = RandomVectorD(M * K, 21);
  const auto B = RandomVectorD(K * N, 22);
  const auto C = RandomVectorD(M * N, 23);
  const double alpha = 1.5, beta = -0.5;
  const auto expected = ReferenceGemm<double>(true, false, M, N, K, alpha, A, B, beta, &C);
  std::vector<double> Y(M * N, 0.0);
  // trans_a: A stored as K x M.
  onnx_light_cpu::GemmFloat64(true, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

TEST(GemmFloat64, TransposeVariants) {
  // The float32 path already covers every trans_a/trans_b combination; do the
  // same for float64 so the double transpose-b packing and both-transpose
  // address arithmetic are exercised too.
  // These dimensions force the general path even with 8-lane AVX-512.
  const std::size_t M = 8, N = 9, K = 40;
  for (int ta = 0; ta < 2; ++ta) {
    for (int tb = 0; tb < 2; ++tb) {
      const bool trans_a = ta != 0;
      const bool trans_b = tb != 0;
      const auto A = RandomVectorD((trans_a ? K * M : M * K), 40 + ta * 2 + tb);
      const auto B = RandomVectorD((trans_b ? N * K : K * N), 50 + ta * 2 + tb);
      const auto expected =
          ReferenceGemm<double>(trans_a, trans_b, M, N, K, 1.0, A, B, 0.0, nullptr);
      std::vector<double> Y(M * N, 0.0);
      onnx_light_cpu::GemmFloat64(trans_a, trans_b, M, N, K, 1.0, A.data(), B.data(), 0.0, nullptr,
                                  Y.data());
      for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(Y[i], expected[i], 1e-9)
            << "trans_a=" << trans_a << " trans_b=" << trans_b << " i=" << i;
      }
    }
  }
}

TEST(GemmFloat64, SpecializedRowVariantsMatchReference) {
  const std::size_t N = 11, K = 40;
  for (std::size_t M = 1; M <= 4; ++M) {
    const auto A = RandomVectorD(M * K, static_cast<unsigned>(100 + M));
    const auto B = RandomVectorD(K * N, static_cast<unsigned>(110 + M));
    const auto expected = ReferenceGemm<double>(false, false, M, N, K, 1.0, A, B, 0.0, nullptr);
    std::vector<double> Y(M * N, 0.0);

    onnx_light_cpu::GemmFloat64(false, false, M, N, K, 1.0, A.data(), B.data(), 0.0, nullptr,
                                Y.data());

    for (std::size_t i = 0; i < M * N; ++i) {
      EXPECT_NEAR(Y[i], expected[i], 1e-9) << "M=" << M << " i=" << i;
    }
  }
}

// Same K-blocking regression as GemmFloat32.LargeKSpansMultipleChunks, for the
// float64 micro-kernels (AVX/SSE2 4/2-wide accumulate paths).
TEST(GemmFloat64, LargeKSpansMultipleChunks) {
  const std::size_t M = 5, N = 7, K = 600;
  const auto A = RandomVectorD(M * K, 70);
  const auto B = RandomVectorD(K * N, 71);
  const auto C = RandomVectorD(M * N, 72);
  const double alpha = 1.25, beta = 0.5;
  const auto expected = ReferenceGemm<double>(false, false, M, N, K, alpha, A, B, beta, &C);
  std::vector<double> Y(M * N, 0.0);
  onnx_light_cpu::GemmFloat64(false, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

TEST(GemmFloat64, LargeKSquareOutputMatchesReference) {
  constexpr std::size_t M = 32;
  constexpr std::size_t N = 32;
  for (const std::size_t K : {1024, 4096}) {
    const auto A = RandomVectorD(M * K, 79 + K);
    const auto B = RandomVectorD(K * N, 80 + K);
    const auto C = RandomVectorD(M * N, 81 + K);
    const auto expected = ReferenceGemm<double>(false, false, M, N, K, 0.75, A, B, 1.25, &C);
    std::vector<double> Y(M * N);

    onnx_light_cpu::GemmFloat64(false, false, M, N, K, 0.75, A.data(), B.data(), 1.25, C.data(),
                                Y.data());

    for (std::size_t i = 0; i < M * N; ++i) {
      EXPECT_NEAR(Y[i], expected[i], 1e-9) << "K=" << K << " i=" << i;
    }
  }
}

TEST(GemmFloat64, PlannedSixRowRegisterTileMatchesReference) {
  const std::size_t M = 41, N = 53, K = 130;
  const auto A = RandomVectorD(M * K, 76);
  const auto B = RandomVectorD(K * N, 77);
  const auto C = RandomVectorD(M * N, 78);
  const auto expected = ReferenceGemm<double>(false, false, M, N, K, 0.75, A, B, 1.25, &C);
  std::vector<double> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{24, 48, 64, 6, 8};

  onnx_light_cpu::detail::GemmFloat64Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, false, M, N, K, 0.75, A.data(), B.data(), 1.25, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

TEST(GemmFloat64, SplitKSupportsTransposedBAndBias) {
  const std::size_t M = 2, N = 2, K = 4096;
  const auto A = RandomVectorD(M * K, 73);
  const auto B = RandomVectorD(N * K, 74);
  const auto C = RandomVectorD(M * N, 75);
  const auto expected = ReferenceGemm<double>(false, true, M, N, K, 0.5, A, B, 2.0, &C);
  std::vector<double> Y(M * N);

  onnx_light_cpu::GemmFloat64(false, true, M, N, K, 0.5, A.data(), B.data(), 2.0, C.data(),
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

TEST(GemmFloat64, NoBiasColumnTail) {
  // N == 7 leaves a column tail (7 % 4 != 0 for float64) that is finished by the
  // scalar micro-kernel; with beta == 0 / C == nullptr this exercises the
  // bias-free branch of the scalar tail handler.
  const std::size_t M = 3, N = 7, K = 5;
  const auto A = RandomVectorD(M * K, 60);
  const auto B = RandomVectorD(K * N, 61);
  const auto expected = ReferenceGemm<double>(false, false, M, N, K, 1.0, A, B, 0.0, nullptr);
  std::vector<double> Y(M * N, 0.0);
  onnx_light_cpu::GemmFloat64(false, false, M, N, K, 1.0, A.data(), B.data(), 0.0, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

namespace {

// Round-trips a float32 vector through the half format under test so the
// reference computation observes the same rounded inputs the kernel does.
std::vector<std::uint16_t> NarrowHalf(const std::vector<float> &values, bool is_bfloat16) {
  std::vector<std::uint16_t> bits(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    bits[i] = is_bfloat16 ? onnx_light_cpu::detail::FloatToBFloat16Bits(values[i])
                          : onnx_light_cpu::detail::FloatToFloat16Bits(values[i]);
  }
  return bits;
}

std::vector<float> WidenHalf(const std::vector<std::uint16_t> &bits, bool is_bfloat16) {
  std::vector<float> values(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) {
    values[i] = is_bfloat16 ? onnx_light_cpu::detail::Bfloat16BitsToFloat(bits[i])
                            : onnx_light_cpu::detail::Float16BitsToFloat(bits[i]);
  }
  return values;
}

// Compares the native convert-while-packing half GEMM against the equivalent
// widen-then-float32 reference for one shape/transpose/bias configuration.
void CheckGemmHalf(bool is_bfloat16, bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                   std::size_t K, bool with_bias, unsigned seed) {
  const auto a_f = RandomVector(M * K, seed);
  const auto b_f = RandomVector(K * N, seed + 1);
  const auto a_bits = NarrowHalf(a_f, is_bfloat16);
  const auto b_bits = NarrowHalf(b_f, is_bfloat16);
  // The reference multiplies the exact values the kernel sees after rounding.
  const auto a_round = WidenHalf(a_bits, is_bfloat16);
  const auto b_round = WidenHalf(b_bits, is_bfloat16);

  const float alpha = 0.75f;
  const float beta = with_bias ? 1.5f : 0.0f;
  std::vector<float> bias_round;
  std::vector<std::uint16_t> bias_bits;
  const std::vector<float> *bias_ptr = nullptr;
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  if (with_bias) {
    const auto bias_f = RandomVector(M * N, seed + 2);
    bias_bits = NarrowHalf(bias_f, is_bfloat16);
    bias_round = WidenHalf(bias_bits, is_bfloat16);
    bias_ptr = &bias_round;
    epilogue.bias = bias_round.data();
    epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kMatrix;
    epilogue.beta = beta;
  }

  const auto expected =
      ReferenceGemm<float>(trans_a, trans_b, M, N, K, alpha, a_round, b_round, beta, bias_ptr);

  std::vector<std::uint16_t> out_bits(M * N, 0);
  std::vector<float> workspace(M * N, -1.0f);
  epilogue.output_conversion = is_bfloat16 ? onnx_light_cpu::GemmOutputConversion::kBFloat16
                                           : onnx_light_cpu::GemmOutputConversion::kFloat16;
  epilogue.converted_output = out_bits.data();
  onnx_light_cpu::GemmHalfWithEpilogue(is_bfloat16, trans_a, trans_b, M, N, K, alpha, a_bits.data(),
                                       b_bits.data(), epilogue, workspace.data());

  const auto out = WidenHalf(out_bits, is_bfloat16);
  // BF16 keeps only 8 mantissa bits, so allow a looser tolerance than FP16.
  const float rtol = is_bfloat16 ? 6e-2f : 8e-3f;
  for (std::size_t i = 0; i < M * N; ++i) {
    const float tol = rtol * std::max(1.0f, std::abs(expected[i]));
    EXPECT_NEAR(out[i], expected[i], tol) << "i=" << i;
  }
}

} // namespace

TEST(GemmHalf, Float16MatchesWidenReference) {
  CheckGemmHalf(false, false, false, 5, 7, 3, false, 101);
  CheckGemmHalf(false, false, false, 4, 6, 8, true, 111);
  // K spanning several blocking chunks and a wide N with column tails.
  CheckGemmHalf(false, false, false, 17, 33, 40, true, 121);
}

TEST(GemmHalf, Float16VectorizedPackingTails) {
  // Contiguous N and K runs that are not multiples of the eight-lane F16C
  // conversion width, so the packing path exercises both the vectorized bulk
  // and the scalar tail on the general blocked algorithm.
  CheckGemmHalf(false, false, false, 19, 37, 45, false, 141);
  CheckGemmHalf(false, false, false, 33, 70, 66, true, 151);
}

TEST(GemmHalf, Float16TransposeVariants) {
  for (bool trans_a : {false, true}) {
    for (bool trans_b : {false, true}) {
      CheckGemmHalf(false, trans_a, trans_b, 6, 5, 7, true, 131);
    }
  }
}

TEST(GemmHalf, BFloat16MatchesWidenReference) {
  CheckGemmHalf(true, false, false, 5, 7, 3, false, 201);
  CheckGemmHalf(true, false, false, 4, 6, 8, true, 211);
  CheckGemmHalf(true, true, true, 6, 5, 7, true, 221);
}

TEST(GemmHalf, BFloat16VectorizedPackingTails) {
  // Same tail coverage as the FP16 case for the AVX2 BF16 packing conversion.
  CheckGemmHalf(true, false, false, 19, 37, 45, false, 241);
  CheckGemmHalf(true, false, false, 33, 70, 66, true, 251);
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
TEST(GemmHalf, Avx2BlockedTransposePackingHandlesTailsAndNaNs) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2) {
    GTEST_SKIP() << "AVX2 transpose packing requires AVX2.";
  }
  constexpr std::size_t rows = 17;
  constexpr std::size_t columns = 19;
  constexpr std::size_t stride = rows + 3;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> values(columns * stride, 0.0f);
  for (std::size_t column = 0; column < columns; ++column) {
    for (std::size_t row = 0; row < rows; ++row) {
      values[column * stride + row] =
          (row == 8 && column == 8)
              ? nan
              : static_cast<float>(static_cast<std::ptrdiff_t>(row * 7 + column) - 50);
    }
  }

  for (const bool is_bfloat16 : {false, true}) {
    if (!is_bfloat16 && !onnx_light_cpu::CpuSupportsF16C()) {
      continue;
    }
    const auto bits = NarrowHalf(values, is_bfloat16);
    std::vector<float> packed(rows * columns);
    if (is_bfloat16) {
      onnx_light_cpu::GemmPackTransposeBFloat16ToFloat32_AVX2(bits.data(), stride, packed.data(),
                                                              rows, columns);
    } else {
#ifdef ONNX_LIGHT_CPU_HAVE_F16C
      onnx_light_cpu::GemmPackTransposeFloat16ToFloat32_F16C(bits.data(), stride, packed.data(),
                                                             rows, columns);
#endif
    }
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        const std::uint16_t expected_bits = bits[column * stride + row];
        const float expected = is_bfloat16
                                   ? onnx_light_cpu::detail::Bfloat16BitsToFloat(expected_bits)
                                   : onnx_light_cpu::detail::Float16BitsToFloat(expected_bits);
        const float actual = packed[row * columns + column];
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          EXPECT_EQ(actual, expected) << "row=" << row << " column=" << column;
        }
      }
    }
  }
}
#endif

// Roadmap PR10.1 correctness gate: run the same mixed corpus for FLOAT16 and
// BFLOAT16 so native kernels and fallback paths (transpose combinations,
// non-trivial tails, and empty-K bias handling) are validated together.
TEST(GemmHalf, HalfCorrectnessGateCorpus) {
  struct Case {
    bool trans_a;
    bool trans_b;
    std::size_t M;
    std::size_t N;
    std::size_t K;
    bool with_bias;
    unsigned seed;
  };
  const Case cases[] = {
      {false, false, 20, 37, 41, true, 271}, // general blocked path with tails
      {true, false, 17, 19, 33, true, 281},  // transposed A with native/fallback B
      {false, true, 11, 13, 29, false, 291}, // transposed B fallback path
      {true, true, 9, 15, 23, true, 301},    // both operands transposed
      {false, false, 3, 9, 5, false, 311},   // short rows with vector/scalar tails
      {false, false, 6, 8, 0, true, 321},    // empty-K keeps scaled bias only
  };

  for (const Case &test_case : cases) {
    CheckGemmHalf(false, test_case.trans_a, test_case.trans_b, test_case.M, test_case.N,
                  test_case.K, test_case.with_bias, test_case.seed);
    CheckGemmHalf(true, test_case.trans_a, test_case.trans_b, test_case.M, test_case.N, test_case.K,
                  test_case.with_bias, test_case.seed + 1000);
  }
}

// Roadmap PR08.1: contiguous FP16/BF16 packing runs whose length crosses the
// eight-lane vectorized conversion width and leaves every possible 1..7 element
// scalar tail. Non-transposed ``A``/``B`` keep the packing contiguous so the
// vectorized ``PackConvertContiguous`` path (AVX2/F16C on x86, NEON ``FCVTL`` /
// shift on ARM) and its scalar tail both run. Every configuration must match the
// widen-then-float32 reference bit-for-bit within the half tolerance.
TEST(GemmHalf, HalfVectorizedPackingTailRemainders) {
  for (std::size_t tail = 1; tail <= 7; ++tail) {
    const std::size_t N = 8 + tail;  // contiguous B run of 8 + tail per row
    const std::size_t K = 16 + tail; // contiguous A run of 8 + tail per row
    const unsigned seed = static_cast<unsigned>(300 + tail);
    CheckGemmHalf(false, false, false, 3, N, K, tail % 2 == 0, seed);
    CheckGemmHalf(true, false, false, 3, N, K, tail % 2 == 0, seed + 50);
  }
}

// Roadmap PR07.3: general FLOAT16 shapes whose N spans several 16-lane native
// AVX-512FP16 column vectors and leaves a scalar tail, with M covering several
// register-row blocks. On an AVX-512FP16 CPU these exercise the native kernel
// (both the vector body and the scalar column tail); elsewhere they keep the
// converting float32 fallback. Both must match the widen-then-float32 reference.
TEST(GemmHalf, Float16NativeGeneralColumnTails) {
  CheckGemmHalf(false, false, false, 20, 48, 40, false, 161); // N == 3 * 16, no tail
  CheckGemmHalf(false, false, false, 20, 35, 40, true, 171);  // N == 2 * 16 + 3 tail
  CheckGemmHalf(false, true, false, 17, 19, 33, true, 181);   // trans_a, N == 16 + 3 tail
}

namespace {

struct ThreadedExecutor {
  std::atomic<std::size_t> dispatches{0};
  std::atomic<std::size_t> maximum_active{0};
  std::atomic<std::size_t> maximum_blocks{0};

  static void Run(void *context, int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<ThreadedExecutor *>(context);
    self.dispatches.fetch_add(1, std::memory_order_relaxed);
    const std::size_t blocks = static_cast<std::size_t>(num_blocks);
    std::size_t maximum_blocks = self.maximum_blocks.load(std::memory_order_relaxed);
    while (blocks > maximum_blocks && !self.maximum_blocks.compare_exchange_weak(
                                          maximum_blocks, blocks, std::memory_order_relaxed)) {
    }
    std::atomic<std::size_t> active{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_blocks));
    for (int64_t block = 0; block < num_blocks; ++block) {
      workers.emplace_back([&, block] {
        const std::size_t now = active.fetch_add(1, std::memory_order_relaxed) + 1;
        std::size_t maximum = self.maximum_active.load(std::memory_order_relaxed);
        while (now > maximum && !self.maximum_active.compare_exchange_weak(
                                    maximum, now, std::memory_order_relaxed)) {
        }
        task(task_context, block);
        active.fetch_sub(1, std::memory_order_relaxed);
      });
    }
    for (std::thread &worker : workers) {
      worker.join();
    }
  }
};

struct ProductiveExecutor {
  const float *output = nullptr;
  std::size_t output_size = 0;
  std::size_t productive_blocks = 0;
  std::vector<std::size_t> writes_per_block;

  static void Run(void *context, int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<ProductiveExecutor *>(context);
    for (int64_t block = 0; block < num_blocks; ++block) {
      const std::size_t before = static_cast<std::size_t>(std::count_if(
          self.output, self.output + self.output_size, [](float value) { return value != 0.0f; }));
      task(task_context, block);
      const std::size_t after = static_cast<std::size_t>(std::count_if(
          self.output, self.output + self.output_size, [](float value) { return value != 0.0f; }));
      self.productive_blocks += static_cast<std::size_t>(after > before);
      self.writes_per_block.push_back(after - before);
    }
  }
};

TEST(GemmFloat32, MediumSquareBalancesRowsAcrossParticipants) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX512) {
    GTEST_SKIP() << "medium-square direct scheduling is specific to AVX-512";
  }
  constexpr std::size_t M = 256;
  constexpr std::size_t N = 256;
  constexpr std::size_t K = 256;
  const std::vector<float> A(M * K, 1.0f);
  const std::vector<float> B(K * N, 1.0f);
  std::vector<float> Y(M * N);
  ProductiveExecutor executor{Y.data(), Y.size()};
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ProductiveExecutor::Run};

  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    const onnx_light_cpu::GemmPlan<float> plan(
        onnx_light_cpu::GemmPlanOptions<float>{false, false, M, N, K});
    plan.Execute(A.data(), B.data(), nullptr, Y.data());
  }

  ASSERT_EQ(executor.writes_per_block.size(), 4u);
  const auto [minimum, maximum] =
      std::minmax_element(executor.writes_per_block.begin(), executor.writes_per_block.end());
  EXPECT_LE(*maximum - *minimum, onnx_light_cpu::kGemmAVX512MR * N);
  for (float value : Y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(K));
  }
}

TEST(GemmFloat32, RawLargeKSplitBoundsParticipants) {
  constexpr std::size_t M = 32;
  constexpr std::size_t N = 32;
  constexpr std::size_t K = 16384;
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 64, &ThreadedExecutor::Run};
  const std::vector<float> A(M * K, 1.0f);
  const std::vector<float> B(K * N, 1.0f);
  std::vector<float> Y(M * N);

  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                                Y.data());
  }

  EXPECT_GT(executor.maximum_blocks.load(std::memory_order_relaxed), 1u);
  EXPECT_LE(executor.maximum_blocks.load(std::memory_order_relaxed), 16u);
  for (float value : Y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(K));
  }
}

// Runs the native FLOAT16 general driver (Roadmap PR07.3) with an injected
// micro-kernel and compares ``alpha * op(A) @ B`` against the equivalent
// widen-then-float32 reference. The driver, FLOAT16 A packing (including
// ``trans_a``), and the kernel's 16-lane body plus scalar column tail are all
// exercised here; the portable scalar member runs everywhere while the
// AVX-512FP16 member only runs on capable hardware.
void CheckGemmFp16NativeDriver(bool trans_a, std::size_t M, std::size_t N, std::size_t K,
                               float alpha, unsigned seed,
                               onnx_light_cpu::GemmFp16MicroKernel kernel,
                               const onnx_light_cpu::GemmBlocking *blocking = nullptr) {
  const auto a_f = RandomVector(trans_a ? K * M : M * K, seed);
  const auto b_f = RandomVector(K * N, seed + 1);
  const auto a_bits = NarrowHalf(a_f, false);
  const auto b_bits = NarrowHalf(b_f, false);
  const auto a_round = WidenHalf(a_bits, false);
  const auto b_round = WidenHalf(b_bits, false);
  const auto expected =
      ReferenceGemm<float>(trans_a, false, M, N, K, alpha, a_round, b_round, 0.0f, nullptr);

  std::vector<float> Y(M * N, -1.0f);
  onnx_light_cpu::detail::GemmFp16NativeGeneral(trans_a, M, N, K, alpha, a_bits.data(),
                                                b_bits.data(), Y.data(), kernel, 6, blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    const float tol = 1e-3f * std::max(1.0f, std::abs(expected[i]));
    EXPECT_NEAR(Y[i], expected[i], tol) << "i=" << i;
  }
}

} // namespace

TEST(GemmFloat32, LargeBPanelPackingFusesWithCompute) {
  constexpr std::size_t M = 16;
  constexpr std::size_t N = 1024;
  constexpr std::size_t K = 512;
  const onnx_light_cpu::GemmBlocking blocking{8, 1024, 512, 4, 16};
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};

  for (bool trans_b : {false, true}) {
    const auto A = RandomVector(M * K, trans_b ? 401 : 400);
    const auto B = RandomVector(K * N, trans_b ? 403 : 402);
    const auto expected = ReferenceGemm<float>(false, trans_b, M, N, K, 1.0f, A, B, 0.0f, nullptr);
    std::vector<float> Y(M * N);
    {
      onnx_light_cpu::ExecutionExecutorScope scope(&view);
      onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
          false, trans_b, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr, Y.data(), &blocking);
    }
    for (std::size_t i = 0; i < Y.size(); ++i) {
      EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "trans_b=" << trans_b << " i=" << i;
    }
  }
  EXPECT_EQ(executor.dispatches.load(std::memory_order_relaxed), 2u);
  EXPECT_GT(executor.maximum_blocks.load(std::memory_order_relaxed), 1u);
}

TEST(GemmFloat32, SmallBPanelPackingStaysInline) {
  constexpr std::size_t M = 8;
  constexpr std::size_t N = 256;
  constexpr std::size_t K = 256;
  const onnx_light_cpu::GemmBlocking blocking{8, 256, 256, 4, 16};
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  const auto A = RandomVector(M * K, 404);
  const auto B = RandomVector(K * N, 405);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N);
  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
        false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr, Y.data(), &blocking);
  }

  for (std::size_t i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
  EXPECT_EQ(executor.dispatches.load(std::memory_order_relaxed), 0u);
}

TEST(GemmFloat32, ImbalancedTileCountUsesAllRequestedBlocks) {
  constexpr std::size_t M = 768;
  constexpr std::size_t N = 512;
  constexpr std::size_t K = 255;
  const onnx_light_cpu::GemmBlocking blocking{256, 128, 255, 4, 16};
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 10, &ThreadedExecutor::Run};
  const std::vector<float> A(M * K, 1.0f);
  const std::vector<float> B(K * N, 1.0f);
  std::vector<float> Y(M * N);
  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
        false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr, Y.data(), &blocking);
  }

  EXPECT_EQ(executor.maximum_blocks.load(std::memory_order_relaxed), 10u);
  for (float value : Y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(K));
  }
}

TEST(GemmFp16Native, ScalarKernelMatchesReference) {
  // Portable scalar member: several row blocks (M > 6), exact-16 and tail N, an
  // empty K, ``trans_a``, and a non-unit alpha.
  CheckGemmFp16NativeDriver(false, 13, 16, 20, 1.0f, 301,
                            &onnx_light_cpu::GemmMicroKernel_ScalarFp16);
  CheckGemmFp16NativeDriver(false, 20, 35, 40, 0.75f, 311,
                            &onnx_light_cpu::GemmMicroKernel_ScalarFp16);
  CheckGemmFp16NativeDriver(true, 9, 19, 33, 0.5f, 321,
                            &onnx_light_cpu::GemmMicroKernel_ScalarFp16);
  CheckGemmFp16NativeDriver(false, 7, 5, 0, 1.0f, 331, &onnx_light_cpu::GemmMicroKernel_ScalarFp16);
}

TEST(GemmFp16Native, CompactBPanelsCrossBlockingBoundaries) {
  // Each dimension spans three blocking chunks and leaves M/N/K tails.
  const onnx_light_cpu::GemmBlocking blocking{12, 16, 7, 6, 8};
  CheckGemmFp16NativeDriver(false, 25, 37, 19, 0.75f, 341,
                            &onnx_light_cpu::GemmMicroKernel_ScalarFp16, &blocking);
  CheckGemmFp16NativeDriver(true, 25, 37, 20, 0.5f, 351,
                            &onnx_light_cpu::GemmMicroKernel_ScalarFp16, &blocking);
}

TEST(GemmFp16Native, CompactBPanelWaveIsSharedAcrossParallelRows) {
  const onnx_light_cpu::GemmBlocking blocking{96, 128, 128, 6, 8};
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    CheckGemmFp16NativeDriver(false, 193, 257, 257, 0.75f, 356,
                              &onnx_light_cpu::GemmMicroKernel_ScalarFp16, &blocking);
  }
  EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0u);
  EXPECT_GT(executor.maximum_active.load(std::memory_order_relaxed), 1u);
}

#if defined(ONNX_LIGHT_CPU_HAVE_AVX2_FMA) && defined(ONNX_LIGHT_CPU_HAVE_F16C)
TEST(GemmFp16Native, Avx2KernelSupportsSixRows) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma() || !onnx_light_cpu::CpuSupportsF16C()) {
    GTEST_SKIP() << "CPU does not support AVX2/FMA/F16C";
  }
  CheckGemmFp16NativeDriver(false, 13, 19, 33, 0.75f, 361,
                            &onnx_light_cpu::GemmMicroKernel_AVX2F16C);
}
#endif

TEST(GemmFp16Native, Avx512Fp16KernelMatchesReferenceWhenSupported) {
  if (!onnx_light_cpu::CpuSupportsAvx512Fp16()) {
    GTEST_SKIP() << "CPU does not support AVX-512FP16";
  }
  // On capable hardware the public FLOAT16 GEMM path dispatches these general
  // shapes to the native AVX-512FP16 kernel; assert it matches the reference for
  // the 16-lane body, the scalar column tail, and a transposed A.
  CheckGemmHalf(false, false, false, 20, 48, 40, false, 401);
  CheckGemmHalf(false, false, false, 20, 35, 40, true, 411);
  CheckGemmHalf(false, true, false, 17, 19, 33, true, 421);
}

// Roadmap PR08.2: general FLOAT16 shapes whose N spans several eight-lane native
// NEON column vectors and leaves the four-lane and scalar column tails, with M
// covering several register-row blocks. On an ARM build these exercise the
// native NEON FLOAT16 kernel (the eight-, four-, and scalar column paths);
// elsewhere they keep the converting float32 fallback. Both must match the
// widen-then-float32 reference.
TEST(GemmHalf, Float16NeonNativeGeneralColumnTails) {
  CheckGemmHalf(false, false, false, 20, 48, 40, false, 471); // N == 6 * 8, no tail
  CheckGemmHalf(false, false, false, 20, 37, 40, true, 481);  // N == 4 * 8 + 5 tail
  CheckGemmHalf(false, true, false, 17, 19, 33, true, 491);   // trans_a, N == 2 * 8 + 3 tail
}

namespace {

// Runs the native BFLOAT16 general driver (Roadmap PR07.4) with an injected
// micro-kernel and compares ``alpha * op(A) @ B`` against the equivalent
// widen-then-float32 reference. The driver, BFLOAT16 A packing (including
// ``trans_a``), and the kernel's paired 16-lane ``vdpbf16ps`` body -- including
// the odd-K leftover and the scalar column tail -- are all exercised here; the
// portable scalar member runs everywhere while the AVX-512BF16 member only runs
// on capable hardware.
void CheckGemmBf16NativeDriver(bool trans_a, std::size_t M, std::size_t N, std::size_t K,
                               float alpha, unsigned seed,
                               onnx_light_cpu::GemmBf16MicroKernel kernel,
                               const onnx_light_cpu::GemmBlocking *blocking = nullptr) {
  const auto a_f = RandomVector(trans_a ? K * M : M * K, seed);
  const auto b_f = RandomVector(K * N, seed + 1);
  const auto a_bits = NarrowHalf(a_f, true);
  const auto b_bits = NarrowHalf(b_f, true);
  const auto a_round = WidenHalf(a_bits, true);
  const auto b_round = WidenHalf(b_bits, true);
  const auto expected =
      ReferenceGemm<float>(trans_a, false, M, N, K, alpha, a_round, b_round, 0.0f, nullptr);

  std::vector<float> Y(M * N, -1.0f);
  onnx_light_cpu::detail::GemmBf16NativeGeneral(trans_a, M, N, K, alpha, a_bits.data(),
                                                b_bits.data(), Y.data(), kernel, 6, blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    const float tol = 8e-3f * std::max(1.0f, std::abs(expected[i]));
    EXPECT_NEAR(Y[i], expected[i], tol) << "i=" << i;
  }
}

} // namespace

TEST(GemmBf16Native, ScalarKernelMatchesReference) {
  // Portable scalar member: several row blocks (M > 6), exact-16 and tail N,
  // even and odd K, an empty K, ``trans_a``, and a non-unit alpha.
  CheckGemmBf16NativeDriver(false, 13, 16, 20, 1.0f, 501,
                            &onnx_light_cpu::GemmMicroKernel_ScalarBf16);
  CheckGemmBf16NativeDriver(false, 20, 35, 41, 0.75f, 511,
                            &onnx_light_cpu::GemmMicroKernel_ScalarBf16);
  CheckGemmBf16NativeDriver(true, 9, 19, 33, 0.5f, 521,
                            &onnx_light_cpu::GemmMicroKernel_ScalarBf16);
  CheckGemmBf16NativeDriver(false, 7, 5, 0, 1.0f, 531, &onnx_light_cpu::GemmMicroKernel_ScalarBf16);
}

TEST(GemmBf16Native, CompactBPanelsCrossBlockingBoundaries) {
  // Each dimension spans three blocking chunks and leaves M/N/K tails.
  const onnx_light_cpu::GemmBlocking blocking{12, 16, 7, 6, 8};
  CheckGemmBf16NativeDriver(false, 25, 37, 19, 0.75f, 541,
                            &onnx_light_cpu::GemmMicroKernel_ScalarBf16, &blocking);
  CheckGemmBf16NativeDriver(true, 25, 37, 20, 0.5f, 551,
                            &onnx_light_cpu::GemmMicroKernel_ScalarBf16, &blocking);
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
TEST(GemmBf16Native, Avx2KernelSupportsSixRows) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "CPU does not support AVX2/FMA";
  }
  CheckGemmBf16NativeDriver(false, 13, 19, 33, 0.75f, 561,
                            &onnx_light_cpu::GemmMicroKernel_AVX2BF16);
}
#endif

TEST(GemmBf16Native, Avx512Bf16KernelMatchesReferenceWhenSupported) {
  if (!onnx_light_cpu::CpuSupportsAvx512Bf16()) {
    GTEST_SKIP() << "CPU does not support AVX-512BF16";
  }
  // On capable hardware the public BFLOAT16 GEMM path dispatches these general
  // shapes to the native AVX-512BF16 kernel; assert it matches the reference for
  // the 16-lane paired body, the odd-K leftover, the scalar column tail, and a
  // transposed A.
  CheckGemmHalf(true, false, false, 20, 48, 40, false, 601); // N == 3 * 16, even K
  CheckGemmHalf(true, false, false, 20, 35, 41, true, 611);  // N tail, odd K
  CheckGemmHalf(true, true, false, 17, 19, 33, true, 621);   // trans_a, N tail, odd K
}

// Roadmap PR08.2: general BFLOAT16 shapes whose N spans several eight-lane
// native NEON column vectors and leaves the four-lane and scalar column tails,
// with M covering several register-row blocks and even/odd K. On an ARM build
// these exercise the native NEON BFLOAT16 kernel (the eight-, four-, and scalar
// column paths); elsewhere they keep the converting float32 fallback. Both must
// match the widen-then-float32 reference.
TEST(GemmHalf, BFloat16NeonNativeGeneralColumnTails) {
  CheckGemmHalf(true, false, false, 20, 48, 40, false, 671); // N == 6 * 8, even K
  CheckGemmHalf(true, false, false, 20, 37, 41, true, 681);  // N == 4 * 8 + 5 tail, odd K
  CheckGemmHalf(true, true, false, 17, 19, 33, true, 691);   // trans_a, N tail, odd K
}

// Roadmap PR08.3: general FLOAT16/BFLOAT16 shapes whose N spans several native
// SVE column vectors and leaves a predicated column tail, with M covering
// several register-row blocks and even/odd K. On an SVE build whose runtime
// vector length selects the SVE profile (at least 256 bits) these exercise the
// native SVE half kernels (the two-vector body and the ``svwhilelt`` predicated
// tail); shorter vectors keep the NEON kernel and every other target keeps the
// converting float32 fallback. All must match the widen-then-float32 reference.
TEST(GemmHalf, Float16SveNativeGeneralColumnTails) {
  CheckGemmHalf(false, false, false, 20, 48, 40, false, 741); // N spans several vectors, no tail
  CheckGemmHalf(false, false, false, 20, 37, 41, true, 751);  // predicated N tail, odd K
  CheckGemmHalf(false, true, false, 17, 19, 33, true, 761);   // trans_a, predicated N tail, odd K
}

TEST(GemmHalf, BFloat16SveNativeGeneralColumnTails) {
  CheckGemmHalf(true, false, false, 20, 48, 40, false, 771); // N spans several vectors, even K
  CheckGemmHalf(true, false, false, 20, 37, 41, true, 781);  // predicated N tail, odd K
  CheckGemmHalf(true, true, false, 17, 19, 33, true, 791);   // trans_a, predicated N tail, odd K
}

TEST(GemmBf16Native, AmxBf16KernelMatchesReferenceWhenSupported) {
  if (!onnx_light_cpu::CpuSupportsAmxBf16() || !onnx_light_cpu::AmxTileStateAvailable()) {
    GTEST_SKIP() << "CPU does not support AMX-BF16 tile state";
  }
  // On capable hardware the public BFLOAT16 GEMM path dispatches these general
  // shapes to the native AMX-BF16 tile kernel; assert it matches the reference
  // for a full 16x16 tile, the row/column/K tails that must be zero-padded into
  // the fixed tiles, and a transposed A.
  CheckGemmHalf(true, false, false, 32, 32, 32, false, 701); // exact tiles, even K
  CheckGemmHalf(true, false, false, 20, 35, 41, true, 711);  // row/col/K tails
  CheckGemmHalf(true, true, false, 17, 19, 33, true, 721);   // trans_a, tails, odd K
}

TEST(GemmHalf, EmptyKGivesBiasOnly) {
  const std::size_t M = 3, N = 4, K = 0;
  const auto bias_f = RandomVector(M * N, 301);
  const auto bias_bits = NarrowHalf(bias_f, false);
  const auto bias_round = WidenHalf(bias_bits, false);

  std::vector<std::uint16_t> out_bits(M * N, 0);
  std::vector<float> workspace(M * N, -1.0f);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.bias = bias_round.data();
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kMatrix;
  epilogue.beta = 2.0f;
  epilogue.output_conversion = onnx_light_cpu::GemmOutputConversion::kFloat16;
  epilogue.converted_output = out_bits.data();
  onnx_light_cpu::GemmHalfWithEpilogue(false, false, false, M, N, K, 1.0f, nullptr, nullptr,
                                       epilogue, workspace.data());

  const auto out = WidenHalf(out_bits, false);
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(out[i], 2.0f * bias_round[i], 8e-3f * std::max(1.0f, std::abs(bias_round[i])))
        << "i=" << i;
  }
}

TEST(GemmHalf, Float16GemvSkinnyNMatchesReference) {
  // N == 1 selects the skinny-N (K-vectorized GEMV) algorithm, which now reads
  // the half operands directly instead of widening the full A and B tensors.
  CheckGemmHalf(false, false, false, 200, 1, 300, false, 301);
  CheckGemmHalf(false, false, true, 200, 1, 300, false, 311);
  CheckGemmHalf(false, true, false, 64, 1, 128, true, 321);
}

TEST(GemmHalf, Float16GemvSkinnyMMatchesReference) {
  // M == 1 selects the skinny-M (N-vectorized GEMV) algorithm.
  CheckGemmHalf(false, false, false, 1, 200, 300, false, 331);
  CheckGemmHalf(false, false, true, 1, 200, 300, false, 341);
  CheckGemmHalf(false, true, false, 1, 64, 128, true, 351);
}

TEST(GemmHalf, Float16SplitKMatchesReference) {
  // A tiny output with a very deep K selects the split-K algorithm.
  CheckGemmHalf(false, false, false, 2, 2, 4096, false, 356);
  CheckGemmHalf(false, false, false, 8, 4, 4096, false, 361);
  CheckGemmHalf(false, false, false, 4, 4, 5000, true, 371);
}

TEST(GemmHalf, Float16DirectSmallKMatchesReference) {
  // K <= 32 with no transpose selects the direct small-K path.
  CheckGemmHalf(false, false, false, 32, 128, 16, false, 380);
  CheckGemmHalf(false, false, false, 40, 48, 16, false, 381);
  CheckGemmHalf(false, false, false, 40, 48, 31, true, 391);
}

TEST(GemmHalf, BFloat16AlgorithmVariantsMatchReference) {
  CheckGemmHalf(true, false, false, 200, 1, 300, false, 401); // skinny-N
  CheckGemmHalf(true, false, false, 1, 200, 300, false, 411); // skinny-M
  CheckGemmHalf(true, false, false, 8, 4, 4096, false, 421);  // split-K
  CheckGemmHalf(true, false, false, 32, 128, 16, false, 430); // AVX2 direct
  CheckGemmHalf(true, false, false, 40, 48, 16, true, 431);   // direct small-K
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
TEST(GemmHalf, Avx2OutputNarrowingMatchesScalarConversion) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2) {
    GTEST_SKIP() << "AVX2 is not available.";
  }
  const std::vector<float> values = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(std::uint32_t{0x7f800001u}),
      std::bit_cast<float>(std::uint32_t{0xffc12345u}),
      1.00048828125f,
      1.00146484375f,
      3.14159265f,
      -2.71828183f,
      65504.0f,
      65520.0f,
  };
  std::vector<std::uint16_t> actual(values.size());
  onnx_light_cpu::GemmConvertFloat32ToBFloat16_AVX2(values.data(), actual.data(), values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(actual[i], onnx_light_cpu::detail::FloatToBFloat16Bits(values[i])) << "i=" << i;
  }

#ifdef ONNX_LIGHT_CPU_HAVE_F16C
  if (!onnx_light_cpu::CpuSupportsF16C()) {
    return;
  }
  onnx_light_cpu::GemmConvertFloat32ToFloat16_F16C(values.data(), actual.data(), values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(actual[i], onnx_light_cpu::detail::FloatToFloat16Bits(values[i])) << "i=" << i;
  }
#endif
}
#endif

namespace {

// Runs ``fn`` once to warm up one-time allocations, then again while recording
// heap allocations, and returns the largest single allocation observed.
template <typename Fn> std::size_t PeakAllocationBytes(Fn fn) {
  fn();
  g_alloc_peak.store(0, std::memory_order_relaxed);
  g_alloc_recording.store(true, std::memory_order_relaxed);
  fn();
  g_alloc_recording.store(false, std::memory_order_relaxed);
  return g_alloc_peak.load(std::memory_order_relaxed);
}

// Confirms the FP16/BF16 path never allocates enough space to widen its larger
// operand in full. Native kernels use a separate compact blocking profile, so
// their bounded panel allocation need not match the float32 path's panel size.
void CheckHalfNoExpandedOperand(bool is_bfloat16, std::size_t M, std::size_t N, std::size_t K,
                                unsigned seed) {
  const auto a_f = RandomVector(M * K, seed);
  const auto b_f = RandomVector(K * N, seed + 1);
  const auto a_bits = NarrowHalf(a_f, is_bfloat16);
  const auto b_bits = NarrowHalf(b_f, is_bfloat16);

  std::vector<float> y_half(M * N, 0.0f);
  std::vector<std::uint16_t> out_bits(M * N, 0);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = is_bfloat16 ? onnx_light_cpu::GemmOutputConversion::kBFloat16
                                           : onnx_light_cpu::GemmOutputConversion::kFloat16;
  epilogue.converted_output = out_bits.data();
  const std::size_t half_peak = PeakAllocationBytes([&] {
    onnx_light_cpu::GemmHalfWithEpilogue(is_bfloat16, false, false, M, N, K, 1.0f, a_bits.data(),
                                         b_bits.data(), epilogue, y_half.data());
  });

  const std::size_t expanded_operand_bytes = std::max(M * K, K * N) * sizeof(float);
  EXPECT_LT(half_peak, expanded_operand_bytes)
      << "half path allocated a full expanded operand for M=" << M << " N=" << N << " K=" << K;
}

} // namespace

TEST(GemmHalf, Float16AlgorithmsDoNotWidenOperands) {
  CheckHalfNoExpandedOperand(false, 512, 1, 256, 501); // skinny-N GEMV
  CheckHalfNoExpandedOperand(false, 1, 512, 256, 511); // skinny-M GEMV
  CheckHalfNoExpandedOperand(false, 8, 2, 49152, 521); // split-K
  CheckHalfNoExpandedOperand(false, 8, 768, 512, 531); // general blocked
}

TEST(GemmHalf, BFloat16AlgorithmsDoNotWidenOperands) {
  CheckHalfNoExpandedOperand(true, 512, 1, 256, 541); // skinny-N GEMV
  CheckHalfNoExpandedOperand(true, 1, 512, 256, 551); // skinny-M GEMV
  CheckHalfNoExpandedOperand(true, 8, 2, 49152, 561); // split-K
  CheckHalfNoExpandedOperand(true, 8, 768, 512, 571); // general blocked
}

TEST(GemmFloat32, DynamicBUsesBoundedWorkerLocalPacking) {
  constexpr std::size_t M = 128;
  constexpr std::size_t N = 512;
  constexpr std::size_t K = 256;
  const onnx_light_cpu::GemmBlocking blocking{32, 1024, 256, 4, 16};
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  const std::vector<float> A(M * K, 1.0f);
  const std::vector<float> B(K * N, 1.0f);
  std::vector<float> Y(M * N);

  const std::size_t peak_allocation = PeakAllocationBytes([&] {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
        false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr, Y.data(), &blocking);
  });

  EXPECT_LE(peak_allocation, K * 64 * sizeof(float));
  EXPECT_GT(executor.maximum_active.load(std::memory_order_relaxed), 1u);
  for (float value : Y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(K));
  }
}

TEST(GemmFloat32, SmallMicroPanelGridKeepsPackingBounded) {
  constexpr std::size_t M = 300;
  constexpr std::size_t N = 600;
  constexpr std::size_t K = 256;
  const onnx_light_cpu::GemmBlocking blocking{64, 200, 128, 4, 16};
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  const std::vector<float> A(M * K, 1.0f);
  const std::vector<float> B(K * N, 1.0f);
  std::vector<float> Y(M * N);

  const std::size_t peak_allocation = PeakAllocationBytes([&] {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
        false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr, Y.data(), &blocking);
  });

  EXPECT_LT(peak_allocation, K * N * sizeof(float));
  EXPECT_GT(executor.maximum_active.load(std::memory_order_relaxed), 1u);
  for (float value : Y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(K));
  }
}

TEST(GemmFloat32, DynamicBUsesAllParticipantsWithTrailingNcPanel) {
  constexpr std::size_t M = 64;
  constexpr std::size_t N = 600;
  constexpr std::size_t K = 256;
  const onnx_light_cpu::GemmBlocking blocking{64, 512, 256, 4, 16};
  const std::vector<float> A(M * K, 1.0f);
  const std::vector<float> B(K * N, 1.0f);
  std::vector<float> Y(M * N);
  ProductiveExecutor executor{Y.data(), Y.size()};
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ProductiveExecutor::Run};

  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
        false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr, Y.data(), &blocking);
  }

  EXPECT_EQ(executor.productive_blocks, 4u);
  for (float value : Y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(K));
  }
}

namespace {

// Independent scalar reference for the contiguous 2D ``MatMulInteger`` used by
// the INT8 dot-product differential tests. It reproduces the ONNX INT32
// accumulation (wrapping modulo 2^32) directly from raw bytes so the shared
// ``IntegerMatMul2D`` dispatch -- the native ARM NEON ``UDOT`` kernel on capable
// AArch64, the native x86 AVX-512 VNNI kernel on capable x86, the portable
// scalar reduction elsewhere -- is checked against it.
std::vector<std::int32_t> ReferenceMatMulInteger(const std::vector<std::uint8_t> &a, bool a_signed,
                                                 const std::vector<std::uint8_t> &b, bool b_signed,
                                                 std::size_t m, std::size_t n, std::size_t k,
                                                 const std::vector<std::int32_t> &a_zero_points,
                                                 const std::vector<std::int32_t> &b_zero_points) {
  const auto read = [](const std::vector<std::uint8_t> &data, std::size_t index,
                       bool is_signed) -> std::int32_t {
    return is_signed ? static_cast<std::int32_t>(static_cast<std::int8_t>(data[index]))
                     : static_cast<std::int32_t>(data[index]);
  };
  std::vector<std::int32_t> out(m * n, 0);
  for (std::size_t row = 0; row < m; ++row) {
    const std::int32_t az = a_zero_points.size() == 1 ? a_zero_points[0] : a_zero_points[row];
    for (std::size_t col = 0; col < n; ++col) {
      const std::int32_t bz = b_zero_points.size() == 1 ? b_zero_points[0] : b_zero_points[col];
      std::uint32_t accumulator = 0;
      for (std::size_t depth = 0; depth < k; ++depth) {
        const std::int32_t av = read(a, row * k + depth, a_signed) - az;
        const std::int32_t bv = read(b, depth * n + col, b_signed) - bz;
        accumulator += static_cast<std::uint32_t>(av * bv);
      }
      out[row * n + col] = static_cast<std::int32_t>(accumulator);
    }
  }
  return out;
}

std::vector<std::uint8_t> RandomBytes(std::size_t count, unsigned seed) {
  std::mt19937 engine(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<std::uint8_t> data(count);
  for (auto &value : data) {
    value = static_cast<std::uint8_t>(dist(engine));
  }
  return data;
}

std::vector<std::int32_t> RandomZeroPoints(std::size_t count, bool is_signed, unsigned seed) {
  std::mt19937 engine(seed);
  std::uniform_int_distribution<int> dist(is_signed ? -128 : 0, is_signed ? 127 : 255);
  std::vector<std::int32_t> data(count);
  for (auto &value : data) {
    value = dist(engine);
  }
  return data;
}

// Runs the shared dispatch over random operands and compares against the
// independent scalar reference. ``per_axis`` selects per-row (A) / per-column
// (B) zero points instead of a single shared value; a K that is not a multiple
// of 16 exercises the ``UDOT`` scalar tail.
void CheckIntegerMatMul2D(bool a_signed, bool b_signed, std::size_t m, std::size_t n, std::size_t k,
                          bool per_axis, unsigned seed) {
  const auto a = RandomBytes(m * k, seed);
  const auto b = RandomBytes(k * n, seed + 1);
  const auto a_zp = RandomZeroPoints(per_axis ? m : 1, a_signed, seed + 2);
  const auto b_zp = RandomZeroPoints(per_axis ? n : 1, b_signed, seed + 3);

  const auto expected = ReferenceMatMulInteger(a, a_signed, b, b_signed, m, n, k, a_zp, b_zp);

  std::vector<std::int32_t> actual(m * n, -1);
  onnx_light_cpu::IntegerMatMul2D(
      a.data(), a_signed, b.data(), b_signed, actual.data(), static_cast<std::int64_t>(m),
      static_cast<std::int64_t>(n), static_cast<std::int64_t>(k), a_zp.data(),
      static_cast<std::int64_t>(a_zp.size()), b_zp.data(), static_cast<std::int64_t>(b_zp.size()));
  ASSERT_EQ(actual, expected);
}

} // namespace

// Roadmap PR09.2 / PR09.3: the contiguous 2D MatMulInteger dispatch must match
// the scalar reference for every signedness combination, scalar and per-axis
// zero points, and a K that leaves a UDOT scalar tail. On capable AArch64 these
// exercise the native NEON dot-product kernel, on capable x86 the AVX-512 VNNI
// kernel; capable AMX-INT8 x86 runs the 16x16 AMX tiles first; every other
// target exercises the portable scalar reduction the kernel shares.
TEST(IntegerMatMul2D, MatchesScalarReferenceAcrossSignednessAndZeroPoints) {
  CheckIntegerMatMul2D(false, false, 5, 7, 64, false, 4001); // uint8 x uint8, aligned K
  CheckIntegerMatMul2D(true, true, 5, 7, 64, false, 4011);   // int8 x int8, aligned K
  CheckIntegerMatMul2D(false, true, 6, 9, 50, false, 4021);  // uint8 x int8, K tail
  CheckIntegerMatMul2D(true, false, 9, 6, 50, false, 4031);  // int8 x uint8, K tail
  CheckIntegerMatMul2D(false, false, 8, 8, 37, true, 4041);  // per-axis zero points, K tail
  CheckIntegerMatMul2D(true, true, 8, 8, 37, true, 4051);    // per-axis zero points, K tail
  CheckIntegerMatMul2D(false, true, 7, 5, 3, true, 4061);    // K below one UDOT vector
  CheckIntegerMatMul2D(true, false, 1, 1, 128, false, 4071); // single output, wide K
  CheckIntegerMatMul2D(false, true, 17, 19, 67, true, 4081); // AMX row/column/K tails
}

namespace {

using onnx_light_cpu::GemmFloat8Format;
using onnx_light_cpu::detail::Float8BitsToFloat;
using onnx_light_cpu::detail::Float8Format;

Float8Format ToDetailFormat(GemmFloat8Format format) { return static_cast<Float8Format>(format); }

float ReferenceFloat8Bits(Float8Format format, std::uint8_t value) {
  const unsigned sign = value >> 7;
  unsigned exponent_bits;
  unsigned mantissa_bits;
  int bias;
  bool fnuz;
  bool ieee;
  switch (format) {
  case Float8Format::kE4M3FN:
    exponent_bits = 4;
    mantissa_bits = 3;
    bias = 7;
    fnuz = false;
    ieee = false;
    break;
  case Float8Format::kE4M3FNUZ:
    exponent_bits = 4;
    mantissa_bits = 3;
    bias = 8;
    fnuz = true;
    ieee = false;
    break;
  case Float8Format::kE5M2:
    exponent_bits = 5;
    mantissa_bits = 2;
    bias = 15;
    fnuz = false;
    ieee = true;
    break;
  case Float8Format::kE5M2FNUZ:
    exponent_bits = 5;
    mantissa_bits = 2;
    bias = 16;
    fnuz = true;
    ieee = false;
    break;
  }

  const unsigned exponent = (value >> mantissa_bits) & ((1u << exponent_bits) - 1u);
  const unsigned mantissa = value & ((1u << mantissa_bits) - 1u);
  const std::uint32_t float_sign = sign << 31;
  if (fnuz && exponent == 0u && mantissa == 0u && sign != 0u) {
    return std::bit_cast<float>(0xffc00000u);
  }
  if (!fnuz && !ieee && exponent == (1u << exponent_bits) - 1u &&
      mantissa == (1u << mantissa_bits) - 1u) {
    return std::bit_cast<float>(float_sign | 0x7fc00000u);
  }
  if (ieee && exponent == (1u << exponent_bits) - 1u) {
    return mantissa == 0u ? std::bit_cast<float>(float_sign | 0x7f800000u)
                          : std::bit_cast<float>(float_sign | 0x7fc00000u);
  }

  const float magnitude =
      exponent == 0u
          ? std::ldexp(static_cast<float>(mantissa), 1 - bias - static_cast<int>(mantissa_bits))
          : std::ldexp(static_cast<float>((1u << mantissa_bits) + mantissa),
                       static_cast<int>(exponent) - bias - static_cast<int>(mantissa_bits));
  return sign == 0u ? magnitude : -magnitude;
}

// Generates finite Float8 inputs for GEMM by cycling through every byte
// encoding. Non-finite encodings are replaced with zero; their exact decoding
// and propagation are covered separately.
std::vector<std::uint8_t> SequentialFiniteFloat8Bytes(std::size_t size, unsigned seed,
                                                      GemmFloat8Format format) {
  const Float8Format fmt = ToDetailFormat(format);
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    auto candidate = static_cast<std::uint8_t>((i + seed) & 0xFF);
    if (!std::isfinite(Float8BitsToFloat(fmt, candidate))) {
      candidate = 0;
    }
    bytes[i] = candidate;
  }
  return bytes;
}

std::vector<float> DecodeFloat8Vector(const std::vector<std::uint8_t> &bytes,
                                      GemmFloat8Format format) {
  const Float8Format fmt = ToDetailFormat(format);
  std::vector<float> out(bytes.size());
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    out[i] = Float8BitsToFloat(fmt, bytes[i]);
  }
  return out;
}

// Runs the Float8 GEMM path (Roadmap PR09.5) for ``format`` and compares
// ``alpha * op(A) @ op(B)`` against the reference computed on the exact decoded
// float32 values. Exercises the per-format decode-while-packing path (including
// the vectorized table gather on the contiguous copies and the per-element
// decode on the transposed gathers), the column and K tails, and the algorithm
// the driver selects for the shape.
void CheckGemmFloat8(GemmFloat8Format format, bool trans_a, bool trans_b, std::size_t M,
                     std::size_t N, std::size_t K, float alpha, unsigned seed) {
  const auto a_bytes = SequentialFiniteFloat8Bytes(trans_a ? K * M : M * K, seed, format);
  const auto b_bytes = SequentialFiniteFloat8Bytes(trans_b ? N * K : K * N, seed + 1, format);
  const auto a_f = DecodeFloat8Vector(a_bytes, format);
  const auto b_f = DecodeFloat8Vector(b_bytes, format);
  // Reference in double so the comparison is against the exact math, and an
  // accumulation-magnitude scale (sum of absolute products) so the tolerance
  // absorbs the float rounding a reordered reduction (e.g. split-K) incurs when
  // the sequential byte sweep decodes large-magnitude Float8 values that cancel.
  std::vector<double> a_d(a_f.begin(), a_f.end());
  std::vector<double> b_d(b_f.begin(), b_f.end());
  const auto expected =
      ReferenceGemm<double>(trans_a, trans_b, M, N, K, alpha, a_d, b_d, 0.0, nullptr);

  std::vector<double> scale(M * N, 0.0);
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      double s = 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        const double a = trans_a ? a_d[k * M + m] : a_d[m * K + k];
        const double b = trans_b ? b_d[n * K + k] : b_d[k * N + n];
        s += std::abs(a * b);
      }
      scale[m * N + n] = std::abs(static_cast<double>(alpha)) * s;
    }
  }

  std::vector<float> workspace(M * N, -1.0f);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  onnx_light_cpu::GemmFloat8WithEpilogue(format, trans_a, trans_b, M, N, K, alpha, a_bytes.data(),
                                         b_bytes.data(), epilogue, workspace.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    const double operations = static_cast<double>(K + 2);
    const double epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double gamma = operations * epsilon / (1.0 - operations * epsilon);
    const double tol = std::max(1e-6, 4.0 * gamma * scale[i]);
    EXPECT_NEAR(static_cast<double>(workspace[i]), expected[i], tol) << "i=" << i;
  }
}

} // namespace

// Roadmap PR09.5: each Float8 format decodes bit for bit like the reference
// ml_dtypes / ONNX conversion, including zero, the smallest subnormal, a normal
// value, the format maximum, a negative value, and the format-specific NaN.
TEST(GemmFloat8, DecodeMatchesReference) {
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FN, 0x00), 0.0f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FN, 0x01), 0.001953125f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FN, 0x38), 1.0f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FN, 0x7e), 448.0f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FN, 0xc0), -2.0f);
  EXPECT_TRUE(std::isnan(Float8BitsToFloat(Float8Format::kE4M3FN, 0x7f)));
  EXPECT_TRUE(std::isnan(Float8BitsToFloat(Float8Format::kE4M3FN, 0xff)));

  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FNUZ, 0x01), 0.0009765625f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FNUZ, 0x40), 1.0f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE4M3FNUZ, 0x7e), 224.0f);
  EXPECT_TRUE(std::isnan(Float8BitsToFloat(Float8Format::kE4M3FNUZ, 0x80)));

  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE5M2, 0x01), 1.52587890625e-05f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE5M2, 0x40), 2.0f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE5M2, 0x78), 32768.0f);
  EXPECT_TRUE(std::isinf(Float8BitsToFloat(Float8Format::kE5M2, 0x7c)));
  EXPECT_TRUE(std::isnan(Float8BitsToFloat(Float8Format::kE5M2, 0x7e)));

  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE5M2FNUZ, 0x40), 1.0f);
  EXPECT_EQ(Float8BitsToFloat(Float8Format::kE5M2FNUZ, 0x78), 16384.0f);
  EXPECT_TRUE(std::isnan(Float8BitsToFloat(Float8Format::kE5M2FNUZ, 0x80)));
}

TEST(GemmFloat8, DecodeAllBytePatternsMatchesReference) {
  for (Float8Format format : {Float8Format::kE4M3FN, Float8Format::kE4M3FNUZ, Float8Format::kE5M2,
                              Float8Format::kE5M2FNUZ}) {
    for (unsigned value = 0; value < 256; ++value) {
      const float actual = Float8BitsToFloat(format, static_cast<std::uint8_t>(value));
      const float expected = ReferenceFloat8Bits(format, static_cast<std::uint8_t>(value));
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(expected))
          << "format=" << static_cast<int>(format) << ", value=" << value;
    }
  }
}

TEST(GemmFloat8, NonFiniteInputsPropagate) {
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  float output = 0.0f;
  const std::uint8_t one_e4m3fn = 0x38;
  const std::uint8_t nan_e4m3fn = 0x7f;
  onnx_light_cpu::GemmFloat8WithEpilogue(GemmFloat8Format::kE4M3FN, false, false, 1, 1, 1, 1.0f,
                                         &nan_e4m3fn, &one_e4m3fn, epilogue, &output);
  EXPECT_TRUE(std::isnan(output));

  const std::uint8_t one_e5m2 = 0x3c;
  const std::uint8_t infinity_e5m2 = 0x7c;
  onnx_light_cpu::GemmFloat8WithEpilogue(GemmFloat8Format::kE5M2, false, false, 1, 1, 1, 1.0f,
                                         &infinity_e5m2, &one_e5m2, epilogue, &output);
  EXPECT_TRUE(std::isinf(output));
  EXPECT_GT(output, 0.0f);
}

TEST(GemmFloat8, RejectsUnknownFormat) {
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  const std::uint8_t value = 0;
  float output = 0.0f;
  EXPECT_THROW(onnx_light_cpu::GemmFloat8WithEpilogue(static_cast<GemmFloat8Format>(99), false,
                                                      false, 1, 1, 1, 1.0f, &value, &value,
                                                      epilogue, &output),
               std::invalid_argument);
}

// Roadmap PR10.2 correctness gate: every Float8 format runs the same transpose,
// tail, and algorithm corpus on each CI architecture. This exercises scalar and
// native packing/GEMM paths without allowing one format to cover gaps for the
// other three.
TEST(GemmFloat8, Float8CorrectnessGateCorpus) {
  const std::array<GemmFloat8Format, 4> formats = {
      GemmFloat8Format::kE4M3FN, GemmFloat8Format::kE4M3FNUZ, GemmFloat8Format::kE5M2,
      GemmFloat8Format::kE5M2FNUZ};
  CheckGemmFloat8(GemmFloat8Format::kE4M3FN, false, false, 17, 37, 45, 1.0f, 5001);
  CheckGemmFloat8(GemmFloat8Format::kE4M3FNUZ, false, false, 19, 33, 40, 0.75f, 5011);
  CheckGemmFloat8(GemmFloat8Format::kE5M2, false, false, 20, 48, 40, 1.0f, 5021);
  CheckGemmFloat8(GemmFloat8Format::kE5M2FNUZ, false, false, 33, 70, 66, 0.5f, 5031);

  unsigned seed = 5001;
  for (GemmFloat8Format format : formats) {
    CheckGemmFloat8(format, false, false, 17, 37, 45, 0.75f, seed++); // general, N/K tails
    for (bool trans_a : {false, true}) {
      for (bool trans_b : {false, true}) {
        CheckGemmFloat8(format, trans_a, trans_b, 6, 5, 7, 1.0f, seed++);
      }
    }
    CheckGemmFloat8(format, false, false, 12, 1, 40, 1.0f, seed++);  // skinny-N
    CheckGemmFloat8(format, false, false, 1, 12, 40, 0.75f, seed++); // skinny-M
    CheckGemmFloat8(format, false, false, 2, 2, 512, 1.0f, seed++);  // split-K
    CheckGemmFloat8(format, false, false, 3, 3, 4, 1.0f, seed++);    // direct
    CheckGemmFloat8(format, false, false, 5, 7, 0, 1.0f, seed++);    // empty K
  }
}
