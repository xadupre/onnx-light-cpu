// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/rotary_embedding_kernel.h"

#include "ml_cpu_compute/math/half_conversion.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;

rt_ns::KernelContext Context() { return rt_ns::KernelContext(rt_ns::OpsetId("", 23)); }

TEST(OnnxLightRotaryEmbeddingKernel, SplitHalfRank4WithIndexedPositions) {
  const onnx_light_cpu::RotaryEmbeddingKernel kernel(Context());
  const auto x =
      rt_ns::Tensor::FromFloat("x", {1, 2, 2, 6}, {1, 2, 3, 4, 7,  8,  5, 6, 7, 8, 9,  10,
                                                   2, 3, 4, 5, 11, 12, 6, 7, 8, 9, 13, 14});
  const auto cos = rt_ns::Tensor::FromFloat("c", {2, 2}, {1, 1, 0, 0});
  const auto sin = rt_ns::Tensor::FromFloat("s", {2, 2}, {0, 0, 1, 1});
  const auto ids = rt_ns::Tensor::FromInt64("p", {1, 2}, {0, 1});
  const auto y = kernel(x, cos, sin, &ids, false, 4);
  ASSERT_EQ(y.shape, x.shape);
  const std::vector<float> expected = {1, 2, 3, 4, 7,  8,  -7, -8, 5, 6, 9,  10,
                                       2, 3, 4, 5, 11, 12, -8, -9, 6, 7, 13, 14};
  EXPECT_EQ(std::vector<float>(y.AsFloat(), y.AsFloat() + expected.size()), expected);
}

TEST(OnnxLightRotaryEmbeddingKernel, InterleavedRank3PerTokenAndHalfTypes) {
  const onnx_light_cpu::RotaryEmbeddingKernel kernel(Context());
  for (const auto type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const auto tensor = [&](const char *name, const rt_ns::Shape &shape,
                            const std::vector<float> &values) {
      if (type == rt_ns::DataType::FLOAT16) {
        return rt_ns::MakeFloat16Tensor(name, shape, values);
      }
      if (type == rt_ns::DataType::BFLOAT16) {
        return rt_ns::MakeBfloat16Tensor(name, shape, values);
      }
      return rt_ns::Tensor::FromFloat(name, shape, values);
    };
    const auto x = tensor("x", {1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    const auto cos = tensor("c", {1, 2, 2}, {0, 0, 1, 1});
    const auto sin = tensor("s", {1, 2, 2}, {1, 1, 0, 0});
    const auto y = kernel(x, cos, sin, nullptr, true, 0, 1);
    EXPECT_EQ(y.data_type, x.data_type);
    EXPECT_EQ(y.shape, x.shape);
    if (type == rt_ns::DataType::FLOAT) {
      EXPECT_EQ(std::vector<float>(y.AsFloat(), y.AsFloat() + 8),
                (std::vector<float>{-2, 1, -4, 3, 5, 6, 7, 8}));
    } else {
      const auto *bits = reinterpret_cast<const std::uint16_t *>(y.bytes());
      const auto expected = std::vector<float>{-2, 1, -4, 3, 5, 6, 7, 8};
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto value = type == rt_ns::DataType::FLOAT16
                               ? ml_cpu_compute::detail::Float16BitsToFloat(bits[i])
                               : ml_cpu_compute::detail::Bfloat16BitsToFloat(bits[i]);
        EXPECT_FLOAT_EQ(value, expected[i]);
      }
    }
  }
}

TEST(OnnxLightRotaryEmbeddingKernel, MicrosoftOffsetAndInvalidPositions) {
  const onnx_light_cpu::RotaryEmbeddingKernel kernel(Context());
  const auto x = rt_ns::Tensor::FromFloat("x", {1, 2, 2}, {1, 2, 3, 4});
  const auto cos = rt_ns::Tensor::FromFloat("c", {3, 1}, {1, 0, 1});
  const auto sin = rt_ns::Tensor::FromFloat("s", {3, 1}, {0, 1, 0});
  const auto offset = rt_ns::Tensor::FromInt64("p", {1}, {1});
  const auto y = kernel(x, cos, sin, &offset, false, 0, 1, true);
  EXPECT_EQ(std::vector<float>(y.AsFloat(), y.AsFloat() + 4), (std::vector<float>{-2, 1, 3, 4}));
  const auto invalid = rt_ns::Tensor::FromInt64("p", {1, 2}, {1, 3});
  EXPECT_THROW(kernel(x, cos, sin, &invalid, false, 0, 1, true), std::invalid_argument);
  EXPECT_THROW(kernel(x, cos, sin, nullptr, false, 3, 1), std::invalid_argument);
}

TEST(OnnxLightRotaryEmbeddingKernel, SplitHalfMatchesScalarInterleavedAcrossVectorTails) {
  const onnx_light_cpu::RotaryEmbeddingKernel kernel(Context());
  for (const auto type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const auto tensor = [&](const rt_ns::Shape &shape, const std::vector<float> &values) {
      if (type == rt_ns::DataType::FLOAT16) {
        return rt_ns::MakeFloat16Tensor("", shape, values);
      }
      if (type == rt_ns::DataType::BFLOAT16) {
        return rt_ns::MakeBfloat16Tensor("", shape, values);
      }
      return rt_ns::Tensor::FromFloat("", shape, values);
    };
    for (std::int64_t half = 1; half <= 33; ++half) {
      for (const bool rank4 : {false, true}) {
        for (const std::int64_t padding : {0, 3}) {
          SCOPED_TRACE(::testing::PrintToString(
              std::vector<std::int64_t>{static_cast<std::int64_t>(type), half, rank4, padding}));
          const auto head = 2 * half + padding;
          const rt_ns::Shape shape =
              rank4 ? rt_ns::Shape{2, 2, 3, head} : rt_ns::Shape{2, 3, 2 * head};
          std::vector<float> split(static_cast<std::size_t>(12 * head)), interleaved(split.size());
          for (std::size_t i = 0; i < split.size(); ++i) {
            split[i] = static_cast<float>(static_cast<int>(i % 97) - 48) / 31.0f;
          }
          interleaved = split;
          for (std::size_t base = 0; base < split.size(); base += head) {
            for (std::int64_t i = 0; i < half; ++i) {
              interleaved[base + 2 * i] = split[base + i];
              interleaved[base + 2 * i + 1] = split[base + half + i];
            }
          }
          std::vector<float> cos(static_cast<std::size_t>(6 * half)), sin(cos.size());
          for (std::size_t i = 0; i < cos.size(); ++i) {
            cos[i] = static_cast<float>(i % 13) / 17.0f;
            sin[i] = static_cast<float>(i % 19) / 23.0f;
          }
          const auto c = tensor({6, half}, cos), s = tensor({6, half}, sin);
          const auto ids = rt_ns::Tensor::FromInt64("", {2, 3}, {5, 1, 3, 2, 0, 4});
          const auto actual = kernel(tensor(shape, split), c, s, &ids, false, 2 * half, 2);
          const auto expected = kernel(tensor(shape, interleaved), c, s, &ids, true, 2 * half, 2);
          const auto bytes = actual.element_size();
          for (std::size_t base = 0; base < split.size(); base += head) {
            for (std::int64_t i = 0; i < head; ++i) {
              const auto index = i < half ? 2 * i : i < 2 * half ? 2 * (i - half) + 1 : i;
              if (type == rt_ns::DataType::FLOAT && i < 2 * half) {
                EXPECT_NEAR(actual.AsFloat()[base + i], expected.AsFloat()[base + index], 1e-6f);
              } else {
                EXPECT_EQ(std::memcmp(actual.bytes() + (base + i) * bytes,
                                      expected.bytes() + (base + index) * bytes, bytes),
                          0);
              }
            }
          }
        }
      }
    }
  }
}

TEST(OnnxLightRotaryEmbeddingKernel, RejectsCacheShapeAndAcceptsEmptySequence) {
  const onnx_light_cpu::RotaryEmbeddingKernel kernel(Context());
  const auto x = rt_ns::Tensor::FromFloat("x", {1, 1, 2}, {1, 2});
  const auto c = rt_ns::Tensor::FromFloat("c", {1, 1, 1}, {1});
  const auto s = rt_ns::Tensor::FromFloat("s", {1, 1, 2}, {0, 0});
  EXPECT_THROW(kernel(x, c, s, nullptr, false, 0, 1), std::invalid_argument);
  const auto empty = rt_ns::Tensor::FromFloat("empty", {1, 0, 2}, {});
  const auto empty_c = rt_ns::Tensor::FromFloat("c", {1, 0, 1}, {});
  const auto empty_s = rt_ns::Tensor::FromFloat("s", {1, 0, 1}, {});
  const auto result = kernel(empty, empty_c, empty_s, nullptr, false, 0, 1);
  EXPECT_EQ(result.shape, empty.shape);
  EXPECT_EQ(result.size_bytes(), 0);
}

TEST(OnnxLightRotaryEmbeddingKernel, BothDomainsDispatchAndInferSymbolicShape) {
  for (const char *domain : {"", onnx_light_cpu::kMicrosoftDomain}) {
    const bool microsoft = std::string(domain) == onnx_light_cpu::kMicrosoftDomain;
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    node.set_domain(domain);
    node.set_op_type("RotaryEmbedding");
    node.add_input("x");
    if (microsoft) {
      node.add_input("p");
    }
    node.add_input("c");
    node.add_input("s");
    if (!microsoft) {
      node.add_input("p");
    }
    node.add_output("y");
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "num_heads", std::int64_t{1});

    shapes_ns::ShapesContext shapes;
    shapes.Set(
        "x", sym_ns::SymTensor(nullptr, sym_ns::TensorType::kFloat, sym_ns::SymShape{"B", "S", 2}));
    shapes.Set("p",
               sym_ns::SymTensor(nullptr, sym_ns::TensorType::kInt64, sym_ns::SymShape{"B", "S"}));
    shapes.Set("c", sym_ns::SymTensor(nullptr, sym_ns::TensorType::kFloat, sym_ns::SymShape{3, 1}));
    shapes.Set("s", sym_ns::SymTensor(nullptr, sym_ns::TensorType::kFloat, sym_ns::SymShape{3, 1}));
    onnx_light_cpu::ComputeShapeRotaryEmbedding(shapes, node);
    EXPECT_EQ(shapes.Get("y").Shape().ToString(), shapes.Get("x").Shape().ToString());
    EXPECT_EQ(onnx_light_cpu::ComputePeakMemoryRotaryEmbedding(sym_ns::Device::kCPU, {}), 0);

    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::OpsetId(domain, microsoft ? 1 : 23)));
    runtime.Set("x", rt_ns::Tensor::FromFloat("x", {1, 2, 2}, {1, 2, 3, 4}));
    runtime.Set("p", rt_ns::Tensor::FromInt64("p", {1, 2}, {0, 1}));
    runtime.Set("c", rt_ns::Tensor::FromFloat("c", {3, 1}, {1, 0, 1}));
    runtime.Set("s", rt_ns::Tensor::FromFloat("s", {3, 1}, {0, 1, 0}));
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, domain, "RotaryEmbedding"));
    rt_ns::RunNode(node, runtime);
    const auto &y = runtime.Get("y");
    EXPECT_EQ(std::vector<float>(y.AsFloat(), y.AsFloat() + 4), (std::vector<float>{1, 2, -4, 3}));
  }
  EXPECT_EQ(onnx_light_cpu::GetMicrosoftOpSchemasWithHistory("RotaryEmbedding").size(), 1);
}

} // namespace
