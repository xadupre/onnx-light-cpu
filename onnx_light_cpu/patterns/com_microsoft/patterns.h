// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace onnx_light_cpu {

/**
 * Fuses an exact Gelu applied to an Add with a rank-one bias.
 *
 * @code
 * Before:
 *                 ┌─────┐
 *   A, bias ─────→│ Add │
 *                 └─────┘
 *                    │
 *                    ↓
 *                    z
 *                    │
 *                    ↓
 *                 ┌──────┐
 *                 │ Gelu │────→ Y
 *                 └──────┘
 *
 * After:
 *                 ┌──────────┐
 *   A, bias ─────→│ BiasGelu │────→ Y
 *                 └──────────┘
 * @endcode
 *
 * The bias must broadcast over the last dimension of a floating-point input,
 * and the Add output must be consumed exclusively by the exact Gelu.
 */
class BiasGeluFusionPattern final
    : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  BiasGeluFusionPattern() : PatternOptimization(0, "MicrosoftBiasGelu") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

/**
 * Fuses the squared-Euclidean distance expansion into CDist.
 *
 * @code
 * Before:
 *          ┌──────────────┐
 *   A ────→│ Unsqueeze(1) │───┐
 *          └──────────────┘   │
 *                             │
 *          ┌──────────────┐   │   ┌─────┐
 *   B ────→│ Unsqueeze(0) │───┴──→│ Sub │────→ difference
 *          └──────────────┘       └─────┘           │
 *                                                   ↓
 *                                               ┌──────────┐
 *                                               │ Mul self │
 *                                               └──────────┘
 *                                                   │
 *                                                   ↓
 *                                           squared difference
 *                                                   │
 *                                                   ↓
 *                                            ┌───────────────┐
 *                                            │ ReduceSum(-1) │────→ Y
 *                                            └───────────────┘
 *
 * After:
 *              ┌──────────────────────────┐
 *   A, B ─────→│ CDist metric=sqeuclidean │────→ Y
 *              └──────────────────────────┘
 * @endcode
 *
 * Both inputs must have rank two. The difference must be squared by an
 * exclusively consumed self-Mul and reduced over the last axis without keeping
 * the reduced dimension.
 */
class CDistFusionPattern final : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  CDistFusionPattern() : PatternOptimization(0, "MicrosoftCDist") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

/**
 * Rewrites the rank-3 grouped-query subset of ``ai.onnx::Attention`` to
 * ``com.microsoft::GroupQueryAttention``.
 *
 * @code
 * Before:
 *                    ┌─────────────────────────┐
 *   query, key, ─────→│ Attention               │────→ output
 *   value             │ q_heads != kv_heads     │
 *                    └─────────────────────────┘
 *
 * After:
 *   key ─────────────────→┌───────┐────→┌────────┐────→ sequence length
 *                         │ Shape │     │ Gather │
 *                         └───────┘     └────────┘
 *                                             │
 *                                             └──→┌─────┐────→┌──────┐──┐
 *                                                 │ Sub │     │ Cast │  │
 *                                                 └─────┘     └──────┘  │
 *                                                                        ↓
 *   query ───────────────→┌───────┐────→┌────────┐────→┌───────────┐────→┌────────┐
 *                         │ Shape │     │ Gather │     │ Unsqueeze │     │ Expand │
 *                         └───────┘     └────────┘     └───────────┘     └────────┘
 *                                                                        │
 *                                                                        ↓
 *                    ┌─────────────────────────────────────────────┐  seqlens_k
 *   query, key, ─────→│ com.microsoft::GroupQueryAttention         │←────┘
 *   value             │ num_heads, kv_num_heads, causal, scale     │────→ output
 *                    └─────────────────────────────────────────────┘
 * @endcode
 *
 * The generated shape subgraph derives the cache-free sequence metadata
 * required by GroupQueryAttention without changing the Attention result.
 */
class GroupQueryAttentionFusionPattern final
    : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  GroupQueryAttentionFusionPattern() : PatternOptimization(0, "MicrosoftGroupQueryAttention") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

/**
 * Rewrites the FLOAT subset of ``ai.onnx::LinearAttention-27`` to the
 * compatible ``com.microsoft::LinearAttention-1`` contract.
 *
 * @code
 * Before:
 *   query, key, value --->+--------------------------+---> output
 *   past, decay, beta --->| ai.onnx::LinearAttention |---> present_state
 *                         +--------------------------+
 *
 * After:
 *   query, key, value --->+----------------------------------+---> output
 *   past, decay, beta --->| com.microsoft::LinearAttention-1 |---> present_state
 *                         +----------------------------------+
 * @endcode
 *
 * Inputs, optional-input slots, outputs, and all semantic attributes are
 * preserved. The Microsoft-domain import is added to the graph.
 */
class LinearAttentionFusionPattern final
    : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  LinearAttentionFusionPattern() : PatternOptimization(0, "MicrosoftLinearAttention") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

/**
 * Fuses a rank-1 FLOAT Add bias into optional MatMulNBits input slot 5.
 *
 * The source must be the exact com.microsoft opset-1 FLOAT/UINT8 restricted
 * subset, have no existing bias, zero_points, or g_idx, and its output must be
 * consumed only by the Add.
 */
class MatMulNBitsBiasFusionPattern final
    : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  MatMulNBitsBiasFusionPattern() : PatternOptimization(0, "MicrosoftMatMulNBitsBias") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

/**
 * Removes FLOAT casts around Microsoft skip RMS normalization for a proven kernel.
 *
 * kernel_name is an explicit execution contract, not a registry lookup: the caller
 * must use that kernel for both the original and rewritten graph, without session
 * overrides. Currently only onnx_light_cpu::NaiveSkipSimplifiedLayerNormalization
 * has identical FP32 arithmetic for FLOAT and FLOAT16. An empty or unsupported
 * name disables the rewrite. The builder must also explicitly target CPU.
 *
 * Weights must originate in FLOAT16 or be exactly representable FLOAT constants.
 * A live residual output must exclusively feed a FLOAT16 Cast; live FLOAT
 * statistics retain their names and types. No stash_type attribute is introduced.
 */
class NormalizationCastEliminationPattern final
    : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  explicit NormalizationCastEliminationPattern(std::string kernel_name = {})
      : PatternOptimization(0, "MicrosoftNormalizationCast",
                            ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU),
        kernel_name_(std::move(kernel_name)) {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;

private:
  std::string kernel_name_;
};

/**
 * Fuses Softmax/TopK routing and two GroupedMatMul-29 expert layers into MoE.
 *
 * @code
 * Before:
 *   logits[T,E] --> Softmax(-1) --> TopK(K)
 *                                     |
 *                                     +--> indices[T,K] --> Reshape[-1,1] --> ids
 *                                     +--> p[T,K] --> [Normalize?] --> Unsqueeze(-1) --> w
 *
 *   X[T,H], W1[E,H,I], indices, b1? --> GroupedMatMul --> a[T,K,I]
 *   a --> [Activation?] --> Reshape[-1,I] --> flat_a
 *   flat_a, W2[E,I,H], ids, b2? --> GroupedMatMul --> Reshape[-1,K,H] --> experts
 *   experts, w --> Mul --> ReduceSum(axis=1,keepdims=0) --> Y[T,H]
 *
 * After:
 *   logits --> Gather(reverse,axis=1) ---------------------> logits'
 *   W1,W2 --> Gather(reverse,axis=0) --> Transpose[0,2,1] ---> W1',W2'
 *   b1?,b2? --> Gather(reverse,axis=0) --------------------> b1'?,b2'?
 *
 *   X, logits', W1', b1'?, W2', b2'? --> com.microsoft::MoE --> Y[T,H]
 *                                       k=K, activation_type=Activation,
 *                                       normalize_routing_weights=Normalize?
 * @endcode
 *
 * T/E/H/I/K denote tokens, experts, hidden width, intermediate width and selected
 * experts. Normalize? is p / ReduceSum(p,axis=1,keepdims=1), or just p when absent.
 * Activation? is identity, Relu, or Gelu(approximate="tanh").
 * Biases b1[E,I] and b2[E,H] are optional. reverse is [E-1,...,0].
 *
 * Supports FLOAT, static positive expert/hidden/intermediate dimensions, constant
 * reshape targets, constant weights and optional constant biases, Relu, tanh-Gelu
 * and identity activations.
 * FLOAT16/BFLOAT16 are excluded: the unfused graph rounds its intermediates.
 * Exact Gelu is excluded: the MoE CPU kernel uses a tanh approximation.
 * SiLU is excluded: MoE and standalone Sigmoid/Mul differ for extreme inputs.
 * Retained inputs must be produced outside the matched block.
 * Expert order is reversed for logits, weights and biases before transposing
 * weights. This preserves ONNX TopK's low-index tie breaking with MoE's
 * high-index tie breaking. Shared source tensors are never modified.
 */
class MoEFusionPattern final : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  MoEFusionPattern() : PatternOptimization(0, "MicrosoftMoE") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

void RegisterCustomOperatorPatterns();

} // namespace onnx_light_cpu
