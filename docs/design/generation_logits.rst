Generation-only logits
======================

Status: upstream integration required
------------------------------------

This optimization is not implemented in onnx-light-cpu. There is no separate
Python generation method or model-preparation API here. Full-sequence logits,
MatMul semantics, and existing generation behavior remain unchanged.

The integration point is onnx-light's existing native generation path:

* `ReferenceEvaluator.generate
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/_reference/_evaluator.py>`_
  delegates generation to the native runtime.
* `GenerationOptions and Generate
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/runtime/generation.h>`_
  own token selection and the generation contract.
* `The native implementation
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/runtime/generation.cc>`_
  owns per-call sessions, masks, positions, and persistent cache state.

Any opt-in last-token projection must be implemented through that existing
native path, including its pipeline overloads, rather than a Python-side
rewrite requiring callers to maintain extra selection feeds. CPU kernels
remain reusable execution primitives; they must not silently truncate MatMul
outputs. Upstream implementation and validation remain outstanding; this page
does not claim that a new generation option is available.

Required upstream contract
--------------------------

An explicitly generation-specific model/export option can select the required
hidden-state row before the LM head, producing ``[B, 1, V]`` instead of
``[B, S, V]``. The native generation path must supply or derive that selection
using its own token-position and padding semantics. Its current documented mask
contract supports unpadded and left-padded prompts; right-padded or arbitrary
masks must not be assumed supported. Extending that contract requires upstream
validation and tests, not a local Python workaround.

Selection must preserve batch dimensions, dtype, token indices, and cache
updates. Define zero-length and all-padding behavior explicitly, and distinguish
physical positions in the current hidden states from accumulated cache lengths.
Never infer the last valid token from the last physical row without proving
that assumption for the supported padding contract.

If implemented as a native graph rewrite, prove that all consumers need only
the selected logits and preserve the original selection and shape semantics.
Do not rewrite a graph exposing full-sequence logits. Do not consume
initializers when generation recreates sessions from the same model; preserve
repeated Generate calls and owning/borrowed pipeline lifetime rules.

Export and acceptance work
--------------------------

TinyLLM's exporter integration is in
`onnx-asv-benchmark/benchmarks/models/llm/tiny_llm.py
<https://github.com/xadupre/onnx-asv-benchmark/blob/main/benchmarks/models/llm/tiny_llm.py>`_
and uses ``modelbuilder.builder.create_model``. Coordinate any export option
there with native Generate support in onnx-light. No upstream code is changed
by this documentation.

Before claiming support:

* Compare reduced logits with the corresponding slice of the unchanged
  full-logits model for batch-one and supported variable-length/padded batches.
* Exercise the existing Generate entry point repeatedly, including prefill,
  decode, fresh calls from the same model, and supported pipeline forms.
* Verify unchanged full-logits behavior by default.
* Measure prefill, isolated projection latency, and logits output storage
  separately on CPU and ORT using the same reduced-output model.
* Add a separately named generation-only ASV scenario; keep the existing
  all-token TinyLLM prefill result comparable and label the different contract.

For ``S=128``, selecting one hidden-state row reduces LM-head matrix arithmetic
128-fold, not necessarily latency. This work remains separate from prepacking
and activation fusion. Standalone synthetic Python benchmark results do not
establish integration with native Generate.
