CausalConvWithState
===================

The ``com.microsoft::CausalConvWithState`` adapter validates geometry, types and
buffer capacities before dispatching. The general implementation handles FLOAT,
FLOAT16 and BFLOAT16, both layouts, optional bias and past state, dilation and
SiLU/Swish activation.

FLOAT16 three-tap path
---------------------

For a three-tap kernel, unit dilation, no activation and sequence length at least
two, the adapter uses a bounded, allocation-free tiled routine in
``ml_cpu_compute``. Channels-first tiles contain 64 time positions plus two
history positions. Channels-last tiles contain up to 16 channels and rotate three
history buffers. Channel tiles never cross batch boundaries, including partial
tiles.

Weights are converted once per channel tile, and each input value is converted
once rather than once per tap. The existing bulk half converters select F16C
where available and retain portable scalar fallbacks. The float32 arithmetic
retains bias followed by the three products in tap order; there is no
half-precision accumulation or approximate activation. Present state is copied
from the original input bits, preserving signed zero and NaN payloads.

Independent channel rows/tiles use ``ExecuteRanges`` and the caller's executor.
Scratch is fixed-size stack storage, not a full-tensor conversion, persistent
cache, worker allocation or private thread pool. Unsupported configurations
continue to use the general implementation.

Validation and measurement
--------------------------

The focused C++ suite is ``test_onnx_light_causal_conv_with_state_kernel``.
It compares against the independent naive kernel across types, layouts,
dilations, activations and state boundaries. Additional FLOAT16 cases cover tile
tails, the benchmark lengths, multiple batches, special values, exact state bits
and serial/parallel/nested scheduling.

The four issue-979 workloads are the existing backend cases matching
``^test_cpu_causal_conv_c64_l(128|513)_float16_(first|last)_benchmark$``.
Their inputs, weights, output obligations and expected-value generation are
unchanged. Reproduce with an AVX2-ceiling Release build and, from the repository
root, run:

.. code-block:: bash

   python -m onnx_light_cpu benchmark \
     --test '^test_cpu_causal_conv_c64_l(128|513)_float16_(first|last)_benchmark$' \
     --dtype float16 --onnxruntime --threads 1 \
     --warmup 100 --repeat 1000 --max-repeat-time 3 \
     --output /tmp/causal-conv.xlsx

Use the same thread limit for both runtimes and for before/after measurements.
The original dashboard does not provide per-case build provenance or complete
execution settings; local results are not a reconstruction of that campaign.
