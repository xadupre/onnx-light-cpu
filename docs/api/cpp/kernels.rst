Kernel classes
--------------

.. doxygenclass:: onnx_light_cpu::AbsKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::ExpKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::LogKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::GemmKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::MatMulNBitsKernel
   :project: onnx_light_cpu
   :members:

The ``com.microsoft::MatMulNBits-1`` CPU implementation targets the
Qwen2/Qwen3 weight layout with matching ``FLOAT``, ``FLOAT16``, or ``BFLOAT16``
activations, scales, optional bias, and output. Packed ``UINT8`` storage holds
2-bit, 4-bit, or 8-bit weights with ``block_size=32`` and an implicit midpoint
zero point of 2, 8, or 128 respectively.
It consumes the packed weights directly without allocating or expanding a
floating-point weight matrix. ``accuracy_level`` 0 and 4 are accepted.
Explicit zero points, ``g_idx``, prepacked provider-specific layouts, other
bit widths, block sizes, mixed floating-point types, and ``DOUBLE`` are
rejected.

.. doxygenclass:: onnx_light_cpu::GatherKernel
   :project: onnx_light_cpu
   :members:

Gather copies fixed-width elements without numerical conversion, accepts
``INT32`` or ``INT64`` indices, and handles scalar/multidimensional indices,
negative axes and indices, and empty outputs. Outputs below 192 KiB remain
serial. Between 192 KiB and 1 MiB, gathers use the runtime executor with
at least 64 KiB per work block and at most eight participants, bounded by the
available threads. Larger outputs retain 256 KiB work blocks, and nested
calls remain serial. In particular, tail-sized outputs just below 1 MiB no
longer fall back to a single worker.
As in onnx-light's built-in Gather, strings, complex values, and packed
sub-byte types are not supported.

.. doxygenclass:: onnx_light_cpu::NonZeroKernel
   :project: onnx_light_cpu
   :members:

NonZero supports ``BOOL`` inputs from opset 9 onwards, including the
``Equal -> NonZero -> Transpose`` token-position pattern. It produces
``INT64`` coordinates in deterministic row-major order with shape
``[rank, count]``. The count stays symbolic during shape inference and is
recomputed for each invocation before checked output allocation, without
worst-case index scratch storage. Empty dimensions and all-zero inputs
produce zero columns.
Other input types retain onnx-light's built-in NonZero implementation.

Scalar inputs follow the ONNX specification: ``[0, 0]`` for false and
``[0, 1]`` for true. ONNX Runtime 1.30 instead returns ``[1, count]`` for
scalars; ranked BOOL outputs match ONNX Runtime.

.. doxygenclass:: onnx_light_cpu::ScatterNDKernel
   :project: onnx_light_cpu
   :members:

ScatterND implements replacement semantics from opset 11 onwards. The
``reduction`` attribute (introduced in opset 16) must be absent or ``"none"``;
``add``, ``mul``, ``min``, ``max``, and unknown reductions are explicitly
rejected, including for empty updates. Payloads use the same fixed-width
types as Gather, including FP32, FP16, and BF16, without numerical conversion.
``INT64`` indices follow ONNX; ``INT32`` indices are an onnx-light-cpu
extension and must be cast to ``INT64`` for portable ONNX graphs.

Index tuples address scalars or contiguous trailing slices. Negative
coordinates are normalized per dimension, and all shapes, buffer sizes, and
bounds are validated before writing. The input is copied exactly once into
independent output storage, then only selected blocks are replaced. Empty
updates still produce an independent copy. Preallocated outputs must not
overlap any input, even partially; inputs may share storage with each other.

ONNX's replacement contract says indices should not contain duplicate
destinations, because update order is unspecified. This implementation
processes tuples in row-major order, with the last tuple winning (including
equivalent positive and negative indices). Do not depend on this ordering
when moving graphs between runtimes; duplicate parity tests use identical
updates. The initial copy can use the runtime executor, but replacement
writes are serial to avoid duplicate-index races.

The embedding fixture combines ``Gather -> Equal -> NonZero -> Transpose
-> ScatterND`` with dynamic ``[visual_tokens, 1]`` positions and
``[visual_tokens, 6656]`` vision features, including zero-image inputs.

AVX-512CD conflict-detection evaluation (2026-10-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ScatterND is the only registered indexed-destination writer; Gather reads
indices but writes contiguous output, and there is no histogram or
ScatterElements kernel. For ``reduction="none"``, ONNX says duplicate
destinations should not occur and does not define their update order. The
local last-tuple-wins guarantee above nevertheless permits a narrow
experiment: for each group of eight *normalized*, four-byte scalar
destinations, reverse the offsets, use ``VPCONFLICTQ`` to retain only each
destination's final lane, and mask-scatter the retained values. Groups remain
ordered, and the scalar tail runs afterwards. Distinct destination bytes do
not overlap; trailing slices wider than four bytes still use the ordered
``memcpy`` loop. This does not implement reduction modes.

The standalone, non-dispatched prototype is
``tools/scatter_nd_conflict_probe.cc``. On an Intel Xeon Platinum 8573C
with AVX-512CD, GCC 13.3, one pinned CPU and the flags shown below, it
checked bitwise parity before measuring. Each trial
copies 65,536 INT32 elements and writes either 8,192 or 65,536 INT32 updates
using already prepared byte offsets. Nine alternating-path runs of 100
trials each give these median *seconds per copy plus update* (one thread):

.. list-table::
   :header-rows: 1

   * - Updates
     - Destinations
     - Ordered scalar (s)
     - AVX-512CD (s)
     - AVX-512CD / scalar
   * - 8,192
     - unique
     - 0.000008138
     - 0.000011916
     - 1.46
   * - 8,192
     - clustered (16)
     - 0.000008570
     - 0.000012376
     - 1.44
   * - 8,192
     - all one destination
     - 0.000008667
     - 0.000012556
     - 1.45
   * - 65,536
     - unique
     - 0.000030537
     - 0.000063837
     - 2.09
   * - 65,536
     - clustered (16)
     - 0.000029877
     - 0.000063727
     - 2.13
   * - 65,536
     - all one destination
     - 0.000030799
     - 0.000065988
     - 2.14

Reproduce on an AVX-512CD host (choose an allowed CPU for affinity)::

   g++ -O3 -std=c++20 -mavx512f -mavx512cd -mavx2 \
       tools/scatter_nd_conflict_probe.cc -o /tmp/scatter_cd
   taskset -c <allowed-core> /tmp/scatter_cd

C++ parity coverage in
``test_onnx_light_scatter_nd_kernel.cc`` also exercises unique, clustered and
repeated destinations, negative aliases, both index widths, and counts around
eight-lane boundaries against the built-in kernel and the local last-wins
contract. Because every measured case regressed, no AVX-512CD compiler or
runtime dispatch is added; the existing scalar replacement path remains.

.. doxygenclass:: onnx_light_cpu::CastKernel
   :project: onnx_light_cpu
   :members:

Cast preserves the input shape while converting its element type. Common
numeric conversions use typed loops and the runtime executor for large
tensors; integer-to-integer casts do not pass through floating point.
This includes the ``INT64`` to ``INT32`` sequence-length conversions used
by Qwen3. Outputs own their storage, including identity casts.
String, float8 and packed low-precision conversions retain the built-in
onnx-light compatibility path, including the ``saturate`` attribute.
The built-in restrictions on extended conversion pairs still apply.
For floating-to-integer values outside the defined ONNX conversion range,
the numeric path provides deterministic behavior: NaN becomes zero and
values outside the integer intermediate range clamp to the destination
bounds. Representable ``INT64`` intermediates retain modular narrowing;
``UINT64`` destinations clamp to their own range.

.. doxygenclass:: onnx_light_cpu::NotKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::SliceKernel
   :project: onnx_light_cpu
   :members:

Slice supports tensor parameters from opset 10 onwards and the legacy
attribute form. It handles optional axes and steps, clipped bounds, negative
steps, empty outputs, and the same fixed-width data types as Gather.
Contiguous trailing dimensions are copied together; strided copies use
bounded-rank coordinates and the runtime executor for large outputs.

.. doxygenclass:: onnx_light_cpu::ConcatKernel
   :project: onnx_light_cpu
   :members:

Concat accepts one or more equal-rank tensors, including empty inputs and
negative axes. It preserves fixed-width element bytes and validates matching
non-axis dimensions, output sizes and non-overlap. Large concatenations use
runtime-owned byte tiles, including concatenations along axis zero.

.. doxygenclass:: onnx_light_cpu::SplitKernel
   :project: onnx_light_cpu
   :members:

Split supports explicit sizes (legacy attributes or an ``INT64`` tensor),
implicit equal partitions, and the opset-18 ``num_outputs`` form with a
smaller final partition. Negative axes and zero-length partitions are
supported for the same fixed-width data types as Gather. Outputs own their
storage, including last-axis QKV partitions across multiple tokens.
Large copies use the runtime executor; small splits remain serial.

.. doxygenclass:: onnx_light_cpu::SimplifiedLayerNormalizationKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::RotaryEmbeddingKernel
   :project: onnx_light_cpu
   :members:

``ai.onnx::RotaryEmbedding`` (opset 23+) accepts rank-3 or rank-4 input and
either per-token rank-3 cosine/sine caches or rank-2 caches with INT64 position
indices. ``com.microsoft::RotaryEmbedding`` (opset 1+) uses the order
``input, position_ids, cos_cache, sin_cache`` and accepts a scalar start
position. Both support split-half/interleaved and partial rotation for FLOAT,
FLOAT16, and BFLOAT16. Packed batching and dynamic cache extension are not
supported.

.. doxygenstruct:: onnx_light_cpu::SimplifiedLayerNormalizationResult
   :project: onnx_light_cpu
   :members:

The experimental default-domain operator normalizes the suffix beginning at
``axis`` and broadcasts Scale to the entire input. Input and Scale may have
independent floating-point types; Y uses Scale's type. Optional inverse
statistics use ``stash_type`` (FLOAT or DOUBLE). Unlike RMSNormalization,
low-precision values are rounded only after scaling, and DOUBLE inputs retain
double arithmetic. The FP32 suffix path reuses the SIMD RMS engine and can
save statistics without a second reduction.
