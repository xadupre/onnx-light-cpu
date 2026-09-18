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
