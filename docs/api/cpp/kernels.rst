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

.. doxygenclass:: onnx_light_cpu::GatherKernel
   :project: onnx_light_cpu
   :members:

Gather copies fixed-width elements without numerical conversion, accepts
``INT32`` or ``INT64`` indices, and handles scalar/multidimensional indices,
negative axes and indices, and empty outputs. Large gathers use the runtime
executor to copy independent slices; smaller gathers remain serial.
As in onnx-light's built-in Gather, strings, complex values, and packed
sub-byte types are not supported.

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
