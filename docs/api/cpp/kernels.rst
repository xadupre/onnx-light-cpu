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
negative axes and indices, and empty outputs. Outputs below 192 KiB remain
serial. Between 192 KiB and 1 MiB, gathers use the runtime executor with
at least 64 KiB per work block and at most eight participants, bounded by the
available threads. Larger outputs retain 256 KiB work blocks, and nested
calls remain serial. In particular, tail-sized outputs just below 1 MiB no
longer fall back to a single worker.
As in onnx-light's built-in Gather, strings, complex values, and packed
sub-byte types are not supported.

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
