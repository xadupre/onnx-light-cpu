Binary Kernel Design
====================

Nineteen binary ONNX operators share one descriptor, broadcast-plan, and
runtime-adapter architecture. The implementation is under
``onnx_light_cpu/impl/math/binary`` and the registered adapter is
``onnx_light_cpu/kernels/elementwise/binary_kernel.cc``.

Prepared execution
------------------

.. code-block:: text

   NodeProto
      |
      v
   BinaryKernelDescriptor
   (operator, opset, attributes, typed adapters)
      |
      +---- concrete input shapes and types
      v
   BinaryBroadcastPlanCache (8-entry LRU)
      |
      v
   immutable BinaryBroadcastPlan
   (output shape, coalesced strides, loop family)
      |
      v
   scalar or bulk typed function

``BinaryKernelDescriptor`` is built once for the node. It validates
operator-specific attributes such as ``Mod.fmod`` and
``BitShift.direction``, resolves the output type, and binds scalar and optional
bulk functions for the exact input/output type triple.

Dynamic shapes use a bounded eight-entry LRU cache. Its key contains a
monotonic descriptor identity, input/output types, and both shapes; it never
stores raw descriptor pointers. A miss constructs and validates a fresh plan,
so stale strides cannot be reused.

Operators and traversal
-----------------------

The manifest covers arithmetic (``Add``, ``Sub``, ``Mul``, ``Div``, ``Mod``,
``Pow``), comparisons, logical operators, bitwise operators, ``BitShift``, and
``PRelu``. It is the source of truth for registered type signatures.

After right-aligning shapes, the plan assigns zero strides to broadcast
dimensions, removes unit dimensions, coalesces compatible adjacent dimensions,
and selects one loop family:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Loop family
     - Purpose
   * - Contiguous
     - Both inputs and the output advance linearly.
   * - Left / right scalar
     - Broadcast one scalar while preserving operand order.
   * - Repeated contiguous block
     - Reuse one broadcast value or block across a contiguous suffix.
   * - Inner-vector / outer broadcast
     - Vectorize a contiguous inner region while advancing coalesced outer
       offsets.
   * - General strided
     - Use the validated coalesced strides as the correctness fallback.

Offsets advance incrementally across outer blocks; division and modulo are not
performed for every output element. Empty dimensions produce no work and read
neither input.

Compute, scheduling, and safety
-------------------------------

Same-type common signatures bind bulk contiguous and scalar-broadcast kernels.
Other legal signatures use typed scalar adapters. Validation occurs before
unchecked loops for integer division, modulo, shifts, and other operations with
invalid inputs. Comparison results and logical tensors use one byte per BOOL.

The tuning snapshot has separate thresholds for bulk, block-broadcast, and
scalar layouts plus a target block size and participant limit. Independent
flat ranges or outer blocks are submitted through the session executor; the
plan itself owns no scheduler and stores no thread count.

An input may alias the output only when its type and shape equal the output and
the selected traversal cannot overwrite a value before its final read. A
broadcast input is never expanded in place.
