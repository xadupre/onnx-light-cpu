Kernel and implementation selection
===================================

Kernel selection has two stages. First, ``onnx-light`` selects a node-kernel
factory from its dispatch table. Then the selected ``onnx-light-cpu`` kernel
validates concrete inputs and selects the best compiled scalar or SIMD
implementation for the current CPU. Registration is described in more detail
in :doc:`registering_kernels`.

Selecting a registered node kernel
----------------------------------

``RegisterAllKernels`` calls every ``Register*Kernel`` function in
`kernels/register_kernels.cc <https://github.com/xadupre/onnx-light-cpu/blob/main/onnx_light_cpu/kernels/register_kernels.cc>`_.
Each function provides a node factory and a ``KernelRegistration`` record to
``RegisterKernel``. The record is normalized and installed in onnx-light's
shared table under its ``(domain, op_type, CPU device)`` key.

``com.microsoft`` operators have an additional family-level choice.
``MicrosoftKernelImplementation::NAIVE`` selects independent scalar reference
kernels, while ``OPTIMIZED`` selects the production implementations.
``RegisterAllKernels()`` defaults to ``OPTIMIZED``; its typed overload and
``RegisterMicrosoftKernels`` make the alternative explicit. Inventory and
usage names include ``Naive`` for reference variants, so callers can verify
the selected family.

An empty domain in source is normalized to ``ai.onnx``, the standard ONNX
domain. Custom kernels explicitly register ``com.microsoft`` entries (and
some traditional machine-learning operators use ``ai.onnx.ml``). The runtime
uses the model node's domain, operator type, and CPU device to obtain the
registered factory; that factory creates the ``KernelBase`` adapter for the
node. ``KernelRegistration`` also records supported element types and optional
opset bounds, which can be inspected without changing dispatch through
:cpp:func:`onnx_light_cpu::CollectRegisteredKernels`. See its API declaration
in
`kernels/kernel_registration.h <https://github.com/xadupre/onnx-light-cpu/blob/main/onnx_light_cpu/kernels/kernel_registration.h>`_.

The adapter validates the actual input count, shapes, attributes, and data
types before calculating. These constraints are operator-specific: for
example, the Abs registration declares FLOAT, DOUBLE, FLOAT16, BFLOAT16, and
selected integer types in
`kernels/math/abs_kernel.cc <https://github.com/xadupre/onnx-light-cpu/blob/main/onnx_light_cpu/kernels/math/abs_kernel.cc>`_.
Unsupported concrete inputs are rejected with the kernel's validation error;
they do not silently select a different implementation or reinstate the
built-in kernel.

Selecting scalar or SIMD code
-----------------------------

After validation, the adapter dispatches by data type and passes concrete
dimensions and attributes to its implementation or execution plan. The
implementation checks runtime CPU capabilities, rather than assuming the
compiler's target machine:

* On x86, :cpp:func:`onnx_light_cpu::DetectSimdLevel` detects SSE2, AVX,
  AVX2, and AVX-512 while checking that the operating system saves the needed
  register state. Feature predicates additionally gate FMA, F16C, AVX-512
  extensions, and AMX. They are declared in
  `impl/simd_level.h <https://github.com/xadupre/onnx-light-cpu/blob/main/onnx_light_cpu/impl/simd_level.h>`_.
* On ARM, :cpp:func:`onnx_light_cpu::DetectArmSimdLevel` distinguishes scalar,
  NEON, SVE, and SVE2 paths; the separate dot-product predicate protects
  INT8-specific implementations. Its interface is
  `impl/arm_simd_level.h <https://github.com/xadupre/onnx-light-cpu/blob/main/onnx_light_cpu/impl/arm_simd_level.h>`_.
* A family combines those capabilities with its data type, tensor layout,
  shape, and operation-specific constraints. For example, a GEMM plan caches
  an algorithm and blocking choice derived from data type, dimensions, and
  transpose attributes; see :doc:`kernels/gemm_kernel_design`.

Specialized translation units are only called after their required feature
check succeeds. When an ISA feature, alignment/layout condition, or profitable
shape is unavailable, the same selected kernel follows its lower-level SIMD
or portable scalar path. Therefore a model can use the registered CPU kernel
on a less capable CPU without executing unsupported instructions. This is
different from an unsupported operator or input contract, which is reported
during adapter validation as described above.

SSSE3 and SSE4 fallback assessment
----------------------------------

The shared ``SimdLevel`` remains SSE2, AVX, AVX2, AVX-512. On x86 without
AVX2, ``Abs`` for INT8 and INT16 additionally checks CPUID.1:ECX[9] once,
then uses a separately compiled SSSE3 ``pabsb``/``pabsw`` range function.
The existing SSE2 compare/XOR/subtract paths remain the fallback if SSSE3
is absent; AVX2 and AVX-512 retain priority. This is a local feature check,
not a new level in every kernel's dispatch. The SSSE3 functions use unaligned
loads, preserve the signed minimum value (modular absolute value), allow
in-place operation, and finish every short tail with the scalar path.

Candidates considered on non-AVX2 x86:

.. list-table::
   :header-rows: 1

   * - Instructions
     - Kernel / current fallback
     - Decision
   * - SSSE3 ``pabsb``/``pabsw``
     - INT8/INT16 Abs, SSE2 compare/XOR/subtract
     - Add a local SSSE3 path: direct-loop measurements below justify it.
   * - SSSE3 ``pabsd``
     - INT32 Abs, SSE2 shift/XOR/subtract
     - No measured kernel benefit yet; leave SSE2 unchanged.
   * - SSSE3 ``pmaddubsw`` and SSE2 ``pmaddwd``
     - UINT8 x INT8 integer dot, scalar below AVX2; the AVX2 short-tail
       implementation already uses the 128-bit pair for depths 16--31.
     - Requires additional packing/overflow parity and whole-matmul evidence;
       do not introduce a second integer GEMM implementation on this evidence.
   * - SSSE3 ``pshufb``
     - Packed 4-bit MatMulNBits weight unpacking, currently scalar below AVX2
     - Potential nibble extraction, but no measured advantage for a full
       non-AVX2 matrix kernel.
   * - SSE4.1 ``packusdw`` / widening conversions
     - Float-to-byte Cast and quantized packing, scalar below AVX2
     - An isolated pack is not a conversion kernel; range checks, exceptional
       values and rounding still need full parity and benchmark coverage.
   * - SSE4.1 integer min/max and comparisons
     - Binary elementwise, existing SSE2 comparisons or scalar cases
     - No demonstrated improvement to current kernels. SSE2 already has
       byte/word comparisons and signed saturation/packing.

SSE3 and SSE4.2 add no currently identified instruction that simplifies these
hot loops. SSE4.1 also has no independent dispatch tier. In particular,
``Not`` already uses an SSE2 byte compare and AND; ``pshufb`` would not
reduce its two data operations. AVX/AVX2 paths and scalar tails are unchanged.

Representative isolated measurement (October 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD EPYC 7763 (four logical CPUs in the guest), pinned to logical CPU 0;
GCC, ``-O3 -DNDEBUG -mno-avx -msse2 -fno-tree-vectorize``. Two
``noinline`` range loops replicate the existing SSE2 Abs arithmetic and the
SSSE3 ``pabs`` candidate, with the latter compiled using a
``target("ssse3")`` function attribute. Identical preallocated input/output
buffers, 200 warmups, 31 median samples of 300 repeated calls per shape;
outputs, including the signed minimum, were compared first. Each loop uses
the same scalar remainder. Values are microseconds per call (one thread);
ratio greater than one favors SSSE3.

.. list-table::
   :header-rows: 1

   * - Type / elements
     - SSE2 (us)
     - SSSE3 (us)
     - SSE2 / SSSE3
   * - INT8 / 15 (scalar-only)
     - 0.0078
     - 0.0081
     - 0.96
   * - INT8 / 17 (one vector + tail)
     - 0.0031
     - 0.0022
     - 1.40
   * - INT8 / 1,024
     - 0.0429
     - 0.0238
     - 1.80
   * - INT8 / 65,536
     - 2.5991
     - 1.4580
     - 1.78
   * - INT8 / 4,194,304
     - 166.6804
     - 105.4775
     - 1.58
   * - INT16 / 15 (one vector + tail)
     - 0.0047
     - 0.0044
     - 1.07
   * - INT16 / 17 (two vectors + tail)
     - 0.0037
     - 0.0025
     - 1.48
   * - INT16 / 1,024
     - 0.0870
     - 0.0471
     - 1.85
   * - INT16 / 65,536
     - 5.1203
     - 2.9842
     - 1.72
   * - INT16 / 4,194,304
     - 331.7831
     - 212.1912
     - 1.56

This host supports AVX2, so these are *direct 128-bit loop* results, not
registered-operator speedups on a non-AVX2 processor. They justify trying
the small Abs path, not generalizing the speedup to other kernels or CPUs.
It adds one feature probe cached per dtype, one translation unit and one
dispatch branch after AVX2. The Release ``abs_kernel_ssse3.cc`` object on
this host contains 1,623 bytes of text (``size``); a global SSE tier would
instead affect every kernel family. No fleet distribution or real non-AVX2
host measurement is available, so this local change does not imply a wider
SSE4.1 dispatch would pay for itself. The exact scope and parity are covered by
``test_abs_kernel`` (direct full vectors, every short tail, unaligned and
in-place buffers) and the existing generic Abs tests.
