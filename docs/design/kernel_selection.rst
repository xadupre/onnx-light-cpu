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
