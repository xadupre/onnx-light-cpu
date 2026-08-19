API
===

C++ API
-------

The public C++ API is declared in ``onnx_light_cpu/impl/math/math_kernels.h``
and ``onnx_light_cpu/impl/logical/logical_kernels.h``. The shared SIMD
dispatch primitives (``SimdLevel`` and ``DetectSimdLevel``) live in the
``onnx_light_cpu/impl/simd_level.h`` header, which both kernel families
include. Every kernel dispatches at runtime to the best available SIMD path.

The signatures below are extracted from the header comments by Doxygen (see
``docs/Doxyfile``) and rendered through the Breathe extension, so they always
reflect the current state of the project.

.. doxygenenum:: onnx_light_cpu::SimdLevel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::DetectSimdLevel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsFloat64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsFloat16
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsInt8
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsInt32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsInt64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ExpFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ExpFloat64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ExpFloat16
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::LogFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::LogFloat64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::LogFloat16
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::GemmFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::GemmFloat64
   :project: onnx_light_cpu

The ``onnx_light_cpu/impl/logical/logical_kernels.h`` header declares the
logical family, currently the elementwise logical negation used by the ``Not``
operator (ONNX ``bool`` tensors are stored one byte per element, so the buffers
are the raw ``uint8_t`` byte patterns):

.. doxygenfunction:: onnx_light_cpu::NotBool
   :project: onnx_light_cpu

Execution ownership
~~~~~~~~~~~~~~~~~~~

``onnx-light-cpu`` owns SIMD computation, not thread scheduling. Direct C++
kernel calls execute synchronously on the calling thread and do not create
workers. When the kernels are registered with ``onnx-light``, the registration
adapter injects the session ``CpuExecutor``. Large ranges are then split into
disjoint SIMD-aligned blocks and dispatched by that executor.

Consequently, participant count, affinity, spin policy, nesting, lifecycle, and
diagnostics all come from the ``onnx-light`` session policy. There are no
``ONNX_LIGHT_CPU_NUM_THREADS`` or ``ONNX_LIGHT_CPU_SPIN_COUNT`` settings and no
second pool that can oversubscribe the runtime.


onnx-light kernel class
~~~~~~~~~~~~~~~~~~~~~~~~

When onnx-light-cpu is built with ``-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`` (which
requires the `onnx-light <https://github.com/xadupre/onnx-light>`_ C++ package),
an additional library ``lib_onnx_light_cpu_kernels`` is produced. It declares
the ``KernelBase`` subclasses (``AbsKernel``, ``ExpKernel``, ``LogKernel``,
``GemmKernel`` and ``NotKernel``) and their per-operator ``Register*`` functions,
plus the ``RegisterAllKernels`` convenience wrapper:

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

.. doxygenclass:: onnx_light_cpu::NotKernel
   :project: onnx_light_cpu
   :members:

.. doxygenfunction:: onnx_light_cpu::RegisterAbsKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterExpKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterLogKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterGemmKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterNotKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterAllKernels
   :project: onnx_light_cpu

``AbsKernel``, ``ExpKernel``, ``LogKernel``, ``GemmKernel`` and ``NotKernel`` are
full ``KernelBase`` subclasses, so once the matching ``Register*`` function (or
``RegisterAllKernels``) has run every ``Abs``/``Exp``/``Log``/``Gemm``/``Not``
node executed by onnx-light's runtime (and therefore any model run through
``ReferenceEvaluator``) resolves to the SIMD-accelerated kernel.

Python API
----------

.. py:module:: onnx_light_cpu.onnx_py._cpukernels

.. py:function:: detect_simd_level() -> int

   Returns the detected SIMD level: ``0=None``, ``1=SSE2``, ``2=AVX``,
   ``3=AVX2``, ``4=AVX512``.

.. py:function:: has_cpu_kernels() -> bool

   Returns ``True`` when the CPU kernel extension is available.

The compiled kernels themselves (``Abs``, ``Exp``, ``Log``, ``Gemm``, ``Not``)
are not exposed as numpy-like Python functions; they are reachable through
onnx-light's runtime after registration (see :func:`register_all_kernels` and
:func:`onnx_light_cpu.register_kernels`).

.. py:module:: onnx_light_cpu.onnx_py._cpuregister

.. py:function:: register_all_kernels() -> None

   Registers every onnx-light-cpu kernel class (``Abs``, ``Exp``, ``Log``,
   ``Gemm`` and ``Not``) into onnx-light's C++ ``KernelDispatchTable`` for the
   CPU device, replacing the corresponding built-in entries for the default
   ONNX domain.
   This is the Python binding for the C++ :cpp:func:`RegisterAllKernels`
   function and is only available in builds compiled with the onnx-light
   integration enabled (``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``).

.. py:function:: registered_kernel_names() -> list[tuple[str, str]]

   Returns the ``(op_type, kernel name)`` pairs of every onnx-light-cpu kernel,
   for example ``("Abs", "onnx_light_cpu::Abs")``. The kernel name is the
   library-qualified name each kernel records when it runs, so callers can check
   that the accelerated kernels are the ones actually used.

.. py:function:: used_kernel_names() -> list[str]

   Returns the library-qualified names of the onnx-light-cpu kernels that ran
   since the last :func:`clear_used_kernel_names` call, in invocation order.

.. py:function:: clear_used_kernel_names() -> None

   Clears the record of onnx-light-cpu kernels that have run.


Registering kernels with onnx-light
-----------------------------------

.. py:module:: onnx_light_cpu

.. py:function:: register_kernels(sess=None)

   Registers the onnx-light-cpu kernels into onnx-light's shared C++
   ``KernelDispatchTable`` for the CPU device by calling
   :func:`register_all_kernels`. After this call, every ``Abs``, ``Exp``,
   ``Log``, ``Gemm`` and ``Not`` node executed by onnx-light's runtime (and
   therefore any model run through ``ReferenceEvaluator``) dispatches to the
   SIMD-accelerated onnx-light-cpu kernel instead of the built-in one. The
   registration is global, so ``sess`` is optional and only returned unchanged
   so calls can be chained.

   .. code-block:: python

      import numpy as np
      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernels

      register_kernels()
      sess = ReferenceEvaluator(model)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})

.. py:function:: registered_kernel_names() -> dict[str, str]

   Returns a ``{op_type: kernel name}`` dictionary mapping each ONNX ``op_type``
   onnx-light-cpu overrides to the library-qualified name of the accelerated
   kernel installed for it (for example ``{"Abs": "onnx_light_cpu::Abs"}``). Use
   it to confirm the accelerated kernels — rather than onnx-light's built-in
   ones — are registered. Wraps
   :func:`onnx_light_cpu.onnx_py._cpuregister.registered_kernel_names`.

.. py:function:: used_kernel_names() -> list[str]

   Returns, in invocation order, the library-qualified names of the
   onnx-light-cpu kernels that have run since the last
   :func:`clear_used_kernel_names` call. After running a model through a
   ``ReferenceEvaluator`` this reports which accelerated kernels the runtime
   actually dispatched to. Wraps
   :func:`onnx_light_cpu.onnx_py._cpuregister.used_kernel_names`.

.. py:function:: clear_used_kernel_names() -> None

   Clears the record of onnx-light-cpu kernels that have run, so a subsequent
   :func:`used_kernel_names` call only reports the kernels used after this call.
   Wraps :func:`onnx_light_cpu.onnx_py._cpuregister.clear_used_kernel_names`.

.. py:function:: register_backend_test_cases() -> None

   Registers the onnx-light-cpu ``test_cpu_*`` backend test cases into
   onnx-light's shared C++ backend test registry, so they are returned by
   :func:`onnx_light.onnx.backend.collect_test_cases` alongside onnx-light's own
   cases and can be driven through the regular ``ReferenceEvaluator`` API. The
   registration is process-wide and idempotent, and only usable when
   :func:`has_backend_test_cases` reports ``True``.

.. py:function:: has_backend_test_cases() -> bool

   Returns whether the ``register_backend_test_cases`` binding is available. It
   is only built when the ``_cpuregister`` extension links onnx-light's backend
   test registry (``lib_onnx_backend_test``). When ``False``,
   :func:`register_backend_test_cases` is not usable.
