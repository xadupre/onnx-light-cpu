Registering kernels
-------------------

.. py:class:: MicrosoftKernelImplementation

   Selects the complete ``com.microsoft`` implementation family. ``NAIVE``
   uses independent scalar correctness oracles; ``OPTIMIZED`` uses production
   kernels.

.. py:function:: register_kernels(sess=None, *, microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED)

   Registers onnx-light-cpu kernels in onnx-light's shared C++
   ``KernelDispatchTable`` for the CPU device. It installs the ``Abs``, ``Exp``,
   ``Log``, ``Gemm``, and ``Not`` kernels, plus the ``com.microsoft`` ``CDist``
   and ``BiasGelu`` kernels, their symbolic shape and peak-memory functions,
   and their fusion patterns. The registration is global; ``sess`` is optional
   and returned unchanged so calls can be chained.

   ``microsoft_implementation`` explicitly selects one complete
   ``com.microsoft`` family. The default is optimized.

   .. code-block:: python

      import numpy as np
      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernels

      register_kernels()
      sess = ReferenceEvaluator(model)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})

.. py:function:: register_backend_test_cases() -> None

   Registers the onnx-light-cpu ``test_cpu_*`` backend cases in onnx-light's
   shared C++ backend test registry.

.. py:function:: has_backend_test_cases() -> bool

   Returns whether the ``register_backend_test_cases`` binding is available.

.. py:class:: BackendCaseResult

   A skipped or failed backend correctness case, including its operator, case
   name, and reason.

.. py:class:: BackendCorrectnessReport

   The executed and passed case counts and skipped and failed case results.

.. py:function:: run_backend_correctness_tests(microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED) -> BackendCorrectnessReport

   Registers onnx-light-cpu kernels and runs applicable onnx-light
   ``TestMode.TEST`` backend cases. Cases use onnx-light's standard
   ``ReferenceEvaluator`` comparison with their declared tolerances; the
   report records unsupported cases and failures.
