Registering kernels
-------------------

.. py:class:: MicrosoftKernelImplementation

   Selects the complete ``com.microsoft`` implementation family. ``NAIVE``
   uses independent scalar correctness oracles; ``OPTIMIZED`` uses production
   kernels.

.. py:function:: register_kernels(*, microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED)

   Legacy name that registers all kernels process-wide. The former ignored
   ``sess`` argument was removed; use :func:`register_kernels_for_session` for
   local registration. New global code can use :func:`register_kernels_global`
   to make the scope explicit.

.. py:function:: register_kernel_global(domain, op_type, *, replace=True, microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED) -> bool

   Registers one native kernel in the process-wide dispatch table. Future
   sessions, and existing sessions that have not resolved the node yet, observe
   it. Already prepared sessions keep their cached kernel. ``replace=True``
   replaces an existing factory; ``replace=False`` keeps it and returns
   ``False``. Unknown domain/operator pairs raise ``ValueError``.

.. py:function:: register_kernels_global(*, replace=True, microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED) -> int

   Registers all shipped kernels globally and returns the number installed.

.. py:function:: register_kernel_for_session(sess, domain, op_type, *, replace=True, microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED) -> bool

   Registers one native compiled kernel on one ``ReferenceEvaluator``. The
   evaluator owns the registration for its lifetime; no other evaluator or the
   process-wide table is modified. The evaluator's cached runtime session is
   reset so its next run observes the change.

.. py:function:: register_kernels_for_session(sess, *, replace=True, microsoft_implementation=MicrosoftKernelImplementation.OPTIMIZED) -> int

   Registers all shipped native kernels on one evaluator and returns the number
   installed. All four explicit APIs are idempotent in resulting state;
   ``replace=False`` also makes duplicate calls no-ops.

   .. code-block:: python

      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernel_for_session, register_kernel_global

      register_kernel_global("", "Abs")
      local_sess = ReferenceEvaluator(model)
      register_kernel_for_session(local_sess, "", "Gemm")

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
   ``ReferenceEvaluator`` comparison with their declared tolerances. The
   report records unsupported cases as skips and execution or comparison
   errors as failures; a kernel without an applicable correctness case is a
   failure.
