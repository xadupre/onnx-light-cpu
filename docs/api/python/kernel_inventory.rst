Kernel inventory and usage
--------------------------

.. py:function:: registered_kernel_names() -> dict[str, str]

   Returns a ``{op_type: kernel name}`` mapping for accelerated registered
   kernels, for example ``{"Abs": "onnx_light_cpu::Abs"}``. Use it to confirm
   that accelerated rather than built-in kernels are registered. The mapping is
   derived from :func:`registered_kernels` rather than a separate operator list.

.. py:class:: RegisteredKernel

   Immutable record describing one kernel registration.

   .. py:attribute:: domain
      :type: str

      ONNX operator domain, e.g. ``"ai.onnx"``.

   .. py:attribute:: op_type
      :type: str

      ONNX operator type name, e.g. ``"Abs"``.

   .. py:attribute:: device
      :type: str

      Device the kernel runs on, e.g. ``"CPU"``.

   .. py:attribute:: kernel_name
      :type: str

      Library-qualified C++ kernel class name, e.g. ``"onnx_light_cpu::Abs"``.

   .. py:attribute:: types
      :type: tuple[str, ...]

      Element type names accepted for primary tensor operands.

   .. py:attribute:: since_version
      :type: int | None

      Inclusive opset lower bound, or ``None`` without a lower bound.

   .. py:attribute:: until_version
      :type: int | None

      Inclusive opset upper bound, or ``None`` without an upper bound.

.. py:function:: registered_kernels() -> tuple[RegisteredKernel, ...]

   Returns registrations collected from the C++
   :cpp:func:`CollectRegisteredKernels` inventory without executing kernels.

.. py:function:: used_kernel_names(sess) -> list[str]

   Returns an independent, non-consuming snapshot of all participating backend
   kernel names recorded by ``sess`` since its last
   :func:`clear_used_kernel_names` call. The session retains its first 1024
   invocations and drops later entries until cleared. Concurrent records are
   ordered by acquisition of the recorder's mutex.

.. py:function:: clear_used_kernel_names(sess) -> None

   Clears the recorded backend-kernel invocations for ``sess`` without
   changing whether recording is enabled.

.. py:function:: set_kernel_usage_recording(sess, enabled) -> None

   Enables or disables per-invocation kernel usage recording for ``sess``.
   Recording is disabled by default; disabling preserves existing entries.

All three functions require an explicit ``onnx_light.onnx.reference.ReferenceEvaluator``
session and raise ``TypeError`` for a missing or invalid session. Recording is
owned by its native ``RuntimeContext``: copies, subgraphs, and functions belonging
to that context share the recording state, while independent sessions are isolated.
The returned names are not filtered to ``onnx_light_cpu`` kernels.

The former process-wide, session-free recording API was intentionally removed;
there is no global compatibility fallback. These functions require an updated
``onnx-light`` containing `PR #4942 <https://github.com/xadupre/onnx-light/pull/4942>`_.
No minimum released version is specified yet because that upstream change has
not been released.
