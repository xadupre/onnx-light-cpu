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

.. py:function:: used_kernel_names() -> list[str]

   Returns, in invocation order, the accelerated kernels run since the last
   :func:`clear_used_kernel_names` call.

.. py:function:: clear_used_kernel_names() -> None

   Clears the recorded accelerated-kernel invocations.

.. py:function:: set_kernel_usage_recording(enabled) -> None

   Enables or disables per-invocation kernel usage recording.
