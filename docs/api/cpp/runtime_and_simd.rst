Runtime and SIMD dispatch
-------------------------

Every kernel dispatches at runtime to the best available SIMD path; the
dispatch implementation is private.

onnx-light-cpu owns SIMD computation, not thread scheduling. Direct C++ kernel
calls execute synchronously on the calling thread and do not create workers.
When the kernels are registered with onnx-light, the registration adapter
injects the session ``CpuExecutor``. Large ranges are then split into disjoint
SIMD-aligned blocks and dispatched by that executor.

Consequently, participant count, affinity, spin policy, nesting, lifecycle,
and diagnostics all come from the onnx-light session policy. There are no
``ONNX_LIGHT_CPU_NUM_THREADS`` or ``ONNX_LIGHT_CPU_SPIN_COUNT`` settings and
no second pool that can oversubscribe the runtime.
