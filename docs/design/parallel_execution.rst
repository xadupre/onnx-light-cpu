Parallel kernel execution
=========================

``onnx-light-cpu`` does not own worker threads. A kernel divides independent
work into contiguous ranges and submits those ranges to the
``onnx-light`` session's ``CpuExecutor``. Consequently, a session has one
executor policy for both its built-in and CPU-extension kernels. This page
describes the shared mechanism; :doc:`kernels/index` explains how individual
kernel families choose their work units.

Execution flow
--------------

#. A kernel turns its independent work into a one-dimensional total: for
   example, elements for unary operations, output tiles for matrix
   multiplication, or rows for attention.
#. The kernel calls ``ExecuteRanges`` or ``ExecuteCostedRanges`` from
   `execution.h <https://github.com/xadupre/onnx-light-cpu/blob/main/ml_cpu_compute/include/ml_cpu_compute/execution.h>`_.
   Its schedule supplies minimum work and block sizes, a participant limit,
   and, where useful, a preferred participant count. Costed calls additionally
   describe bytes read, bytes written, and estimated compute cycles.
#. The helper keeps small jobs serial. For a scheduled call it limits the
   number of blocks by the available threads, requested maximum, and work that
   can meet the minimum block size. For a costed call, the executor's
   ``PlanParallelFor`` selects its grain size and participants. Block boundaries
   are rounded to a requested multiple, such as a SIMD lane count, and each
   callback receives its ``[begin, end)`` range.
#. When more than one block is useful, the helper calls the executor's
   ``ParallelFor`` bridge. Workers execute the block callbacks; each callback
   writes only its assigned range. The caller waits for the bridge to finish,
   then the kernel returns its output.

Obtaining the executor
----------------------

During a normal ``onnx-light`` model run, the runtime places the session's
``CpuExecutor`` on the calling thread. Each ONNX kernel entry point installs a
``RuntimeExecutionScope`` that borrows that executor as an
``ExecutionExecutorView`` with its effective thread count, range runner, and
optional cost planner. The adapter is implemented in
`kernels/execution.cc <https://github.com/xadupre/onnx-light-cpu/blob/main/onnx_light_cpu/kernels/execution.cc>`_.
It preserves an already installed explicit executor and restores the previous
view when the call returns, including when validation throws.

The independent ``ml_cpu_compute`` library knows only this generic
executor view. ``CurrentExecutionExecutor()`` reads the installed view without
discovering ONNX runtime state. A direct low-level caller can install its own
``ExecutionExecutorScope``; without one, execution is serial even if an ONNX
session executor exists on the thread. The standalone project can be copied,
built, installed, and consumed with CMake without ONNX or Python.

The public :doc:`registering_kernels` guide covers the equivalent
``onnx-light`` ``CpuExecutorScope`` requirement for direct ``KernelBase`` use.

Nested work stays serial
------------------------

Every submitted range callback enters an execution-region scope. If code
inside that callback invokes ``ExecuteRanges`` again, the nonzero
thread-local execution depth makes the inner call use one block. This avoids
recursive thread-pool submissions and oversubscribing the session's workers.
The same rule applies to an outer kernel that has already chosen serial
execution: its callback has a region scope, so nested helpers remain direct.

Thus parallelism is selected at one level of the active kernel plan, while
small jobs, direct low-level calls without an executor, and nested helpers
all take the same serial range path.
