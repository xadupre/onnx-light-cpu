Registering kernels
===================

``onnx-light-cpu`` does not run models on its own: it installs its
SIMD-accelerated kernels into `onnx-light
<https://github.com/xadupre/onnx-light>`_'s shared C++
``KernelDispatchTable``. Once installed, every node that ``onnx-light``'s
runtime executes (and therefore any model run through a
``ReferenceEvaluator``) resolves to the accelerated kernel instead of the
built-in one.

This page explains how to register the shipped kernels, how to add a brand new
kernel, and — most importantly — why a registration can appear to be *ignored*
and how to avoid that.

Register the shipped kernels
----------------------------

The shipped kernels are installed with a single call.

From Python:

.. code-block:: python

    import numpy as np
    from onnx_light.onnx.reference import ReferenceEvaluator

    from onnx_light_cpu import register_kernels

    # Install the SIMD kernels into onnx-light's dispatch table. Import
    # onnx-light *before* calling this so its built-in kernels are already
    # registered; the onnx-light-cpu kernels then override them.
    register_kernels()
    sess = ReferenceEvaluator(model)  # any model containing an Abs node
    (y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.0], dtype=np.float32)})

``register_kernels`` is only available in builds compiled with the onnx-light
integration (``-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``); it wraps the compiled
``onnx_light_cpu.onnx_py._cpuregister.register_all_kernels()`` binding.

``register_kernels()`` installs the kernels **process-wide**: it populates
onnx-light's shared dispatch table, so every ``ReferenceEvaluator`` created in
the process afterwards uses them. To override an operator for a single session
only — without touching the shared table — use onnx-light's per-session
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`
hook instead:

.. code-block:: python

    sess = ReferenceEvaluator(model)
    sess.register_custom_kernel("", "Abs", my_abs)  # this session only

From C++:

.. code-block:: cpp

    #include <onnx_light_cpu/kernels/register_kernels.h>

    onnx_light_cpu::RegisterAllKernels();  // the shipped kernels now use the SIMD implementations

The registration overrides the built-in entry for the default ONNX domain, and
it wins **regardless of the order** in which the built-in and the onnx-light-cpu
kernels are registered: ``onnx-light``'s bulk built-in registration never
replaces an entry that is already present, and an explicit registration always
replaces the built-in one.

Checking which kernels are used
-------------------------------

Because the onnx-light-cpu kernels are drop-in replacements, a model produces
the same numbers whether it runs them or ``onnx-light``'s built-in kernels. To
tell them apart, every onnx-light-cpu kernel carries a unique,
library-qualified **name** (for example ``"onnx_light_cpu::Abs"``) that it
records every time it runs. The names can be inspected from Python:

.. code-block:: python

    from onnx_light_cpu import (
        clear_used_kernel_names,
        register_kernels,
        registered_kernel_names,
        used_kernel_names,
    )

    register_kernels()
    registered_kernel_names()  # {'Abs': 'onnx_light_cpu::Abs', 'Exp': ...}

    clear_used_kernel_names()
    sess.run(None, feeds)      # run a model containing e.g. an Abs node
    used_kernel_names()        # ['onnx_light_cpu::Abs', ...] in run order

If ``used_kernel_names()`` is empty after a run whose operators onnx-light-cpu
overrides, the registration did not take effect — see
:ref:`l-registration-ignored` below. The same names are available in C++ as the
static ``AbsKernel::kName`` (etc.) members and through
``onnx_light_cpu::RegisteredKernelNames()`` /
``onnx_light_cpu::UsedKernelNames()`` in
``onnx_light_cpu/kernels/kernel_usage.h``.

Add a new kernel
----------------

A brand new kernel is a subclass of ``onnx-light``'s
``onnx_light::core::runtime::KernelBase`` that is registered into the dispatch
table with ``RegisterKernelFn``. The shipped kernels are the template to copy;
``onnx_light_cpu/kernels/math/abs_kernel.cc`` is the smallest complete example.
The three steps are:

#. Derive from ``KernelBase`` and override ``Run(RuntimeContext &)`` to read the
   node's inputs and write its outputs (see ``AbsKernel::Run``). Give the class a
   unique ``static constexpr const char *kName`` (for example
   ``"onnx_light_cpu::MyOp"``) and call ``RecordKernelUsage(kName)`` at the top
   of ``Run`` so the kernel can be recognised in ``UsedKernelNames()``; add its
   ``{op_type, kName}`` pair to ``RegisteredKernelNames()`` in
   ``onnx_light_cpu/kernels/kernel_usage.cc``.
#. Wrap the class in a ``NodeKernelFn`` factory that constructs the kernel and
   calls ``set_node``.
#. Call ``RegisterKernelFn(domain, op_type, symbolic::Device::kCPU, factory)``.
   An empty ``domain`` is normalised to the default ONNX domain, so it overrides
   the corresponding built-in operator.

.. code-block:: cpp

    void RegisterMyKernel() {
      onnx_light::core::runtime::NodeKernelFn factory =
          [](const onnx_light::NodeProto &node,
             onnx_light::core::runtime::RuntimeContext &rt)
          -> std::unique_ptr<onnx_light::core::runtime::KernelBase> {
        auto kernel = std::make_unique<MyKernel>(rt.kernel_ctx());
        kernel->set_node(node);
        return kernel;
      };
      onnx_light::core::runtime::RegisterKernelFn(
          "", "MyOp", onnx_light::core::symbolic::Device::kCPU, std::move(factory));
    }

Then add ``RegisterMyKernel()`` to ``RegisterAllKernels`` in
``onnx_light_cpu/kernels/register_kernels.cc`` so it is installed together with
the other kernels.

For a kernel that only needs to run from Python — for example a quick,
model-specific override written in NumPy — use ``onnx-light``'s per-evaluator
``ReferenceEvaluator.register_custom_kernel(domain, op_type, fn)`` hook instead;
it does not require rebuilding this package.

.. _l-registration-ignored:

Troubleshooting: the registration is ignored
---------------------------------------------

If ``register_kernels()`` (or ``RegisterAllKernels()``) runs without error but
the model still uses ``onnx-light``'s built-in kernels, the most common cause is
that the registration and the runtime end up using **two different copies of
onnx-light's dispatch table**.

The dispatch table is a singleton that lives inside onnx-light's
``lib_onnx_core``. It only works as a single shared registry when *every*
extension links the **same** ``lib_onnx_core`` at run time. Building the
integration from an onnx-light *source tree* while a *separately installed*
onnx-light Python package is what actually runs the model breaks that
assumption:

* ``python setup.py build_ext --inplace --onnx-light-source`` (equivalently
  ``-DONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR=...``) compiles onnx-light from source
  with its Python build disabled, which produces a **static** ``lib_onnx_core``
  that is embedded privately into ``_cpuregister``.
* The installed ``onnx_light`` Python package ships its **own** shared
  ``lib_onnx_core`` that its ``ReferenceEvaluator`` uses.

``register_kernels()`` then populates the private copy inside ``_cpuregister``,
but the evaluator reads the copy inside ``onnx_light`` — so the SIMD kernels are
never seen and the registration silently appears to be ignored.

To make the registration take effect, build the integration so it links the
same shared ``lib_onnx_core`` that the running ``onnx_light`` uses. Install
onnx-light so that its ``onnx_lightConfig.cmake`` is available and build with
``find_package`` instead of building from source:

.. code-block:: bash

   python setup.py build_ext --inplace --onnx-light

Use ``--onnx-light-source`` only for a self-contained **C++** integration where
``onnx-light-cpu`` owns the whole runtime (there is no separately installed
onnx-light Python package to share state with).
