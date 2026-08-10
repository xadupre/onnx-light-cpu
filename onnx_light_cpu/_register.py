# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Register onnx-light-cpu kernels into onnx-light's C++ dispatch table.

``onnx-light`` evaluates every node against its C++ ``KernelDispatchTable``.
``onnx-light-cpu`` ships the SIMD-accelerated ``Abs``, ``Exp``, ``Log``,
``Gemm`` and ``Not`` kernels as full C++ ``KernelBase`` subclasses and exposes a
single ``register_all_kernels`` binding that installs those classes into that
shared table for the CPU device. This module wraps that binding so that *any*
ONNX model containing those nodes runs the optimized kernels instead of the
built-in ones.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _register_all_kernels() -> None:
    """Imports and calls the compiled ``register_all_kernels`` binding.

    The binding lives in the ``_cpuregister`` extension, which links onnx-light
    and is only built with the onnx-light integration enabled
    (``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``). It is imported lazily so that
    ``import onnx_light_cpu`` keeps working in builds without that extension.
    """
    # Load onnx-light's Python runtime first. Besides initializing its built-in
    # kernels, this loads lib_onnx_proto next to lib_onnx_core before the
    # registration extension links to that exact runtime.
    import_module("onnx_light.onnx.reference")

    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        register_all_kernels,
    )

    register_all_kernels()


def register_kernels(sess: Any = None) -> Any:
    """Registers the onnx-light-cpu kernels into onnx-light's dispatch table.

    This installs the SIMD-accelerated ``Abs``, ``Exp``, ``Log``, ``Gemm`` and
    ``Not`` kernel classes into onnx-light's shared C++ ``KernelDispatchTable``
    for the CPU device, replacing the corresponding built-in entries for the
    default ONNX domain. After this call every such node executed by
    onnx-light's runtime (and therefore any model run through
    ``ReferenceEvaluator``) resolves to the onnx-light-cpu kernel.

    Parameters
    ----------
    sess:
        Optional ``onnx_light.onnx.reference.ReferenceEvaluator`` (or any
        object). The registration is global to onnx-light's dispatch table, so
        this argument is only used as a convenience return value. When provided
        it is returned unchanged so calls can be chained.

    Returns
    -------
    The ``sess`` argument, so calls can be chained.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from onnx_light.onnx.reference import ReferenceEvaluator
        from onnx_light_cpu import register_kernels

        register_kernels()
        sess = ReferenceEvaluator(model)
        (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})
    """
    _register_all_kernels()
    return sess


def registered_kernel_names() -> dict[str, str]:
    """Returns ``{op_type: kernel name}`` for every onnx-light-cpu kernel.

    The kernel name is the library-qualified name (for example
    ``"onnx_light_cpu::Abs"``) that each kernel records when it runs. This maps
    each ONNX ``op_type`` onnx-light-cpu overrides to the name of the
    accelerated kernel installed for it, so callers can check the accelerated
    kernels — rather than onnx-light's built-in ones — are the kernels used.
    """
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        registered_kernel_names as _registered_kernel_names,
    )

    return dict(_registered_kernel_names())


def used_kernel_names() -> list[str]:
    """Returns the onnx-light-cpu kernels that ran since the last clear.

    Every onnx-light-cpu kernel records its library-qualified name when it
    executes, so after running a model through onnx-light's ``ReferenceEvaluator``
    this returns, in invocation order, the names of the accelerated kernels the
    runtime actually dispatched to. Use :func:`clear_used_kernel_names` to reset
    the record before a run.
    """
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        used_kernel_names as _used_kernel_names,
    )

    return list(_used_kernel_names())


def clear_used_kernel_names() -> None:
    """Clears the record of onnx-light-cpu kernels that have run."""
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        clear_used_kernel_names as _clear_used_kernel_names,
    )

    _clear_used_kernel_names()
