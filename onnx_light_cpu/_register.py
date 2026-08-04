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

from typing import Any


def _register_all_kernels() -> None:
    """Imports and calls the compiled ``register_all_kernels`` binding.

    The binding lives in the ``_cpuregister`` extension, which links onnx-light
    and is only built with the onnx-light integration enabled
    (``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``). It is imported lazily so that
    ``import onnx_light_cpu`` keeps working in builds without that extension.
    """
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
