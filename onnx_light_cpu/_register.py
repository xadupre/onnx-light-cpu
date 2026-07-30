# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Register onnx-light-cpu kernels on an onnx-light ``ReferenceEvaluator``.

``onnx-light`` evaluates every node against its C++ ``KernelDispatchTable`` but
exposes a :meth:`register_custom_kernel` hook that overrides a built-in kernel
with a Python callable invoked as ``fn(node, *numpy_inputs)``. This module wires
the SIMD-accelerated ``Abs`` kernel provided by ``onnx-light-cpu`` into that
hook so that *any* ONNX model containing an ``Abs`` node runs the optimized
kernel instead of the built-in one.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .onnx_py._cpukernels import abs as _abs

# NumPy dtypes handled by the optimized ``abs`` kernel. Any other dtype (e.g.
# int8, int16, float16) falls back to :func:`numpy.abs` so the registration is
# safe to install for models using any numeric ``Abs`` input type.
_ABS_DTYPES = frozenset(
    {
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int32),
        np.dtype(np.int64),
    }
)


def _abs_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise absolute value for an ``Abs`` node using the SIMD kernel.

    The compiled ``abs`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.abs`.
    """
    arr = np.ascontiguousarray(x)
    if arr.dtype in _ABS_DTYPES:
        return _abs(arr.reshape(-1)).reshape(arr.shape)
    return np.abs(arr)


def register_kernels(sess: Any, domain: str = "") -> Any:
    """Registers the onnx-light-cpu kernels on an onnx-light evaluator.

    Parameters
    ----------
    sess:
        An ``onnx_light.onnx.reference.ReferenceEvaluator`` (or any object
        exposing a compatible ``register_custom_kernel(domain, op_type, fn)``
        method). After this call every ``Abs`` node dispatched by ``sess`` runs
        the SIMD-accelerated onnx-light-cpu kernel.
    domain:
        Operator domain of the ``Abs`` operator. Defaults to the standard ONNX
        domain (the empty string, treated as ``ai.onnx``).

    Returns
    -------
    The ``sess`` argument, so calls can be chained.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from onnx_light.onnx.reference import ReferenceEvaluator
        from onnx_light_cpu import register_kernels

        sess = ReferenceEvaluator(model)
        register_kernels(sess)
        (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})
    """
    register: Callable[[str, str, Any], Any] = sess.register_custom_kernel
    register(domain, "Abs", _abs_kernel)
    return sess
