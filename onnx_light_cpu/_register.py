# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Register onnx-light-cpu kernels on an onnx-light ``ReferenceEvaluator``.

``onnx-light`` evaluates every node against its C++ ``KernelDispatchTable`` but
exposes a :meth:`register_custom_kernel` hook that overrides a built-in kernel
with a Python callable invoked as ``fn(node, *numpy_inputs)``. This module wires
the SIMD-accelerated ``Abs``, ``Exp``, ``Log`` and ``Not`` kernels provided by
``onnx-light-cpu`` into that hook so that *any* ONNX model containing those
nodes runs the optimized kernels instead of the built-in ones.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .onnx_py._cpukernels import abs as _abs  # pyrefly: ignore[missing-import]
from .onnx_py._cpukernels import exp as _exp  # pyrefly: ignore[missing-import]
from .onnx_py._cpukernels import log as _log  # pyrefly: ignore[missing-import]
from .onnx_py._cpukernels import logical_not as _logical_not  # pyrefly: ignore[missing-import]

# NumPy dtypes handled by the optimized ``abs`` kernel. Any other dtype (e.g.
# int16, bfloat16) falls back to :func:`numpy.abs` so the registration is safe
# to install for models using any numeric ``Abs`` input type.
_ABS_DTYPES = frozenset(
    {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int8),
        np.dtype(np.int32),
        np.dtype(np.int64),
    }
)

# NumPy dtypes handled by the optimized ``exp``/``log`` kernels. Any other dtype
# (e.g. bfloat16) falls back to the corresponding NumPy function.
_FLOAT_DTYPES = frozenset(
    {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }
)


def _as_kernel_input(x: np.ndarray) -> np.ndarray:
    """Returns a writable, C-contiguous view of ``x`` for the compiled kernels.

    The nanobind ``ndarray`` arguments accepted by the compiled kernels require a
    writable, C-contiguous array. ``onnx-light`` passes read-only inputs to
    custom kernels, and :func:`numpy.ascontiguousarray` preserves the read-only
    flag when the input is already contiguous, so a copy is forced whenever the
    array is not writable.
    """
    arr = np.ascontiguousarray(x)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _abs_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise absolute value for an ``Abs`` node using the SIMD kernel.

    The compiled ``abs`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.abs`.
    """
    arr = _as_kernel_input(x)
    if arr.dtype in _ABS_DTYPES:
        return _abs(arr.reshape(-1)).reshape(arr.shape)
    return np.abs(arr)


def _exp_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise natural exponential for an ``Exp`` node using the SIMD kernel.

    The compiled ``exp`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.exp`.
    """
    arr = _as_kernel_input(x)
    if arr.dtype in _FLOAT_DTYPES:
        return _exp(arr.reshape(-1)).reshape(arr.shape)
    return np.exp(arr)


def _log_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise natural logarithm for a ``Log`` node using the SIMD kernel.

    The compiled ``log`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.log`.
    """
    arr = _as_kernel_input(x)
    if arr.dtype in _FLOAT_DTYPES:
        return _log(arr.reshape(-1)).reshape(arr.shape)
    return np.log(arr)


def _not_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise logical negation for a ``Not`` node using the SIMD kernel.

    The compiled ``logical_not`` entry point operates on a contiguous 1-D
    ``bool`` array, so the input is flattened, processed, and reshaped back to
    its original shape. Non-``bool`` dtypes fall back to
    :func:`numpy.logical_not`.
    """
    arr = _as_kernel_input(x)
    if arr.dtype == np.bool_:
        return _logical_not(arr.reshape(-1)).reshape(arr.shape)
    return np.logical_not(arr)


def register_kernels(sess: Any, domain: str = "") -> Any:
    """Registers the onnx-light-cpu kernels on an onnx-light evaluator.

    Parameters
    ----------
    sess:
        An ``onnx_light.onnx.reference.ReferenceEvaluator`` (or any object
        exposing a compatible ``register_custom_kernel(domain, op_type, fn)``
        method). After this call every ``Abs``, ``Exp``, ``Log`` and ``Not``
        node dispatched by ``sess`` runs the SIMD-accelerated onnx-light-cpu
        kernel.
    domain:
        Operator domain of the ``Abs``/``Exp``/``Log``/``Not`` operators. Defaults to
        the standard ONNX domain (the empty string, treated as ``ai.onnx``).

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
    register(domain, "Exp", _exp_kernel)
    register(domain, "Log", _log_kernel)
    register(domain, "Not", _not_kernel)
    return sess
