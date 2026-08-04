# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Register onnx-light-cpu kernels on an onnx-light ``ReferenceEvaluator``.

``onnx-light`` evaluates every node against its C++ ``KernelDispatchTable`` but
exposes a :meth:`register_custom_kernel` hook that overrides a built-in kernel
with a Python callable invoked as ``fn(node, *numpy_inputs)``. This module wires
the SIMD-accelerated ``Abs``, ``Exp``, ``Log``, ``Not`` and ``Gemm`` kernels
provided by ``onnx-light-cpu`` into that hook so that *any* ONNX model
containing those nodes runs the optimized kernels instead of the built-in
ones.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .onnx_py._cpukernels import abs as _abs  # pyrefly: ignore[missing-import]
from .onnx_py._cpukernels import exp as _exp  # pyrefly: ignore[missing-import]
from .onnx_py._cpukernels import gemm as _gemm  # pyrefly: ignore[missing-import]
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


# NumPy dtypes handled by the optimized ``gemm`` kernel. Any other dtype falls
# back to :func:`numpy.matmul` so the registration is safe to install for models
# using any numeric ``Gemm`` input type.
_GEMM_DTYPES = frozenset(
    {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }
)


def _writeable_1d(arr: np.ndarray) -> np.ndarray:
    """Returns a writeable, C-contiguous 1-D view of ``arr``.

    The compiled kernels bind to a mutable ``nb::ndarray`` and reject read-only
    buffers. ``onnx-light`` hands over zero-copy DLPack views that are read-only,
    so the data is copied when it is not already a writeable contiguous buffer.
    """
    flat = np.ascontiguousarray(arr).reshape(-1)
    if not flat.flags.writeable:
        flat = flat.copy()
    return flat


def _abs_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise absolute value for an ``Abs`` node using the SIMD kernel.

    The compiled ``abs`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.abs`.
    """
    arr = np.ascontiguousarray(x)
    if arr.dtype in _ABS_DTYPES:
        return _abs(_writeable_1d(arr)).reshape(arr.shape)
    return np.abs(arr)


def _exp_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise natural exponential for an ``Exp`` node using the SIMD kernel.

    The compiled ``exp`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.exp`.
    """
    arr = np.ascontiguousarray(x)
    if arr.dtype in _FLOAT_DTYPES:
        return _exp(_writeable_1d(arr)).reshape(arr.shape)
    return np.exp(arr)


def _log_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise natural logarithm for a ``Log`` node using the SIMD kernel.

    The compiled ``log`` entry point operates on a contiguous 1-D array, so the
    input is flattened, processed, and reshaped back to its original shape.
    Unsupported dtypes fall back to :func:`numpy.log`.
    """
    arr = np.ascontiguousarray(x)
    if arr.dtype in _FLOAT_DTYPES:
        return _log(_writeable_1d(arr)).reshape(arr.shape)
    return np.log(arr)


def _not_kernel(node: Any, x: np.ndarray) -> np.ndarray:
    """Elementwise logical negation for a ``Not`` node using the SIMD kernel.

    The compiled ``logical_not`` entry point operates on a contiguous 1-D
    ``bool`` array, so the input is flattened, processed, and reshaped back to
    its original shape. Non-``bool`` dtypes fall back to
    :func:`numpy.logical_not`.
    """
    arr = np.ascontiguousarray(x)
    if arr.dtype == np.bool_:
        return _logical_not(_writeable_1d(arr)).reshape(arr.shape)
    return np.logical_not(arr)


# ONNX ``AttributeProto.AttributeType`` codes for the scalar attributes read by
# the ``Gemm`` kernel: ``FLOAT`` scalars live in the ``f`` field and ``INT``
# scalars in the ``i`` field.
_ATTR_FLOAT = 1
_ATTR_INT = 2


def _gemm_attr(node: Any, name: str, default: float) -> float:
    """Returns the ``Gemm`` scalar attribute ``name`` from ``node`` or ``default``.

    The attribute value is read straight from the ``NodeProto`` so no ``onnx``
    (or ``onnx-light``) import is required. ``FLOAT`` attributes are read from
    the ``f`` field and ``INT`` attributes from the ``i`` field.
    """
    for attr in node.attribute:
        if str(attr.name) == name:
            if int(attr.type) == _ATTR_INT:
                return float(attr.i)
            if int(attr.type) == _ATTR_FLOAT:
                return float(attr.f)
    return default


def _gemm_kernel(
    node: Any, a: np.ndarray, b: np.ndarray, c: np.ndarray | None = None
) -> np.ndarray:
    """General matrix multiplication for a ``Gemm`` node using the AVX kernel.

    Computes ``Y = alpha * op(A) @ op(B) + beta * C`` where ``op(A)`` transposes
    ``A`` when the ``transA`` attribute is set and ``op(B)`` transposes ``B``
    when ``transB`` is set. ``C`` is the optional bias input. Unsupported dtypes
    fall back to a NumPy implementation.
    """
    alpha = _gemm_attr(node, "alpha", 1.0)
    beta = _gemm_attr(node, "beta", 1.0)
    trans_a = _gemm_attr(node, "transA", 0.0) != 0.0
    trans_b = _gemm_attr(node, "transB", 0.0) != 0.0

    mat_a = np.ascontiguousarray(a)
    mat_b = np.ascontiguousarray(b)
    mat_c = None if c is None else np.ascontiguousarray(c)
    if mat_a.dtype in _GEMM_DTYPES and mat_b.dtype == mat_a.dtype:
        return _gemm(
            mat_a,
            mat_b,
            c=mat_c,
            alpha=alpha,
            beta=beta,
            trans_a=trans_a,
            trans_b=trans_b,
        )

    op_a = mat_a.T if trans_a else mat_a
    op_b = mat_b.T if trans_b else mat_b
    result = alpha * (op_a @ op_b)
    if mat_c is not None:
        result = result + beta * mat_c
    return result


def register_kernels(sess: Any, domain: str = "") -> Any:
    """Registers the onnx-light-cpu kernels on an onnx-light evaluator.

    Parameters
    ----------
    sess:
        An ``onnx_light.onnx.reference.ReferenceEvaluator`` (or any object
        exposing a compatible ``register_custom_kernel(domain, op_type, fn)``
        method). After this call every ``Abs``, ``Exp``, ``Log``, ``Not`` and
        ``Gemm`` node dispatched by ``sess`` runs the SIMD-accelerated
        onnx-light-cpu kernel.
    domain:
        Operator domain of the ``Abs``/``Exp``/``Log``/``Not``/``Gemm`` operators.
        Defaults to the standard ONNX domain (the empty string, treated as
        ``ai.onnx``).

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
    register(domain, "Gemm", _gemm_kernel)
    return sess
