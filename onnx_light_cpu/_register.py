# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Register onnx-light-cpu kernels into onnx-light's C++ dispatch table.

``onnx-light`` evaluates every node against its C++ ``KernelDispatchTable``.
``onnx-light-cpu`` ships SIMD-accelerated C++ ``KernelBase`` subclasses and
exposes a single ``register_all_kernels`` binding that installs every available
class into that shared table for the CPU device. This module wraps the binding
and exposes read-only kernel and operator-support inventories.
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any, NamedTuple


class RegisteredKernel(NamedTuple):
    """Immutable record describing one onnx-light-cpu kernel registration.

    One instance mirrors exactly what a single ``Register*Kernel[s]`` call
    handed to the C++ ``RegisterKernel`` helper (see
    ``onnx_light_cpu::KernelRegistration``): the ONNX operator domain and
    ``op_type``, the device the kernel runs on, the library-qualified C++
    kernel name it records when it runs, the element types it supports (as
    ``TensorProto::DataType`` names, e.g. ``"FLOAT"``), and its optional
    inclusive opset bounds (``None`` when a bound does not apply).
    """

    domain: str
    op_type: str
    device: str
    kernel_name: str
    types: tuple[str, ...]
    since_version: int | None
    until_version: int | None


class MicrosoftKernelImplementation(Enum):
    """Selects the complete ``com.microsoft`` kernel implementation family."""

    NAIVE = "naive"
    OPTIMIZED = "optimized"


class OperatorSupport(NamedTuple):
    """Describes the additional runtime support for one custom operator."""

    domain: str
    op_type: str
    shape_inference_function: str
    peak_memory_function: str
    fusion_patterns: tuple[str, ...]
    has_gradient: bool


def _cpp_microsoft_implementation(implementation: MicrosoftKernelImplementation) -> Any:
    if not isinstance(implementation, MicrosoftKernelImplementation):
        raise TypeError("microsoft_implementation must be a MicrosoftKernelImplementation")
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        MicrosoftKernelImplementation as CppMicrosoftKernelImplementation,
    )

    return getattr(CppMicrosoftKernelImplementation, implementation.name)


def _register_all_kernels(
    microsoft_implementation: MicrosoftKernelImplementation = (
        MicrosoftKernelImplementation.OPTIMIZED
    ),
) -> None:
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
    import_module("onnx_light.onnx_op")

    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        register_all_kernels,
    )

    register_all_kernels(_cpp_microsoft_implementation(microsoft_implementation))


def custom_op_schemas(op_type: str = "", init_doc: bool = True) -> tuple[Any, ...]:
    """Returns the ``com.microsoft`` light schemas shipped by this package."""
    import_module("onnx_light.onnx_op")
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        microsoft_op_schemas,
    )

    return tuple(microsoft_op_schemas(op_type, init_doc))


def operator_schema_lookup(op_type: str) -> list[Any]:
    """Returns standard ONNX schemas plus this package's custom schemas."""
    onnx_op = import_module("onnx_light.onnx_op")

    return [
        *onnx_op.GetAllOnnxOpSchemasWithHistory(op_type),
        *custom_op_schemas(op_type),
    ]


def operator_support() -> tuple[OperatorSupport, ...]:
    """Returns the custom operator support available from onnx-light-cpu."""
    import_module("onnx_light.onnx_op")
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        operator_support as _operator_support,
    )

    return tuple(
        OperatorSupport(
            domain=domain,
            op_type=op_type,
            shape_inference_function=shape_inference_function,
            peak_memory_function=peak_memory_function,
            fusion_patterns=tuple(fusion_patterns),
            has_gradient=has_gradient,
        )
        for (
            domain,
            op_type,
            shape_inference_function,
            peak_memory_function,
            fusion_patterns,
            has_gradient,
        ) in _operator_support()
    )


def register_operator_support() -> None:
    """Registers custom shape, peak-memory, and fusion-pattern support."""
    import_module("onnx_light.onnx_op")
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        register_custom_operator_support,
    )

    register_custom_operator_support()


def register_custom_gradients(registry: Any = None) -> Any:
    """Adds the ``CDist`` and ``BiasGelu`` backward rules to a gradient registry."""
    gradient_module = import_module("onnx_light.onnx_core.gradient")
    if registry is None:
        registry = gradient_module.GradRegistry.default()
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        register_custom_gradients as _register_custom_gradients,
    )

    _register_custom_gradients(registry)
    return registry


def register_kernels(
    sess: Any = None,
    *,
    microsoft_implementation: MicrosoftKernelImplementation = (
        MicrosoftKernelImplementation.OPTIMIZED
    ),
) -> Any:
    """Registers the onnx-light-cpu kernels into onnx-light's dispatch table.

    This installs the SIMD-accelerated ``Abs``, ``Exp``, ``Log``, ``Gemm`` and
    ``Not`` kernel classes into onnx-light's shared C++ ``KernelDispatchTable``
    for the CPU device, replacing the corresponding built-in entries for the
    default ONNX domain. After this call every such node executed by
    onnx-light's runtime (and therefore any model run through
    ``ReferenceEvaluator``) resolves to the onnx-light-cpu kernel.

    The registration is process-wide: it affects every ``ReferenceEvaluator``
    created afterwards, not just one session. To override an operator for a
    single session only, use onnx-light's per-session
    ``ReferenceEvaluator.register_custom_kernel`` hook instead.

    Parameters
    ----------
    sess:
        Optional ``onnx_light.onnx.reference.ReferenceEvaluator`` (or any
        object). The registration is global to onnx-light's dispatch table, so
        this argument is only used as a convenience return value. When provided
        it is returned unchanged so calls can be chained.
    microsoft_implementation:
        Complete ``com.microsoft`` implementation family to install. The
        default is :attr:`MicrosoftKernelImplementation.OPTIMIZED`.

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
    _register_all_kernels(microsoft_implementation)
    return sess


def registered_kernels(
    microsoft_implementation: MicrosoftKernelImplementation = (
        MicrosoftKernelImplementation.OPTIMIZED
    ),
) -> tuple[RegisteredKernel, ...]:
    """Returns one immutable record per onnx-light-cpu kernel registration.

    Each :class:`RegisteredKernel` mirrors exactly one C++ registration
    collected (without installing or executing any kernel) from
    ``onnx_light_cpu::CollectRegisteredKernels``: its ONNX domain, ``op_type``,
    device, library-qualified C++ kernel name, supported element types, and
    optional inclusive opset bounds. Records are sorted deterministically by
    ``(domain, op_type, device, kernel_name)`` -- the same order every call
    returns, and the order :func:`registered_kernel_names` derives its
    ``{op_type: kernel name}`` mapping from.

    Returns
    -------
    A tuple of :class:`RegisteredKernel` records, one per registration.
    """
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        registered_kernels as _registered_kernels,
    )

    return tuple(
        RegisteredKernel(
            domain=domain,
            op_type=op_type,
            device=device,
            kernel_name=kernel_name,
            types=tuple(types),
            since_version=since_version,
            until_version=until_version,
        )
        for domain, op_type, device, kernel_name, types, since_version, until_version in (
            _registered_kernels(_cpp_microsoft_implementation(microsoft_implementation))
        )
    )


def registered_kernel_names(
    microsoft_implementation: MicrosoftKernelImplementation = (
        MicrosoftKernelImplementation.OPTIMIZED
    ),
) -> dict[str, str]:
    """Returns ``{op_type: kernel name}`` for every onnx-light-cpu kernel.

    The kernel name is the library-qualified name (for example
    ``"onnx_light_cpu::Abs"``) that each kernel records when it runs. This maps
    each ONNX ``op_type`` onnx-light-cpu overrides to the name of the
    accelerated kernel installed for it, so callers can check the accelerated
    kernels — rather than onnx-light's built-in ones — are the kernels used.

    Derived from :func:`registered_kernels` instead of maintaining a second
    operator list, so it always stays in sync with the structured inventory.
    """
    return {
        record.op_type: record.kernel_name
        for record in registered_kernels(microsoft_implementation)
    }


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


def set_kernel_usage_recording(enabled: bool) -> None:
    """Enables or disables per-invocation kernel usage recording."""
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        set_kernel_usage_recording as _set_kernel_usage_recording,
    )

    _set_kernel_usage_recording(enabled)


def register_backend_test_cases() -> None:
    """Registers the onnx-light-cpu backend test cases into onnx-light's registry.

    onnx-light ships its ONNX backend test cases as a C++-registered registry
    exposed to Python through
    :func:`onnx_light.onnx.backend.collect_test_cases`. onnx-light-cpu adds its
    own ``test_cpu_*`` cases (covering every element type each accelerated kernel
    implements) into that same shared registry. After this call those cases are
    returned by :func:`onnx_light.onnx.backend.collect_test_cases` alongside
    onnx-light's own, so they can be driven through onnx-light's regular
    ``ReferenceEvaluator`` API exactly like a built-in operator category.

    The registration is process-wide and idempotent. It relies on the
    ``register_backend_test_cases`` binding, which is only present when the
    ``_cpuregister`` extension was built with onnx-light's backend test registry
    available; :data:`onnx_light_cpu.has_backend_test_cases` reports whether it
    is.
    """
    import_module("onnx_light.onnx.reference")

    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        register_backend_test_cases as _register_backend_test_cases,
    )

    _register_backend_test_cases()


def has_backend_test_cases() -> bool:
    """Returns whether the ``register_backend_test_cases`` binding is available.

    It is only built when the ``_cpuregister`` extension links onnx-light's
    backend test registry (``lib_onnx_backend_test``). When ``False``,
    :func:`register_backend_test_cases` is not usable.
    """
    from .onnx_py._cpuregister import (  # pyrefly: ignore[missing-import]
        has_backend_test_cases as _has_backend_test_cases,
    )

    return bool(_has_backend_test_cases)
