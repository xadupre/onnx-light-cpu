# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Scan the C++ sources for the kernels provided by this repository.

The public kernels are declared in ``onnx_light_cpu/impl/math/math_kernels.h``
and ``onnx_light_cpu/impl/logical/logical_kernels.h`` as free functions of the
form::

    void AbsFloat32(const float *input, float *output, std::size_t count);

This module extracts ``(operator, data type, function)`` triples from those
declarations so the documentation can present an always up-to-date table of the
available CPU kernels without any manual bookkeeping.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

# Matches ``void AbsFloat32(`` style kernel declarations and captures the
# operator (leading CamelCase part) and the trailing data-type suffix.
_KERNEL_RE = re.compile(
    r"void\s+(?P<op>[A-Z][A-Za-z]*?)"
    r"(?P<dtype>Float16|Float32|Float64|Int8|Int16|Int32|Int64|Uint8|Uint16|Uint32|Uint64|Bool)"
    r"\s*\(",
)

# Human-readable ONNX data-type names keyed by the C++ function suffix.
_DTYPE_NAMES = {
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "Uint8": "uint8",
    "Uint16": "uint16",
    "Uint32": "uint32",
    "Uint64": "uint64",
    "Bool": "bool",
}

# Source globs (relative to the repository root) that declare kernels.
DEFAULT_GLOBS: Tuple[str, ...] = (
    "onnx_light_cpu/impl/math/math_kernels.h",
    "onnx_light_cpu/impl/logical/logical_kernels.h",
)

Kernel = Tuple[str, str, str]


def scan_source(text: str) -> List[Kernel]:
    """Return the ``(operator, data type, function)`` triples found in ``text``."""
    kernels: List[Kernel] = []
    for m in _KERNEL_RE.finditer(text):
        op = m.group("op")
        suffix = m.group("dtype")
        dtype = _DTYPE_NAMES.get(suffix, suffix.lower())
        kernels.append((op, dtype, f"{op}{suffix}"))
    return kernels


def iter_source_files(root: Path, globs: Tuple[str, ...] = DEFAULT_GLOBS) -> List[Path]:
    """Return the sorted list of source files matched by ``globs`` under ``root``."""
    files: List[Path] = []
    for pattern in globs:
        files.extend(root.glob(pattern))
    return sorted(set(files))


def find_registered_kernels(root: Path, globs: Tuple[str, ...] = DEFAULT_GLOBS) -> List[Kernel]:
    """Return the sorted, de-duplicated kernels declared under ``root``."""
    kernels: List[Kernel] = []
    seen = set()
    for path in iter_source_files(root, globs):
        for kernel in scan_source(path.read_text(encoding="utf-8")):
            if kernel not in seen:
                seen.add(kernel)
                kernels.append(kernel)
    kernels.sort()
    return kernels


def group_by_operator(kernels: List[Kernel]) -> OrderedDict[str, List[Tuple[str, str]]]:
    """Group kernels by operator, collecting the supported data types.

    Returns a mapping ``operator -> [(data type, function), ...]`` so each
    operator can be rendered on a single row with all of its supported data
    types listed together instead of one row per data type.
    """
    grouped: OrderedDict[str, List[Tuple[str, str]]] = OrderedDict()
    for operator, dtype, function in kernels:
        grouped.setdefault(operator, []).append((dtype, function))
    return grouped


def operator_function(operator: str) -> str:
    """Return the single public function exposed for ``operator``.

    Each operator is exposed through a single numpy-like Python function that
    dispatches on the array data type (for example ``Abs`` -> ``abs``), so the
    documentation lists one function per operator rather than one per data type.
    The ``Not`` operator maps to ``logical_not`` because ``not`` is a Python
    keyword.
    """
    return {"Not": "logical_not"}.get(operator, operator.lower())
