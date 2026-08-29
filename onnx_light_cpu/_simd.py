# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public SIMD capability inspection."""

from enum import IntEnum


class SimdLevel(IntEnum):
    """SIMD instruction-set levels used by onnx-light-cpu kernel dispatch."""

    NONE = 0
    SSE2 = 1
    AVX = 2
    AVX2 = 3
    AVX512 = 4


def detect_simd_level() -> SimdLevel:
    """Returns the highest SIMD instruction-set level available to this process."""
    from .onnx_py._cpukernels import (  # pyrefly: ignore[missing-import]
        detect_simd_level as _detect_simd_level,
    )

    return SimdLevel(_detect_simd_level())


def has_cpu_kernels() -> bool:
    """Returns whether the compiled onnx-light-cpu kernel extension is available."""
    from .onnx_py._cpukernels import (  # pyrefly: ignore[missing-import]
        has_cpu_kernels as _has_cpu_kernels,
    )

    return bool(_has_cpu_kernels())
