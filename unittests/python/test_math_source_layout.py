# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the math implementation's operation/ISA layout."""

from pathlib import Path

_MATH = Path(__file__).resolve().parents[2] / "onnx_light_cpu" / "impl" / "math"


def test_simd_files_are_in_isa_subfolders():
    misplaced = [
        str(path.relative_to(_MATH))
        for folder in (_MATH, _MATH / "binary", _MATH / "gemm")
        for path in folder.iterdir()
        if path.suffix in {".cc", ".h"} and ("_avx" in path.stem or "_f16c" in path.stem)
    ]
    assert not misplaced, f"SIMD files outside ISA subfolders: {sorted(misplaced)}"


def test_binary_files_are_in_binary_subtree():
    misplaced = [
        str(path.relative_to(_MATH))
        for path in _MATH.rglob("binary_*")
        if path.suffix in {".cc", ".h"} and path.relative_to(_MATH).parts[0] != "binary"
    ]
    assert not misplaced, f"Binary files outside binary subtree: {sorted(misplaced)}"
