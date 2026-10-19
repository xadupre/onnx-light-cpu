# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the math implementation's operation/ISA layout."""

import re
from pathlib import Path

_COMPUTE = Path(__file__).resolve().parents[2] / "ml_cpu_compute"
_MATH = _COMPUTE / "src" / "math"


def test_compute_names_are_runtime_independent():
    files = [
        path for path in _COMPUTE.rglob("*") if path.suffix in {".cc", ".h", ".hpp", ".cmake"}
    ]
    misplaced = [
        str(path.relative_to(_COMPUTE))
        for path in files
        if "kernel" in str(path.relative_to(_COMPUTE)).lower()
        or re.search(r"\bnamespace\s+onnx_light_cpu\b", path.read_text(encoding="utf-8"))
    ]
    assert not misplaced, f"Runtime-specific computation names: {sorted(misplaced)}"


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
