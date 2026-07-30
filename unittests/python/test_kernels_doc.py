# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

from onnx_kernels import _group_by_operator  # noqa: E402


class TestGroupByOperator:
    def test_one_row_per_operator(self):
        kernels = [
            ("Abs", "float32", "AbsFloat32"),
            ("Abs", "float64", "AbsFloat64"),
            ("Abs", "int32", "AbsInt32"),
            ("Abs", "int64", "AbsInt64"),
        ]
        grouped = _group_by_operator(kernels)
        # A single operator must not be multiplied into several rows.
        assert list(grouped) == ["Abs"]
        assert grouped["Abs"] == [
            ("float32", "AbsFloat32"),
            ("float64", "AbsFloat64"),
            ("int32", "AbsInt32"),
            ("int64", "AbsInt64"),
        ]

    def test_multiple_operators_preserve_order(self):
        kernels = [
            ("Abs", "float32", "AbsFloat32"),
            ("Neg", "float32", "NegFloat32"),
            ("Abs", "int32", "AbsInt32"),
        ]
        grouped = _group_by_operator(kernels)
        assert list(grouped) == ["Abs", "Neg"]
        assert grouped["Abs"] == [("float32", "AbsFloat32"), ("int32", "AbsInt32")]
        assert grouped["Neg"] == [("float32", "NegFloat32")]
