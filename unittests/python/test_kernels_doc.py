# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

from kernel_scan import group_by_operator, operator_function  # noqa: E402


class TestGroupByOperator:
    def test_one_row_per_operator(self):
        kernels = [
            ("Abs", "float32", "AbsFloat32"),
            ("Abs", "float64", "AbsFloat64"),
            ("Abs", "int32", "AbsInt32"),
            ("Abs", "int64", "AbsInt64"),
        ]
        grouped = group_by_operator(kernels)
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
        grouped = group_by_operator(kernels)
        assert list(grouped) == ["Abs", "Neg"]
        assert grouped["Abs"] == [("float32", "AbsFloat32"), ("int32", "AbsInt32")]
        assert grouped["Neg"] == [("float32", "NegFloat32")]


class TestOperatorFunction:
    def test_single_function_per_operator(self):
        # Each operator exposes a single numpy-like function (e.g. Abs -> abs),
        # not one function per data type.
        assert operator_function("Abs") == "abs"
        assert operator_function("Neg") == "neg"
