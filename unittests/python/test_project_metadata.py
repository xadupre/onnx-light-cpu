# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the Python project metadata."""

from pathlib import Path


def test_onnx_light_version():
    root = Path(__file__).resolve().parents[2]
    metadata = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"onnx-light==0.1.20"' in metadata
