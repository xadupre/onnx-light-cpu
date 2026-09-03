# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the project dependency metadata."""

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_onnx_light_dev_dependency_version():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["dev"]
    onnx_light = [
        dependency for dependency in dependencies if dependency.startswith("onnx-light @")
    ]

    assert len(onnx_light) == 3
    assert all("/0.1.24/onnx_light-0.1.24-" in dependency for dependency in onnx_light)
